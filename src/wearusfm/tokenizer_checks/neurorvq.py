"""Involucro sottile attorno al tokenizer EMG di NeuroRVQ (repo di terzi, licenza CC BY-NC 4.0,
commit 926e770): carica il checkpoint congelato e restituisce, per un batch, il dominio
standardizzato (originale e ricostruito) e i codici RVQ. Nessuna riga di codice del repo e'
copiata qui: si importa dal clone, che deve stare su Leonardo FUORI da questo repo
(`repo_dir`). torch/einops/yaml si importano dentro le funzioni: il modulo si importa anche
senza torch (test locali sulla parte numpy).

Non si puo' eseguire sul Mac (niente torch, niente GPU): la prova e' su Leonardo.

Fatti dal codice (fatto n. 4 in docs/fatti_da_verificare.md):
- `forward` restituisce (std_x, std_xrec): std_x e' (B, N*A, T) mentre std_xrec e' (B, N, A, T)
  (lo `squeeze(2)` non fa nulla se A > 1); `tokens_from_forward` porta entrambi a (B*N*A, T)
  nello stesso ordine (b, canale, patch).
- `get_tokens` del repo fa `.view(8, ...)` (copiato dall'EEG) ma qui i livelli sono 16: non lo
  si usa; si chiama `encode` e si impilano i codici a mano.
- I quantizzatori hanno `kmeans_init=True`: se il buffer `initted` non fosse True dopo il
  caricamento, il primo forward rifarebbe il k-means sui dati e altererebbe il codebook.
  `load_tokenizer` lo verifica e si ferma.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PATCH = 200
MAX_PATCHES = 256  # n_patches di flags/NeuroRVQ_EMG_v1.yml
SPATIAL_INDEX_FIXED = 0  # V1: un solo indice spaziale fisso (il primo dei 16 elettrodi globali)


def tokens_from_forward(std_x: np.ndarray, std_xrec: np.ndarray, n_channels: int, n_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Porta gli output di `forward` a due array (B*n_channels*n_time, 200) nello stesso
    ordine dei token (batch, canale, patch)."""
    b = std_x.shape[0]
    n_tok = b * n_channels * n_time
    if std_x.shape != (b, n_channels * n_time, PATCH):
        raise ValueError(f"std_x forma inattesa {std_x.shape}")
    if std_xrec.shape not in ((b, n_channels, n_time, PATCH), (b, n_channels * n_time, PATCH)):
        raise ValueError(f"std_xrec forma inattesa {std_xrec.shape}")
    return std_x.reshape(n_tok, PATCH), std_xrec.reshape(n_tok, PATCH)


def stack_codes(code_ind: list[np.ndarray], n_tokens: int) -> np.ndarray:
    """Lista di 4 rami, ciascuno (16, n_tokens) -> (4, 16, n_tokens) int32."""
    out = np.stack([np.asarray(c) for c in code_ind]).astype(np.int32)
    if out.shape != (4, 16, n_tokens):
        raise ValueError(f"codici con forma inattesa {out.shape}, attesa (4, 16, {n_tokens})")
    return out


class NeuroRVQRunner:
    def __init__(self, repo_dir: str | Path, checkpoint: str | Path, device: str = "cuda"):
        import torch
        import yaml

        repo_dir = Path(repo_dir)
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        from inference.modules.NeuroRVQ_EMG_tokenizer_inference_modules import ch_names_global, create_embedding_ix
        from NeuroRVQ_EMG.NeuroRVQ import NeuroRVQTokenizer
        from NeuroRVQ_EMG.NeuroRVQ_modules import get_encoder_decoder_params

        self.torch = torch
        self.device = torch.device(device)
        self.ch_names_global = ch_names_global
        self._create_embedding_ix = create_embedding_ix
        args = yaml.safe_load((repo_dir / "flags" / "NeuroRVQ_EMG_v1.yml").read_text())
        args["n_global_electrodes"] = len(ch_names_global)
        enc_cfg, dec_cfg = get_encoder_decoder_params(args)
        model = NeuroRVQTokenizer(
            enc_cfg, dec_cfg, n_code=args["n_code"], code_dim=args["code_dim"], decoder_out_dim=args["decoder_out_dim"]
        )
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state)  # strict: si ferma se le chiavi non coincidono
        model.to(self.device).eval()
        for name, mod in model.named_modules():
            if hasattr(mod, "initted") and not bool(mod.initted):
                raise RuntimeError(f"{name}: buffer `initted` falso dopo il caricamento (rischio di k-means sui dati)")
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.args = args

    def _indices(self, spatial_idx: list[int], n_time: int, batch: int):
        names = np.array([self.ch_names_global[i] for i in spatial_idx])
        t_ix, s_ix = self._create_embedding_ix(n_time, MAX_PATCHES, names, self.ch_names_global)
        t_ix = t_ix.int().to(self.device).expand(batch, -1).contiguous()
        s_ix = s_ix.int().to(self.device).expand(batch, -1).contiguous()
        return t_ix, s_ix

    def run(self, x: np.ndarray, spatial_idx: list[int], *, want_recon: bool = True):
        """x: (B, n_ch, n_time*200) nella vista canonica scalata; spatial_idx: indice spaziale
        (posizione in `ch_names_global`) per ciascuno degli n_ch canali. Ritorna un dict con
        `codes` (4, 16, B*n_ch*n_time) e, se want_recon, `std_x` e `std_xrec` (B*n_ch*n_time, 200)."""
        torch = self.torch
        from einops import rearrange

        b, n_ch, length = x.shape
        if length % PATCH or len(spatial_idx) != n_ch:
            raise ValueError(f"forma {x.shape} o {len(spatial_idx)} indici spaziali incoerenti")
        n_time = length // PATCH
        if n_ch * n_time > MAX_PATCHES:
            raise ValueError(f"{n_ch * n_time} token > {MAX_PATCHES}")
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        t_ix, s_ix = self._indices(spatial_idx, n_time, b)
        out = {}
        with torch.no_grad():
            x4 = rearrange(xt, "B N (A T) -> B N A T", T=PATCH).contiguous()
            _, code_ind, _, _ = self.model.encode(x4, t_ix, s_ix)
            out["codes"] = stack_codes([c.detach().cpu().numpy() for c in code_ind], b * n_ch * n_time)
            if want_recon:
                std_x, std_xrec = self.model(xt, t_ix, s_ix)
                out["std_x"], out["std_xrec"] = tokens_from_forward(
                    std_x.detach().cpu().numpy(), std_xrec.detach().cpu().numpy(), n_ch, n_time
                )
        return out
