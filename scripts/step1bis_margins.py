# ANALISI DESCRITTIVA LOCALE, NON CONGELATA (D5a resta com'e'): margini del vicino piu prossimo e norma del residuo, per ramo e livello RVQ, e legame coi flip sotto rumore.
# Richiede l'ambiente ~/.venvs/wearusfm-tok (torch CPU, einops, pyyaml), il clone di NeuroRVQ (commit 926e770) e il
# checkpoint NeuroRVQ_EMG_tokenizer_v1.pt (sha256 0d255bcc...dce49) e gli array del run 59048369. I percorsi
# in testa allo script sono quelli dello scratchpad della sessione: vanno adattati.
"""Margini del vicino piu' prossimo e norma del residuo, per ramo e livello RVQ, e loro legame con i flip sotto rumore.
Replica il calcolo dei codici SENZA chiamare i quantizzatori (che in eval aggiornano un buffer statistico)."""
import sys, json
from pathlib import Path
import numpy as np, torch
from einops import rearrange
sys.path.insert(0, '/Users/simonebenatti/dev/wearusfm/src')
from wearusfm.tokenizer_checks.neurorvq import NeuroRVQRunner
S = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/57944b46-716a-4677-b1f7-a66f65922403/scratchpad')
runner = NeuroRVQRunner(S / 'neurorvq_repo', S / 'models' / 'NeuroRVQ_EMG_tokenizer_v1.pt', device='cpu')
m = runner.model
torch.set_num_threads(8)
rep = json.load(open('/Users/simonebenatti/dev/wearusfm/results/step1bis/step1bis_59048369.json'))

def trace(x, branches=(0, 1, 2, 3)):
    """x: (B, 1, 3200) float32. Ritorna per ramo: codici (16, n), margine d2-d1 (16, n), norma del residuo (16, n)."""
    b = x.shape[0]
    t_ix, s_ix = runner._indices([0], 16, b)
    x4 = rearrange(torch.from_numpy(x), 'B N (A T) -> B N A T', T=200).contiguous()
    with torch.no_grad():
        f = m.encoder(x4, temporal_embedding_ix=t_ix, spatial_embedding_ix=s_ix, return_patch_tokens=True)
        out = {}
        for k in branches:
            head = getattr(m, f'encode_task_layer_{k+1}')
            q = head(f[k].float())
            q = rearrange(q, 'b (h w) c -> b c h w', h=1, w=16).contiguous()
            residual = q
            codes, margins, norms = [], [], []
            for layer in getattr(m, f'quantize_{k+1}').layers:
                z = rearrange(residual, 'b c h w -> b h w c')
                norms.append(z.reshape(-1, z.shape[-1]).norm(dim=-1))
                zn = z / (z.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                flat = zn.reshape(-1, zn.shape[-1])
                w = layer.embedding.weight
                d = flat.pow(2).sum(1, keepdim=True) + w.pow(2).sum(1) - 2 * flat @ w.T
                top = d.topk(2, dim=1, largest=False)
                codes.append(top.indices[:, 0]); margins.append(top.values[:, 1] - top.values[:, 0])
                zq = w[top.indices[:, 0]].reshape(zn.shape)
                residual = residual - rearrange(zq, 'b h w c -> b c h w')
            out[k] = (torch.stack(codes).numpy(), torch.stack(margins).numpy(), torch.stack(norms).numpy())
    return out

z = np.load(S / 'step1bis_arrays' / 'emg2pose.npz')
wins = z['tokens'].reshape(-1, 3200).astype(np.float32)
sel = np.arange(0, len(wins), 6)[:200]
X = wins[sel][:, None, :]
floor = rep['v4_per_dataset']['emg2pose']['noise_floor_rms']
rng = np.random.default_rng(0)
Xn = X + rng.normal(scale=floor, size=X.shape).astype(np.float32)
clean, noisy = trace(X), trace(Xn)
# controllo: la mia replica dei codici coincide con quelli salvati dal run su A100 (stesse finestre)
tok_idx = np.concatenate([np.arange(w * 16, w * 16 + 16) for w in sel])
agree = [(clean[k][0] == z['codes'][k][:, tok_idx]).mean() for k in range(4)]
print('accordo replica-vs-A100 (tutti i livelli), per ramo:', np.round(agree, 4))
print('\nramo livello | norma residuo mediana | margine d2-d1: 5% / 25% / mediana | flip sotto rumore: tutti / margine nel 25% basso / margine nel 25% alto')
for k in range(4):
    c, mg, nr = clean[k]; cn = noisy[k][0]
    for l in (0, 1, 2, 4, 8, 15):
        flip = (c[l] != cn[l])
        lo = mg[l] <= np.quantile(mg[l], 0.25); hi = mg[l] >= np.quantile(mg[l], 0.75)
        print(f'  {k}   {l:2d}   | {np.median(nr[l]):8.3f} | {np.quantile(mg[l],0.05):.4f} / {np.quantile(mg[l],0.25):.4f} / {np.median(mg[l]):.4f} | {flip.mean():.3f} / {flip[lo].mean():.3f} / {flip[hi].mean():.3f}')
