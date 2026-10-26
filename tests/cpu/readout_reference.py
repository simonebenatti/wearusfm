"""Riferimento d'uguaglianza dell'estrazione del readout (seconda ondata del pilot, `docs/pilot_posa_v1.md` §3, condizione (b1)).

Il file `tests/cpu/data/readout_reference_705ec04.npz` contiene i pesi di un modello piccolo (seme 0) e le uscite `pool`, `local`, `dec` di
`harness.channel_readout.extract_channel_readouts` su 6 finestre casuali (seme 0) del montaggio Myo a 200 Hz. E' stato scritto il 26/10/2026 col
codice di `src/wearusfm/` identico a 705ec04 (il checkout di campagna del pilot: `git diff 705ec04 -- src/wearusfm scripts/channel_readout.py`
vuoto), PRIMA di aggiungere la testa di posa al modello. Il test `test_readout_reference.py` pretende che il codice corrente riproduca quelle
uscite bit per bit: e' la garanzia che R3 si confronta con l'R0 del pilot attraverso la stessa estrazione.

Per riscriverlo (solo con una decisione registrata, perche' cambia il riferimento):  python3 tests/cpu/readout_reference.py --write
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "data" / "readout_reference_705ec04.npz"
CODE_SHA = "705ec0465420e80d5d818a8bbc7acc3a92c06efc"
LAYERS = ("pool", "local", "dec")


def tiny_config():
    from wearusfm.model.fm import FMConfig

    return FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4)


def windows():
    rng = np.random.default_rng(0)
    return rng.normal(size=(6, 1000, 8)) * np.linspace(0.5, 2.0, 8)[None, None, :], np.array(["a", "a", "a", "b", "b", "b"])


def extract(model) -> dict:
    from wearusfm.harness import channel_readout as CR
    from wearusfm.harness import fm_features as FF

    w, subjects = windows()
    out = CR.extract_channel_readouts(model, w, 200.0, FF.myo8_montage("epn612", 200.0), subjects, batch=4)
    return {k: out[k] for k in LAYERS}


def fresh_model():
    import torch

    from wearusfm.model.fm import WearUsFM

    torch.manual_seed(0)
    return WearUsFM(tiny_config()).eval()


def state_bytes(model) -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    return buf.getvalue()


def load_reference() -> tuple[dict, dict]:
    """(state_dict del modello di riferimento, {strato: uscite})."""
    import torch

    z = np.load(REFERENCE, allow_pickle=False)
    state = torch.load(io.BytesIO(z["state_dict"].tobytes()), map_location="cpu", weights_only=True)
    return state, {k: z[k] for k in LAYERS}


def write() -> Path:
    model = fresh_model()
    out = extract(model)
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REFERENCE, state_dict=np.frombuffer(state_bytes(model), dtype=np.uint8), code_sha=np.asarray(CODE_SHA), **out)
    return REFERENCE


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="scrive (o sovrascrive) il riferimento")
    args = ap.parse_args()
    if not args.write:
        ap.error("serve --write")
    p = write()
    print(f"scritto {p} ({p.stat().st_size} byte)")
