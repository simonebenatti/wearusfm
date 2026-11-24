"""Quante query delle loss mascherate restano con raggio 0 e con raggio 2, per tipo di maschera: generatore vero, quote D10 del preset, sei
montaggi veri dei dataset, 1.080 finestre da 1, 2,5 e 4 s (nessun segnale, nessun dato). Seme 0.

Uso, dalla radice del repository: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python \\
    results/infra/tokenizer_default_20261124/raggio_maschere.py
"""
import inspect
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from wearusfm.ingest import emg2pose, emg2qwerty, grabmyo, ninapro_db5, ninapro_std, putemg  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.jepa import core_hidden  # noqa: E402
from wearusfm.training.masking import KIND_NAMES, generate_mask  # noqa: E402


def montage(name):
    if name == "emg2pose":
        return emg2pose.build_montage_metadata("u1", "s", "left")
    if name == "putemg":
        return putemg.build_montage_metadata(1, "a", "b")
    if name == "grabmyo":
        return grabmyo.build_montage_metadata(1, 1)
    if name == "db5":
        return ninapro_db5.build_montage_metadata(1, "right")
    if name == "db2":
        return ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "r")
    build = emg2qwerty.build_montage_metadata
    return build(*(["u1", "s", "x"][:len(inspect.signature(build).parameters)]))


def main():
    spec = R.sanity_config().loader.mask
    print("quote", spec.share_short, spec.share_medium, spec.share_long, spec.share_spatial, "frazione", spec.ratio)
    counts, by_length, windows, empty = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), 0, 0
    rng = np.random.default_rng(0)
    for name in ("emg2qwerty", "emg2pose", "putemg", "grabmyo", "db2", "db5"):
        meta = montage(name)
        d = montage_to_dict(meta, [True] * meta.n_channels)
        layout, compartment = CC.layout_from_montage(d), np.asarray(CC.anatomy_codes(d).compartment_weights).argmax(axis=1)
        for n in (40, 100, 160):  # passi da 25 ms
            for _ in range(60):
                visible, kind = generate_mask(layout, compartment, n, n, False, spec, rng)
                vis = torch.as_tensor(visible[:, :n])
                hidden = ~vis
                core = core_hidden(hidden, vis, 2)
                windows += 1
                empty += int(core.sum() == 0)
                by_length[n][0] += int(hidden.sum())
                by_length[n][1] += int(core.sum())
                for code, label in KIND_NAMES.items():
                    of_kind = torch.as_tensor(kind[:, :n] == code) & hidden
                    counts[label][0] += int(of_kind.sum())
                    counts[label][1] += int((of_kind & core).sum())
    total, kept = (sum(v[i] for v in counts.values()) for i in (0, 1))
    for label, (h, c) in counts.items():
        if h:
            print(f"{label:9s} nascosti {h:8d}  query con raggio 2 {c:8d} = {100 * c / h:5.1f}%  quota sulle query {100 * h / total:4.1f}% -> {100 * c / kept:4.1f}%")
    print(f"totale    nascosti {total}  query con raggio 2 {kept} = {100 * kept / total:.1f}%")
    print("per lunghezza (passi)", {n: round(100 * c / h, 1) for n, (h, c) in by_length.items()})
    print("finestre", windows, "senza query con raggio 2:", empty)


if __name__ == "__main__":
    main()
