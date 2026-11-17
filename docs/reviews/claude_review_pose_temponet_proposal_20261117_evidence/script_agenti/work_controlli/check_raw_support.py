"""Controllo ESEGUITO (indici sintetici): blocchi a 200 Hz assegnati alle patch da window_pose (snapshot c15aac9) e passato raw di TEMPONet
(256 campioni, worktree claude-temponet a 8917e2a) per i centri del crop6."""
import sys
from types import SimpleNamespace
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9"
sys.path.insert(0, SNAP + "/src")
import numpy as np
from wearusfm.data import pose_targets as PT

FS, PATCH_MS, NP_, CROP, WINDOW = 2000.0, 25.0, 160, 6, 256
n_samples = 40000
n_blocks = n_samples // PT.BLOCK
# la posa del blocco j vale j: la media per patch rivela quali blocchi entrano
blocks = np.repeat(np.arange(n_blocks, dtype=np.float64)[:, None], PT.N_JOINTS, axis=1)
valid = np.ones(n_blocks, dtype=bool)
mean, std = np.zeros(PT.N_JOINTS, np.float32), np.ones(PT.N_JOINTS, np.float32)
worst = {"min_past": 10**9, "max_future": -10**9, "counts": set(), "min_past_sample": 10**9, "max_future_sample": -10**9}
for start in range(5000, 5000 + 400):  # tutti gli sfasamenti rispetto alla griglia dei blocchi (e oltre)
    win = SimpleNamespace(trial=None, start=start, n_patches=NP_)
    bounds = PT.patch_bounds(NP_, PATCH_MS, FS) + start
    assert (np.diff(bounds) == 50).all()
    target, mask = PT.window_pose(blocks, valid, win, FS, PATCH_MS, mean, std)
    j = np.arange(n_blocks)
    centers = j * PT.BLOCK + PT.BLOCK / 2.0
    p_of = np.searchsorted(bounds, centers, side="right") - 1
    for t in range(CROP, NP_ - CROP):
        jj = j[(p_of == t) & (centers >= bounds[0]) & (centers < bounds[-1])]
        worst["counts"].add(len(jj))
        assert abs(target[t, 0] - jj.mean()) < 1e-3, (start, t)
        # (a) un endpoint per blocco, all'ultimo campione del blocco (bersaglio del training raw): finestra [10j+9-255, 10j+9]
        first_a, last_a = int(jj.min()) * 10 + 9 - (WINDOW - 1), int(jj.max()) * 10 + 9
        # (b) un endpoint per campione nativo del blocco: finestre [s-255, s], s in [10j, 10j+9]
        first_b = int(jj.min()) * 10 - (WINDOW - 1)
        worst["min_past"] = min(worst["min_past"], first_a - start)
        worst["min_past_sample"] = min(worst["min_past_sample"], first_b - start)
        worst["max_future"] = max(worst["max_future"], last_a - (start + NP_ * 50 - 1))
print("blocchi per patch centrale (tutti validi):", sorted(worst["counts"]))
print("primo campione raw usato meno inizio finestra, endpoint a fine blocco: minimo", worst["min_past"], "campioni (>= 0: dentro la finestra)")
print("primo campione raw usato meno inizio finestra, endpoint a ogni campione: minimo", worst["min_past_sample"], "campioni")
print("ultimo campione raw usato meno ultimo campione della finestra: massimo", worst["max_future"], "campioni (<= 0: dentro la finestra)")
print("margine del crop6:", CROP * 50, "campioni =", CROP * 25, "ms; passato di TEMPONet:", WINDOW - 1, "campioni =", (WINDOW - 1) / 2, "ms")
print("endpoint per patch: a fine blocco", 5, "; a ogni campione nativo", 50, "; rapporto", 50 // 5)
# la griglia degli endpoint del training raw
sys.path.insert(0, "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/src")
from wearusfm.harness import emg2pose_regression as R
jb = R.train_blocks(valid, n_samples)
ends = R.block_end(jb)
print("training raw: blocchi j multipli di", R.TRAIN_STRIDE_BLOCKS, "; fine finestra = 10 j + 9; (fine mod 10) =", sorted(set((ends % 10).tolist())), "; prima fine", int(ends[0]))
inst = R.eval_instants(np.array([1234]))
print("lettura ufficiale: istanti = inizio + 255 + 10 m; (istante mod 10) per inizio 1234 =", sorted(set((inst % 10).ravel().tolist())), "; passo", R.EVAL_STEP, "campioni =", FS / R.EVAL_STEP, "Hz")
