"""Controllo ESEGUITO: griglia degli endpoint del training e della lettura ufficiale di TEMPONet raw (worktree claude-temponet, 8917e2a)."""
import importlib.util
import numpy as np
p = "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/src/wearusfm/harness/emg2pose_regression.py"
spec = importlib.util.spec_from_file_location("emg2pose_regression", p)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
n_samples = 40000
valid = np.ones(n_samples // R.BLOCK, dtype=bool)
jb = R.train_blocks(valid, n_samples)
ends = R.block_end(jb)
print("training raw: blocchi j multipli di", R.TRAIN_STRIDE_BLOCKS, "; fine finestra = 10 j + 9; (fine mod 10) =", sorted(set((ends % 10).tolist())),
      "; prime fini", ends[:3].tolist(), "; finestra", R.WINDOW, "campioni, contesto", R.CONTEXT)
for start in (1230, 1234, 1237):
    inst = R.eval_instants(np.array([start]))
    print("lettura ufficiale, inizio", start, ": istanti = inizio + 255 + 10 m; (istante mod 10) =", sorted(set((inst % 10).ravel().tolist())),
          "; istanti per finestra", inst.shape[1], "; passo", R.EVAL_STEP, "campioni =", R.FS_HZ / R.EVAL_STEP, "Hz")
