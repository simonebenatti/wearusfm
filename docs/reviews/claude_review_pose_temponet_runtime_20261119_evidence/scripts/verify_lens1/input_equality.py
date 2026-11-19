"""L1-1 side claim: the two eval input paths give bitwise-identical FP32 inputs (cost-only defect)."""
import numpy as np, torch
src = open("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/poseheads/agents/verify_lens1/eval_paths.py").read()
exec(src.split("def prod():")[0])
same = all(torch.equal(torch.from_numpy(D.view_inputs(cache, "post", held[s:s+32])),
                       validation.features[s:s+32].to(dtype=torch.float32)) for s in range(0, len(held), 32))
print("bitwise-identical FP32 eval inputs, all", len(range(0, len(held), 32)), "batches:", same)
pad = int((~(np.arange(160)[None] < lengths[held][:, None])).sum())
print("padded positions in held fixture:", pad)
