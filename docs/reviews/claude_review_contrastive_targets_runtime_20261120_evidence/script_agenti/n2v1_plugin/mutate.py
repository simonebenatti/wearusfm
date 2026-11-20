import sys
from pathlib import Path
root = Path(sys.argv[1]); which = set(sys.argv[2].split(","))
M = {
 "M1": ("scripts/contrastive_target_resume.py", "    if ranks > 1:\n        rank_rng = state.get(\"rank_rng\")", "    if False:\n        rank_rng = state.get(\"rank_rng\")"),
 "M2": ("scripts/contrastive_target_resume.py", "expected_rng = fork[\"rank_rng\"][rank] if ranks > 1 else {", "expected_rng = fork[\"rank_rng\"][0] if ranks > 1 else {"),
 "M3": ("scripts/contrastive_run.py", "checkpoint.name != \"checkpoint_step0000544.pt\"", "checkpoint.name == \"checkpoint_step0000544.pt\""),
 "M4": ("scripts/contrastive_target_resume.py", "**{key: data[key] for key in fields if key in data}}) + \"\\n\")", "**{key: data[key] for key in fields}}) + \"\\n\")\n            raise RuntimeError(\"M4 DDP branch reached\")"),
 "M5": ("scripts/contrastive_run.py", "                torch.distributed.barrier()\n", "                raise RuntimeError(\"M5 runner reload path reached\")\n"),
}
for key in sorted(which):
    path, old, new = M[key]
    p = root / path; text = p.read_text()
    assert text.count(old) == 1, (key, text.count(old))
    p.write_text(text.replace(old, new)); print("applied", key, path)
