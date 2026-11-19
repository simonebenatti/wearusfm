"""Call the frozen diagnostics' entry checks from a given tree (no GPU, no Leonardo paths)."""
import importlib.util, json, sys, traceback
from pathlib import Path
root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
ck = load("diag_ck", "scripts/diagnostics/contrastive_common_checkpoint.py")
print("tree source_digest:", ck.source_digest(), "driver SOURCE:", ck.SOURCE)
cu = load("diag_cu", "scripts/diagnostics/contrastive_common_state_cuda.py")
try:
    cu.read_inputs()
    print("cuda.read_inputs: returned (unexpected)")
except Exception as exc:
    print("cuda.read_inputs ->", type(exc).__name__ + ":", exc)
target = out / "resume_CN"
target.mkdir(parents=True, exist_ok=False)
try:
    ck.run("CN", target)
    print("checkpoint.run: returned (unexpected)")
except Exception as exc:
    print("checkpoint.run ->", type(exc).__name__ + ":", exc)
print("files written in out:", sorted(str(p.relative_to(out)) for p in out.rglob("*")))
