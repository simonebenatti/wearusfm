import json, os, sys
from pathlib import Path
CLONE = Path(os.environ["CLONE"])
sys.path[:0] = [str(CLONE / p) for p in ("src", "scripts", "scripts/diagnostics")]
import torch
import contrastive_cn_restart_compare as C
import contrastive_cn_restart_observe as O

base = Path(sys.argv[1]); res = base / "res"
F = int(os.environ.get("FORK", 5)); N = int(os.environ.get("STEPS", 9))
def R(mode, rank=0): return json.loads((res / f"{mode}.rank{rank}.json").read_text())
def ck(mode): return torch.load(str(base / mode / "checkpoint.pt"), map_location="cpu", weights_only=False)

out = {}
# per-update reduced gradient hashes (rank 0). cont/hookcont: updates 1..N; resumes: F+1..N
cont = R("cont")["clip_log"]; hook = R("hookcont")["clip_log"]
print("summary steps", {m: R(m)["summary"]["steps"] for m in ("cont","plain","ordinary","warm","hookcont")})
print("Q1a hookcont vs cont per update grad equal:", [a["grad"] == b["grad"] for a, b in zip(cont, hook)])
for m in ("plain", "ordinary", "warm"):
    log = R(m)["clip_log"]
    eq = [a["grad"] == b["grad"] for a, b in zip(cont[F:], log)]
    first = next((F + 1 + i for i, e in enumerate(eq) if not e), None)
    nd = [sum(x != y for x, y in zip(a["grads"], b["grads"])) for a, b in zip(cont[F:], log)]
    print(f"{m}: vs cont per update (F+1..N) equal={eq} first_diff_update={first} n_param_tensors_diff={nd}")
for a, b in (("plain", "ordinary"),):
    print(f"Q1b {a} vs {b} per update grad equal:", [x["grad"] == y["grad"] for x, y in zip(R(a)["clip_log"], R(b)["clip_log"])])
# all ranks agree within leg
for m in ("ordinary", "warm", "cont"):
    print(m, "ranks agree:", all(R(m, r)["clip_log"] == R(m, 0)["clip_log"] for r in range(4)))
# endpoints
refs = {m: ck(m) for m in ("cont", "plain", "ordinary", "warm", "hookcont")}
for a, b in (("cont", "hookcont"), ("plain", "ordinary"), ("cont", "plain"), ("cont", "ordinary"), ("cont", "warm"), ("ordinary", "warm")):
    d = C.checkpoint_differences(refs[a], refs[b])
    print(f"endpoint {a} vs {b}: exact={d['exact_payload']} nfloat={len(d['floating_tensors_different'])} meta={d['metadata_or_schema_different'][:6]}")
print("ddp logs:")
for m in ("cont", "plain", "ordinary", "warm", "hookcont"):
    print(" ", m, R(m)["ddp"])
# observations
recs = {}
for m in ("ordinary", "warm"):
    recs[m] = [O.load_observation(base / m, r, expected_parameters=106) for r in range(4)]
    print(m, "buckets", [(b["index"], b["nbytes"], b["dtype"], len(b["parameter_ids"])) for b in recs[m][0][0]["buckets"]])
logged = R("cont")["ddp"]
try:
    lay = C.layouts(recs["ordinary"][0][0], recs["warm"][0][0], logged)
    print("layouts OK")
except Exception as e:
    print("layouts FAILED:", type(e).__name__, e)
any_local = any_reduced = excessive = False
maxrel = 0
for r in range(4):
    for pid in range(106):
        st = {s: C.compare_vector(recs["ordinary"][r][1][s][pid], recs["warm"][r][1][s][pid]) for s in ("local", "reduced")}
        any_local |= bool(st["local"]["different_elements"])
        any_reduced |= st["reduced"]["difference_l2"] > 0
        excessive |= st["reduced"]["relative_l2"] > C.LIMIT
        maxrel = max(maxrel, st["reduced"]["relative_l2"])
print("any_local", any_local, "any_reduced", any_reduced, "excessive", excessive, "max rel", maxrel)
# warm reduced at F+1 vs cont reduced at F+1 (clip-log hash of the actual preclip gradients)
w = R("warm")["clip_log"][0]; o = R("ordinary")["clip_log"][0]
print("warm first update == cont update F+1:", w["grad"] == cont[F]["grad"], "ordinary first == cont:", o["grad"] == cont[F]["grad"])
d = C.checkpoint_differences(refs["cont"], refs["warm"])
print("classify:", C.classify(any_local, any_reduced, excessive, d["exact_payload"], w["grad"] == cont[F]["grad"]))
