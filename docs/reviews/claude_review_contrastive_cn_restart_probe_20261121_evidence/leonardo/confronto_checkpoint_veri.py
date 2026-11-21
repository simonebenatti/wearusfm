import collections, json, sys, time
from collections.abc import Mapping
import torch
import numpy as np
R = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118"
P = {"continuo": R + "/benchmark/CN/checkpoint.pt", "A": R + "/reloads/CN_A/checkpoint.pt", "B": R + "/reloads/CN_B/checkpoint.pt"}

def identical(a, b):
    """Strict IEEE bytes, including signed zero, after dtype/shape checks."""
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)))


def checkpoint_differences(reference, measured):
    """Compare all payload fields and leaves. File container metadata is outside this comparison."""
    changed, invalid = [], []

    def visit(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                invalid.append(path)
            elif a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                invalid.append(path)
            elif not identical(a, b):
                kind = changed if a.is_floating_point() and not path.endswith(".step") else invalid
                kind.append(path)
        elif isinstance(a, Mapping):
            if not isinstance(b, Mapping) or a.keys() != b.keys():
                invalid.append(path)
                return
            for k in a:
                visit(a[k], b[k], f"{path}.{k}" if path else str(k))
        elif isinstance(a, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                invalid.append(path)
                return
            for i, value in enumerate(a):
                visit(value, b[i], f"{path}.{i}")
        elif type(a) is not type(b) or a != b:
            invalid.append(path)

    visit(reference, measured, "")
    return {"floating_tensors_different": changed, "metadata_or_schema_different": invalid,
            "exact_payload": not changed and not invalid}

def load(p):
    return torch.load(str(p), map_location="cpu", weights_only=False, mmap=True)

def leaf_types(o, acc, path=""):
    if isinstance(o, torch.Tensor):
        acc["Tensor:" + str(o.dtype)] += 1
    elif isinstance(o, Mapping):
        for k, v in o.items(): leaf_types(v, acc, path + "." + str(k))
    elif isinstance(o, (tuple, list)):
        for i, v in enumerate(o): leaf_types(v, acc, path + "." + str(i))
    else:
        acc[type(o).__module__ + "." + type(o).__name__] += 1
        if isinstance(o, np.ndarray): acc["NUMPY_AT:" + path] += 1

t0 = time.time()
print("torch", torch.__version__)
cont = load(P["continuo"]); a = load(P["A"])
print("chiavi continuo:", sorted(cont.keys()))
print("chiavi A:", sorted(a.keys()))
for name, o in (("continuo", cont), ("A", a)):
    acc = collections.Counter(); leaf_types(o, acc)
    print("foglie", name, dict(acc))
for k in ("step", "world_size", "alarm_counts", "data_signature", "pose_stats_sha256", "contrastive_initial_sha256"):
    print(k, "continuo:", repr(cont.get(k))[:160], "| A:", repr(a.get(k))[:160])
print("stream tipo:", type(cont["stream"]).__name__, "| repr:", repr(cont["stream"])[:300])
print("torch_rng uguale:", torch.equal(cont["torch_rng"], a["torch_rng"]))
print("cuda_rng tipo:", type(cont["cuda_rng"]).__name__, "len", len(cont["cuda_rng"]) if cont["cuda_rng"] is not None else None,
      "| uguali per dispositivo:", [bool(torch.equal(x, y)) for x, y in zip(cont["cuda_rng"], a["cuda_rng"])])
print("rank_rng tipo:", type(cont["rank_rng"]).__name__, "chiavi[0]:", sorted(cont["rank_rng"][0].keys()))
for label, x, y in (("continuo vs A", cont, a),):
    try:
        d = checkpoint_differences(x, y)
        print(label, "| exact_payload", d["exact_payload"], "| tensori float diversi", len(d["floating_tensors_different"]),
              "| metadata_or_schema_different", d["metadata_or_schema_different"][:40], "n", len(d["metadata_or_schema_different"]))
        ch = d["floating_tensors_different"]
        pref = collections.Counter(p.split(".")[0] + ("." + p.split(".")[1] if p.split(".")[0] in ("optimizer", "contrastive_state") else "") for p in ch)
        print("   tensori diversi per parte:", dict(pref))
    except Exception as e:
        print(label, "| ECCEZIONE", type(e).__name__, str(e)[:300])
del cont
b = load(P["B"])
try:
    d = checkpoint_differences(a, b)
    print("A vs B | exact_payload", d["exact_payload"], "| float diversi", len(d["floating_tensors_different"]), "| metadati diversi", d["metadata_or_schema_different"][:20])
except Exception as e:
    print("A vs B | ECCEZIONE", type(e).__name__, str(e)[:300])
print("secondi", round(time.time() - t0, 1))
