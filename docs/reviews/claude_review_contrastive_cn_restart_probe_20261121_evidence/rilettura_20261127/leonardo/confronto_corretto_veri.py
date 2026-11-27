import collections, json, sys, time
from collections.abc import Mapping
import torch
RANKS = 4
PROCESS_ONLY_CUDA_RNG = ("cuda_rng.1", "cuda_rng.2", "cuda_rng.3")
R = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118"
P = {"continuo": R + "/benchmark/CN/checkpoint.pt", "A": R + "/reloads/CN_A/checkpoint.pt", "B": R + "/reloads/CN_B/checkpoint.pt"}

def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def validate_rng_schema(payload):
    """Check the actual state format before interpreting a difference as an unused generator."""
    def check_bytes(value, path):
        require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1
                and value.numel() > 0, f"invalid RNG state at {path}")

    check_bytes(payload.get("torch_rng"), "torch_rng")
    for key in ("cuda_rng", "rank_rng"):
        values = payload.get(key)
        require(isinstance(values, list) and len(values) == RANKS, f"expected {RANKS} entries at {key}")
    for rank in range(RANKS):
        check_bytes(payload["cuda_rng"][rank], f"cuda_rng.{rank}")
        record = payload["rank_rng"][rank]
        require(isinstance(record, Mapping), f"invalid rank_rng.{rank}")
        for key in ("torch_rng", "cuda_rng"):
            check_bytes(record.get(key), f"rank_rng.{rank}.{key}")


def warm_endpoint_differences(continuous, warm, ordinary_cuda_rng):
    """Separate the three unused rank-0 device generators while checking them against the other restart."""
    validate_rng_schema(continuous)
    validate_rng_schema(warm)
    raw = checkpoint_differences(continuous, warm)
    process_rng = []
    scientific_invalid = list(raw["metadata_or_schema_different"])
    for index, path in enumerate(PROCESS_ONLY_CUDA_RNG, start=1):
        reference = continuous["cuda_rng"][index]
        measured = warm["cuda_rng"][index]
        ordinary = ordinary_cuda_rng[index]
        same_shape = reference.shape == measured.shape == ordinary.shape
        matches_continuous = identical(reference, measured)
        matches_ordinary = identical(ordinary, measured)
        process_rng.append({"path": path, "same_shape": same_shape, "matches_ordinary": matches_ordinary,
                            "matches_continuous": matches_continuous})
        # Only valid, same-shape uint8 state reproducing the ordinary restart may differ
        # from continuous. A malformed state, or a warm-only change, remains disqualifying.
        if same_shape and matches_ordinary:
            if path in scientific_invalid:
                scientific_invalid.remove(path)
        elif path not in scientific_invalid:
            scientific_invalid.append(path)
    return {**raw, "scientific_metadata_or_schema_different": scientific_invalid,
            "process_rng": process_rng, "exact_scientific_payload": not raw["floating_tensors_different"]
            and not scientific_invalid}


def load(p):
    return torch.load(str(p), map_location="cpu", weights_only=False, mmap=True)

t0 = time.time()
print("torch", torch.__version__)
cont, a, b = load(P["continuo"]), load(P["A"]), load(P["B"])
for name, o in (("continuo", cont), ("A", a), ("B", b)):
    try:
        validate_rng_schema(o); print("schema RNG", name, "OK | torch_rng", tuple(o["torch_rng"].shape), "| cuda_rng", [tuple(x.shape) for x in o["cuda_rng"]], "| rank_rng[0]", {k: tuple(v.shape) for k, v in o["rank_rng"][0].items()})
    except Exception as e:
        print("schema RNG", name, "ECCEZIONE", type(e).__name__, e)
# simulazione sui dati veri: B fa da gamba ordinaria, A fa da gamba riscaldata (e' una ripresa vera)
ordinary_cuda_rng = [v.clone() for v in b["cuda_rng"]]
d = warm_endpoint_differences(cont, a, ordinary_cuda_rng)
print("riscaldata simulata (CN_A) contro continuo:")
print("  metadata_or_schema_different (grezzo):", d["metadata_or_schema_different"])
print("  scientific_metadata_or_schema_different:", d["scientific_metadata_or_schema_different"])
print("  process_rng:", json.dumps(d["process_rng"]))
print("  tensori float diversi:", len(d["floating_tensors_different"]), "| exact_scientific_payload:", d["exact_scientific_payload"], "| exact_payload:", d["exact_payload"])
# controllo negativo: una copia inattiva alterata rispetto all'ordinaria deve restare squalificante
bad = [v.clone() for v in b["cuda_rng"]]; bad[2] = bad[2].clone(); bad[2][0] ^= 1
d2 = warm_endpoint_differences(cont, a, bad)
print("  controllo negativo (ordinaria con cuda_rng.2 alterato): scientific =", d2["scientific_metadata_or_schema_different"])
# continuo contro se stesso con ordinaria B: le copie inattive coincidono col continuo ma non con l'ordinaria
d3 = warm_endpoint_differences(cont, cont, ordinary_cuda_rng)
print("  controllo negativo (riscaldata = continuo): scientific =", d3["scientific_metadata_or_schema_different"], "| float diversi", len(d3["floating_tensors_different"]))
dab = checkpoint_differences(b, a)
print("CN_B contro CN_A (come ordinaria contro storico): exact_payload", dab["exact_payload"])
print("secondi", round(time.time() - t0, 1))
