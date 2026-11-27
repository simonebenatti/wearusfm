"""C1 verification: B-2 rule, scalar screen, classify, early INCONCLUSIVE (synthetic, real RNG format)."""
import copy, sys
import torch
import contrastive_cn_restart_compare as c

g = torch.Generator().manual_seed(0)


def rng_cpu():  # same format as torch.get_rng_state(): 1-D uint8, 5056 bytes
    return torch.randint(0, 256, (5056,), dtype=torch.uint8, generator=g)


def rng_cuda():  # torch.cuda.get_rng_state(): 1-D uint8, 16 bytes (seed + offset)
    return torch.randint(0, 256, (16,), dtype=torch.uint8, generator=g)


assert torch.get_rng_state().dtype == torch.uint8 and torch.get_rng_state().ndim == 1
base_cuda0 = rng_cuda()
rank_rng = [{"torch_rng": rng_cpu(), "cuda_rng": (base_cuda0.clone() if r == 0 else rng_cuda())} for r in range(4)]
torch_rng = rank_rng[0]["torch_rng"].clone()
continuous = {"step": 576, "world_size": 4, "config": {"a": 1}, "alarm_counts": {"x": 0},
              "student": {"w": torch.randn(3, generator=g)},
              "optimizer": {"state": {0: {"step": torch.tensor(576.), "exp_avg": torch.randn(3, generator=g)}}},
              "torch_rng": torch_rng, "cuda_rng": [base_cuda0.clone(), rng_cuda(), rng_cuda(), rng_cuda()],
              "rank_rng": rank_rng}
ordinary = copy.deepcopy(continuous)
for k in (1, 2, 3):
    ordinary["cuda_rng"][k] = rng_cuda()  # resumed run: inactive copies differ from continuous
ord_cuda = [t.clone() for t in ordinary["cuda_rng"]]


def run(name, warm, expect_exact, expect_paths=None, raises=None):
    try:
        r = c.warm_endpoint_differences(continuous, warm, ord_cuda)
    except ValueError as e:
        ok = raises is not None and raises in str(e)
        print(f"{'OK ' if ok else 'BAD'} {name}: ValueError({e})")
        return
    if raises:
        print(f"BAD {name}: expected raise {raises}, got {r['scientific_metadata_or_schema_different']}")
        return
    paths = r["scientific_metadata_or_schema_different"]
    ok = r["exact_scientific_payload"] == expect_exact and (expect_paths is None or set(paths) == set(expect_paths))
    print(f"{'OK ' if ok else 'BAD'} {name}: exact_sci={r['exact_scientific_payload']} sci_invalid={paths} "
          f"raw={r['metadata_or_schema_different']} float={r['floating_tensors_different']}")


expected = copy.deepcopy(continuous)
for k in (1, 2, 3):
    expected["cuda_rng"][k] = ord_cuda[k].clone()
run("atteso", copy.deepcopy(expected), True, [])

w = copy.deepcopy(expected); w["cuda_rng"][2] = rng_cuda(); run("copia inattiva diversa (nuova)", w, False, ["cuda_rng.2"])
w = copy.deepcopy(expected); w["cuda_rng"][3] = continuous["cuda_rng"][3].clone(); run("copia inattiva = continuo", w, False, ["cuda_rng.3"])
w = copy.deepcopy(expected); w["cuda_rng"][0][0] ^= 1; run("cuda_rng.0 diverso", w, False, ["cuda_rng.0"])
w = copy.deepcopy(expected); w["rank_rng"][1]["cuda_rng"][5] ^= 1; run("rank_rng.1.cuda_rng diverso", w, False, ["rank_rng.1.cuda_rng"])
w = copy.deepcopy(expected); w["rank_rng"][3]["torch_rng"][7] ^= 1; run("rank_rng.3.torch_rng diverso", w, False, ["rank_rng.3.torch_rng"])
w = copy.deepcopy(expected); w["rank_rng"][0]["cuda_rng"][0] ^= 1; run("rank_rng.0.cuda_rng diverso", w, False, ["rank_rng.0.cuda_rng"])
w = copy.deepcopy(expected); w["torch_rng"][100] ^= 1; run("torch_rng diverso", w, False, ["torch_rng"])
w = copy.deepcopy(expected); w["cuda_rng"].append(rng_cuda()); run("lista lunga 5", w, False, raises="expected 4 entries at cuda_rng")
w = copy.deepcopy(expected); w["cuda_rng"].pop(); run("lista lunga 3", w, False, raises="expected 4 entries at cuda_rng")
w = copy.deepcopy(expected); w["cuda_rng"] = tuple(w["cuda_rng"]); run("tuple invece di list", w, False, raises="expected 4 entries")
w = copy.deepcopy(expected); w["rank_rng"].pop(); run("rank_rng lunga 3", w, False, raises="expected 4 entries at rank_rng")
w = copy.deepcopy(expected); w["cuda_rng"][1] = w["cuda_rng"][1].to(torch.int8); run("cuda_rng.1 int8", w, False, raises="cuda_rng.1")
w = copy.deepcopy(expected); w["cuda_rng"][1] = w["cuda_rng"][1].reshape(4, 4); run("cuda_rng.1 2-D", w, False, raises="cuda_rng.1")
w = copy.deepcopy(expected); w["cuda_rng"][2] = w["cuda_rng"][2][:8].clone(); run("cuda_rng.2 corto", w, False, ["cuda_rng.2"])
w = copy.deepcopy(expected); w["torch_rng"] = w["torch_rng"].float(); run("torch_rng float", w, False, raises="torch_rng")
w = copy.deepcopy(expected); del w["rank_rng"][2]["cuda_rng"]; run("chiave rank_rng.2.cuda_rng mancante", w, False, raises="rank_rng.2.cuda_rng")
w = copy.deepcopy(expected); del w["torch_rng"]; run("chiave torch_rng mancante", w, False, raises="torch_rng")
w = copy.deepcopy(expected); w["rank_rng"][1]["extra"] = 1; run("chiave in piu' in rank_rng.1", w, False, ["rank_rng.1"])
w = copy.deepcopy(expected); w["rank_rng"][1] = None; run("rank_rng.1 None", w, False, raises="rank_rng.1")
w = copy.deepcopy(expected); w["step"] = 575; run("step diverso", w, False, ["step"])
w = copy.deepcopy(expected); w["alarm_counts"]["x"] = 1; run("alarm_counts diverso", w, False, ["alarm_counts.x"])
w = copy.deepcopy(expected); w["config"] = {"a": 2}; run("config diverso", w, False, ["config.a"])
w = copy.deepcopy(expected); w["optimizer"]["state"][0]["step"] += 1; run("adam step float diverso", w, False, ["optimizer.state.0.step"])
w = copy.deepcopy(expected); w["student"]["w"][0] += 1e-3; run("float diverso (atteso exact_sci falso, nessun invalid)", w, False, [])
w = copy.deepcopy(expected); w["new_key"] = 1; run("chiave top-level in piu'", w, False, [""])
w = copy.deepcopy(expected); w["cuda_rng"][1] = None; run("cuda_rng.1 None", w, False, raises="cuda_rng.1")
# continuous format checked as well
cbad = copy.deepcopy(continuous); cbad["cuda_rng"] = cbad["cuda_rng"][:3]
try:
    c.warm_endpoint_differences(cbad, copy.deepcopy(expected), ord_cuda); print("BAD continuo malformato accettato")
except ValueError as e:
    print("OK  continuo malformato rifiutato:", e)

# --- classify: all branches
print("\n--- classify")
import itertools
for loc, red, exc, end, first in itertools.product([0, 1], repeat=5):
    s, _ = c.classify(bool(loc), bool(red), bool(exc), bool(end), bool(first))
    want_sup = (not loc) and red and (not exc) and end and first
    if (s == "SUPPORTED") != bool(want_sup):
        print("BAD", loc, red, exc, end, first, s)
print("SUPPORTED reachable only with local=F, reduced=T, excess=F, endpoint=T, first=T: checked 32 combos")
