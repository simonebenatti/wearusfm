"""CPU chain on the corrected clone.

1. Real 4-process Gloo DDP, real leg.instrument_leg (ordinary and warm), real
   ReductionObservation.finish, 270-parameter fixture (from tests/cpu/test_cn_restart_leg.py).
2. load_observation for every rank and both legs under a SIMULATED torch 2.2 load
   (ValueError when mmap=True and f is not str).
3. Corrected comparator functions: compare_vector, screen_reduced_gradients,
   warm_endpoint_differences on payloads with the real cuda_rng/rank_rng structure, classify.
Usage: python chain.py OUTDIR
"""
import copy
import json
from pathlib import Path
import sys

import torch
import torch.multiprocessing as mp

out = Path(sys.argv[1])

import test_cn_restart_leg as legtest
import contrastive_cn_restart_compare as reader
import contrastive_cn_restart_observe as observe

if __name__ == "__main__":
    out.mkdir(parents=True, exist_ok=False)
    mp.spawn(legtest.run_gloo, args=(4, str(out), (out / "store").as_uri()), nprocs=4, join=True)

    native = torch.load
    calls = []

    def torch22_load(f, *a, **k):
        if k.get("mmap") and not isinstance(f, str):
            raise ValueError("f must be a string filename in order to use mmap argument")
        calls.append((type(f).__name__, bool(k.get("mmap"))))
        return native(f, *a, **k)
    torch.load = torch22_load

    # Old (pre-fix) reader behaviour under the shim, for contrast: Path + mmap.
    try:
        torch.load(out / "ordinary" / "vectors.rank0.pt", map_location="cpu", mmap=True, weights_only=True)
        old = "loaded"
    except ValueError as e:
        old = f"ValueError: {e}"

    rec, vec = {}, {}
    for mode in ("ordinary", "warm"):
        for rank in range(4):
            rec[mode, rank], vec[mode, rank] = observe.load_observation(out / mode, rank, expected_parameters=270)
    norms = [type(r[s + "_norm"]).__name__ for (m, k), rcd in rec.items() for r in rcd["parameters"]
             for s in ("local", "reduced")]
    zero_norms = sum(1 for (m, k), rcd in rec.items() for r in rcd["parameters"]
                     for s in ("local", "reduced") if r[s + "_norm"] == 0)

    def readout(perturb):
        details, any_local, any_reduced = [], False, False
        for rank in range(4):
            for pid in range(270):
                o = {s: vec["ordinary", rank][s][pid] for s in ("local", "reduced")}
                w = {s: vec["warm", rank][s][pid].clone() for s in ("local", "reduced")}
                if perturb and pid == 7:
                    w["reduced"].view(-1)[0] = torch.nextafter(w["reduced"].view(-1)[0], torch.tensor(1.))
                stages = {s: reader.compare_vector(o[s], w[s]) for s in ("local", "reduced")}
                any_local |= bool(stages["local"]["different_elements"])
                any_reduced |= stages["reduced"]["difference_l2"] > 0
                details.append({"rank": rank, "id": pid, "name": rec["ordinary", rank]["parameters"][pid]["name"], **stages})
        saved = reader.SCALAR_ADDEND_PARAMETERS
        reader.SCALAR_ADDEND_PARAMETERS = {}  # fixture has no 1-element anchor biases
        try:
            excessive = reader.screen_reduced_gradients(details, {})
        finally:
            reader.SCALAR_ADDEND_PARAMETERS = saved
        try:
            reader.screen_reduced_gradients(details, {253: 0., 257: 0.})
            named = "accepted"
        except ValueError as e:
            named = f"ValueError: {e}"
        json.dumps(details, allow_nan=False)
        return details, any_local, any_reduced, excessive, named

    # Payloads with the real structure read on Leonardo: cuda_rng list of 4 uint8,
    # rank_rng list of 4 dicts {cuda_rng, torch_rng}; ordinary and continuous differ
    # only in cuda_rng.1-3.
    g = torch.Generator().manual_seed(3)
    def ubytes(n, seed):
        return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=torch.Generator().manual_seed(seed))
    base = {"step": 576, "world_size": 4, "student": {"w": torch.randn(3, generator=g)},
            "optimizer": {"state": {0: {"step": torch.tensor(576.), "exp_avg": torch.randn(3, generator=g)}}},
            "torch_rng": ubytes(5056, 1), "cuda_rng": [ubytes(16, 10 + i) for i in range(4)],
            "rank_rng": [{"cuda_rng": ubytes(16, 20 + i), "torch_rng": ubytes(5056, 30 + i)} for i in range(4)]}
    continuous = copy.deepcopy(base)
    ordinary_ep = copy.deepcopy(base)
    for i in (1, 2, 3):
        ordinary_ep["cuda_rng"][i] = ubytes(16, 90 + i)
    warm_ep = copy.deepcopy(continuous)
    for i in (1, 2, 3):
        warm_ep["cuda_rng"][i] = ordinary_ep["cuda_rng"][i].clone()
    reader.validate_rng_schema(ordinary_ep)
    endpoint = reader.warm_endpoint_differences(continuous, warm_ep,
                                                [v.clone() for v in ordinary_ep["cuda_rng"]])
    json.dumps(endpoint, allow_nan=False)

    result = {"old_path_mmap_under_shim": old, "loads": len(calls), "load_types": sorted(set(calls)),
              "norm_types": sorted(set(norms)), "zero_norms": zero_norms,
              "warm_endpoint": {k: endpoint[k] for k in ("scientific_metadata_or_schema_different",
                                                         "metadata_or_schema_different", "exact_scientific_payload")}}
    for perturb in (False, True):
        details, any_local, any_reduced, excessive, named = readout(perturb)
        status = reader.classify(any_local, any_reduced, excessive, endpoint["exact_scientific_payload"], True)
        result["perturbed" if perturb else "as_observed"] = {
            "any_local": any_local, "any_reduced": any_reduced, "excessive": excessive,
            "max_screen_relative_l2": max(d["reduced"]["screen_relative_l2"] for d in details),
            "real_scalar_map_on_fixture": named, "status": status}
    print(json.dumps(result, indent=1))
