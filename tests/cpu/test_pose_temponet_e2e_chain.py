"""CPU execution of all 24 runner stages with the real neural readers.

This is a wiring/regression test, not a calibration or convergence claim. The
SLURM allocation, numeric census, external cache/preflight, benchmark receipts,
and historical predictions are fixtures. CUDA transfers are redirected to CPU;
the bounded test recipe has one/two epochs. Only the *outcome* of the neural
positive-control gate is forced to passed, retaining its unforced diagnostics.
Selection, actual backward updates, refits, inference, metrics and collection
are executed by their unmodified public entry points through run_stage.
"""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet import benchmark, data as D, execution as E, metrics as M
from pose_temponet import provenance as P, runner as R, train as T, workflow as W
from test_pose_temponet_provenance import prepared, release, numeric_fixture
from test_pose_temponet_resources import receipts as cost_receipts


def _cache(users, windows, split, rng, weights):
    n = len(users) * windows
    lengths = np.full(n, 160, dtype=np.int64)
    lengths[3::7] = 150
    available = np.arange(160)[None, :] < lengths[:, None]
    # Low-rank synthetic latents let the actual ridge qualify with forty
    # windows; dimension and all production head shapes stay unchanged.
    x = np.tile(rng.normal(size=(n, 160, 16)), (1, 1, 24)).astype(np.float16)
    pre = (.7 * x.astype(float) + .5 * rng.normal(size=x.shape)).astype(np.float16)
    x[~available] = pre[~available] = 0
    z = x.astype(float)
    z -= z.mean(-1, keepdims=True)
    z /= np.sqrt(np.mean(z*z, axis=-1, keepdims=True) + 1e-5)
    signal = z @ weights
    target = signal + 1.9 * rng.normal(size=signal.shape)
    scales = np.ones(n)
    if split == "val":
        # One static and one dynamic window for every development user.
        scales[::windows] = .1
    target = (target * scales[:, None, None]).astype(np.float32)
    mask = np.repeat(available[..., None], 20, axis=2)
    target[~mask] = 0
    return {
        "pre": pre, "lat": x, "key_time_valid": available,
        "pred": (.8 * signal * scales[:, None, None]).astype(np.float32),
        "target": target, "mask": mask, "std": np.full((n, 20), .3, np.float32),
        "pose_var": M.dynamic_variance(target, mask).astype(np.float32),
        "meta": [{"user": users[i // windows], "recording": i // windows,
                  "window": i % windows, "n_patches": int(lengths[i])} for i in range(n)],
        "config": {"split": split, "which": "teacher", "checkpoint_step": 49046},
        "has_head": True, "dim": 384, "n_recordings": len(users),
    }


def _cpu_execution(monkeypatch):
    """Change device and test horizon, never replace a selection/refit driver."""
    recipe = T._TestRecipe(initial_epochs=1, maximum_epochs=2, last_epochs=1, batch_size=32)
    monkeypatch.setattr(T, "Recipe", lambda: recipe)
    for function in (W._run_selection, W._run_refits):
        monkeypatch.setattr(function, "__kwdefaults__", {**function.__kwdefaults__, "recipe": recipe})
    original_init = T.Trajectory.__init__

    def cpu_trajectory(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(T.Trajectory, "__init__", cpu_trajectory)
    original_to = torch.Tensor.to

    def cpu_transfer(tensor, *args, **kwargs):
        args = list(args)
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith("cuda"):
            args[0] = "cpu"
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    monkeypatch.setattr(T, "configure_numerics", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


@pytest.fixture
def real_chain(prepared, monkeypatch):
    repo, plan = prepared
    plan = release(repo, plan, monkeypatch)
    plan["jobs"] = E.job_graph()
    plan["operator_checks"] = {"work_free_bytes": 13*1024**3, "pending_write_bytes": 0}
    _cpu_execution(monkeypatch)
    original_validate = P.validate_plan
    monkeypatch.setattr(P, "validate_plan", lambda p, **kw: original_validate(p, **{**kw, "source_root": repo}))
    monkeypatch.setattr(P, "numeric_context", lambda kind: numeric_fixture(kind))
    rng = np.random.default_rng(20261119)
    weights = rng.normal(size=(384, 20)) / np.sqrt(384)
    caches = {split: _cache(plan["producers"][split]["users"], 1 if split == "train" else 2,
                            split, rng, weights) for split in ("train", "val")}
    folds = D.alternating_user_folds(caches["train"])
    monkeypatch.setattr(P, "preflight", lambda p: {
        "integrity": True, "historical_R1": {"passed": True, "fixture": True},
        "train": {"folds": copy.deepcopy(folds)},
    })
    opened, completed = [], []

    def load(p, split, *, stage, depth=None):
        assert "cache_" + split in P.stage_inputs(p, stage)
        if split == "val":
            assert len([j for j in completed if j["stage"] == "refit"]) == 5
            assert (Path(p["output_root"]) / "locked_inference/inference_lock.json").is_file()
        opened.append((stage, split, depth))
        cache = copy.deepcopy(caches[split])
        cache["lengths"] = D.lengths_for(cache)
        cache["central_mask"] = D.central_support(cache["mask"], cache["lengths"], cache["key_time_valid"])
        if depth is not None:
            cache["features"] = cache["pre" if depth == "pre" else "lat"]
            del cache["pre"], cache["lat"]
            cache["feature_depth"] = depth
        cache["input_verification"] = {"fixture": True}
        return cache

    monkeypatch.setattr(P, "read_verified_inputs", load)
    cpu, gpu = cost_receipts()
    for measurement in (cpu, gpu):
        measurement["input_identity"] = R._input_identity(plan)
    gpu["persistent_output_bytes"] = 7*1024**3
    monkeypatch.setattr(benchmark, "benchmark_cpu", lambda cache, **kw: copy.deepcopy(cpu))
    monkeypatch.setattr(benchmark, "benchmark_gpu", lambda cache, **kw: copy.deepcopy(gpu))

    references = {
        "post_ridge": .9*caches["val"]["pred"][:, 6:-6].astype(float),
        "trained_head": caches["val"]["pred"][:, 6:-6].astype(float),
    }

    def read_predictions(pin):
        name = "post_ridge" if pin == plan["inputs"]["post_ridge_predictions"] else "trained_head"
        assert pin == plan["inputs"][name + "_predictions"]
        return references[name].copy()

    monkeypatch.setattr(D, "read_predictions", read_predictions)
    # Keep the R-1 arithmetic and support checks real, with fixture references.
    policies = {}
    for name, prediction in references.items():
        dynamic = M.evaluate(prediction, caches["val"])["groups"]["dynamic"]
        centered = dynamic["centered"]
        policies[name + "_E_dyn"] = centered["centered_mse_ratio"]
        policies.update(dynamic_selected_windows=dynamic["n_selected_windows"],
                        dynamic_centered_windows=centered["n_windows"],
                        dynamic_centered_elements=centered["n_valid_elements"],
                        dynamic_window_joint_pairs=centered["n_window_joints"],
                        dynamic_users=centered["n_users"])
    monkeypatch.setattr(M, "HISTORICAL_REFERENCE", policies)

    actual_qualify = W.qualify_controls
    qualified = []

    def admit_short_test_recipe(artifacts, selections):
        gate = actual_qualify(artifacts, selections)
        assert gate["complete"] and len(gate["selection_locks"]) == 2
        qualified.append(gate["status"])
        gate.pop("gate_sha256")
        gate.update(fixture_forced_admission=True, fixture_unforced_status=gate["status"], status="passed")
        return W._lock(gate, "gate_sha256")

    monkeypatch.setattr(W, "qualify_controls", admit_short_test_recipe)
    current = {}
    original_output = R.subprocess.check_output

    def slurm_output(args, *a, **kw):
        if list(args[:3]) != ["scontrol", "show", "job"]:
            return original_output(args, *a, **kw)
        r = current["job"]["resources"]
        minutes = (r["requested_seconds"] + 59)//60
        tres = f"cpu={r['cpus']},mem={r['host_ram_gib']}G,node=1" + (",gres/gpu=1" if r["gpus"] else "")
        return (f"JobId={args[3]} TimeLimit={minutes//60:02}:{minutes%60:02}:00 "
                f"Partition={r['partition']} NumNodes=1 NumCPUs={r['cpus']} AllocTRES={tres}\n")

    monkeypatch.setattr(R.subprocess, "check_output", slurm_output)

    def execute(p, job):
        current["job"] = job
        r = job["resources"]
        environment = {"SLURM_JOB_ID": str(70000000 + len(completed)), "SLURM_CPUS_PER_TASK": str(r["cpus"]),
                       "SLURM_GPUS_ON_NODE": str(r["gpus"]), "SLURM_NNODES": "1",
                       "SLURM_MEM_PER_NODE": str(r["host_ram_gib"]*1024), "SLURM_RESTART_COUNT": "0"}
        for key, value in environment.items():
            monkeypatch.setenv(key, value)
        pin = R.run_stage(p, job["id"], source_root=repo)
        receipt = P.verified_receipt(pin, p, source_root=repo)
        assert receipt["complete"] is True
        completed.append(job)
        return pin

    return repo, plan, execute, opened, completed, qualified


@pytest.mark.parametrize("budget_transition", [False, True])
def test_all_24_stages_use_real_selection_refits_inference_and_report(real_chain, capsys,
                                                                  monkeypatch, budget_transition):
    repo, plan, execute, opened, completed, qualified = real_chain
    originals = W.run_selection, W.run_refits
    pins = {}
    for job in plan["jobs"][:3]:
        pins[job["stage"]] = execute(plan, job)
    if budget_transition:
        # Wiring simulation of A1 after three successful predecessors. Only the
        # campaign-specific base/pins and external review are fixtures here;
        # test_pose_temponet_budget_amendment verifies the real immutable pins.
        from pose_temponet import budget_amendment as A1
        base = copy.deepcopy(plan)
        monkeypatch.setattr(A1, "base_plan", lambda root: copy.deepcopy(base))
        monkeypatch.setattr(A1, "LEGACY", {j["id"]: (pins[j["stage"]]["sha256"], pins[j["stage"]]["bytes"])
                                         for j in plan["jobs"][:3]})
        plan = A1.prepare(base, source_root=repo)
        plan = release(repo, plan, monkeypatch)
    plan["operator_checks"]["observed_at"] = datetime.now(timezone.utc).isoformat()
    plan = R.measured_plan(plan, pins["cpu_benchmark"], pins["gpu_benchmark"], source_root=repo)
    for job in plan["jobs"][3:]:
        pins[job["id"]] = execute(plan, job)
    assert (W.run_selection, W.run_refits) == originals
    assert len(completed) == len(plan["jobs"]) == 24
    assert len(qualified) >= 1
    assert all(split == "train" for stage, split, depth in opened if stage not in ("locked_inference", "metrics"))
    assert [(stage, depth) for stage, split, depth in opened if split == "val"] == [
        ("locked_inference", "post"), ("locked_inference", "pre"), ("metrics", "post")]
    root = Path(plan["output_root"])
    selections = [json.loads(path.read_text()) for path in root.rglob("selection.json")]
    assert len(selections) == 11
    assert all(s["selection_locked"] and s["curves"] and s["recipe"]["test_only"] for s in selections)
    refits = [json.loads(path.read_text()) for path in root.rglob("refits.json")]
    assert len(refits) == 5 and all(set(r["cells"]) == {"0", "1", "2"} for r in refits)
    assert all(cell["updates"] > 0 for r in refits for cell in r["cells"].values())
    inference = R._receipt(plan, E.job_id("locked_inference"), source_root=repo)[0]
    assert len([key for key in inference["artifacts"] if key.endswith("_predictions")]) == 15
    final = json.loads((root / "metrics/report.json").read_text())
    assert final["complete"] is True and len(final["contrasts"]) == 5
    parity = json.loads((root / "metrics/historical_R1.json").read_text())
    assert parity["passed"] is True and parity == final["historical_R1"]
    assert parity["crop"] == [6, 154] and parity["observed"]["trained_head"]["users"] == 15
    collection = R._receipt(plan, E.job_id("collection"), source_root=repo)[0]
    assert collection["details"]["scientific_results"]["sha256"] == P.file_pin(root / "metrics/report.json")["sha256"]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    zero_points = [event for event in events if event.get("event") == "selection_checkpoint"
                   and event["point"]["exposure"] == 0]
    assert len(zero_points) >= 11 * 4
    assert all(set(event["initial_sha256_per_fold"]) == {"0", "1"} and
               len(set(event["initial_sha256_per_fold"].values())) == 1 for event in zero_points)
    initial_refits = [event for event in events if event.get("event") == "refit_initialization"]
    assert len(initial_refits) == 15 and all(event["updates"] == 0 for event in initial_refits)
