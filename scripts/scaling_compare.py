"""Audited 5000-step scaling comparison from completed pose and probe outputs.

No fits, checkpoint loading, job submission or manual quality overrides. Numerical
points and paired bootstrap intervals are recomputed from pinned predictions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_probe as C
import scaling_pose_readout as P
import scaling_projection as SP
import scaling_reading as S8
import scaling_train as ST

PROTOCOL = "scaling-comparison-5000-20261104-v1"
INPUT_PROTOCOL = "scaling-comparison-inputs-20261104-v1"
require = C.require
PRIMARY = "natural_300m_minus_30m"
CONTROL, CANDIDATE = "T100-G-30M-natural", "T100-G-300M-natural"


def source_hashes():
    return {str(Path(m.__file__).relative_to(ROOT)): C.sha(m.__file__) for m in (C, P, SP, ST, S8)} | {
        str(Path(__file__).relative_to(ROOT)): C.sha(__file__)}


def _pin(pin, states):
    ST.pin_schema(pin)
    require(Path(pin["path"]).is_absolute(), "comparison inputs require absolute paths")
    state = ST.resolve_pin(pin)
    previous = states.setdefault(state["path"], state)
    require(previous == state, "artifact changed between uses")
    return state


def _read(pin, states):
    return json.loads(Path(_pin(pin, states)["path"]).read_text())


def make_inputs(probe_plan, pose_plan, probe_reports, pose_report):
    return {"protocol": INPUT_PROTOCOL, "checkpoint_step": 5000, "sources": source_hashes(),
            "probe_plan": probe_plan, "pose_plan": pose_plan, "probe_reports": probe_reports, "pose_report": pose_report}


def validate_inputs(inputs):
    require(inputs.get("protocol") == INPUT_PROTOCOL and type(inputs.get("checkpoint_step")) is int and
            inputs["checkpoint_step"] == 5000, "wrong comparison input protocol/dose")
    require(inputs.get("sources") == source_hashes(), "comparison source pins differ")
    reports = inputs.get("probe_reports")
    require(isinstance(reports, list) and len(reports) == 6 and len({r["path"] for r in reports}) == 6, "need six distinct probe reports")
    for pin in [inputs["probe_plan"], inputs["pose_plan"], inputs["pose_report"], *reports]:
        ST.pin_schema(pin)


def validate_shared_plans(probe, pose, states):
    context, projection = C.validate_plan(probe)
    P.validate_plan(pose)
    for key in ("projection", "training_source_contract"):
        a, b = _pin(probe[key], states), _pin(pose[key], states)
        require(a == b, "reader plans have different " + key)
    source = context["contract"]
    checkpoints = {}
    for arm in C.ARMS:
        pcs = {p["checkpoint_sha256"] for p in probe["caches"][arm].values()}
        qcs = {p["checkpoint_sha256"] for p in pose["caches"][arm].values()}
        require(len(pcs) == 1 and pcs == qcs, "pose/classification checkpoint mismatch")
        checkpoints[arm] = next(iter(pcs))
    require(checkpoints[C.ARMS[0]] == source["reference_checkpoint_sha256"], "historical control differs")
    for record in context["states"].values():
        states[record["path"]] = record
    # The projection loader validates both NPY files, algorithm and fixed seeds.
    for path in projection.states:
        _pin({"path": path, "sha256": C.sha(path)}, states)
    return checkpoints, projection


def _blas_identity(pools):
    keys = ("internal_api", "prefix", "version", "architecture", "threading_layer")
    require(isinstance(pools, list) and pools, "missing numeric library provenance")
    return sorted([{k: p.get(k) for k in keys} for p in pools], key=lambda p: json.dumps(p, sort_keys=True))


def runtime_coherence(probes, pose_runtime, expected_code):
    require(all(r["code_sha"] == expected_code for r in probes) and pose_runtime.get("code_sha") == expected_code,
            "mixed reader code")
    require(len({r["numeric_sha256"] for r in probes}) == 1, "mixed probe numerical runtime")
    numeric = probes[0]["numeric_context"]
    require(all(C.jhash(r["numeric_context"]) == r["numeric_sha256"] for r in probes), "invalid probe numeric signature")
    versions = pose_runtime.get("versions", {})
    require(all(versions.get(k) == numeric.get(k) for k in ("python", "numpy")), "pose/probe numeric versions differ")
    require(_blas_identity(pose_runtime.get("threadpools")) == _blas_identity(numeric.get("threadpools")), "pose/probe numeric libraries differ")
    require(pose_runtime.get("sources") == P.source_hashes(), "pose reader source identity differs")
    return {"code_sha": expected_code, "probe_numeric_sha256": probes[0]["numeric_sha256"],
            "pose_runtime": pose_runtime, "thread_count_policy": "exact within probes; pose may use its separately recorded CPU allocation"}


def validate_pose_selection(selection, folds):
    grid = selection.get("grid", [])
    require(isinstance(grid, list) and len(grid) >= 7 and all(set(r) == {"alpha", "mse"} and
            type(r["alpha"]) in (int, float) and type(r["mse"]) in (int, float) and
            np.isfinite(r["alpha"]) and r["alpha"] > 0 and np.isfinite(r["mse"]) and r["mse"] >= 0 for r in grid),
            "invalid ridge CV grid")
    scores = {r["alpha"]: r["mse"] for r in grid}
    require(len(scores) == len(grid), "duplicate ridge alpha")
    try:
        expected = P.Q.E.choose_alpha(lambda alpha: scores[alpha])
    except KeyError as exc:
        raise ValueError("ridge CV extension/grid incomplete") from exc
    expected["fold_users"] = folds
    require(selection == expected, "ridge CV selection/boundary audit differs")


def validate_pose_outputs(plan, report, plan_sha, states):
    require(report.get("protocol") == P.PROTOCOL and report.get("complete") is True and report.get("status") == "completed" and
            type(report.get("step")) is int and report["step"] == 5000 and report.get("recipe") == P.recipe() and
            report.get("plan_sha256") == plan_sha, "incomplete/wrong pose report")
    for key, expected in (("input_pins", plan["caches"]), ("reference_pins", plan["references"]),
                          ("projection_pin", plan["projection"]), ("training_source_contract", plan["training_source_contract"])):
        require(report.get(key) == expected, "pose report input provenance differs: " + key)
    require(set(report.get("cells", {})) == set(P.CELLS), "incomplete pose cells")
    references = {s: P.Q.validate_reference(plan["references"][s]) for s in ("train", "val")}
    require(references["train"]["users"] == sorted(sum(plan["fold_users"], [])) and
            set(references["train"]["users"]).isdisjoint(references["val"]["users"]), "pose train/val user leakage")
    for pin in plan["references"].values():
        _pin({k: pin[k] for k in ("path", "sha256", "bytes")}, states)
    inspections = {}
    for arm in P.ARMS:
        for split in ("train", "val"):
            pin = plan["caches"][arm][split]
            inspection = P.validate_artifact(pin, arm, split, plan, references[split])
            inspections[arm + "/" + split] = inspection
            for path_key, hash_key in (("path", "sha256"), ("provenance_path", "provenance_sha256"), ("receipt_path", "receipt_sha256")):
                _pin({"path": pin[path_key], "sha256": pin[hash_key]}, states)
    require(report.get("inspections") == inspections, "pose inspection receipt differs")
    cells = copy.deepcopy(report["cells"])
    train_support = {a: P.Q.arrays_hash(P.Q.read_targets(plan["caches"][a]["train"]["path"])["central_mask"]) for a in P.ARMS}
    for name, (arm, view) in P.CELLS.items():
        cell = cells[name]
        require(cell.get("checkpoint_sha256") == plan["caches"][arm]["val"]["checkpoint_sha256"] and
                cell.get("projection_manifest_sha256") == (plan["projection"]["sha256"] if view == "matched" else None),
                "pose cell checkpoint/projection mismatch")
        fit = cell["fit"]
        dim = 384 if view == "matched" else P.V.dimension(arm)
        require(fit.get("n_features") == dim and fit.get("n_users") == 40 and
                fit.get("fit_support_sha256") == train_support[arm] and fit.get("selection", {}).get("fold_users") == plan["fold_users"],
                "pose fit support/folds/dimension differs")
        validate_pose_selection(fit["selection"], plan["fold_users"])
        coefficient = _pin(cell["coefficients"], states)
        prediction = _pin(cell["predictions"], states)
        with np.load(coefficient["path"], allow_pickle=False) as z:
            require(set(z.files) == {"weights", "bias"}, "unexpected coefficient arrays")
            w, b = z["weights"], z["bias"]
        require(w.shape == (dim, 20) and b.shape == (20,) and w.dtype == b.dtype == np.float64 and
                np.isfinite(w).all() and np.isfinite(b).all() and fit.get("coefficients_sha256") == P.Q.arrays_hash(w, b),
                "invalid/nonfinite pose coefficients")
        pred = np.load(prediction["path"], allow_pickle=False)
        data = P.Q.read_targets(plan["caches"][arm]["val"]["path"])
        require(pred.dtype == np.float64 and pred.shape == data["central_mask"].shape and np.isfinite(pred).all(),
                "invalid/nonfinite pose predictions including padding")
        analysis = P.Q.summarize(data, pred)
        require(cell.get("analysis") == analysis, "pose analysis differs from pinned predictions/targets")
        cell["analysis"] = analysis
        del pred, data
    paired = P.paired_comparison(cells)
    require(report.get("paired") == paired, "pose paired statistics differ from recomputation")
    return cells, paired


def _classification(plan, pins, states, plan_sha, expected_code):
    loaded = []
    for pin in pins:
        path = _pin(pin, states)["path"]
        r, arrays, records = C.read_completed(path, "aggregate", plan)
        require(r["plan_sha256"] == plan_sha and r["code_sha"] == expected_code, "probe plan/code mismatch")
        for p, h in ((r["oof"], r["oof_sha256"]), (r["audit"], r["audit_sha256"])):
            _pin({"path": p, "sha256": h}, states)
        # Recheck immutable shard receipts as well as the aggregate files.
        require(r.get("shards") and len({s["path"] for s in r["shards"]}) == len(r["shards"]), "missing/duplicate shard provenance")
        shard_records, shard_arrays = [], []
        for shard in r["shards"]:
            shard_path = _pin(shard, states)["path"]
            sr, sa, audit = C.read_completed(shard_path, "shard", plan)
            shard_records.extend(audit)
            shard_arrays.append(sa)
            require(sr["cell"] == r["cell"] and sr["task"] == r["task"] and sr["code_sha"] == expected_code and
                    sr["plan_sha256"] == plan_sha, "mixed shard provenance")
            for p, h in ((sr["oof"], sr["oof_sha256"]), (sr["audit"], sr["audit_sha256"])):
                _pin({"path": p, "sha256": h}, states)
        require(shard_records == records, "aggregate audit differs from source shards")
        for ds in C.TASKS[r["task"]]:
            parts = [sa for sa in shard_arrays if ds + "_row" in sa]
            require(parts, "dataset missing in source shards")
            joined = {k: np.concatenate([sa[ds+"_"+k] for sa in parts]) for k in ("row", "y", "pred", "s", "fold")}
            C.T.validate_oof(*(joined[k] for k in ("row", "y", "pred", "s", "fold")))
            left, right = np.argsort(joined["row"]), np.argsort(arrays[ds+"_row"])
            require(all(np.array_equal(joined[k][left], arrays[ds+"_"+k][right]) for k in joined),
                    "aggregate OOF differs from source shards")
        loaded.append((r, arrays, records))
    require({(r["cell"], r["task"]) for r, _, _ in loaded} == {(c, t) for c in C.CELLS for t in C.TASKS}, "incomplete/duplicate probe grid")
    cells = {c: {} for c in C.CELLS}
    for r, arrays, _ in loaded:
        for ds in C.TASKS[r["task"]]:
            cells[r["cell"]][ds] = {k: arrays[f"{ds}_{k}"] for k in ("row", "y", "pred", "s", "fold")}
    # Cache, extraction receipt and provenance are all rehashed at completion.
    for arm in C.ARMS:
        for pin in plan["caches"][arm].values():
            for p, h in (("path", "sha256"), ("receipt_path", "receipt_sha256"), ("provenance_path", "provenance_sha256")):
                _pin({"path": pin[p], "sha256": pin[h]}, states)
    for pin in [plan["splits"], *plan["references"].values()]:
        _pin(pin, states)
    return [r for r, _, _ in loaded], C.compare_arrays(cells)


def compose_summary(classification, pose_paired, probe_reports, pose_cells):
    """Called only with validated producer artifacts; quality has no CLI override."""
    dynamic = pose_paired["dynamic"]
    control, candidate = dynamic["cell_ratio"][CONTROL], dynamic["cell_ratio"][CANDIDATE]
    raw_delta = dynamic["contrasts"][PRIMARY]["delta_ratio"]
    require(np.isclose(candidate - control, raw_delta, rtol=1e-10, atol=1e-12), "recomputed pose point/delta disagreement")
    solver = all(r["audit_summary"]["converged_by_criterion"] for r in probe_reports) and all(
        cell["fit"]["selection"]["boundary_unresolved"] is False for cell in pose_cells.values())
    summary = {"protocol": PROTOCOL, "complete": True, "checkpoint_step": 5000, "candidate": "T100-G-300M", "control": "T100-G-30M",
               "bootstrap": S8.POLICY["bootstrap"], "quality": {"artifact_integrity": True, "paired_support": True, "solver_converged": solver},
               "pose": {"control_E_dyn": control, "candidate_E_dyn": candidate,
                        "delta_E_dyn": candidate - control, "ci95": dynamic["contrasts"][PRIMARY]["ci95"]},
               "secondary": {"classification": classification, "pose": pose_paired},
               "scope": "One pretraining seed and development subjects; natural widths primary, fixed projection secondary. No automatic continuation."}
    for name, block in (("epn", classification["datasets"]["epn612"]), ("p1", classification["tasks"]["p1"]),
                        ("p2", classification["tasks"]["p2"])):
        contrast = block["contrasts"][PRIMARY]
        summary[name] = {"delta_pp": contrast["delta_pp"], "ci95": contrast["ci95_pp"]}
    summary["reading"] = S8.read_comparison(summary)
    return summary


def compare(inputs, out, expected_code, input_state=None):
    validate_inputs(inputs)
    C.E.P.code_identity(expected_code)
    states = {} if input_state is None else {input_state["path"]: input_state}
    probe = _read(inputs["probe_plan"], states)
    pose = _read(inputs["pose_plan"], states)
    checkpoints, projection = validate_shared_plans(probe, pose, states)
    reports, classification = _classification(probe, inputs["probe_reports"], states, inputs["probe_plan"]["sha256"], expected_code)
    pose_report = _read(inputs["pose_report"], states)
    runtime = runtime_coherence(reports, pose_report["runtime"], expected_code)
    pose_cells, paired = validate_pose_outputs(pose, pose_report, inputs["pose_plan"]["sha256"], states)
    result = compose_summary(classification, paired, reports, pose_cells)
    result.update(inputs=inputs, inputs_content_sha256=C.jhash(inputs), checkpoints=checkpoints,
                  source_contract=probe["training_source_contract"], projection=probe["projection"], runtime=runtime,
                  sources=source_hashes())
    C.E.P.check_unchanged(states)
    require(projection.states == {p: SP.file_state(p) for p in projection.states}, "projection changed during comparison")
    validate_inputs(inputs)
    require(C.E.P.code_identity(expected_code) == expected_code, "comparison code changed")
    C.T.fresh(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    C.T.write_json(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    state = ST.resolve_pin({"path": str(args.inputs.resolve()), "sha256": args.expected_inputs_sha256})
    inputs = json.loads(Path(state["path"]).read_text())
    result = compare(inputs, args.out, args.expected_code_sha, input_state=state)
    require(C.E.P.file_identity(state["path"]) == state, "comparison input manifest changed")
    print(json.dumps({"complete": result["complete"], "reading_valid": result["reading"]["reading_valid"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
