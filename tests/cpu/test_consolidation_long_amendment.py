"""Amendment A1 of the long evaluation, verification by verification (V-1...V-8 of the design review).

No job, no cluster access and no real feature is read: the inputs are the two
plans, the 28 collected receipts, the cache manifest and the Git objects of the
two original phase commits.
"""
from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import consolidation_long_amendment as A
import consolidation_long_common as C
import consolidation_long_classification as M
import consolidation_long_classification_runtime as R
import consolidation_long_pose_runtime as U
import test_consolidation_long_classification as F

CAMPAIGN = ROOT / "results/infra/consolidation_long_20261105"
TOOLS = CAMPAIGN / "amendment_a1"
DECISION = {"by": "Simone", "date": "2026-11-06", "quote": "decision used only by this test"}
PINNED = ("path", "sha256", "bytes")


def plan():
    return json.loads((CAMPAIGN / "plan.json").read_text())


def launch():
    return json.loads((CAMPAIGN / "launch_plan.json").read_text())


def manifest():
    return json.loads((ROOT / A.INPUT_MANIFEST["path"]).read_text())


def current():
    """Covered files and changed texts of this checkout, as the runtime reads them."""
    return (A.source_files(ROOT, C.T.R.sha256_file),
            {path: (ROOT / path).read_text() for path in A.CHANGED_PATHS})


def original(path):
    """Text of a file at the original benchmark phase commit, from Git."""
    return subprocess.check_output(["git", "show", A.ORIGINAL_PHASES["benchmark"]["code_sha"] + ":" + path],
                                   cwd=ROOT, text=True)


def receipt(ident):
    return json.loads((CAMPAIGN / "execution/receipts" / (ident + ".json")).read_text())


def benchmarks():
    reports = {}
    for row in launch()["jobs"]:
        if row["stage"] == "benchmark-classification":
            report = receipt(row["id"])
            reports.setdefault(report["dataset"], {})[report["view"]] = report
    return reports


def registered():
    return {R.shard_key(*shard): shard for shard in A.registered_shards(M.DOSES, M.VIEWS, M.DATASETS)}


def segment(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    return ast.get_source_segment(text, nodes[0])


# ----------------------------------------------------------------------------------------------
# Fold coverage and walltime rule
# ----------------------------------------------------------------------------------------------

def test_registered_shards_cover_each_new_cell_and_each_fold_exactly_once():
    shards = A.registered_shards(M.DOSES, M.VIEWS, M.DATASETS)
    assert len(shards) == 69 == A.FIT_POLICY["physical_fit_jobs"] == len(registered())
    covered = {}
    for step, view, dataset, folds in shards:
        assert folds == sorted(folds) and folds in A.fit_shapes(dataset, view)
        covered.setdefault((step, view, dataset), []).extend(folds)
    assert len(covered) == 40 == A.FIT_POLICY["new_fit_cells"]
    assert set(covered) == {(s, v, d) for s in M.DOSES for v in M.VIEWS for d in M.DATASETS if (s, v) != (19618, "pool")}
    assert all(folds == [0, 1, 2, 3] for folds in covered.values())
    assert Counter(dataset for _, _, dataset, _ in shards) == {
        "ninapro_db2": 26, "ninapro_db3": 17, "ninapro_db6": 10, "epn612": 8, "uci_emg": 8}
    # Exactly the four packagings of the proposal changed; the others are the original ones.
    changed = {(d, v) for d in M.DATASETS for v in M.VIEWS
               if A.fit_shapes(d, v) != R.BENCHMARK_POLICY["fixed_folds_by_dataset"][d]}
    assert changed == {("ninapro_db2", "pre"), ("ninapro_db2", "pool"), ("ninapro_db3", "pre"), ("ninapro_db6", "pool")}
    assert A.fit_shapes("ninapro_db2", "pre") == A.fit_shapes("ninapro_db2", "pool") == A.fit_shapes("ninapro_db3", "pre") \
        == [[0], [1], [2], [3]] and A.fit_shapes("ninapro_db6", "pool") == [[0, 1], [2, 3]]
    # A caller cannot alter the registered shapes through the returned lists.
    shape = A.fit_shapes("ninapro_db2", "pre")
    shape[0].append(3)
    shape.append([0, 1, 2, 3])
    assert A.fit_shapes("ninapro_db2", "pre") == [[0], [1], [2], [3]]


@pytest.mark.parametrize("projection,expected", [
    (1., 900), (300., 900), (300.01, 1800), (1200., 1800), (1200.5, 2700), (12900., 13500),
    (13800., 14400), (13800.01, 15300)])
def test_walltime_is_the_projection_plus_the_reserve_rounded_up_to_the_quantum(projection, expected):
    assert A.walltime_seconds(projection) == expected and type(A.walltime_seconds(projection)) is int
    assert projection <= expected - 600 and expected - 900 < projection + 600


@pytest.mark.parametrize("view", ["pre", "pool"])
def test_fold_packaging_changes_no_fold_fit_and_no_prediction(tmp_path, view):
    """Folds are fitted independently: whole, pairs and single folds give the same out-of-fold rows."""
    pin, _, _, _ = F.cache(tmp_path)

    def fit(name, folds):
        probe = M.fit_cache(pin["path"], view, tmp_path / name / "probe.json", expected=pin,
                            identity={"code_sha": "c"*40, "plan_sha256": "e"*64}, folds=folds)
        with np.load(tmp_path / name / "probe.oof.npz", allow_pickle=False) as archive:
            return probe, {key: archive[key] for key in M.OOF_KEYS}

    whole, reference = fit("whole", [0, 1, 2, 3])
    for name, shapes in (("single", A.SINGLES), ("pair", A.PAIRS)):
        parts = [fit(f"{name}{index}", folds) for index, folds in enumerate(shapes)]
        assert [probe["fold_indices"] for probe, _ in parts] == shapes
        for key in M.OOF_KEYS:
            np.testing.assert_array_equal(np.concatenate([arrays[key] for _, arrays in parts]), reference[key])
        # Same selected C, validation scores, scaler and held score, fold by fold.
        assert [fold for probe, _ in parts for fold in probe["folds"]] == whole["folds"]
        assert all(fold["c"] in M.C_GRID and len(fold["val_scores"]) == len(M.C_GRID) for fold in whole["folds"])


# ----------------------------------------------------------------------------------------------
# V-1  Closed list of covered source differences
# ----------------------------------------------------------------------------------------------

def test_v1_checkout_differs_from_the_original_227_file_source_only_inside_the_closed_list():
    inputs, (files, texts) = manifest(), current()
    assert {**C.file_pin(ROOT / A.INPUT_MANIFEST["path"]), "path": A.INPUT_MANIFEST["path"]} == A.INPUT_MANIFEST
    assert A.validate_inputs(inputs, files, texts, C.digest) is inputs
    old = inputs["source_files"]
    assert len(old) == 227 and C.digest(old) == A.ORIGINAL_SOURCE == inputs["source_sha256"]
    assert len(files) == 228 and C.digest(files) == C.source_identity() == plan()["source_sha256"] != A.ORIGINAL_SOURCE
    assert {p for p in files if files[p] != old.get(p)} == set(A.CHANGED_PATHS) | set(A.ADDED_PATHS)
    assert set(A.CHANGED_PATHS) == {A.COMMON, A.CLASSIFICATION, A.POSE, A.LAUNCHER} and len(A.ADDED_PATHS) == 1
    # Scientific modules, entry point, extractor, trainer and every file of src/ are byte-identical.
    for path in ("scripts/consolidation_long.py", "scripts/consolidation_long_classification.py",
                 "scripts/consolidation_long_pose.py", "scripts/tokenizer_evaluation.py", "scripts/tokenizer_train.py",
                 "scripts/tokenizer_probe.py", "scripts/train_jepa.py", "scripts/slurm/lib_code_dir.sh"):
        assert files[path] == old[path], path
    package = [p for p in old if p.startswith("src/")]
    assert len(package) > 50 and all(files[p] == old[p] for p in package)
    assert sorted(p for p in files if p.startswith("src/")) == sorted(package)


@pytest.mark.parametrize("mutation", ["scientific", "package", "extra", "removed", "reverted", "module_missing"])
def test_v1_any_other_covered_difference_is_refused(mutation):
    inputs, (files, texts) = manifest(), current()
    if mutation == "scientific":
        files["scripts/consolidation_long_classification.py"] = "0"*64
    elif mutation == "package":
        files[next(p for p in files if p.startswith("src/"))] = "0"*64
    elif mutation == "extra":
        files["scripts/consolidation_long_extra.py"] = "0"*64
    elif mutation == "removed":
        del files["scripts/consolidation_long_pose.py"]
    elif mutation == "reverted":
        files[A.LAUNCHER] = inputs["source_files"][A.LAUNCHER]
    else:
        del files[A.ADDED_PATHS[0]]
    with pytest.raises(ValueError, match="outside the closed amendment list"):
        A.validate_inputs(inputs, files, texts, C.digest)


def test_v1_manifest_must_name_the_original_source_recipe_and_phase_commits():
    inputs, (files, texts) = manifest(), current()
    for change in ({"source_sha256": "0"*64}, {"recipe_sha256": "0"*64}, {"amendment": "A2"}, {"kind": "other"},
                   {"phases": {**inputs["phases"], "extract": {"code_sha": "0"*40, "plan_sha256": "0"*64}}},
                   {"source_files": {**inputs["source_files"], "scripts/consolidation_long.py": "0"*64}}):
        with pytest.raises(ValueError, match="manifest identity differs"):
            A.validate_inputs({**inputs, **change}, files, texts, C.digest)
    # The same source on both sides is not an amendment.
    with pytest.raises(ValueError, match="outside the closed amendment list"):
        A.validate_inputs(inputs, dict(inputs["source_files"]), texts, C.digest)


def test_v1_manifest_is_reproduced_from_git_objects_and_collected_receipts():
    result = subprocess.run([sys.executable, str(TOOLS / "build_inputs.py"), "--check"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == A.INPUT_MANIFEST


# ----------------------------------------------------------------------------------------------
# V-2  Frozen text inside the changed files
# ----------------------------------------------------------------------------------------------

def test_v2_extraction_gates_and_policies_keep_the_text_of_the_original_phase_commit():
    texts = current()[1]
    functions = {
        A.COMMON: ("load_teacher", "verify_training_receipt", "source_identity", "recipe_identity", "finish_output",
                   "file_pin", "pinned", "read_pin", "write_json", "canonical", "digest"),
        A.CLASSIFICATION: ("resource_gate", "_check_benchmark", "_cache", "_historical", "_historical_report",
                           "_historical_reports", "_historical_numeric", "_historical_timing", "_numeric_match",
                           "_threads", "_policy", "_receipt", "shard_key"),
        A.POSE: ("_extract", "_reference", "_entry", "_benchmark", "project_benchmark", "_fit", "_compare",
                 "_threads", "environment", "peak_rss_bytes", "_save_array"),
    }
    constants = {A.COMMON: ("PROTOCOL", "DOSES", "OUTPUT_ROOT", "READING_RULE", "RESOURCE_POLICY"),
                 A.CLASSIFICATION: ("BENCHMARK_POLICY",), A.POSE: ("STAGES", "FIT_RESOURCES", "BENCHMARK_POLICY")}
    for path, names in functions.items():
        old = original(path)
        for name in names:
            assert segment(texts[path], name) == segment(old, name), (path, name)
        new, before = A.definitions(texts[path]), A.definitions(old)
        assert all(new[name] == before[name] for name in constants[path]), path
    # The extraction and benchmark branches of the classification dispatcher: same text, now unreachable.
    spec = A.FROZEN_BLOCKS[A.CLASSIFICATION]
    assert A.blocks(texts[A.CLASSIFICATION], spec) == A.blocks(original(A.CLASSIFICATION), spec) \
        == manifest()["blocks"][A.CLASSIFICATION]
    assert set(spec["execute"]) == {"stage == 'extract-classification'", "stage == 'benchmark-classification'"}


def test_v2_only_the_listed_definitions_and_launcher_lines_differ_from_git():
    inputs, texts = manifest(), current()[1]
    for path in A.AMENDED_DEFINITIONS:
        new, old = A.definitions(texts[path]), A.definitions(original(path))
        assert old == inputs["definitions"][path] and list(old) == inputs["order"][path]
        assert list(old)[-1] == A.END_OF_FILE and set(A.ADDED_AFTER[path]) == set(A.ADDED_DEFINITIONS[path])
        assert {key for key in old if new.get(key) != old[key]} == set(A.AMENDED_DEFINITIONS[path])
        assert set(new) - set(old) == set(A.ADDED_DEFINITIONS[path]) and not set(old) - set(new)
    new, old, frame = texts[A.LAUNCHER].split("\n"), original(A.LAUNCHER).split("\n"), inputs["launcher"]
    assert (frame["prefix_lines"], frame["suffix_lines"], frame["replaced_lines"], frame["lines"]) == (51, 15, [52, 53], 67)
    assert len(old) == 51 + 2 + 15 and new[:51] == old[:51] and new[-15:] == old[-15:]
    # The two replaced lines are the DB2 special case; the new ones only choose the walltime of the shard.
    assert old[51:53] == ["if args.stage=='fit-classification' and args.dataset=='ninapro_db2':",
                          " policy=plan['resources']['classification_fit_db2']"]
    assert len(new[51:-15]) == 6 and new[51] == "if args.stage=='fit-classification':"
    assert new[56] == " policy={**policy,'seconds':table[shard]['walltime_seconds']}"


def test_v2_regions_partition_each_changed_file_byte_for_byte():
    """Every byte of a changed file belongs to exactly one keyed region: nothing escapes the hashes."""
    for path in A.AMENDED_DEFINITIONS:
        for text in (current()[1][path], original(path)):
            parts = list(A.regions(text).values())
            assert "\n".join(parts[:-1]) + parts[-1] == text, path
            assert list(A.regions(text)) == list(A.definitions(text))
    # A decorator, a comment and the blank lines before a statement belong to its region.
    sample = "import os\n\n# note\n@decorate\ndef f():\n    return 1\nX = 2"
    assert A.regions(sample) == {"import os": "import os", "f": "\n# note\n@decorate\ndef f():\n    return 1",
                                 "X": "X = 2", A.END_OF_FILE: ""}
    assert list(A.regions(sample + "\n").values())[-1] == "\n"
    with pytest.raises(ValueError, match="top-level statements share a line"):
        A.regions("X = 1; Y = 2\n")


@pytest.mark.parametrize("mutation,message", [
    ("decorator", "frozen definition changed"),
    ("comment", "frozen definition changed"),
    ("blank_line", "frozen definition changed"),
    ("final_newline", "frozen definition changed"),
    ("swap", "top-level statements differ from the closed list or its order"),
    ("moved_added", "top-level statements differ from the closed list or its order"),
    ("removed_frozen", "top-level statements differ from the closed list or its order"),
])
def test_v2_decorators_comments_and_order_outside_the_closed_list_are_refused(mutation, message):
    inputs, (files, texts) = manifest(), current()
    A.validate_inputs(inputs, files, texts, C.digest)

    def once(path, old, new):
        assert texts[path].count(old) == 1, old
        texts[path] = texts[path].replace(old, new)

    if mutation == "decorator":     # a frozen function wrapped without touching its own lines
        once(A.CLASSIFICATION, "\ndef resource_gate(",
             '\n@(lambda f: (lambda *a, **k: {**f(*a, **k), "all_passed": True}))\ndef resource_gate(')
    elif mutation == "comment":
        once(A.POSE, "\ndef _fit(ctx, out, step, benchmark):", "\n# note\ndef _fit(ctx, out, step, benchmark):")
    elif mutation == "blank_line":
        once(A.COMMON, "\nDOSES = (19618, 39237, 49046)\n", "\n\nDOSES = (19618, 39237, 49046)\n")
    elif mutation == "final_newline":
        assert texts[A.POSE].endswith("\n") and not texts[A.POSE].endswith("\n\n")
        texts[A.POSE] = texts[A.POSE][:-1]
    elif mutation == "swap":        # same statements, same text, another order
        protocol = "PROTOCOL = 'consolidation-long-evaluation-20261105-v1'\n"
        doses = "DOSES = (19618, 39237, 49046)\n"
        once(A.COMMON, protocol + doses, doses + protocol)
    elif mutation == "moved_added":  # an added definition anywhere but its declared place
        text = texts[A.CLASSIFICATION]
        gate = A.regions(text)["amended_gate"]
        assert text.count(gate) == 1
        texts[A.CLASSIFICATION] = text.replace(gate, "", 1) + gate.lstrip("\n") + "\n"
    else:
        text = texts[A.POSE]
        region = A.regions(text)["peak_rss_bytes"]
        assert text.count(region) == 1
        texts[A.POSE] = text.replace(region, "", 1)
    with pytest.raises(ValueError, match=message):
        A.validate_inputs(inputs, files, texts, C.digest)


def test_v2_closed_list_of_definitions_must_be_consistent_with_the_manifest_order():
    inputs, (files, texts) = manifest(), current()
    for change in (lambda v: v["order"][A.COMMON].reverse(),
                   lambda v: v["order"][A.POSE].pop(),
                   lambda v: v["order"][A.CLASSIFICATION].append("extra"),
                   lambda v: v["definitions"][A.COMMON].pop("digest")):
        value = copy.deepcopy(inputs)
        change(value)
        with pytest.raises(ValueError, match="closed list of definitions is inconsistent|top-level statements differ"):
            A.validate_inputs(value, files, texts, C.digest)


@pytest.mark.parametrize("path,old,new,message", [
    (A.CLASSIFICATION, '"safety_factor": 3.0', '"safety_factor": 2.0', "frozen definition changed"),
    (A.CLASSIFICATION, "fold_costs.append(max(scaled,", "fold_costs.append(min(scaled,", "frozen definition changed"),
    (A.CLASSIFICATION, "_threads(1)", "_threads(8)", "frozen stage block changed"),
    (A.CLASSIFICATION, "benchmark requires one dataset/view at19618", "benchmark requires a dataset",
     "frozen stage block changed"),
    (A.COMMON, "DOSES = (19618, 39237, 49046)", "DOSES = (19618, 39237)", "frozen definition changed"),
    (A.COMMON, "def load_teacher(ctx, step, device='cpu'):", "def load_teacher(ctx, step, device='cuda'):",
     "frozen definition changed"),
    (A.POSE, '"safety_factor": 2., "fixed_seconds": 120.,', '"safety_factor": 1., "fixed_seconds": 120.,',
     "frozen definition changed"),
    (A.POSE, "def _fit(ctx, out, step, benchmark):", "def _fit(ctx, out, step, benchmark=None):",
     "frozen definition changed"),
    (A.LAUNCHER, "#SBATCH --mem=30G", "#SBATCH --mem=60G", "launcher changed outside its allocation lines"),
    (A.LAUNCHER, "raise SystemExit('wrong walltime')", "pass", "launcher changed outside its allocation lines"),
])
def test_v2_a_change_outside_the_listed_definitions_is_refused(path, old, new, message):
    inputs, (files, texts) = manifest(), current()
    assert texts[path].count(old) == 1
    texts[path] = texts[path].replace(old, new)
    with pytest.raises(ValueError, match=message):
        A.validate_inputs(inputs, files, texts, C.digest)


def test_v2_new_top_level_statements_and_shadowing_redefinitions_are_refused():
    inputs, (files, texts) = manifest(), current()
    for path in A.AMENDED_DEFINITIONS:
        for extra in ("\nEXTRA = 1\n", "\nif True:\n    pass\n", "\nimport os as system\n"):
            with pytest.raises(ValueError, match="top-level statements differ from the closed list"):
                A.validate_inputs(inputs, files, {**texts, path: texts[path] + extra}, C.digest)
    # A frozen function or constant cannot be replaced by a later definition of the same name.
    for path, extra in ((A.CLASSIFICATION, "\ndef resource_gate(reports):\n    return {}\n"),
                        (A.CLASSIFICATION, "\nBENCHMARK_POLICY = {}\n"), (A.POSE, "\nFIT_RESOURCES = {}\n"),
                        (A.COMMON, "\ndef load_teacher(ctx, step):\n    return None\n")):
        with pytest.raises(ValueError, match="duplicate top-level definition"):
            A.validate_inputs(inputs, files, {**texts, path: texts[path] + extra}, C.digest)
    with pytest.raises(ValueError, match="top-level statements differ from the closed list"):
        A.validate_inputs(inputs, files, {**texts, A.CLASSIFICATION: texts[A.CLASSIFICATION]
                                          + '\nBENCHMARK_POLICY["safety_factor"] = 1.0\n'}, C.digest)


# ----------------------------------------------------------------------------------------------
# V-3  Closed stages
# ----------------------------------------------------------------------------------------------

def test_v3_closed_stages_are_the_two_original_phases_and_every_such_row_already_ran():
    import consolidation_long as entry
    assert set(entry.STAGES) - set(A.CLOSED_STAGES) == {"fit-classification", "fit-pose", "compare-classification",
                                                        "compare-pose", "summary"}
    assert plan()["amendment"]["closed_stages"] == list(A.CLOSED_STAGES)
    rows = launch()["jobs"]
    assert all((row["stage"] in A.CLOSED_STAGES) == (row["phase"] in A.ORIGINAL_PHASES) for row in rows)
    closed = [row for row in rows if row["stage"] in A.CLOSED_STAGES]
    assert Counter(row["stage"] for row in closed) == A.INPUT_KINDS and sum(A.INPUT_KINDS.values()) == 28
    assert all(type(row["slurm_job_id"]) is int for row in closed)


# ----------------------------------------------------------------------------------------------
# V-4  Closed list of the 28 inputs, exact pins
# ----------------------------------------------------------------------------------------------

def test_v4_closed_list_is_the_28_collected_receipts_of_the_original_phases():
    inputs = manifest()
    rows = {row["id"]: row for row in launch()["jobs"] if row["phase"] in A.ORIGINAL_PHASES}
    listed = inputs["receipts"]
    assert len(listed) == 28 == len(rows) and {r["id"] for r in listed} == set(rows)
    assert Counter(r["kind"] for r in listed) == A.INPUT_KINDS
    assert len({r["path"] for r in listed}) == len({r["sha256"] for r in listed}) == 28
    for r in listed:
        raw = (CAMPAIGN / "execution/receipts" / (r["id"] + ".json")).read_bytes()
        report, row = json.loads(raw), rows[r["id"]]
        assert hashlib.sha256(raw).hexdigest() == r["sha256"] and len(raw) == r["bytes"]
        assert r["path"] == row["out"] + "/report.json" and r["kind"] == row["stage"] == report["kind"]
        assert r["phase"] == row["phase"] and report["complete"] is True and report["protocol"] == C.PROTOCOL
        assert report["source_sha256"] == A.ORIGINAL_SOURCE and report["recipe_sha256"] == A.ORIGINAL_RECIPE
        assert {k: report[k] for k in ("code_sha", "plan_sha256")} == A.ORIGINAL_PHASES[r["phase"]]
        assert "amendment" not in report and "artifacts" in r["fields"] and "checkpoint_sha256" in r["fields"]
        assert all(report[key] == value for key, value in r["fields"].items())
    pins = [{k: r[k] for k in PINNED} for r in listed]
    caches = json.loads((ROOT / inputs["cache_manifest"]["path"]).read_text())
    assert inputs["cache_manifest"] == plan()["cache_manifest"] and caches["source_sha256"] == A.ORIGINAL_SOURCE
    entries = [entry for task in ("classification", "pose") for by in caches[task].values() for entry in by.values()]
    assert len(entries) == 21 == inputs["cache_payloads"]
    assert all(pins.count(entry["receipt"]) == 1 for entry in entries)
    assert {json.dumps(entry["receipt"], sort_keys=True) for entry in entries} \
        == {json.dumps(pin, sort_keys=True) for pin, r in zip(pins, listed) if r["phase"] == "extract"}
    # Each benchmark names a listed extraction receipt (classification) or the pinned cache manifest (pose).
    for r in listed:
        if r["kind"] == "benchmark-classification":
            assert pins.count(r["fields"]["extraction_receipt"]) == 1
        elif r["kind"] == "benchmark-pose":
            assert r["fields"]["cache_manifest"] == inputs["cache_manifest"]
            assert r["fields"]["train_cache"] == caches["pose"]["19618"]["train"]


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Context of the concrete plan in this checkout.

    Only what lives on the cluster is replaced: the 28 receipts are read from
    their collected copies with the same SHA-256 and size check, the other
    cluster files are not opened, and the Git state of the working tree is not
    consulted. Sources, plan, manifests and receipts are the real ones.
    """
    local = {r["path"]: CAMPAIGN / "execution/receipts" / (r["id"] + ".json") for r in manifest()["receipts"]}
    real = C.pinned

    def pinned(pin):
        if pin["path"] in local:
            raw = local[pin["path"]].read_bytes()
            C.require(hashlib.sha256(raw).hexdigest() == pin["sha256"], "SHA mismatch: " + pin["path"])
            C.require("bytes" not in pin or pin["bytes"] == len(raw), "pinned size differs")
            return local[pin["path"]]
        if str(pin["path"]).startswith("/leonardo"):
            return Path(pin["path"])
        return real(pin)

    monkeypatch.setattr(C, "pinned", pinned)
    monkeypatch.setattr(C.T.P, "code_identity", lambda expected: expected)

    def load(mutate=None, decision=DECISION):
        value = plan()
        value["amendment"]["decision"] = decision
        if mutate is not None:
            mutate(value)
        path = tmp_path / f"plan{len(list(tmp_path.glob('plan*.json')))}.json"
        path.write_text(json.dumps(value))
        return C.load_context(path, "c"*40)

    return load


def test_v4_context_of_the_concrete_plan_reads_each_real_receipt_only_as_its_listed_kind(campaign):
    ctx = campaign()
    assert ctx["source_sha256"] == C.source_identity() != ctx["inputs"]["source_sha256"] == A.ORIGINAL_SOURCE
    assert ctx["amendment"] == A.record() == {"id": "A1", "input_compatibility": A.INPUT_MANIFEST,
                                              "input_source_sha256": A.ORIGINAL_SOURCE}
    assert ctx["caches"]["source_sha256"] == A.ORIGINAL_SOURCE and ctx["plan"]["recipe_sha256"] == A.ORIGINAL_RECIPE
    for r in ctx["inputs"]["receipts"]:
        pin = {k: r[k] for k in PINNED}
        assert C.verified_input(ctx, pin, r["kind"]) == receipt(r["id"])
        for other in set(A.INPUT_KINDS) - {r["kind"]}:
            with pytest.raises(ValueError, match="not a listed amendment input"):
                C.verified_input(ctx, pin, other)
        for change in ({"sha256": "0"*64}, {"bytes": r["bytes"] + 1}, {"path": r["path"] + ".copy"}):
            with pytest.raises(ValueError, match="not a listed amendment input"):
                C.verified_input(ctx, {**pin, **change}, r["kind"])
    # Every cache of the manifest is reached through its listed extraction receipt.
    for step, entries in ctx["caches"]["classification"].items():
        for entry in entries.values():
            C.verified_extraction(ctx, entry, int(step), "extract-classification")
    for step, entries in ctx["caches"]["pose"].items():
        for split, entry in entries.items():
            C.verified_extraction(ctx, entry, int(step), "extract-pose-" + split)
            with pytest.raises(ValueError, match="extraction receipt differs"):
                C.verified_extraction(ctx, entry, 19618 if int(step) != 19618 else 39237, "extract-pose-" + split)


def test_v4_listed_pin_with_other_content_and_new_source_inputs_are_refused(campaign, tmp_path, monkeypatch):
    ctx = campaign()
    r = next(r for r in ctx["inputs"]["receipts"] if r["kind"] == "benchmark-pose")
    pin = {k: r[k] for k in PINNED}
    report = receipt(r["id"])
    for name, change in (("new source", {"source_sha256": ctx["source_sha256"]}), ("recipe", {"recipe_sha256": "0"*64}),
                         ("code", {"code_sha": A.ORIGINAL_PHASES["extract"]["code_sha"]}),
                         ("plan", {"plan_sha256": A.ORIGINAL_PHASES["extract"]["plan_sha256"]}),
                         ("incomplete", {"complete": False}), ("field", {"checkpoint_step": 39237}),
                         ("artifacts", {"artifacts": {}})):
        monkeypatch.setattr(C, "read_pin", lambda _, value={**report, **change}: value)
        with pytest.raises(ValueError, match="amendment input receipt provenance differs"):
            C.verified_input(ctx, pin, "benchmark-pose")
    monkeypatch.undo()


@pytest.mark.parametrize("mutation,message", [
    ("no_decision", "decision of Simone"), ("source", "evaluation sources differ from reviewed plan"),
    ("pin", "amendment A1 policy or inputs differ"), ("policy", "amendment A1 policy or inputs differ"),
    ("table", "amended fit shard walltime differs"), ("shard", "amended fit shard matrix differs"),
    ("cache_manifest", "amendment inputs differ from this evaluation"), ("review", "Claude review has not converged"),
    ("benchmark_policy", "benchmark policy changed")])
def test_context_refuses_a_plan_without_decision_or_with_another_source_table_or_input_list(campaign, tmp_path, mutation, message):
    def mutate(value):
        block = value["amendment"]
        key = R.shard_key(39237, "pre", "ninapro_db2", [0])
        if mutation == "source":
            value["source_sha256"] = A.ORIGINAL_SOURCE
        elif mutation == "pin":
            block["input_compatibility"] = {**A.INPUT_MANIFEST, "sha256": "0"*64}
        elif mutation == "policy":
            block["fit_policy"] = {**A.FIT_POLICY, "walltime_reserve_seconds": 0.}
        elif mutation == "table":
            block["fit_shards"][key]["walltime_seconds"] += 900
        elif mutation == "shard":
            block["fit_shards"][R.shard_key(39237, "pre", "ninapro_db2", [0, 1])] = block["fit_shards"].pop(key)
        elif mutation == "cache_manifest":
            copy_path = tmp_path / "cache_manifest.json"
            copy_path.write_bytes((ROOT / value["cache_manifest"]["path"]).read_bytes() + b"\n")
            value["cache_manifest"] = C.file_pin(copy_path)
        elif mutation == "review":
            value["review"]["status"] = "pending"
        elif mutation == "benchmark_policy":
            value["classification_benchmark_policy"]["physical_fit_jobs"] = 69
    with pytest.raises(ValueError, match=message):
        campaign(mutate, decision=None if mutation == "no_decision" else DECISION)


# ----------------------------------------------------------------------------------------------
# V-5  Pose: original source in caches and benchmark, original policy and resources
# ----------------------------------------------------------------------------------------------

def test_v5_real_pose_benchmark_is_admitted_through_the_bridge_with_the_original_policy(campaign):
    ctx = campaign()
    r = next(r for r in ctx["inputs"]["receipts"] if r["kind"] == "benchmark-pose")
    pin, report = {k: r[k] for k in PINNED}, receipt(r["id"])
    ctx["plan"]["benchmark_receipts"] = {"classification": None, "pose": pin}
    assert U._benchmark_gate(ctx, check_environment=False) == pin
    assert report["benchmark_policy"] == U.BENCHMARK_POLICY == ctx["plan"]["pose_benchmark_policy"]
    assert report["fit_resources"] == U.FIT_RESOURCES == ctx["plan"]["resources"]["pose_fit"]
    projection = U.project_benchmark(report["timings_seconds"], report["peak_rss_bytes"])
    assert projection == report["projection"] and projection["admitted"] is True
    assert projection["seconds_per_dose"] <= U.BENCHMARK_POLICY["max_seconds_per_dose"] < U.FIT_RESOURCES["seconds"]
    assert report["source_sha256"] == A.ORIGINAL_SOURCE and report["held_payload_opened"] is False
    # The numeric environment of the fit must still equal the benchmark one: unchanged, checked on the node.
    with pytest.raises(ValueError, match="numeric environment differs"):
        U._benchmark_gate(ctx)
    # The three pose rows are unchanged and are released before the classification fits.
    rows = launch()["jobs"]
    pose = [i for i, row in enumerate(rows) if row["stage"] == "fit-pose"]
    fits = [i for i, row in enumerate(rows) if row["stage"] == "fit-classification"]
    assert len(pose) == 3 and max(pose) < min(fits)
    assert all(rows[i]["allocation"] == U.FIT_RESOURCES and "amendment" not in rows[i] for i in pose)


# ----------------------------------------------------------------------------------------------
# V-6 / V-7  Original gate intact, amended gate, frozen table, launcher
# ----------------------------------------------------------------------------------------------

def test_v6_real_receipts_keep_the_original_policy_and_fail_the_original_gate_15_of_48():
    reports, value = benchmarks(), plan()
    assert all(reports[d][v]["policy"] == R.BENCHMARK_POLICY == value["classification_benchmark_policy"]
               for d in M.DATASETS for v in M.VIEWS)
    assert R.BENCHMARK_POLICY["physical_fit_jobs"] == 48 and R.BENCHMARK_POLICY["max_projected_seconds"] == 13800.
    gate = R.resource_gate(reports)
    rejected = sorted(key for key, shard in gate["shards"].items() if not shard["passed"])
    assert gate["all_passed"] is False and len(gate["shards"]) == 48 and len(rejected) == 15
    assert sum(not cell["passed"] for cell in gate["cells"].values()) == 10
    assert {key.split("/")[2] for key in rejected} == {"ninapro_db2", "ninapro_db3", "ninapro_db6"}
    assert gate["projected_core_hours"] == pytest.approx(1087.621716, abs=1e-6)
    assert R.amended_gate(reports)["original_gate"] == {"all_passed": False, "physical_fit_jobs": 48,
                                                        "rejected_shards": rejected}


def test_v7_amended_gate_on_the_real_receipts_equals_the_table_frozen_in_the_plan():
    reports, table = benchmarks(), plan()["amendment"]["fit_shards"]
    original, gate = R.resource_gate(reports), R.amended_gate(reports, table)
    assert gate["all_passed"] is True and gate["physical_fit_jobs"] == 69 == len(table) and gate["new_fit_cells"] == 40
    assert set(table) == set(registered()) == set(gate["shards"])
    for key, shard in gate["shards"].items():
        cell = original["cells"]["/".join(key.split("/")[:3])]
        bench = reports[shard["dataset"]][shard["view"]]["benchmark"]
        # Unchanged per-fold projection, fixed seconds and load/scale overhead of the original policy.
        assert shard["projected_seconds"] == sum(cell["projected_fit_seconds_by_fold"][f] for f in shard["folds"]) \
            + 600. + 2*(bench["load_s"] + bench["scale_s"])
        assert table[key]["walltime_seconds"] == shard["walltime_seconds"] == A.walltime_seconds(shard["projected_seconds"])
        assert table[key]["projected_seconds"] == shard["projected_seconds"] and table[key]["folds"] == shard["folds"]
        assert shard["projected_seconds"] <= shard["walltime_seconds"] - 600 and shard["walltime_seconds"] <= 14400
        assert shard["projected_peak_rss_bytes"] == cell["projected_peak_rss_bytes"] <= .85*30*1024**3
    unchanged = [key for key in gate["shards"] if key in original["shards"]]
    assert len(unchanged) == 33 and all(original["shards"][key]["passed"] and gate["shards"][key]["projected_seconds"]
                                        == original["shards"][key]["projected_seconds"] for key in unchanged)
    assert Counter(shard["walltime_seconds"]//60 for shard in gate["shards"].values()) == {
        30: 8, 45: 3, 75: 3, 90: 3, 120: 6, 135: 6, 150: 15, 165: 6, 195: 9, 210: 5, 225: 5}
    assert gate["allocated_core_hours"] == 1276. and gate["projected_core_hours"] == pytest.approx(1115.658874, abs=1e-6)
    slack = min(s["walltime_seconds"] - 600 - s["projected_seconds"] for s in gate["shards"].values())
    assert slack == pytest.approx(40.48, abs=.01)
    assert max(s["projected_peak_rss_bytes"] for s in gate["shards"].values())/1024**3 == pytest.approx(4.078, abs=1e-3)
    # A table that is not the one implied by the receipts stops every fit.
    for change in ({"walltime_seconds": 14400}, {"projected_seconds": 1.}, {"folds": [0, 1]}):
        changed = copy.deepcopy(table)
        changed[R.shard_key(49046, "pre", "ninapro_db3", [2])].update(change)
        with pytest.raises(ValueError, match="frozen amendment walltime table differs"):
            R.amended_gate(reports, changed)
    # A slower receipt cannot be absorbed by a longer walltime: the frozen table no longer matches.
    slower = copy.deepcopy(reports)
    for fit in slower["ninapro_db2"]["pre"]["benchmark"]["fits"]:
        fit["elapsed_s"] *= 1.5
    with pytest.raises(ValueError, match="frozen amendment walltime table differs"):
        R.amended_gate(slower, table)


def test_v7_real_receipts_pass_the_runtime_bridge_and_the_amended_gate(campaign, tmp_path, monkeypatch):
    ctx = campaign()
    listed = {r["id"]: {k: r[k] for k in PINNED} for r in ctx["inputs"]["receipts"]}
    pins = {}
    for row in launch()["jobs"]:
        if row["stage"] == "benchmark-classification":
            report = receipt(row["id"])
            pins.setdefault(report["dataset"], {})[report["view"]] = listed[row["id"]]
    C.write_json(tmp_path / "benchmarks.json", {"reports": pins})
    ctx["plan"]["benchmark_receipts"] = {"classification": C.file_pin(tmp_path / "benchmarks.json"), "pose": None}
    # The historical reports live on the cluster: their numeric reference is the one each receipt recorded.
    any_report = receipt(next(iter(listed.keys() & {row["id"] for row in launch()["jobs"]
                                                    if row["stage"] == "benchmark-classification"})))
    evidence = any_report["historical_numeric_match"]
    monkeypatch.setattr(R, "_historical_numeric", lambda ctx, historical: {
        "numeric_sha256": any_report["historical_timing"]["numeric_sha256"],
        "numeric_context": evidence["historical_numeric_context"], "reports": evidence["historical_reports"]})
    reports = R._benchmarks(ctx)
    assert reports == benchmarks()
    gate = R.amended_gate(reports, ctx["plan"]["amendment"]["fit_shards"])
    assert gate["all_passed"] and not gate["original_gate"]["all_passed"]
    # An unlisted copy of a real receipt is not an input, even with the right content.
    copy_path = tmp_path / "copy.json"
    copy_path.write_bytes((CAMPAIGN / "execution/receipts" / "LONG-BENCH-epn612-post-20261105.json").read_bytes())
    pins["epn612"]["post"] = C.file_pin(copy_path)
    C.write_json(tmp_path / "copies.json", {"reports": pins})
    ctx["plan"]["benchmark_receipts"]["classification"] = C.file_pin(tmp_path / "copies.json")
    with pytest.raises(ValueError, match="not a listed amendment input"):
        R._benchmarks(ctx)


def allocation(monkeypatch, capsys, arguments, limit, cpus=8, memory=30720):
    """Run the allocation check embedded in the launcher against a given SLURM time limit."""
    text = (ROOT / A.LAUNCHER).read_text()
    code = text.split("python3 - \"$@\" <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["-", "--plan", str(CAMPAIGN / "plan.json"), "--expected-code", "c"*40,
                                      "--out", "/unused", *arguments])
    for name, value in {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": cpus, "SLURM_MEM_PER_NODE": memory,
                        "SLURM_JOB_NUM_NODES": 1}.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **k: f"JobId=1 JobName=x TimeLimit={limit} Partition=lrd_all_serial\n")
    capsys.readouterr()
    exec(compile(code, "launcher-allocation", "exec"), {"__name__": "launcher"})
    return json.loads(capsys.readouterr().out)


def hhmmss(seconds):
    return "%02d:%02d:%02d" % (seconds//3600, seconds//60 % 60, seconds % 60)


def test_v7_launcher_accepts_only_the_walltime_frozen_for_the_registered_shard(monkeypatch, capsys):
    table = plan()["amendment"]["fit_shards"]
    for row in [r for r in launch()["jobs"] if r["stage"] == "fit-classification"]:
        frozen = table[row["shard"]]["walltime_seconds"]
        assert row["allocation"]["seconds"] == frozen
    cases = {"19618/pre/ninapro_db2/folds0": ["--step", "19618", "--dataset", "ninapro_db2", "--view", "pre", "--folds", "0"],
             "49046/pool/ninapro_db6/folds2-3": ["--step", "49046", "--dataset", "ninapro_db6", "--view", "pool", "--folds", "2", "3"],
             "39237/post/epn612/folds0-1-2-3": ["--step", "39237", "--dataset", "epn612", "--view", "post", "--folds", "0", "1", "2", "3"]}
    for key, arguments in cases.items():
        frozen = table[key]["walltime_seconds"]
        accepted = allocation(monkeypatch, capsys, ["--stage", "fit-classification", *arguments], hhmmss(frozen))
        assert accepted["event"] == "allocation_accepted" and accepted["allocation"] == {
            "gpus": 0, "cpus": 8, "memory_gib": 30, "seconds": frozen}
        for wrong in {frozen + 900, frozen - 900, 14400} - {frozen}:
            with pytest.raises(SystemExit, match="wrong walltime"):
                allocation(monkeypatch, capsys, ["--stage", "fit-classification", *arguments], hhmmss(wrong))
    assert len({table[key]["walltime_seconds"] for key in cases}) == 3
    # Superseded or missing fold shards have no allocation at all.
    for arguments in (["--step", "39237", "--dataset", "ninapro_db2", "--view", "pre", "--folds", "0", "1"],
                      ["--step", "39237", "--dataset", "ninapro_db3", "--view", "pre", "--folds", "0", "1", "2", "3"],
                      ["--step", "39237", "--dataset", "ninapro_db6", "--view", "pool"],
                      ["--step", "19618", "--dataset", "epn612", "--view", "pool", "--folds", "0", "1", "2", "3"]):
        with pytest.raises(SystemExit, match="unregistered amendment fit shard"):
            allocation(monkeypatch, capsys, ["--stage", "fit-classification", *arguments], "04:00:00")
    # CPUs and memory are still checked, and the other stages keep the allocations of the original policy.
    arguments = ["--stage", "fit-classification", *cases["19618/pre/ninapro_db2/folds0"]]
    limit = hhmmss(table["19618/pre/ninapro_db2/folds0"]["walltime_seconds"])
    with pytest.raises(SystemExit, match="wrong CPUs"):
        allocation(monkeypatch, capsys, arguments, limit, cpus=4)
    with pytest.raises(SystemExit, match="wrong RAM"):
        allocation(monkeypatch, capsys, arguments, limit, memory=61440)
    assert allocation(monkeypatch, capsys, ["--stage", "fit-pose", "--step", "39237"], "01:00:00")["allocation"] \
        == C.RESOURCE_POLICY["pose_fit"]
    assert allocation(monkeypatch, capsys, ["--stage", "compare-classification"], hhmmss(
        C.RESOURCE_POLICY["classification_compare"]["seconds"]))["allocation"] == C.RESOURCE_POLICY["classification_compare"]
    with pytest.raises(SystemExit, match="wrong walltime"):
        allocation(monkeypatch, capsys, ["--stage", "fit-pose", "--step", "39237"], "02:00:00")


# ----------------------------------------------------------------------------------------------
# V-8  Plans: policy block, decision gate, derivation
# ----------------------------------------------------------------------------------------------

def test_v8_plan_block_freezes_policy_table_inputs_and_requires_the_recorded_decision():
    value, shards = plan(), registered()
    block = value["amendment"]
    assert A.validate_block(block, shards, require_decision=False) is block
    assert block["fit_policy"] == A.FIT_POLICY and block["input_compatibility"] == A.INPUT_MANIFEST
    assert "before any fit" in block["post_hoc"] and "not a pre-registered rule" in block["post_hoc"]
    assert "post hoc" in A.FIT_POLICY["status"] and "not a pre-registered rule" in A.FIT_POLICY["status"]
    assert A.validate_block({**block, "decision": DECISION}, shards)["decision"] == DECISION
    for decision in (None, {}, "Simone", {**DECISION, "by": "GPT"}, {**DECISION, "date": "06/11"},
                     {**DECISION, "quote": "  "}, {**DECISION, "quote": None}, {"by": "Simone"}):
        with pytest.raises(ValueError, match="decision of Simone"):
            A.validate_block({**block, "decision": decision}, shards)
    for change in ({"id": "A2"}, {"closed_stages": []}, {"closed_stages": list(A.CLOSED_STAGES[:-1])},
                   {"fit_policy": {**A.FIT_POLICY, "max_walltime_seconds": 86400}},
                   {"input_compatibility": {**A.INPUT_MANIFEST, "bytes": 1}}):
        with pytest.raises(ValueError, match="policy or inputs differ"):
            A.validate_block({**block, "decision": DECISION, **change}, shards)
    key = R.shard_key(39237, "pool", "ninapro_db6", [0, 1])
    for change in ({"walltime_seconds": block["fit_shards"][key]["walltime_seconds"] + 900},
                   {"walltime_seconds": float(block["fit_shards"][key]["walltime_seconds"])},
                   {"projected_seconds": 1.}, {"folds": [0]}, {"step": 49046}, {"dataset": "ninapro_db2"}):
        table = copy.deepcopy(block["fit_shards"])
        table[key].update(change)
        with pytest.raises(ValueError, match="amended fit shard walltime differs"):
            A.validate_block({**block, "decision": DECISION, "fit_shards": table}, shards)
    table = copy.deepcopy(block["fit_shards"])
    del table[key]
    with pytest.raises(ValueError, match="amended fit shard matrix differs"):
        A.validate_block({**block, "decision": DECISION, "fit_shards": table}, shards)
    # The explicit exception replaces the original policy text, which is kept for the record.
    policy = value["phase_release"]["source_change_policy"]
    assert policy == launch()["phase_source_policy"]
    assert "accepted-source whitelist" in policy["original_r2i"]["action"]
    assert "generic accepted-source list" in policy["amendment_A1"]["not_allowed"]
    assert policy["amendment_A1"]["original_source_sha256"] == A.ORIGINAL_SOURCE
    assert policy["amendment_A1"]["amended_source_sha256"] == value["source_sha256"] == C.source_identity()
    assert value["recipe_sha256"] == A.ORIGINAL_RECIPE == C.recipe_identity(value)


def tool(name):
    spec = importlib.util.spec_from_file_location("amendment_a1_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v8_recording_the_decision_changes_only_the_decision_fields_and_opens_the_runtime_gate():
    value, matrix = plan(), launch()
    for block in (value["amendment"], value["phase_release"]["source_change_policy"]["amendment_A1"],
                  matrix["phase_source_policy"]["amendment_A1"]):
        block["decision"] = None
    with pytest.raises(ValueError, match="decision of Simone"):
        C.validate_plan(value)
    module = tool("record_decision")
    decided, decided_launch = module.record(value, matrix, "2026-11-06", "question of the test", "«answer of the test»")
    assert C.validate_plan(decided)["amendment"]["decision"] == {
        "by": "Simone", "date": "2026-11-06", "question": "question of the test", "quote": "«answer of the test»",
        "scope": module.SCOPE}
    assert decided["amendment"]["status"] == decided_launch["amendment_A1"]["status"] == "approved_by_simone"
    assert decided["phase_release"]["source_change_policy"] == decided_launch["phase_source_policy"]
    assert value["amendment"]["decision"] is None
    # Everything else is untouched: after blanking the decision fields the plans are the same.
    neutral = tool("apply_to_plans").neutral
    assert neutral(decided, decided_launch) == neutral(value, matrix)
    for arguments in (("06/11/2026", "q", "a"), ("2026-11-06", " ", "a"), ("2026-11-06", "q", ""), ("2026-11-06", "q", None)):
        with pytest.raises(SystemExit, match="literal quote required"):
            module.record(value, matrix, *arguments)
    with pytest.raises(SystemExit, match="already recorded"):
        module.record(decided, decided_launch, "2026-11-07", "q", "a")


def test_v8_fit_pin_is_derived_from_the_listed_receipts_and_writes_nothing_in_a_dry_run():
    before = sorted(p.name for p in (CAMPAIGN / "execution/manifests").iterdir())
    result = subprocess.run([sys.executable, str(TOOLS / "pin_phase.py"), "fit", "--dry-run"], cwd=ROOT,
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src:scripts"})
    assert result.returncode == 0, result.stderr[-2000:]
    summary = json.loads(result.stdout)
    assert summary["dry_run"] is True and summary["classification_fit_jobs"] == 69
    assert summary["original_rejected_shards"] == 15 and summary["allocated_core_hours"] == 1276.
    assert summary["pose_seconds_per_dose"] <= U.BENCHMARK_POLICY["max_seconds_per_dose"]
    pose = next(r for r in manifest()["receipts"] if r["kind"] == "benchmark-pose")
    assert summary["pins"]["benchmark_receipts"]["pose"] == {k: pose[k] for k in PINNED}
    assert sorted(p.name for p in (CAMPAIGN / "execution/manifests").iterdir()) == before
    held = plan()["benchmark_receipts"]
    assert held in ({"classification": None, "pose": None}, summary["pins"]["benchmark_receipts"])


def test_v8_committed_plans_are_the_derivation_from_the_base_plans_and_the_receipts():
    result = subprocess.run([sys.executable, str(TOOLS / "apply_to_plans.py"), "--check"], cwd=ROOT,
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src:scripts"})
    assert result.returncode == 0, result.stderr[-2000:]
    summary = json.loads(result.stdout)
    assert summary["source_sha256"] == C.source_identity() and summary["jobs"] == 103 and summary["superseded"] == 48
    assert summary["classification_fit_allocated_core_hours"] == 1276. and summary["future_allocated_core_hours"] == 1317.
    assert summary["new_gpu_hours"] == 0. and summary["conservative_total_core_hours"] <= 1711.666667
