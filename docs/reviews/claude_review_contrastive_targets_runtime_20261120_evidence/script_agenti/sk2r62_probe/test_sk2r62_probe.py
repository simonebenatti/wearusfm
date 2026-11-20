"""Skeptic n.2 probe for R6-2 (readout receipt vs common-fork resume evidence).

Only reads the code under test; every file written lives under pytest's tmp dirs.
"""
import hashlib
import json
from pathlib import Path

import contrastive_benchmark_readout as B
import contrastive_target_resume as C
from test_contrastive_targets_readout import readout_fixture
from test_contrastive_target_resume import common_paths  # noqa: F401  (module fixture: real CPU trainer)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _phase_dirs(kwargs, arm):
    return {"continuous": Path(kwargs["full_dirs"][arm]),
            "reload_a": Path(kwargs["resumed_dirs"][arm + "_A"]),
            "reload_b": Path(kwargs["resumed_dirs"][arm + "_B"])}


def _write_trace_files(kwargs):
    for arm in ("CN", "CR"):
        for phase, folder in _phase_dirs(kwargs, arm).items():
            for rank in range(4):
                (folder / f"trace-{rank}.jsonl").write_text(json.dumps({"update": 545, "rank": rank, "phase": phase}) + "\n")
            if phase != "continuous":
                (folder / "start.json").write_text(json.dumps({"protocol": "p", "phase": phase}))


def _compare_with_own_evidence(monkeypatch):
    """Same evidence block as compare_three_paths (target_resume.py:347-361), hashes taken now."""
    def compare(paths, *args, ranks=4, **kwargs):
        evidence = {}
        for phase, folder in paths.items():
            folder = Path(folder)
            row = {"summary.json": _sha(folder / "summary.json")} if (folder / "summary.json").exists() else {}
            if phase != "continuous":
                row["start.json"] = _sha(folder / "start.json")
            for rank in range(ranks):
                row[f"trace-{rank}.jsonl"] = _sha(folder / f"trace-{rank}.jsonl")
            evidence[phase] = row
        return {"status": "PASSED", "passed": True, "evidence_sha256": evidence}
    monkeypatch.setattr(C, "compare_three_paths", compare)


def test_p1_receipt_resume_evidence_keys(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    _write_trace_files(kwargs)
    _compare_with_own_evidence(monkeypatch)
    result = B.read_preflight(**kwargs)
    print("\nP1 status:", result["status"])
    for arm in ("CN", "CR"):
        ev = result["arms"][arm]["resume"]["evidence_sha256"]
        print("P1", arm, "resume.evidence_sha256 =", {k: sorted(v) for k, v in ev.items()})
    text = json.dumps(result)
    print("P1 'trace-' in receipt:", "trace-" in text, "| 'start.json' in receipt:", "start.json" in text,
          "| 'continuous' key in resume evidence:", any("continuous" in result["arms"][a]["resume"]["evidence_sha256"] for a in ("CN", "CR")))


def test_p2_authentication_after_trace_and_start_byte_changes(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    _write_trace_files(kwargs)
    _compare_with_own_evidence(monkeypatch)
    receipt = B.read_preflight(**kwargs)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    B.authenticate_preflight(path)
    print("\nP2 baseline authentication: OK")
    # Byte changes after the receipt, in files the comparator hashes but the receipt does not keep.
    changed = [kwargs["resumed_dirs"]["CN_A"] / "trace-2.jsonl", kwargs["full_dirs"]["CR"] / "trace-0.jsonl",
               kwargs["resumed_dirs"]["CR_B"] / "start.json"]
    for file in changed:
        file.write_text(file.read_text() + " ")
    try:
        B.authenticate_preflight(path)
        print("P2 after byte changes in", [f.parent.name + "/" + f.name for f in changed], "-> AUTHENTICATED")
    except ValueError as error:
        print("P2 after byte changes -> REJECTED:", error)


def test_p3_real_comparator_evidence_and_neutral_mutations(common_paths):
    paths, cfg, initial_sha, initial, fork_sha, _ = common_paths
    first = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
    print("\nP3 real compare_three_paths status:", first["status"], "passed:", first["passed"])
    print("P3 real evidence_sha256 keys:", {k: sorted(v) for k, v in first["evidence_sha256"].items()})
    originals = {}
    try:
        # (a) same JSON content, different bytes; (b) one consistent rewrite of a trace field in all three paths.
        start = paths["reload_a"] / "start.json"
        originals[start] = start.read_bytes()
        start.write_text(json.dumps(json.loads(start.read_text()), indent=4) + "\n\n")
        for phase, folder in paths.items():
            trace = folder / "trace-0.jsonl"
            originals[trace] = trace.read_bytes()
            rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
            for row in rows:
                row["rng_cpu_after_sha256"] = "0" * 64
            trace.write_text("".join(json.dumps(row, separators=(", ", ":  ")) + "\n" for row in rows))
        second = C.compare_three_paths(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
        strip = lambda r: {k: v for k, v in r.items() if k != "evidence_sha256"}
        print("P3 after mutations: status", second["status"], "passed", second["passed"],
              "| result without evidence identical:", strip(first) == strip(second),
              "| evidence identical:", first["evidence_sha256"] == second["evidence_sha256"])
        changed = {phase: sorted(name for name in first["evidence_sha256"][phase]
                                 if first["evidence_sha256"][phase][name] != second["evidence_sha256"][phase][name])
                   for phase in first["evidence_sha256"]}
        print("P3 changed pins per phase:", changed)
    finally:
        for file, data in originals.items():
            file.write_bytes(data)
