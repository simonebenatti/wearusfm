"""Scettico 1, R2-M2: real compare_three_paths output vs what the readout publishes.

The common fork is the real 1-rank CPU trainer fixture of the suite (fork 2, end 4).
Only the readout's own surrounding evidence uses the suite's readout fixture stubs.
"""
import copy
import json
from pathlib import Path

import contrastive_benchmark_readout as B
import contrastive_target_resume as C
from test_contrastive_targets_readout import readout_fixture
from test_contrastive_target_resume import common_paths  # noqa: F401 (module fixture)

REAL_COMPARE = C.compare_three_paths  # captured before any monkeypatch


def _real(common):
    paths, cfg, initial_sha, initial, fork_sha, _ = common
    return REAL_COMPARE(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)


def _show(label, evidence):
    print(label, sorted(evidence), {phase: sorted(pins) for phase, pins in evidence.items()})


def test_real_compare_three_paths_pins_traces_and_start(common_paths):
    result = _real(common_paths)
    assert result["passed"] and result["status"] == "PASSED"
    _show("REAL compare_three_paths evidence:", result["evidence_sha256"])
    assert set(result["evidence_sha256"]) == {"continuous", "reload_a", "reload_b"}
    assert all("trace-0.jsonl" in pins for pins in result["evidence_sha256"].values())
    assert all("start.json" in result["evidence_sha256"][p] for p in ("reload_a", "reload_b"))


def _rewrite(paths, mutate):
    originals = {}
    for folder in paths.values():
        path = folder / "trace-0.jsonl"
        originals[path] = path.read_bytes()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            mutate(row)
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    return originals


def test_receipt_drops_real_pins_and_authentication_misses_post_receipt_trace_edits(tmp_path, monkeypatch, common_paths):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    produced = []

    def compare(paths, *args, **kw):
        out = _real(common_paths)
        produced.append(copy.deepcopy(out["evidence_sha256"]))
        return out

    monkeypatch.setattr(C, "compare_three_paths", compare)
    result = B.read_preflight(**kwargs)
    assert result["status"] == "PASSED_FOR_REVIEW"
    _show("PRODUCED (CN):", produced[0])
    for arm in ("CN", "CR"):
        published = result["arms"][arm]["resume"]["evidence_sha256"]
        _show(f"PUBLISHED {arm}:", published)
        assert set(published) == {"reload_a", "reload_b"}
        assert not any(n.startswith("trace-") or n == "start.json" for pins in published.values() for n in pins)
    flat = json.dumps(result)
    for phase in ("continuous", "reload_a", "reload_b"):
        assert produced[0][phase]["trace-0.jsonl"] not in flat, "trace pin survives elsewhere in the receipt"
    for phase in ("reload_a", "reload_b"):
        assert produced[0][phase]["start.json"] not in flat, "start.json pin survives elsewhere"
    print("trace/start pins absent from the whole receipt: True")

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(result, allow_nan=False))
    paths = common_paths[0]
    cases = {
        "extra_key_and_compact_format": lambda row: row.update(post_receipt_edit=True),
        "same_fake_batch_sha_in_all_three_phases": lambda row: row.update(batch_sha256="0" * 64, input_sha256="1" * 64),
    }
    for name, mutate in cases.items():
        originals = _rewrite(paths, mutate)
        try:
            before = len(produced)
            assert B.authenticate_preflight(receipt) == result
            after = produced[before]
            changed = sorted(p for p in after if after[p]["trace-0.jsonl"] != produced[0][p]["trace-0.jsonl"])
            print(f"{name}: authenticate_preflight ACCEPTS; discarded trace pins would differ in {changed}")
            assert changed == ["continuous", "reload_a", "reload_b"]
        finally:
            for path, data in originals.items():
                path.write_bytes(data)
    # Control: a trace edit that changes the gate IS caught by recomputation.
    originals = _rewrite({"reload_b": paths["reload_b"]}, lambda row: row.update(batch_sha256="f" * 64))
    try:
        try:
            B.authenticate_preflight(receipt)
        except ValueError as exc:
            print("control (reload_b only, gate-changing):", exc)
        else:
            raise AssertionError("gate-changing edit was not caught")
    finally:
        for path, data in originals.items():
            path.write_bytes(data)
