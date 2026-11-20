"""r2: the resume check's own evidence pins (traces, start.json) survive in the published receipt?"""
import contrastive_target_resume
from test_contrastive_targets_readout import readout_fixture, B


def test_resume_trace_pins_in_receipt(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    produced = {}
    def compare(paths, *args, **kw):
        ev = {phase: {**{f"trace-{r}.jsonl": f"{phase}-trace-{r}" for r in range(4)},
                      **({"start.json": phase + "-start"} if phase != "continuous" else {})} for phase in paths}
        produced[len(produced)] = ev
        return {"status": "PASSED", "passed": True, "evidence_sha256": ev}
    monkeypatch.setattr(contrastive_target_resume, "compare_three_paths", compare)
    result = B.read_preflight(**kwargs)
    published = result["arms"]["CN"]["resume"]["evidence_sha256"]
    print("produced by compare_three_paths:", sorted(produced[0]), sorted(produced[0]["reload_a"]))
    print("published in receipt:", sorted(published), {k: sorted(v) for k, v in published.items()})
    assert result["status"] == "PASSED_FOR_REVIEW"
    assert "continuous" not in published and all("trace-0.jsonl" not in v for v in published.values())
