"""Criteri del collaudo su dbg dei bracci nuovi (`scripts/arms_collaudo.py`): un run sintetico che li supera, e ogni criterio che cade."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("arms_collaudo", REPO / "scripts" / "arms_collaudo.py")
AC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AC)

SHA = "72cf24abec5b8332011d83ce2b2fc7ae6bcc9cec"
STEPS = 1300
LISTING = """total 1349036
-rw-r--r--. 1 sbenatti interactive 454073464 Oct 20 17:06 checkpoint_postdecay.pt
-rw-r--r--. 1 sbenatti interactive 454073464 Oct 20 16:06 checkpoint_predecay.pt
-rw-r--r--. 1 sbenatti interactive 454073464 Oct 20 17:06 checkpoint.pt
-rw-r--r--. 1 sbenatti interactive      2677 Oct 20 12:03 config.json
"""


def _run(tmp_path, arm, steps=STEPS, edit=None):
    wj, wa = AC.ARM_WEIGHTS[arm]
    d = tmp_path / arm
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"jepa": {"jepa_weight": wj, "anchor_weight": wa, "keep_weight": 0.0, "branch_weight": 0.0}}))
    (d / "summary.json").write_text(json.dumps({"stopped": "max_steps", "steps": steps, "elapsed_s": 1000.0, "code": {"sha": SHA},
                                                "world_size": 4, "batch_per_process": 8, "accum_steps": 1, "gpu_max_memory_gib": 14.5}))
    recs = []
    for s in range(1, steps + 1):
        decay = 1.0 / (1.0 + s / 100)
        r = {"step": s, "windows": 32, "t_step_s": 0.74, "t_data_s": 0.005, "jepa": 0.5 * decay + 0.01,
             **{a: (0.3 + i / 10) * decay + 0.1 for i, a in enumerate(AC.ANCHORS)}}
        r["total"] = wj * r["jepa"] + wa * sum(r[a] for a in AC.ANCHORS)
        if s in AC.expected_evaluations(steps):
            r.update(student_collapse_ratio=3.0, teacher_collapse_ratio=2.5, student_erank=100.0, teacher_erank=90.0)
        recs.append(r)
    if edit:
        recs = edit(recs)
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


@pytest.mark.parametrize("arm", sorted(AC.ARM_WEIGHTS))
def test_a_clean_run_passes(tmp_path, arm):
    res = AC.check(_run(tmp_path, arm), arm, STEPS, SHA, LISTING)
    assert res["passed"], {k: v for k, v in res["criteria"].items() if not v["ok"]}
    assert res["descriptive"]["s_per_passo_mediana"] == pytest.approx(0.74)
    assert [r["step"] for r in res["descriptive"]["non_collasso"]] == [500, 1000, 1300]


def test_listing_parser_reads_sizes_and_names():
    assert AC.listing_sizes(LISTING) == {"checkpoint_postdecay.pt": 454073464, "checkpoint_predecay.pt": 454073464,
                                         "checkpoint.pt": 454073464, "config.json": 2677}


def test_expected_evaluations_include_the_last_step():
    assert AC.expected_evaluations(1300) == [500, 1000, 1300]
    assert AC.expected_evaluations(1000) == [500, 1000]


def _failed(res):
    return sorted(k for k, v in res["criteria"].items() if not v["ok"])


def test_weights_of_the_other_arm_fail(tmp_path):
    d = _run(tmp_path, "JEPA-only")
    assert _failed(AC.check(d, "ancore-only", STEPS, SHA, LISTING)) == ["composizione_della_loss", "pesi"]


def test_total_off_the_weighted_sum_fails(tmp_path):
    def edit(recs):
        recs[700]["total"] += 1e-3
        return recs
    res = AC.check(_run(tmp_path, "ancore-only", edit=edit), "ancore-only", STEPS, SHA, LISTING)
    assert _failed(res) == ["composizione_della_loss"]
    assert res["criteria"]["composizione_della_loss"]["detail"]["passi_fuori_tolleranza"] == [701]


def test_missing_anchor_component_fails(tmp_path):
    def edit(recs):
        del recs[10]["ms_slow"]
        return recs
    assert _failed(AC.check(_run(tmp_path, "JEPA-only", edit=edit), "JEPA-only", STEPS, SHA, LISTING)) == ["composizione_della_loss"]


def test_gap_in_the_records_fails(tmp_path):
    res = AC.check(_run(tmp_path, "JEPA-only", edit=lambda recs: recs[:400] + recs[401:]), "JEPA-only", STEPS, SHA, LISTING)
    assert "record" in _failed(res)


def test_non_finite_value_fails(tmp_path):
    def edit(recs):
        recs[5]["grad_norm"] = float("nan")
        return recs
    assert "record" in _failed(AC.check(_run(tmp_path, "JEPA-only", edit=edit), "JEPA-only", STEPS, SHA, LISTING))


def test_alarm_and_early_stop_fail(tmp_path):
    def edit(recs):
        recs[999]["alarm"] = "collasso"
        return recs[:1000]
    d = _run(tmp_path, "JEPA-only", edit=edit)
    s = json.loads((d / "summary.json").read_text())
    (d / "summary.json").write_text(json.dumps({**s, "stopped": "allarme: collasso", "steps": 1000}))
    (d / "STOP").write_text("allarme: collasso")
    res = AC.check(d, "JEPA-only", STEPS, SHA, LISTING)
    assert {"completo", "non_collasso", "record"} <= set(_failed(res))
    assert res["criteria"]["non_collasso"]["detail"]["allarmi"] == ["collasso"]


def test_loss_not_decreasing_fails(tmp_path):
    def edit(recs):
        return [dict(r, total=r["total"] + 1e-3 * r["step"], jepa=r["jepa"]) for r in recs]
    res = AC.check(_run(tmp_path, "JEPA-only", edit=edit), "JEPA-only", STEPS, SHA, LISTING)
    assert "apprendimento" in _failed(res)


def test_missing_or_empty_checkpoint_fails(tmp_path):
    d = _run(tmp_path, "ancore-only")
    no_post = "\n".join(l for l in LISTING.splitlines() if "postdecay" not in l)
    assert _failed(AC.check(d, "ancore-only", STEPS, SHA, no_post)) == ["checkpoint"]
    empty = LISTING.replace("454073464 Oct 20 16:06 checkpoint_predecay.pt", "0 Oct 20 16:06 checkpoint_predecay.pt")
    assert _failed(AC.check(d, "ancore-only", STEPS, SHA, empty)) == ["checkpoint"]


def test_wrong_code_or_ddp_shape_fails(tmp_path):
    d = _run(tmp_path, "ancore-only")
    s = json.loads((d / "summary.json").read_text())
    (d / "summary.json").write_text(json.dumps({**s, "world_size": 1}))
    assert _failed(AC.check(d, "ancore-only", STEPS, SHA, LISTING)) == ["codice_e_ddp"]
    (tmp_path / "other").mkdir()
    assert _failed(AC.check(_run(tmp_path / "other", "ancore-only"), "ancore-only", STEPS, "0" * 40, LISTING)) == ["codice_e_ddp"]


def test_unknown_arm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="braccio"):
        AC.check(_run(tmp_path, "JEPA-only"), "A", STEPS, SHA, LISTING)


def test_cli_exit_code_and_output(tmp_path):
    d = _run(tmp_path, "JEPA-only")
    ls = tmp_path / "ls.txt"
    ls.write_text(LISTING)
    out = tmp_path / "esito.json"
    assert AC.main([str(d), "--arm", "JEPA-only", "--steps", str(STEPS), "--expected-sha", SHA, "--listing", str(ls), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True
    assert AC.main([str(d), "--arm", "JEPA-only", "--steps", "1200", "--expected-sha", SHA, "--listing", str(ls)]) == 1


def test_cli_writes_the_outcome_even_with_a_non_finite_collapse_ratio(tmp_path):
    """Re-review del critico (21/10), rilievo 5: un rapporto di collasso NaN (che il trainer puo' scrivere) non deve impedire l'esito."""
    def edit(recs):
        recs[499]["student_collapse_ratio"] = float("nan")
        recs[-1]["total"] = float("inf")
        return recs
    d = _run(tmp_path, "JEPA-only", edit=edit)
    ls = tmp_path / "ls.txt"
    ls.write_text(LISTING)
    out = tmp_path / "esito.json"
    assert AC.main([str(d), "--arm", "JEPA-only", "--steps", str(STEPS), "--expected-sha", SHA, "--listing", str(ls), "--out", str(out)]) == 1
    res = json.loads(out.read_text())
    assert res["passed"] is False and not res["criteria"]["record"]["ok"] and not res["criteria"]["composizione_della_loss"]["ok"]
    assert res["descriptive"]["non_collasso"][0]["student_collapse_ratio"] is None
