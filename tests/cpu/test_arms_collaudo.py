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


@pytest.mark.parametrize("arm", sorted(a for a in AC.ARM_WEIGHTS if a not in AC.ARM_POSE_WEIGHT))  # R3: il suo test, con la posa
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
        AC.check(_run(tmp_path, "JEPA-only"), "inesistente", STEPS, SHA, LISTING)


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


# --- R3: posa (seconda ondata, docs/pilot_posa_v1.md §2 (iii) e §6 punto 7) -----------------------------------------------------------------

def _run_r3(tmp_path, steps=STEPS, edit=None, cfg_edit=None, rate=0.74):
    wj, wa = AC.ARM_WEIGHTS["R3"]
    wp = AC.ARM_POSE_WEIGHT["R3"]
    d = tmp_path / "R3"
    d.mkdir(parents=True)
    cfg = {"jepa": {"jepa_weight": wj, "anchor_weight": wa, "keep_weight": 0.0, "branch_weight": 0.0, "pose_weight": wp},
           "model": {"pose_head": True}, "loader": {"pose_root": "/scratch/emg2pose_pose"}, "pose_stats_sha256": "ab" * 32}
    if cfg_edit:
        cfg = cfg_edit(cfg)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "summary.json").write_text(json.dumps({"stopped": "max_steps", "steps": steps, "elapsed_s": 1000.0, "code": {"sha": SHA},
                                                "world_size": 4, "batch_per_process": 8, "accum_steps": 1, "gpu_max_memory_gib": 14.5}))
    recs = []
    for s in range(1, steps + 1):
        decay = 1.0 / (1.0 + s / 100)
        r = {"step": s, "windows": 32, "t_step_s": rate, "t_data_s": 0.005, "jepa": 0.5 * decay + 0.01,
             **{a: (3.0 + i) * decay + 0.5 for i, a in enumerate(AC.ANCHORS)}}  # ancore ~ 25 all'avvio (R0: 29,8), ~8 a 500 passi
        r["pose"] = 0.3 + 0.7 * decay  # parte da ~1 e scende (testa a zero su target standardizzati)
        r["pose_windows"] = 0.0 if s % 97 == 0 else 7.0  # un passo su 97 senza finestre di posa (~1%)
        r["total"] = wj * r["jepa"] + wa * sum(r[a] for a in AC.ANCHORS) + wp * r["pose"]
        if s in AC.expected_evaluations(steps):
            r.update(student_collapse_ratio=3.0, teacher_collapse_ratio=2.5, student_erank=100.0, teacher_erank=90.0)
        recs.append(r)
    if edit:
        recs = edit(recs)
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


def test_r3_clean_run_passes_the_gate_and_the_rate(tmp_path):
    res = AC.check(_run_r3(tmp_path), "R3", STEPS, SHA, LISTING)
    assert res["passed"], {k: v for k, v in res["criteria"].items() if not v["ok"]}
    c = res["criteria"]
    assert {"posa_configurata", "posa_scala_avvio", "posa_tetto", "posa_discesa", "posa_conteggio", "ritmo"} <= set(c)
    assert 0.1 <= c["posa_scala_avvio"]["detail"]["rapporto"] < 1.0 and c["posa_tetto"]["detail"]["rapporto"] <= 2.0
    assert c["posa_conteggio"]["detail"]["frazione_passi_con_posa"] > 0.98 and c["ritmo"]["detail"]["soglia"] == 0.81
    assert "+ 1 x pose" in c["composizione_della_loss"]["detail"]["formula"] and res["descriptive"]["posa_inizio_fine"][1] < res["descriptive"]["posa_inizio_fine"][0]


def test_r3_gate_failures(tmp_path):
    def scaled(f):
        def edit(recs):
            for r in recs:
                r["pose"] *= f
                r["total"] = 0.2 * sum(r[a] for a in AC.ANCHORS) + 1.0 * r["pose"]
            return recs
        return edit
    # scala sbagliata della standardizzazione: posa cento volte piu' piccola -> rapporto all'avvio sotto 0,1
    res = AC.check(_run_r3(tmp_path / "small", edit=scaled(0.01)), "R3", STEPS, SHA, LISTING)
    assert not res["criteria"]["posa_scala_avvio"]["ok"] and not res["passed"]
    # divergenza: posa cento volte piu' grande -> sopra il tetto
    res = AC.check(_run_r3(tmp_path / "big", edit=scaled(100.0)), "R3", STEPS, SHA, LISTING)
    assert not res["criteria"]["posa_tetto"]["ok"]

    def flat(recs):
        for r in recs:
            r["pose"] = 0.8
            r["total"] = 0.2 * sum(r[a] for a in AC.ANCHORS) + 0.8
        return recs
    res = AC.check(_run_r3(tmp_path / "flat", edit=flat), "R3", STEPS, SHA, LISTING)
    assert not res["criteria"]["posa_discesa"]["ok"] and res["criteria"]["posa_scala_avvio"]["ok"]

    def sparse(recs):
        for r in recs:
            if r["step"] % 5 == 0:
                r["pose_windows"] = 0.0
        return recs
    res = AC.check(_run_r3(tmp_path / "sparse", edit=sparse), "R3", STEPS, SHA, LISTING)
    assert not res["criteria"]["posa_conteggio"]["ok"]
    # ritmo sopra la soglia congelata
    res = AC.check(_run_r3(tmp_path / "slow", rate=0.85), "R3", STEPS, SHA, LISTING)
    assert not res["criteria"]["ritmo"]["ok"] and res["criteria"]["ritmo"]["detail"]["s_per_passo_mediana"] == 0.85
    # configurazione senza testa o senza albero della posa
    res = AC.check(_run_r3(tmp_path / "nohead", cfg_edit=lambda c: {**c, "model": {"pose_head": False}}), "R3", STEPS, SHA, LISTING)
    assert not res["criteria"]["posa_configurata"]["ok"]
    # pesi di un altro braccio (A) letti come R3: il peso della posa manca
    res = AC.check(_run(tmp_path, "A"), "R3", STEPS, SHA, LISTING)
    assert not res["criteria"]["pesi"]["ok"] and not res["criteria"]["composizione_della_loss"]["ok"]


def test_old_arms_ignore_the_pose_fields_and_old_configs_count_as_zero_pose_weight(tmp_path):
    res = AC.check(_run(tmp_path, "ancore-only"), "ancore-only", STEPS, SHA, LISTING)
    assert res["passed"] and "posa_scala_avvio" not in res["criteria"] and "ritmo" not in res["criteria"]
    assert res["criteria"]["pesi"]["detail"]["config"]["pose_weight"] == 0.0


def test_r3_gradient_diagnostic_is_descriptive_and_flags_below_one_percent(tmp_path):
    def grads(scale):
        def edit(recs):
            for r in recs:
                if r["step"] % 100 == 0:
                    r["shared_grad_anchors"], r["shared_grad_pose"], r["shared_grad_cos"] = 2.0, 2.0 * scale, 0.1
            return recs
        return edit
    res = AC.check(_run_r3(tmp_path / "ok", edit=grads(0.3)), "R3", STEPS, SHA, LISTING)
    g = res["descriptive"]["gradienti_condivisi"]
    assert res["passed"] and g["n_record"] == 9 and abs(g["rapporto_norme_medio"] - 0.3) < 1e-9 and not g["flag_sotto_1pct"]
    res = AC.check(_run_r3(tmp_path / "weak", edit=grads(0.001)), "R3", STEPS, SHA, LISTING)
    assert res["passed"] and res["descriptive"]["gradienti_condivisi"]["flag_sotto_1pct"]  # descrittivo: non cambia l'esito del collaudo
    res = AC.check(_run_r3(tmp_path / "none"), "R3", STEPS, SHA, LISTING)
    assert res["descriptive"]["gradienti_condivisi"] is None
