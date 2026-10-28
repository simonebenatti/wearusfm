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

def _run_r3(tmp_path, steps=STEPS, edit=None, cfg_edit=None, rate=0.74, sha_in=("summary", "provenance")):
    """Cartella di un run R3 come la scrive il trainer: lo SHA di pose_stats.json sta in summary.json e in pose_provenance.json, NON nel
    config (`training/run.py`: il config e' l'identita' del run). `sha_in` dice dove metterlo (il lettore fino al 27/10 lo cercava solo nel
    config e bocciava un run vero, collaudo 62429050)."""
    wj, wa = AC.ARM_WEIGHTS["R3"]
    wp = AC.ARM_POSE_WEIGHT["R3"]
    d = tmp_path / "R3"
    d.mkdir(parents=True)
    cfg = {"jepa": {"jepa_weight": wj, "anchor_weight": wa, "keep_weight": 0.0, "branch_weight": 0.0, "pose_weight": wp},
           "model": {"pose_head": True}, "loader": {"pose_root": "/scratch/emg2pose_pose"}}
    if "config" in sha_in:
        cfg["pose_stats_sha256"] = "ab" * 32
    if cfg_edit:
        cfg = cfg_edit(cfg)
    (d / "config.json").write_text(json.dumps(cfg))
    summary = {"stopped": "max_steps", "steps": steps, "elapsed_s": 1000.0, "code": {"sha": SHA}, "world_size": 4, "batch_per_process": 8,
               "accum_steps": 1, "gpu_max_memory_gib": 14.5, "pose_stats_sha256": "ab" * 32 if "summary" in sha_in else None}
    (d / "summary.json").write_text(json.dumps(summary))
    if "provenance" in sha_in:
        (d / "pose_provenance.json").write_text(json.dumps({"pose_root": "/scratch/emg2pose_pose", "pose_stats_sha256": "ab" * 32,
                                                            "pose_unprompted": True}))
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
    # lo SHA delle statistiche: basta che stia in summary.json O in pose_provenance.json (dove lo scrive il trainer) O nel config; in nessuno
    # dei tre -> NO. Regressione del 27/10/2026: il lettore cercava solo nel config e dava NO al collaudo vero 62429050
    for where, ok in ((("summary",), True), (("provenance",), True), (("config",), True), ((), False)):
        res = AC.check(_run_r3(tmp_path / ("sha_" + "_".join(where or ("none",))), sha_in=where), "R3", STEPS, SHA, LISTING)
        assert res["criteria"]["posa_configurata"]["ok"] is ok, (where, res["criteria"]["posa_configurata"])
        if ok:
            assert res["criteria"]["posa_configurata"]["detail"]["sha_in"] == [{"summary": "summary.json", "provenance": "pose_provenance.json",
                                                                               "config": "config.json"}[where[0]]]
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


# --- R3-lungo: paired fixed-panel decision; legacy gates above remain unchanged --------------------------------------

LONG_STEPS = 5000
LISTING_LONG = "-rw-r--r--. 1 sbenatti interactive 454073464 Oct 28 03:06 checkpoint.pt\n"


def _panel(steps=LONG_STEPS, every=500, users=32, windows=8, final=0.7, pred_var=0.2, flat=False, drop=()):
    out = []
    for step in range(0, steps + 1, every):
        if step in drop:
            continue
        frac = 1.0 if flat else 1.0 - (1.0 - final) * step / steps
        per = []
        for u in range(users):
            for w in range(windows):
                const = 1.0 + 0.05 * ((u * 7 + w) % 5)
                mse = const * frac
                start = 10000 * (w % 4)
                per.append({"user": f"u{u:02d}", "session": f"emg2pose/u{u:02d}/s{w // 4}", "window": [None, start, start + 8000],
                            "n_valid": 400, "mse": mse, "mae": mse ** 0.5 * 0.8,
                            "mse_const": const, "pred_var": 0.0 if step == 0 else pred_var,
                            "n_valid_per_joint": [20] * 20, "pred_mean_per_joint": [0.0] * 20})
        row = {"step": step, "n_windows": len(per), "n_windows_valid": len(per), "n_users": users, "per_window": per}
        _aggregates(row)
        out.append(row)
    return out


def _aggregates(row):
    for key in ("mse", "mae", "mse_const", "pred_var"):
        row[key] = sum(w[key] for w in row["per_window"]) / len(row["per_window"])


def _manifest(panel):
    per = panel[0]["per_window"]
    ident = {"schema_version": 1, "window_s": 4.0, "sampling": "nonoverlapping_pose_valid", "seed": [0, 7272],
             "mask_strategy": "training_mask_fixed",
             "users": sorted({w["user"] for w in per}), "window_users": [w["user"] for w in per],
             "windows": [[w["session"], *w["window"]] for w in per], "n_valid": [w["n_valid"] for w in per],
             "batch_sha256": ["cd" * 32] * 64, "pose_stats_sha256": AC.B_POSE_STATS_SHA}
    for row in panel:
        row["panel_sha256"] = AC._canonical_sha256(ident)
    return ident


def _long_config():
    return {"jepa": {"jepa_weight": 0.0, "anchor_weight": 0.2, "keep_weight": 0.0, "branch_weight": 0.0,
                     "pose_weight": 1.0, "loss_per_window": True},
            "model": {"pose_head": True, "dim": 384, "patch_ms": 25.0, "n_heads": 6, "k_latents": 64, "local_layers": 2,
                      "backbone_layers": 8, "decoder_layers": 2, "muscle_dropout": 0.4, "grad_checkpoint": True},
            "loader": {"pose_root": "/scratch/emg2pose_pose", "pose_unprompted": True, "patch_ms": 25.0},
            "max_steps": 5000, "warmup_steps": 1962, "lr": 1e-4, "weight_decay": 0.05, "grad_clip": 1.0,
            "batch_size": 32, "accum_steps": 1, "seed": 0, "freeze_encoder": False, "pose_grad_every": 500,
            "pose_panel_every": 500, "pose_panel_users": 32, "pose_panel_recordings": 2, "pose_panel_windows": 4,
            "nested": "1", "eval_every": 500, "amp_bf16": True, "prefetch": True,
            "schedule": {"kind": "wsd", "total_steps": 49046, "decay_fraction": 0.2, "keep_checkpoints": [19618]}}


def _run_long(tmp_path, steps=LONG_STEPS, edit=None, panel=None):
    d = _run_r3(tmp_path, steps=steps)
    (d / "config.json").write_text(json.dumps(_long_config()))
    for filename in ("summary.json", "pose_provenance.json"):
        obj = json.loads((d / filename).read_text())
        obj["pose_stats_sha256"] = AC.B_POSE_STATS_SHA
        (d / filename).write_text(json.dumps(obj))
    recs = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()]
    for r in recs:
        r["lr"] = 1e-4 * min(1.0, r["step"] / 1962)
        r["pose"] = 0.1 + 0.9 / (1.0 + r["step"] / 500)
        r["total"] = 0.2 * sum(r[a] for a in AC.ANCHORS) + r["pose"]
    if edit:
        recs = edit(recs)
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    reference = _panel()
    manifest = _manifest(reference)
    if panel is None:
        panel = reference
    else:
        for row in panel:
            row["panel_sha256"] = AC._canonical_sha256(manifest)
    (d / "pose_panel_set.json").write_text(json.dumps(manifest))
    (d / "pose_panel.jsonl").write_text("".join(json.dumps(p) + "\n" for p in panel))
    return d


def _gate(panel, manifest):
    return {name: {"ok": ok, "detail": detail} for name, ok, detail in AC.panel_gate(panel, manifest)}


def test_r3_long_clean_run_uses_only_paired_panel_learning_gate(tmp_path):
    res = AC.check(_run_long(tmp_path), "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    assert res["passed"] and res["evaluable"] and res["status"] == "superato", _failed(res)
    c = res["criteria"]
    assert {"posa_scala_avvio", "posa_tetto", "posa_conteggio", "ritmo", "pannello_valutazioni", "pannello_integrita",
            "pannello_baseline", "pannello_discesa", "pannello_ic95_utenti", "configurazione_b", "file_integri"} <= set(c)
    assert not {"posa_discesa", "pannello_sotto_costante", "pannello_varianza_predizioni", "apprendimento"} & set(c)
    assert c["pannello_discesa"]["detail"]["rapporto"] == pytest.approx(0.7)
    ic = c["pannello_ic95_utenti"]["detail"]
    assert ic["n_utenti"] == 32 and ic["ic95"][1] < 1.0 and ic["utenti_in_discesa"] == 32
    assert ic["n_boot"] == 1000 and ic["seme"] == 20261027
    assert c["checkpoint"]["detail"] == {"checkpoint.pt": 454073464}
    trend = res["descriptive"]["posa_training_blocchi"]
    assert trend["passi_da"] == [2001, 2500] and trend["passi_a"] == [4501, 5000]
    assert len(res["descriptive"]["pannello_posa"]) == 11


def test_r3_long_flat_panel_is_valid_but_fails_learning(tmp_path):
    res = AC.check(_run_long(tmp_path, panel=_panel(flat=True)), "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    assert res["evaluable"] and not res["passed"] and res["status"] == "non_superato"
    assert not res["criteria"]["pannello_discesa"]["ok"]
    assert not res["criteria"]["pannello_ic95_utenti"]["ok"]


def test_r3_long_flat_training_and_zero_prediction_variance_are_descriptive(tmp_path):
    def flat(recs):
        for r in recs:
            if r["step"] >= 1500:
                r["pose"] = 0.3
                r["total"] = 0.2 * sum(r[a] for a in AC.ANCHORS) + 0.3
        return recs
    res = AC.check(_run_long(tmp_path, edit=flat, panel=_panel(pred_var=0.0)), "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    assert res["passed"], _failed(res)
    assert res["descriptive"]["posa_training_blocchi"]["media_posa_pesata_da"] == res["descriptive"]["posa_training_blocchi"]["media_posa_pesata_a"]


@pytest.mark.parametrize("corruption", ["missing_eval", "duplicate_eval", "out_of_order", "missing_window", "duplicate_window", "user",
                                         "recording", "coordinates", "n_valid", "nonfinite", "nonfinite_extra", "negative", "aggregate", "const_changed", "hash", "null_row"])
def test_panel_integrity_rejects_corruptions_before_computing_gate(corruption):
    panel = _panel()
    manifest = _manifest(panel)
    middle = panel[5]
    row = middle["per_window"][0]
    if corruption == "missing_eval":
        panel.pop(5)
    elif corruption == "duplicate_eval":
        panel.insert(5, panel[5])
    elif corruption == "out_of_order":
        panel[4], panel[5] = panel[5], panel[4]
    elif corruption == "missing_window":
        middle["per_window"].pop()
    elif corruption == "duplicate_window":
        middle["per_window"][0] = dict(middle["per_window"][1])
    elif corruption == "user":
        row["user"] = "u31"
    elif corruption == "recording":
        row["session"] += "other"
    elif corruption == "coordinates":
        row["window"][1] += 1
    elif corruption == "n_valid":
        row["n_valid"] -= 1
    elif corruption == "nonfinite":
        row["mae"] = float("nan")
    elif corruption == "nonfinite_extra":
        middle["diagnostic"] = {"extra": [float("inf")]}
    elif corruption == "negative":
        row["mse"] = -1
    elif corruption == "aggregate":
        middle["mse"] *= 0.5
    elif corruption == "const_changed":
        row["mse_const"] += 0.1
        _aggregates(middle)
    elif corruption == "hash":
        middle["panel_sha256"] = "0" * 64
    elif corruption == "null_row":
        middle["per_window"][0] = None
    gates = _gate(panel, manifest)
    assert not gates["pannello_integrita"]["ok"]
    assert not gates["pannello_discesa"]["ok"] and not gates["pannello_ic95_utenti"]["ok"]


@pytest.mark.parametrize("bad", ["size", "duplicate", "overlap", "users", "support", "digest", "seed"])
def test_manifest_must_describe_the_complete_nonoverlapping_panel(bad):
    panel = _panel()
    manifest = _manifest(panel)
    if bad == "size":
        manifest["windows"].pop()
    elif bad == "duplicate":
        manifest["windows"][0] = manifest["windows"][1]
    elif bad == "overlap":
        manifest["windows"][1][2] = 7999
    elif bad == "users":
        manifest["users"].pop()
    elif bad == "support":
        manifest["n_valid"][0] = 0
    elif bad == "digest":
        manifest["batch_sha256"][0] = "x" * 64
    elif bad == "seed":
        manifest["seed"] = [1, 7272]
    gates = _gate(panel, manifest)
    assert not gates["pannello_integrita"]["ok"]


def test_baseline_is_consistency_check_not_an_additional_learning_target():
    panel = _panel()
    manifest = _manifest(panel)
    panel[0]["per_window"][0]["mse"] *= 1.1
    _aggregates(panel[0])
    gates = _gate(panel, manifest)
    assert gates["pannello_integrita"]["ok"] and not gates["pannello_baseline"]["ok"]
    assert not gates["pannello_discesa"]["ok"]


def test_long_rejects_nonzero_joint_bias_even_with_unchanged_constant_mse_and_zero_variance(tmp_path):
    panel = _panel()
    w = panel[0]["per_window"][0]
    # A temporally constant prediction 2 * target mean has the same MSE as prediction zero.
    w["pred_mean_per_joint"][0] = 0.5
    assert w["mse"] == w["mse_const"] and w["pred_var"] == 0.0
    res = AC.check(_run_long(tmp_path, panel=panel), "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    assert res["status"] == "non_valutabile" and not res["criteria"]["pannello_integrita"]["ok"]


@pytest.mark.parametrize("field,value", [
    ("pred_mean_per_joint", None), ("n_valid_per_joint", None),
    ("pred_mean_per_joint", [0.0] * 19), ("n_valid_per_joint", [20] * 19),
    ("n_valid_per_joint", [20.0] * 20), ("n_valid_per_joint", [-1, 41] + [20] * 18),
    ("n_valid_per_joint", [21] * 20), ("pred_mean_per_joint", [None] + [0.0] * 19),
    ("pred_mean_per_joint", [float("nan")] + [0.0] * 19),
])
def test_zero_step_joint_support_and_mean_fields_are_required(field, value):
    panel = _panel()
    manifest = _manifest(panel)
    if value is None:
        del panel[0]["per_window"][0][field]
    else:
        panel[0]["per_window"][0][field] = value
    assert not _gate(panel, manifest)["pannello_integrita"]["ok"]


def test_zero_step_joint_mean_can_be_null_only_without_valid_support():
    panel = _panel()
    manifest = _manifest(panel)
    w = panel[0]["per_window"][0]
    w["n_valid_per_joint"] = [0, 40] + [20] * 18
    w["pred_mean_per_joint"] = [None, 1e-6] + [0.0] * 18
    assert all(c["ok"] for c in _gate(panel, manifest).values())


@pytest.mark.parametrize("zero_users", [1, 31, 32])
def test_long_baseline_requires_global_positive_mse_and_valid_bootstrap_denominators(tmp_path, zero_users):
    panel = _panel()
    for row in panel:
        for w in row["per_window"]:
            if int(w["user"][1:]) < zero_users:
                w.update(mse=0.0, mse_const=0.0, mae=0.0)
        _aggregates(row)
    res = AC.check(_run_long(tmp_path, panel=panel), "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    gates = res["criteria"]
    assert gates["pannello_integrita"]["ok"]
    assert gates["pannello_baseline"]["ok"] == (zero_users < 32)
    if zero_users == 1:
        assert res["status"] == "superato" and gates["pannello_bootstrap_valido"]["ok"]
    else:
        assert res["status"] == "non_valutabile" and not gates["pannello_bootstrap_valido"]["ok"]
        if zero_users == 31:
            assert gates["pannello_bootstrap_valido"]["detail"]["repliche_denominatore_non_positivo_o_non_finito"] > 0


def test_long_gradient_diagnostic_includes_every_measurement_through_endpoint(tmp_path):
    def grads(recs):
        for r in recs:
            if r["step"] % 500 == 0:
                r.update(shared_grad_anchors=2.0, shared_grad_pose=r["step"] / 1000, shared_grad_cos=0.1)
        return recs
    res = AC.check(_run_long(tmp_path, edit=grads), "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    g = res["descriptive"]["gradienti_condivisi"]
    assert res["passed"] and g["passi"] == [500, 5000] and g["n_record"] == 10
    assert g["rapporto_norme_medio"] == pytest.approx(1.375)
    assert g["rapporto_norme_min_max"] == [0.25, 2.5]


def test_paired_bootstrap_detects_user_heterogeneity_even_when_mean_drop_passes():
    panel = _panel()
    manifest = _manifest(panel)
    for w in panel[-1]["per_window"]:
        w["mse"] = w["mse_const"] * (8.0 if w["user"] == "u00" else 0.65)
    _aggregates(panel[-1])
    gates = _gate(panel, manifest)
    assert gates["pannello_discesa"]["ok"] and not gates["pannello_ic95_utenti"]["ok"]


@pytest.mark.parametrize("field,value", [("warmup_steps", 1000), ("max_steps", 49046), ("pose_panel_users", 31),
                                        ("lr", 3e-4), ("freeze_encoder", True), ("pose_grad_every", 100)])
def test_long_configuration_cannot_silently_change_the_recipe(tmp_path, field, value):
    d = _run_long(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    cfg[field] = value
    (d / "config.json").write_text(json.dumps(cfg))
    res = AC.check(d, "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    assert not res["criteria"]["configurazione_b"]["ok"] and res["status"] == "non_valutabile"


def test_long_rejects_short_schedule_and_mismatched_pose_provenance(tmp_path):
    d = _run_long(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    cfg["schedule"]["total_steps"] = 5000
    (d / "config.json").write_text(json.dumps(cfg))
    prov = json.loads((d / "pose_provenance.json").read_text())
    prov["pose_stats_sha256"] = "ff" * 32
    (d / "pose_provenance.json").write_text(json.dumps(prov))
    res = AC.check(d, "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    differences = res["criteria"]["configurazione_b"]["detail"]["differenze"]
    assert {"schedule.total_steps", "pose_stats_sha256"} <= set(differences)
    assert not res["evaluable"]


def test_long_rejects_lr_from_the_old_dbg(tmp_path):
    def change(recs):
        recs[1040]["lr"] = 1e-4
        return recs
    res = AC.check(_run_long(tmp_path, edit=change), "R3-lungo", LONG_STEPS, SHA, LISTING_LONG)
    assert res["criteria"]["configurazione_b"]["detail"]["differenze"]["lr_metrics"]["passi_errati"] == [1041]


@pytest.mark.parametrize("kind", ["broken_json", "duplicate_key", "list_record", "missing_file"])
def test_long_cli_always_emits_non_evaluable_result_for_bad_panel_file(tmp_path, kind):
    d = _run_long(tmp_path)
    path = d / "pose_panel.jsonl"
    if kind == "broken_json":
        path.write_text(path.read_text() + "{truncated\n")
    elif kind == "duplicate_key":
        path.write_text(path.read_text() + '{"step":0,"step":5000}\n')
    elif kind == "list_record":
        path.write_text(path.read_text() + "[]\n")
    else:
        path.unlink()
    listing, output = tmp_path / "ls.txt", tmp_path / "result.json"
    listing.write_text(LISTING_LONG)
    assert AC.main([str(d), "--arm", "R3-lungo", "--steps", "5000", "--expected-sha", SHA,
                    "--listing", str(listing), "--out", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["status"] == "non_valutabile" and not result["passed"]
