"""Derive the 19,618 evaluation chain from the frozen 5,000 one by declared, counted replacements.

The frozen sources are never modified. Every target is the source text with each (old, new)
pair applied in order, and each `old` must occur exactly once at that moment (or exactly the
declared count). `--check` proves that the committed targets are exactly these derivations and
that the frozen sources still have their pinned SHA-256; `--write` regenerates the targets.
What changes: dose and checkpoint name, protocol names, module names, the candidate certified
by the exact-continuation receipt, the reused T100-G control at 19,618 with its original
tokenizer-campaign identity, the three 300M extractions, the per-dataset benchmark limits of
the budget freeze v2 and a descriptive reading without recommendation. Nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = "scaling_derive19618.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


EVALUATION = [
    ('''"""Attest and extract the frozen 30M/300M teacher endpoints at 5,000 updates.

This is a separate campaign contract, not an override of the tokenizer guards.
''', '''"""Attest and extract the 300M teacher endpoint at 19,618 updates; the 30M control is reused.

Derived from scaling_evaluation.py (frozen 5,000 contract) by the replacements declared in
scaling_derive19618.py. The candidate is the exact continuation certified by scaling_continue.py;
the existing T100-G 19,618 caches of the control are neither regenerated nor rewritten.
This is a separate campaign contract, not an override of the tokenizer guards.
'''),
    ('''PROTOCOL = "scaling-evaluation-20261104-v1"
TRAIN_PROTOCOL = "scaling-training-20261104-v1"
''', '''PROTOCOL = "scaling-evaluation-19618-20261106-v1"
TRAIN_PROTOCOL = "scaling-continuation-training-20261106-v1"
'''),
    ('''STEP = 5000
''', '''STEP, START = 19618, 5000
# The existing T100-G 30M endpoint at 19,618 (budget freeze v2, control19618): reused, never re-extracted.
CONTROL_CHECKPOINT_SHA256 = "a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056"
'''),
    ('''    Checkpoint pins include config, completion receipt and phase-summary file pins.
    In particular the historical summary is the immutable 5000 phase copy at
    $WORK/wearusfm_runs/results/tokenizer_training_20261103/T100-G/5000/summary.json,
    not the mutable scratch run summary (which now describes 19618).
    The historical 30M has its original tokenizer receipt; the new 300M has the
    scaling-training receipt, including the exact scientific source contract.
    """
    return copy.deepcopy({"protocol": PROTOCOL, "step": STEP, "arms": list(ARMS),
''', '''    Only the 300M endpoint is pinned: checkpoint_step0019618.pt of the continuation run with
    the config, receipt and summary written by scaling_continue.py in its receipt directory.
    The 30M control is identified by its registered checkpoint hash and is not loaded here.
    """
    return copy.deepcopy({"protocol": PROTOCOL, "step": STEP, "arms": list(ARMS),
        "control_checkpoint_sha256": CONTROL_CHECKPOINT_SHA256,
'''),
    ('''    This does not certify a future 19618 resume: the historical optimizer check
    does not bind every moment to the full parameter list. A continuation needs
    its own resume validation; extraction loads and verifies the teacher exactly.
''', '''    The exact resume from 5,000 (optimizer moments bound to the full parameter list,
    counters and original schedule) was certified by scaling_continue.py before its
    receipt; this is the endpoint check. Extraction loads and verifies the teacher exactly.
'''),
    ('path.name == "checkpoint_step0005000.pt"', 'path.name == "checkpoint_step0019618.pt"'),
    ('''                "scientific training source identity mismatch")
''', '''                "scientific training source identity mismatch")
        require(receipt.get("start_step") == START and receipt.get("additional_steps") == STEP - START and
                receipt.get("scientific_endpoint") is True and receipt.get("previous_run_unchanged") is True and
                receipt.get("control_checkpoint_sha256") == CONTROL_CHECKPOINT_SHA256,
                "not the completed exact continuation verified against the registered control")
'''),
    ('''    require(type(step) is int and step == STEP, "unregistered scaling dose")
    model_recipe(expected_config["model"], arm)
''', '''    require(type(step) is int and step == STEP, "unregistered scaling dose")
    require(arm == ARMS[1], "only the 300M endpoint is extracted at 19618; the 30M control is reused")
    model_recipe(expected_config["model"], arm)
'''),
    ('''    require(set(plan["checkpoints"]) == set(ARMS), "both actual endpoint pins required")
''', '''    require(arm == ARMS[1] and set(plan["checkpoints"]) == {ARMS[1]} and
            plan.get("control_checkpoint_sha256") == CONTROL_CHECKPOINT_SHA256,
            "exactly the actual 300M endpoint pin and the registered control are required; the control is not loaded")
'''),
    ('''    require(plan["checkpoints"][ARMS[0]]["sha256"] == contract["reference_checkpoint_sha256"], "wrong historical control checkpoint")
''', ''),
    ('''    require(R.config_to_dict(cfg) == R.config_to_dict(checked["config_30m" if arm == ARMS[0] else "config_300m"]),
            "effective configuration differs from training source contract")
''', '''    require(R.config_to_dict(cfg) == R.config_to_dict(replace(checked["config_30m" if arm == ARMS[0] else "config_300m"],
                                                              max_steps=STEP)),
            "effective configuration differs from training source contract at the registered dose")
'''),
    ('''    parser.add_argument("--arm", choices=ARMS, required=True)
''', '''    parser.add_argument("--arm", choices=ARMS[1:], required=True)
'''),
]

POSE = [
    ('''"""Two-size, three-view teacher pose readout at the frozen 5,000-step endpoint.
''', '''"""Two-size, three-view teacher pose readout at the 19,618-step endpoint.

Derived from scaling_pose_readout.py (frozen 5,000 contract) by the replacements declared in
scaling_derive19618.py; the 30M control is the reused tokenizer-campaign extraction.
'''),
    ('import scaling_evaluation as V\n', 'import scaling_evaluation19618 as V\n'),
    ('''PROTOCOL = "scaling-pose-readout-20261104-v1"
ARMS, STEP = V.ARMS, 5000
''', '''PROTOCOL = "scaling-pose-readout-19618-20261106-v1"
ARMS, STEP = V.ARMS, V.STEP
'''),
    ('''    legacy = arm == ARMS[0] and prov.get("protocol") == V.E.PROTOCOL
''', '''    legacy = arm == ARMS[0]  # the 19618 control is always the reused tokenizer-campaign extraction
'''),
    ('''    require(plan["caches"][ARMS[0]]["train"]["checkpoint_sha256"] == checked["contract"]["reference_checkpoint_sha256"],
            "wrong historical control checkpoint")
''', '''    require(plan["caches"][ARMS[0]]["train"]["checkpoint_sha256"] == V.CONTROL_CHECKPOINT_SHA256,
            "wrong registered control checkpoint")
'''),
]

PROBE = [
    ('''"""Frozen 5000-step 30M/300M classification probes; no job submission.
''', '''"""19618-step 30M/300M classification probes; no job submission.

Derived from scaling_probe.py (frozen 5,000 contract) by the replacements declared in
scaling_derive19618.py; the 30M control caches are the reused tokenizer-campaign extraction.
'''),
    ('import scaling_evaluation as E\n', 'import scaling_evaluation19618 as E\n'),
    ('''PROTOCOL = "scaling-probe-20261104-v1"
RESOURCE_PROTOCOL = "scaling-probe-resources-20261104-v1"
STEP = 5000
''', '''PROTOCOL = "scaling-probe-19618-20261106-v1"
RESOURCE_PROTOCOL = "scaling-probe-resources-19618-20261106-v1"
STEP = E.STEP
'''),
    ('''            require(cps == {context["contract"]["reference_checkpoint_sha256"]}, "wrong historical 30M5000 checkpoint")
''', '''            require(cps == {E.CONTROL_CHECKPOINT_SHA256}, "wrong registered 30M 19618 control checkpoint")
'''),
    ('''              "training_source_contract_sha256": plan["training_source_contract"]["sha256"]}
    for k, v in common.items():
''', '''              "training_source_contract_sha256": plan["training_source_contract"]["sha256"]}
    if arm == ARMS[0]:  # reused tokenizer-campaign extraction of the control: its own identity, no scaling contract
        common = {**{k: v for k, v in common.items() if k != "training_source_contract_sha256"},
                  "protocol": E.E.PROTOCOL, "arm": "T100-G"}
    for k, v in common.items():
'''),
]

COMPARE = [
    ('''"""Audited 5000-step scaling comparison from completed pose and probe outputs.
''', '''"""Audited 19618-step scaling comparison from completed pose and probe outputs.

Derived from scaling_compare.py (frozen 5,000 contract) by the replacements declared in
scaling_derive19618.py. It describes the 19,618 endpoint and recommends nothing.
'''),
    ('''import scaling_probe as C
import scaling_pose_readout as P
import scaling_projection as SP
import scaling_reading as S8
''', '''import scaling_probe19618 as C
import scaling_pose_readout19618 as P
import scaling_projection as SP
import scaling_reading19618 as S8
'''),
    ('''PROTOCOL = "scaling-comparison-5000-20261104-v1"
INPUT_PROTOCOL = "scaling-comparison-inputs-20261104-v1"
''', '''PROTOCOL = "scaling-comparison-19618-20261106-v1"
INPUT_PROTOCOL = "scaling-comparison-inputs-19618-20261106-v1"
STEP = C.STEP
'''),
    ('"checkpoint_step": 5000, "sources": source_hashes(),', '"checkpoint_step": STEP, "sources": source_hashes(),'),
    ('''            inputs["checkpoint_step"] == 5000, "wrong comparison input protocol/dose")
''', '''            inputs["checkpoint_step"] == STEP, "wrong comparison input protocol/dose")
'''),
    ('''    require(checkpoints[C.ARMS[0]] == source["reference_checkpoint_sha256"], "historical control differs")
''', '''    require(checkpoints[C.ARMS[0]] == C.E.CONTROL_CHECKPOINT_SHA256, "registered control differs")
'''),
    ('report["step"] == 5000 and report.get("recipe") == P.recipe() and',
     'report["step"] == STEP and report.get("recipe") == P.recipe() and'),
    ('"checkpoint_step": 5000, "candidate": "T100-G-300M"', '"checkpoint_step": STEP, "candidate": "T100-G-300M"'),
    ('natural widths primary, fixed projection secondary. No automatic continuation."}',
     'natural widths primary, fixed projection secondary. '
     'Describes the 19618 endpoint only: no recommendation, no checkpoint selection, no further training."}'),
]

READING = [
    ('''"""Prespecified descriptive reading of the paired 300M/30M 5k comparison.

Consumes audited summary contrasts; it does not compute probes or bootstrap,
certify their provenance, authorize a continuation, or submit any job.
"""
''', '''"""Descriptive reading of the paired 300M/30M comparison at 19,618 updates.

Derived from scaling_reading.py (frozen 5,000 policy) by the replacements declared in
scaling_derive19618.py. The S8 definitions are unchanged: EPN-612 and dynamic pose error as
separate primaries, practical pose signal at a 2% point reduction with the interval below zero,
P1/P2 non-inferiority margin of 1 pp, subject bootstrap 2,000/20261102. The S8 operational table
decided the continuation from the 5,000 result and is not reapplied: this reading describes the
19,618 endpoint, recommends no dose, selects no checkpoint and authorizes no further training.
Consumes audited summary contrasts; it does not compute probes or bootstrap,
certify their provenance or submit any job.
"""
'''),
    ('PROTOCOL = "scaling-reading-5000-20261104-v1"\n', 'PROTOCOL = "scaling-reading-19618-20261106-v1"\n'),
    ('''    "checkpoint_step": 5000,
''', '''    "checkpoint_step": 19618,
'''),
    ('''    "continuation_is_authorized": False,
    "continuation_endpoint": 19618,
    "continuation_additional_steps": 14618,
    "budget_formula": "4/3600 * ceil900(14618 * max(trainer300M5000_seconds/5000, preflight300M_112_update_mean_seconds) * 1.20 + 1200)",
    "budget_freeze": "numeric training, extraction GPU and reader CPU caps before opening 5000 downstream scores",
}
''', '''    "definitions_inherited_from": "scaling-reading-5000-20261104-v1",
    "s8_operational_table_reapplied": False,
    "recommendation_emitted": False,
    "checkpoint_selection_between_doses": False,
    "further_training_authorized": False,
    "pretraining_seeds": 1,
}
'''),
    ('''    require(summary.get("protocol") == "scaling-comparison-5000-20261104-v1", "wrong summary protocol")
    require(type(summary.get("checkpoint_step")) is int and summary["checkpoint_step"] == 5000, "mixed or wrong dose")
''', '''    require(summary.get("protocol") == "scaling-comparison-19618-20261106-v1", "wrong summary protocol")
    require(type(summary.get("checkpoint_step")) is int and summary["checkpoint_step"] == 19618, "mixed or wrong dose")
'''),
    ('''              "continuation_authorized": False, "source_quality": quality}
    if not result["reading_valid"]:
        return {**result, "recommendation": "repair_evaluation_no_scientific_reading",
''', '''              "further_training_authorized": False, "source_quality": quality}
    if not result["reading_valid"]:
        return {**result, "reading_status": "repair_evaluation_no_scientific_reading",
'''),
    ('''    if "worse" in (epn, pose_reading) or "loss_beyond_margin" in ni.values():
        next_step = "pause_this_recipe_review_before_19618"
    elif "better" in (epn, pose_reading):
        next_step = "propose_19618_confirmation"
    else:
        next_step = "propose_19618_dose_question"
    return {**result, "primary_reading": {"epn": epn, "pose": pose_reading},
''', '''    return {**result, "reading_status": "descriptive_endpoint_reading",
            "primary_reading": {"epn": epn, "pose": pose_reading},
'''),
    ('''            "classification_noninferiority": ni, "recommendation": next_step,
            "scope": "Conditional on one training seed and development subjects. Separate marginal CIs; no joint 95% guarantee. A pause concerns this recipe, not scaling generally."}


def continuation_training_cap(trainer5000_seconds, preflight_update_mean_seconds):
    trainer, update = number(trainer5000_seconds), number(preflight_update_mean_seconds)
    require(trainer > 0 and update > 0, "positive measured timings required")
    seconds = 900 * math.ceil((14618 * max(trainer / 5000, update) * 1.20 + 1200) / 900)
    return {"additional_steps": 14618, "walltime_seconds": seconds, "gpu_count": 4,
            "gpu_hours_cap": 4 * seconds / 3600,
            "extractions_and_cpu_included": False, "submission_authorized": False}
''', '''            "classification_noninferiority": ni,
            "scope": "Conditional on one training seed and development subjects. Separate marginal CIs; no joint 95% guarantee. "
                     "An unresolved outcome is not equivalence; a worse outcome concerns this recipe at this dose, not scaling generally. "
                     "The 5000 and 19618 readings stay side by side: no rule selects between doses."}
'''),
]

EVALUATION_LAUNCHER = [
    ('#SBATCH -J scaling_evaluation\n', '#SBATCH -J scaling_evaluation19618\n'),
    ('/wearusfm_runs/logs/scaling_evaluation_%j.out', '/wearusfm_runs/logs/scaling_evaluation19618_%j.out', 2),
    ('''# Exactly five 5000-step teacher extractions, each validates its own checkpoint:
# 300M pose train+val 150min; 300M P1 120min; 300M P2 150min;
# 30M P1 30min; 30M P2 45min. Total ceiling8.25GPUh; no validation-only job.
# Default150min; submit the exact --time override for each shorter allocation.
# Historical30M pose is reused, never regenerated by this launcher.
''', '''# Exactly three 19618-step 300M teacher extractions, each validates the checkpoint:
# pose train+val 150min; P1 120min; P2 150min. Total ceiling 7GPUh (budget freeze v2);
# no validation-only job. Default150min; submit the exact --time override for P1.
# The 30M control at 19618 (two pose and five classification caches) is reused,
# never regenerated by this launcher.
'''),
    ("echo 'usage: scaling_evaluation.sbatch <relative plan>", "echo 'usage: scaling_evaluation19618.sbatch <relative plan>"),
    ('''  T100-G-30M:extract-classification:p1) SECONDS_EXPECTED=1800;;
  T100-G-30M:extract-classification:p2) SECONDS_EXPECTED=2700;;
  *) echo 'unregistered extraction: only five fixed5000 cells, no30M pose or validation-only job' >&2; exit 2;;
''', '''  *) echo 'unregistered extraction: only three fixed19618 cells of the300M, no30M or validation-only job' >&2; exit 2;;
'''),
    ('''if plan.get('protocol') != 'scaling-evaluation-20261104-v1' or type(plan.get('step')) is not int or plan['step'] != 5000 or plan.get('arms') != arms or set(plan.get('checkpoints', {})) != set(arms):
''', '''if plan.get('protocol') != 'scaling-evaluation-19618-20261106-v1' or type(plan.get('step')) is not int or plan['step'] != 19618 or plan.get('arms') != arms or set(plan.get('checkpoints', {})) != {arms[1]}:
'''),
    ('--arm "$ARM" --step 5000\n', '--arm "$ARM" --step 19618\n'),
    ('exec python3 scripts/scaling_evaluation.py "${ARGS[@]}"', 'exec python3 scripts/scaling_evaluation19618.py "${ARGS[@]}"'),
]

READERS_LAUNCHER = [
    ('scaling_readers_recovery', 'scaling_readers19618', 3),
    ('''# pose ridge 8CPU/60min; recovery benchmark only300M-natural, one dataset:
# EPN612 8CPU/120min, UCI 8CPU/10min; fit <=15shards, <=180min, <=240coreh;
''', '''# pose ridge 8CPU/60min; probe benchmark 8CPU, one cell and one dataset per job,
# limits of the budget freeze v2; fit <=15shards, <=180min, <=240coreh;
'''),
    ('# Every CPU process preloads scaling_compare, including', '# Every CPU process preloads scaling_compare19618, including'),
    ("description='Frozen scaling5000 CPU dispatch; no scientific overrides'",
     "description='Scaling19618 CPU dispatch; no scientific overrides'"),
    ("p.add_argument('--dataset', choices=TASKS['p2'])", "p.add_argument('--dataset', choices=DATASETS)"),
    ('''require(expected_code == 'aef665ba7ea66fc80291846487907bb3ef9107df', 'recovery requires the original frozen reader code')
''', ''),
    ('''if a.stage == 'probe-benchmark':
    require(a.cell == 'T100-G-300M-natural', 'recovery benchmark only for300M-natural')
''', '''# Budget freeze v2, readers.stages.probe_benchmark: fifteen jobs, one dataset each.
BENCHMARK_SECONDS = {
    'T100-G-30M-natural': {'ninapro_db2': 2700, 'ninapro_db3': 1200, 'ninapro_db6': 1200, 'epn612': 1800, 'uci_emg': 600},
    'T100-G-300M-natural': {'ninapro_db2': 3600, 'ninapro_db3': 1800, 'ninapro_db6': 3600, 'epn612': 7200, 'uci_emg': 600},
    'T100-G-300M-matched': {'ninapro_db2': 2700, 'ninapro_db3': 1200, 'ninapro_db6': 1200, 'epn612': 1800, 'uci_emg': 600}}
'''),
    ("'probe-benchmark': 7200 if a.dataset == 'epn612' else 600,",
     "'probe-benchmark': BENCHMARK_SECONDS[a.cell][a.dataset] if a.stage == 'probe-benchmark' else None,"),
    ("'scaling-probe-resources-20261104-v1'", "'scaling-probe-resources-19618-20261106-v1'"),
    ('import scaling_compare\n', 'import scaling_compare19618\n'),
    ('    import scaling_pose_readout as P\n', '    import scaling_pose_readout19618 as P\n', 2),
    ("'scaling-pose-resources-20261104-v1'", "'scaling-pose-resources-19618-20261106-v1'"),
    ("report['step'] == 5000 and report.get('recipe') == P.recipe()", "report['step'] == 19618 and report.get('recipe') == P.recipe()"),
    ("plan.get('protocol') == 'scaling-comparison-inputs-20261104-v1' and type(plan.get('checkpoint_step')) is int and plan['checkpoint_step'] == 5000",
     "plan.get('protocol') == 'scaling-comparison-inputs-19618-20261106-v1' and type(plan.get('checkpoint_step')) is int and plan['checkpoint_step'] == 19618"),
    ("report.get('protocol') == 'scaling-pose-readout-20261104-v1' and report.get('complete') is True and report.get('status') == 'completed' and report.get('step') == 5000",
     "report.get('protocol') == 'scaling-pose-readout-19618-20261106-v1' and report.get('complete') is True and report.get('status') == 'completed' and report.get('step') == 19618"),
    ("'scripts/scaling_pose_readout.py'", "'scripts/scaling_pose_readout19618.py'"),
    ("'scripts/scaling_probe.py'", "'scripts/scaling_probe19618.py'", 3),
    ("'scripts/scaling_compare.py'", "'scripts/scaling_compare19618.py'"),
    ('# Exactly one missing dataset; any failure stops without retry.',
     '# Exactly one dataset per job; any failure stops without retry.'),
    ("sys.path.insert(0,'scripts');import scaling_compare;", "sys.path.insert(0,'scripts');import scaling_compare19618;"),
]

# ---------------------------------------------------------------------------------------------
# Tests: the 5,000 tests with the same declared replacements, plus the cases that exist only at
# 19,618 (certified continuation, reused control identity, descriptive reading, benchmark table).
# ---------------------------------------------------------------------------------------------
NOTE_TEST = "Derived from {} by the replacements declared in scaling_derive19618.py.\n"

EVALUATION_TEST = [
    ('"""Scaling cache regressions and real, small teacher checkpoint restoration.\n',
     '"""Scaling 19,618 cache regressions and real, small teacher checkpoint restoration.\n\n'
     + NOTE_TEST.format("test_scaling_evaluation.py")),
    ('import scaling_evaluation as S\n', 'import scaling_evaluation19618 as S\n'),
    ('''@pytest.mark.parametrize("arm", S.ARMS)
def test_public_checkpoint_gate_checks_fixed_dose_and_architecture_before_io(arm, monkeypatch):
    cfg = {"model": model_config(arm), "max_steps": 5000}
''', '''@pytest.mark.parametrize("arm", S.ARMS[1:])
def test_public_checkpoint_gate_checks_fixed_dose_and_architecture_before_io(arm, monkeypatch):
    cfg = {"model": model_config(arm), "max_steps": 19618}
'''),
    (', 5000, STATS,', ', 19618, STATS,', 2),
    ('    for step in (19618, 4999, True, 5000.):\n', '    for step in (5000, 19617, True, 19618.):\n'),
    ('''    assert len(observed) == 1


def endpoint(tmp_path, arm):
    cfg = replace(R.sanity_config(), max_steps=5000, model=FMConfig(''', '''    assert len(observed) == 1


def test_public_gate_never_loads_the_reused_control_or_a_5000_configuration(monkeypatch):
    monkeypatch.setattr(S, "load_teacher_endpoint", lambda *args: pytest.fail("no checkpoint I/O expected"))
    control = {"model": model_config(S.ARMS[0]), "max_steps": 19618}
    with pytest.raises(ValueError, match="control is reused"):
        S.validate_checkpoint({}, control, S.ARMS[0], 19618, STATS, {}, {})
    with pytest.raises(ValueError, match="endpoint configuration"):
        S.validate_checkpoint({}, {"model": model_config(S.ARMS[1]), "max_steps": 5000}, S.ARMS[1], 19618, STATS, {}, {})


def endpoint(tmp_path, arm):
    cfg = replace(R.sanity_config(), max_steps=19618, model=FMConfig('''),
    ('    optimizer.state[model.tokenizer.proj.weight]["step"].fill_(5000)\n',
     '    optimizer.state[model.tokenizer.proj.weight]["step"].fill_(19618)\n'),
    ('    state = {"step": 5000, "config": config,', '    state = {"step": 19618, "config": config,'),
    ('checkpoint_step0005000.pt', 'checkpoint_step0019618.pt', 4),
    ('''        "arm": "T100-G" if old else arm, "checkpoint_step": 5000,
''', '''        "arm": "T100-G" if old else arm, "checkpoint_step": 19618,
        "start_step": 5000, "additional_steps": 14618, "scientific_endpoint": True, "previous_run_unchanged": True,
        "control_checkpoint_sha256": S.CONTROL_CHECKPOINT_SHA256,
'''),
    ('    summary = {"stopped": "max_steps", "steps": 5000}\n', '    summary = {"stopped": "max_steps", "steps": 19618}\n'),
    ('    if corruption == "step": state["step"] = 4999\n', '    if corruption == "step": state["step"] = 19617\n'),
    ('''    if corruption == "config": state["config"] = {**config, "max_steps": 19618}
''', '''    if corruption == "config": state["config"] = {**config, "max_steps": 5000}
'''),
    ('''def test_historical_control_does_not_accept_a_new_training_code_identity(tmp_path):
''', '''@pytest.mark.parametrize("key,value", [("start_step", 0), ("additional_steps", 19618), ("scientific_endpoint", False),
    ("scientific_endpoint", 1), ("previous_run_unchanged", False), ("control_checkpoint_sha256", CKPT_SHA),
    ("start_step", None), ("control_checkpoint_sha256", None)])
def test_candidate_must_be_the_certified_exact_continuation_against_the_registered_control(tmp_path, key, value):
    arm = S.ARMS[1]
    scene = endpoint(tmp_path, arm)
    p, receipt = scene[0], scene[4]
    assert CKPT_SHA != S.CONTROL_CHECKPOINT_SHA256
    if value is None:
        del receipt[key]
    else:
        receipt[key] = value
    save_json(Path(p["receipt_path"]), receipt)
    p["receipt_sha256"] = R.sha256_file(p["receipt_path"])
    with pytest.raises(ValueError, match="exact continuation"):
        load(scene, arm)


def test_the_5000_training_receipt_protocol_does_not_certify_the_19618_endpoint(tmp_path):
    arm = S.ARMS[1]
    scene = endpoint(tmp_path, arm)
    p, receipt = scene[0], scene[4]
    receipt["protocol"] = "scaling-training-20261104-v1"
    save_json(Path(p["receipt_path"]), receipt)
    p["receipt_sha256"] = R.sha256_file(p["receipt_path"])
    with pytest.raises(ValueError, match="wrong training receipt"):
        load(scene, arm)


def test_historical_control_does_not_accept_a_new_training_code_identity(tmp_path):
'''),
    ('"sha256": CKPT_SHA} for arm in S.ARMS})', '"sha256": CKPT_SHA} for arm in S.ARMS[1:]})'),
    ('''        assert config == R.config_to_dict(checks["config_30m" if arm == S.ARMS[0] else "config_300m"])
''', '''        assert config == R.config_to_dict(replace(checks["config_30m" if arm == S.ARMS[0] else "config_300m"],
                                                  max_steps=19618))
'''),
    ('        assert step == 5000 and source["sha256"] == contract_pin["sha256"]\n',
     '        assert step == 19618 and source["sha256"] == contract_pin["sha256"]\n'),
    ('''@pytest.mark.parametrize("arm", S.ARMS)
def test_context_binds_both_full_recipes_to_training_source_contract(context, arm):
''', '''@pytest.mark.parametrize("arm", S.ARMS[1:])
def test_context_binds_the_300m_recipe_at_the_registered_dose_to_training_source_contract(context, arm):
'''),
    (', 5000, CODE)', ', 19618, CODE)', 2),
    ('''@pytest.mark.parametrize("mutation", ("comparison", "input", "control", "base_cfg", "large_cfg", "dose", "missing_endpoint"))
''', '''def test_context_never_loads_the_reused_control(context):
    with pytest.raises(ValueError, match="control is not loaded"):
        S.load_context(context["path"], S.ARMS[0], 19618, CODE)
    assert not context["seen"]


@pytest.mark.parametrize("mutation", ("comparison", "input", "control", "no_control", "base_cfg", "large_cfg", "dose",
                                      "missing_endpoint", "extra_endpoint"))
'''),
    ('''    if mutation == "control": plan["checkpoints"][S.ARMS[0]]["sha256"] = "3" * 64
''', '''    if mutation == "control": plan["control_checkpoint_sha256"] = "3" * 64
    if mutation == "no_control": del plan["control_checkpoint_sha256"]
'''),
    ('''    if mutation == "dose": plan["step"] = 19618
    if mutation == "missing_endpoint": plan["checkpoints"].pop(S.ARMS[0])
''', '''    if mutation == "dose": plan["step"] = 5000
    if mutation == "missing_endpoint": plan["checkpoints"].pop(S.ARMS[1])
    if mutation == "extra_endpoint": plan["checkpoints"][S.ARMS[0]] = dict(plan["checkpoints"][S.ARMS[1]])
'''),
    (', 5000, CODE, out', ', 19618, CODE, out', 2),
    ('        assert receipt["complete"] and receipt["checkpoint_step"] == 5000\n',
     '        assert receipt["complete"] and receipt["checkpoint_step"] == 19618\n'),
]

POSE_TEST = [
    ('"""Natural-width regressions, S5-before-LN, leakage and paired-support tests."""\n',
     '"""Natural-width regressions, S5-before-LN, leakage and paired-support tests at 19,618.\n\n'
     + NOTE_TEST.format("test_scaling_pose_readout.py") + '"""\n'),
    ('import scaling_pose_readout as S\n', 'import scaling_pose_readout19618 as S\n'),
    ('"view": view, "which": "teacher", "checkpoint_step": 5000,', '"view": view, "which": "teacher", "checkpoint_step": 19618,'),
    ('    if mutation == "dose": last["checkpoint_step"] = 19618\n', '    if mutation == "dose": last["checkpoint_step"] = 5000\n'),
    ('''    caches = {a: copy.deepcopy(old["caches"][b]) for a, b in zip(S.ARMS, ("T25-L", "T100-G"))}
''', '''    caches = {a: copy.deepcopy(old["caches"][b]) for a, b in zip(S.ARMS, ("T25-L", "T100-G"))}
    for pin in caches[S.ARMS[0]].values():  # the control is the registered 19618 checkpoint
        pin["checkpoint_sha256"] = S.V.CONTROL_CHECKPOINT_SHA256
'''),
    ('lambda p: p.update(step=19618)', 'lambda p: p.update(step=5000)'),
    ('''    d, pin, plan, ref, prov, receipt = OLD.artifact_fixture(tmp_path)
    dim = S.V.dimension(arm)
''', '''    d, pin, plan, ref, prov, receipt = OLD.artifact_fixture(tmp_path)
    plan["step"] = prov["checkpoint_step"] = receipt["checkpoint_step"] = S.STEP
    dim = S.V.dimension(arm)
'''),
    ('''@pytest.mark.parametrize("arm,legacy", ((S.ARMS[0], True), (S.ARMS[0], False), (S.ARMS[1], False)))
def test_reader_accepts_only_explicit_historical_or_scaling_provenance(tmp_path, arm, legacy):
    _, pin, plan, ref, _, _ = artifact(tmp_path, arm, legacy)
    assert S.validate_artifact(pin, arm, "train", plan, ref)["users"] == ["a", "b"]
    if legacy:
        assert Q.validate_artifact(pin, "T100-G", "train", plan, ref)["targets"] == ref["targets"]
''', '''@pytest.mark.parametrize("arm,legacy", ((S.ARMS[0], True), (S.ARMS[1], False)))
def test_reader_accepts_the_reused_control_identity_and_the_scaling_candidate(tmp_path, arm, legacy):
    _, pin, plan, ref, _, _ = artifact(tmp_path, arm, legacy)
    assert S.validate_artifact(pin, arm, "train", plan, ref)["users"] == ["a", "b"]
    if legacy:
        assert Q.validate_artifact(pin, "T100-G", "train", plan, ref)["targets"] == ref["targets"]


def rewrite(pin, prov, receipt):
    Path(pin["provenance_path"]).write_text(json.dumps(prov))
    pin["provenance_sha256"] = Q.D.file_sha256(pin["provenance_path"])
    receipt["outputs"]["train"]["provenance_sha256"] = pin["provenance_sha256"]
    Path(pin["receipt_path"]).write_text(json.dumps(receipt))
    pin["receipt_sha256"] = Q.D.file_sha256(pin["receipt_path"])


@pytest.mark.parametrize("arm,legacy", ((S.ARMS[0], False), (S.ARMS[1], True)))
def test_reader_rejects_a_control_with_scaling_identity_and_a_candidate_with_tokenizer_identity(tmp_path, arm, legacy):
    _, pin, plan, ref, _, _ = artifact(tmp_path, arm, legacy)
    with pytest.raises(ValueError, match="extraction identity mismatch"):
        S.validate_artifact(pin, arm, "train", plan, ref)


@pytest.mark.parametrize("arm,legacy", ((S.ARMS[0], True), (S.ARMS[1], False)))
def test_reader_rejects_caches_of_the_5000_dose(tmp_path, arm, legacy):
    _, pin, plan, ref, prov, receipt = artifact(tmp_path, arm, legacy)
    prov["checkpoint_step"] = receipt["checkpoint_step"] = 5000
    rewrite(pin, prov, receipt)
    with pytest.raises(ValueError, match="extraction identity mismatch: checkpoint_step"):
        S.validate_artifact(pin, arm, "train", plan, ref)
'''),
    ('''    assert report["status"] == "train_only_timing" and report["projection_pin"] == projection.pin
''', '''    assert report["status"] == "train_only_timing" and report["projection_pin"] == projection.pin


def test_main_refuses_a_control_that_is_not_the_registered_19618_checkpoint(tmp_path, monkeypatch, projection):
    plan = fake_plan(tmp_path, projection)
    for pin in plan["caches"][S.ARMS[0]].values():
        pin["checkpoint_sha256"] = "9" * 64
    S.validate_plan(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(Q, "runtime_identity", lambda _: {"sources": plan["sources"]})
    source = tmp_path / "source.json"
    source.write_text('{}')
    monkeypatch.setattr(S.ST, "resolve_pin", lambda _: {"path": str(source)})
    monkeypatch.setattr(S.ST, "check_source_contract", lambda _: {"states": {}, "contract": {}})
    monkeypatch.setattr(Q, "validate_reference", lambda pin: pytest.fail("no artifact may be read"))
    out = tmp_path / "validate"
    with pytest.raises(ValueError, match="registered control"):
        S.main(["validate", "--plan", str(plan_path), "--expected-code-sha", "a" * 40, "--out", str(out)])
    assert not out.exists()
'''),
]

PROBE_TEST = [
    ('"""CPU regression tests for dimension-aware scaling probes; no cluster actions."""\n',
     '"""CPU regression tests for dimension-aware scaling probes at 19,618; no cluster actions.\n\n'
     + NOTE_TEST.format("test_scaling_probe.py") + '"""\n'),
    ('import scaling_probe as M\n', 'import scaling_probe19618 as M\n'),
    ('''            cp = context["contract"]["reference_checkpoint_sha256"] if arm == M.ARMS[0] else "2"*64
''', '''            cp = M.E.CONTROL_CHECKPOINT_SHA256 if arm == M.ARMS[0] else "2"*64
'''),
    ('''            common = {"protocol": M.E.PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": 5000,
                      "checkpoint_sha256": cp, "postload_recipe_passed": True, "code_sha": "c"*40,
                      "plan_sha256": "d"*64, "training_source_contract_sha256": contract["sha256"], "model_config": cfg}
''', '''            common = {"protocol": M.E.PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": 19618,
                      "checkpoint_sha256": cp, "postload_recipe_passed": True, "code_sha": "c"*40,
                      "plan_sha256": "d"*64, "training_source_contract_sha256": contract["sha256"], "model_config": cfg}
            if arm == M.ARMS[0]:  # reused tokenizer-campaign extraction: its own protocol and arm name, no scaling contract
                common = {**{k: v for k, v in common.items() if k != "training_source_contract_sha256"},
                          "protocol": M.E.E.PROTOCOL, "arm": "T100-G", "stats_sha256": "b"*64}
'''),
    ('    if mutation == "dose": bad["step"] = 19618\n', '    if mutation == "dose": bad["step"] = 5000\n'),
    ('bad["sources"]["scripts/scaling_probe.py"]', 'bad["sources"]["scripts/scaling_probe19618.py"]'),
    ('[("receipt", "checkpoint_step", 19618),', '[("receipt", "checkpoint_step", 5000),'),
    ('''    with pytest.raises(ValueError): M.load_cache(plan, M.ARMS[1], "epn612")
''', '''    with pytest.raises(ValueError): M.load_cache(plan, M.ARMS[1], "epn612")


@pytest.mark.parametrize("where,key,value", [("receipt", "protocol", M.E.PROTOCOL), ("provenance", "protocol", M.E.PROTOCOL),
    ("receipt", "arm", M.ARMS[0]), ("provenance", "arm", M.ARMS[0]), ("receipt", "checkpoint_step", 5000),
    ("provenance", "checkpoint_step", 5000), ("provenance", "plan_sha256", "a"*64), ("receipt", "code_sha", "a"*40),
    ("receipt", "stage", "extract-pose"), ("receipt", "checkpoint_unchanged", False), ("provenance", "pool_shape", [1, 1792])])
def test_reused_control_is_accepted_only_with_its_tokenizer_campaign_identity(plan, where, key, value):
    x, _, _ = M.load_cache(plan, M.ARMS[0], "epn612")
    assert x.shape[1] == 768
    p = plan["caches"][M.ARMS[0]]["epn612"]
    file = Path(p[where+"_path"])
    record = json.loads(file.read_text())
    assert record["protocol"] == M.E.E.PROTOCOL and record["arm"] == "T100-G" and "training_source_contract_sha256" not in record
    record[key] = value
    save_json(file, record)
    p[where+"_sha256"] = M.sha(file)
    if where == "provenance":
        receipt = json.loads(Path(p["receipt_path"]).read_text())
        receipt["outputs"]["epn612"]["provenance_sha256"] = p["provenance_sha256"]
        save_json(p["receipt_path"], receipt)
        p["receipt_sha256"] = M.sha(p["receipt_path"])
    with pytest.raises(ValueError): M.load_cache(plan, M.ARMS[0], "epn612")
'''),
]

COMPARE_TEST = [
    ('"""Final comparison with real small cache, OOF, audit and pose artifacts.\n',
     '"""Final 19,618 comparison with real small cache, OOF, audit and pose artifacts.\n\n'
     + NOTE_TEST.format("test_scaling_compare.py")),
    ('import scaling_compare as M\nimport test_scaling_probe as TP\n',
     'import scaling_compare19618 as M\nimport test_scaling_probe19618 as TP\n'),
    ('''            common = {"protocol": M.P.V.PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": 5000,
                      "checkpoint_sha256": cp, "code_sha": "c"*40,
                      "training_source_contract_sha256": probe["training_source_contract"]["sha256"]}
''', '''            common = {"protocol": M.P.V.PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": 19618,
                      "checkpoint_sha256": cp, "code_sha": "c"*40,
                      "training_source_contract_sha256": probe["training_source_contract"]["sha256"]}
            if arm == M.P.ARMS[0]:  # reused tokenizer-campaign pose extraction of the control
                common = {"protocol": M.P.V.E.PROTOCOL, "arm": "T100-G", "which": "teacher", "checkpoint_step": 19618,
                          "checkpoint_sha256": cp, "code_sha": "c"*40}
'''),
    ('"view": view, "which": "teacher", "checkpoint_step": 5000,', '"view": view, "which": "teacher", "checkpoint_step": 19618,'),
    ('''    pose_report = {"protocol": M.P.PROTOCOL, "status": "completed", "complete": True, "step": 5000,
''', '''    pose_report = {"protocol": M.P.PROTOCOL, "status": "completed", "complete": True, "step": 19618,
'''),
    ('''    assert result["reading"]["reading_valid"] and not result["reading"]["continuation_authorized"]
''', '''    assert result["reading"]["reading_valid"] and not result["reading"]["further_training_authorized"]
    assert result["reading"]["reading_status"] == "descriptive_endpoint_reading" and "recommendation" not in result["reading"]
    assert result["checkpoint_step"] == 19618 and result["checkpoints"][M.C.ARMS[0]] == M.C.E.CONTROL_CHECKPOINT_SHA256
'''),
    ('    if mutation == "dose": report["step"] = 19618\n', '    if mutation == "dose": report["step"] = 5000\n'),
    ('inputs["sources"]["scripts/scaling_compare.py"]', 'inputs["sources"]["scripts/scaling_compare19618.py"]'),
    ('        if mutation == "mixed_plan": pose["step"] = 19618\n', '        if mutation == "mixed_plan": pose["step"] = 5000\n'),
    ('''    assert not result["reading"]["reading_valid"] and result["reading"]["recommendation"] == "repair_evaluation_no_scientific_reading"
''', '''    assert not result["reading"]["reading_valid"] and result["reading"]["reading_status"] == "repair_evaluation_no_scientific_reading"
    assert "recommendation" not in result["reading"] and "primary_reading" not in result["reading"]
'''),
]

READING_TEST = [
    ('spec = importlib.util.spec_from_file_location("scaling_reading", ROOT / "scripts/scaling_reading.py")\n',
     'spec = importlib.util.spec_from_file_location("scaling_reading19618", ROOT / "scripts/scaling_reading19618.py")\n'),
    ('    return {"protocol": "scaling-comparison-5000-20261104-v1", "checkpoint_step": 5000,\n',
     '    return {"protocol": "scaling-comparison-19618-20261106-v1", "checkpoint_step": 19618,\n'),
    ('''def test_unresolved_supports_a_dose_question_not_a_claim_or_submission():
    result = S.read_comparison(example())
    assert result["recommendation"] == "propose_19618_dose_question"
    assert result["primary_reading"] == {"epn": "unresolved", "pose": "unresolved"}
    assert not result["pose_practical_signal"] and not result["continuation_authorized"]
''', '''def test_unresolved_is_described_without_a_claim_recommendation_or_submission():
    result = S.read_comparison(example())
    assert result["reading_status"] == "descriptive_endpoint_reading" and "recommendation" not in result
    assert result["primary_reading"] == {"epn": "unresolved", "pose": "unresolved"}
    assert not result["pose_practical_signal"] and not result["further_training_authorized"]
'''),
    ('    assert result["recommendation"] == "pause_this_recipe_review_before_19618"\n',
     '    assert result["reading_status"] == "descriptive_endpoint_reading" and "recommendation" not in result\n'),
    ('    assert result["recommendation"] == "propose_19618_confirmation"\n',
     '    assert "recommendation" not in result\n'),
    ('''    assert (result["recommendation"] == "pause_this_recipe_review_before_19618") == (expected == "loss_beyond_margin")
''', '''    assert "recommendation" not in result and result["reading_status"] == "descriptive_endpoint_reading"
'''),
    ('''    assert not result["reading_valid"] and not result["continuation_authorized"]
    assert "primary_reading" not in result
''', '''    assert not result["reading_valid"] and not result["further_training_authorized"]
    assert "primary_reading" not in result and result["reading_status"] == "repair_evaluation_no_scientific_reading"
    assert result["failed_checks"] == ["solver_converged"] and "recommendation" not in result
'''),
    ('''def test_cost_uses_slower_measurement_margin_reserve_and_15min_rounding():
    result = S.continuation_training_cap(15000, 4.)
    # 14618 * 4 * 1.2 + 1200 = 71366.4 seconds, rounded up to 72000.
    assert result["walltime_seconds"] == 72000 and result["gpu_hours_cap"] == 80.
    assert not result["extractions_and_cpu_included"] and not result["submission_authorized"]
''', '''def test_reading_has_no_budget_formula_dose_proposal_or_checkpoint_selection():
    source = (ROOT / "scripts/scaling_reading19618.py").read_text()
    assert not hasattr(S, "continuation_training_cap")
    for forbidden in ("propose_", "pause_this_recipe", "49046", "best_checkpoint", "continuation_is_authorized"):
        assert forbidden not in source
    assert S.POLICY["recommendation_emitted"] is False and S.POLICY["s8_operational_table_reapplied"] is False
    assert S.POLICY["checkpoint_selection_between_doses"] is False and S.POLICY["further_training_authorized"] is False
    assert S.POLICY["pretraining_seeds"] == 1 and S.POLICY["checkpoint_step"] == 19618
'''),
    ('''    policy = json.loads((ROOT / "results/infra/scaling_300m_20261104/scientific_reading_policy.json").read_text())
    assert policy == S.POLICY
''', '''    policy = json.loads((ROOT / "results/infra/scaling_300m_20261104/continuation19618_20261106/"
                         "scientific_reading_policy_19618.json").read_text())
    assert policy == S.POLICY


def frozen_reading():
    spec5 = importlib.util.spec_from_file_location("scaling_reading", ROOT / "scripts/scaling_reading.py")
    module = importlib.util.module_from_spec(spec5)
    spec5.loader.exec_module(module)
    return module


def test_definitions_are_those_frozen_for_the_5000_reading():
    import json
    frozen = json.loads((ROOT / "results/infra/scaling_300m_20261104/scientific_reading_policy.json").read_text())
    shared = ("candidate", "control", "bootstrap", "epn_margin_pp", "pose_practical_point_reduction",
              "p1_p2_noninferiority_margin_pp", "historical_pair_is_noise_floor")
    assert {key: S.POLICY[key] for key in shared} == {key: frozen[key] for key in shared}
    assert S.POLICY["definitions_inherited_from"] == frozen["protocol"] == frozen_reading().PROTOCOL
    assert frozen["checkpoint_step"] == 5000 and S.POLICY["bootstrap"] == {"unit": "subject", "replicates": 2000, "seed": 20261102}


def test_labels_equal_the_frozen_5000_definitions_on_a_grid_of_intervals_and_points():
    frozen = frozen_reading()
    intervals = [[-2.5, -1.5], [-2., -1.], [-1.5, -.5], [-1., 0.], [-.5, 0.], [-.5, .5], [0., .5], [0., 1.], [.5, 1.5]]
    pose_intervals = [[-.06, -.01], [-.03, 0.], [-.02, .01], [0., .02], [.001, .02]]
    compared = 0
    for epn in intervals:
        for p1 in intervals:
            for pose_ci in pose_intervals:
                for candidate in (.97, .98, .99, 1.01):
                    row = example()
                    row["epn"]["ci95"], row["p1"]["ci95"], row["p2"]["ci95"] = epn, p1, [-p1[1], -p1[0]]
                    row["pose"] = {"control_E_dyn": 1., "candidate_E_dyn": candidate, "delta_E_dyn": candidate - 1.,
                                   "ci95": pose_ci}
                    new = S.read_comparison(row)
                    old = frozen.read_comparison({**row, "protocol": "scaling-comparison-5000-20261104-v1",
                                                  "checkpoint_step": 5000})
                    for key in ("reading_valid", "source_quality", "primary_reading", "pose_relative_point_reduction",
                                "pose_practical_signal", "classification_noninferiority"):
                        assert new[key] == old[key]
                    assert "recommendation" in old and "recommendation" not in new
                    compared += 1
    assert compared == 9 * 9 * 5 * 4
'''),
]

EVALUATION_LAUNCHER_TEST = [
    ('''"""Execute the launcher with fake scheduler/GPU commands, never the controller.
''', '''"""Execute the 19618 launcher with fake scheduler/GPU commands, never the controller.

''' + NOTE_TEST.format("test_scaling_evaluation_launcher.py")),
    ('LAUNCHER = ROOT / "scripts/slurm/scaling_evaluation.sbatch"\n', 'LAUNCHER = ROOT / "scripts/slurm/scaling_evaluation19618.sbatch"\n'),
    ('''         ("T100-G-300M", "extract-classification", "p2", "02:30:00", 9000),
         ("T100-G-30M", "extract-classification", "p1", "00:30:00", 1800),
         ("T100-G-30M", "extract-classification", "p2", "00:45:00", 2700)]
''', '''         ("T100-G-300M", "extract-classification", "p2", "02:30:00", 9000)]
'''),
    ('''    plan = {"protocol": "scaling-evaluation-20261104-v1", "step": 5000,
            "arms": ["T100-G-30M", "T100-G-300M"], "checkpoints": {a: {"pin": "validated by controller"} for a in ("T100-G-30M", "T100-G-300M")}}
    for name, value in (("plan.json", plan), ("wrong_step.json", {**plan, "step": 19618}),
                        ("legacy.json", {**plan, "protocol": "tokenizer-evaluation-20261103-v1"}),
                        ("missing_pin.json", {**plan, "checkpoints": {"T100-G-300M": {}}})):
''', '''    plan = {"protocol": "scaling-evaluation-19618-20261106-v1", "step": 19618,
            "arms": ["T100-G-30M", "T100-G-300M"], "checkpoints": {"T100-G-300M": {"pin": "validated by controller"}}}
    for name, value in (("plan.json", plan), ("wrong_step.json", {**plan, "step": 5000}),
                        ("legacy.json", {**plan, "protocol": "scaling-evaluation-20261104-v1"}),
                        ("missing_pin.json", {**plan, "checkpoints": {}}),
                        ("control_pin.json", {**plan, "checkpoints": {a: {} for a in ("T100-G-30M", "T100-G-300M")}})):
'''),
    ("elif argv[0]=='scripts/scaling_evaluation.py':", "elif argv[0]=='scripts/scaling_evaluation19618.py':"),
    ('def test_each_registered_extraction_runs_guards_and_invokes_fixed5000_once(harness, cell):',
     'def test_each_registered_extraction_runs_guards_and_invokes_fixed19618_once(harness, cell):'),
    ('''    assert argv[0] == "scripts/scaling_evaluation.py"
    assert args["--step"] == "5000" and''', '''    assert argv[0] == "scripts/scaling_evaluation19618.py"
    assert args["--step"] == "19618" and'''),
    ('''def test_five_cell_gpu_budget_and_no_separate_validation():
    assert sum(c[4] for c in CELLS)/3600 == 8.25
''', '''def test_three_cell_gpu_budget_of_the_freeze_and_no_separate_validation():
    assert sum(c[4] for c in CELLS)/3600 == 7. and all(c[0] == "T100-G-300M" for c in CELLS)
    freeze = json.loads((ROOT/"results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json").read_text())
    assert freeze["extractions"]["gpu_hours_cap"] == 7. and not freeze["extractions"]["new_30m_extractions_included"]
    assert [(j["arm"], j["stage"], j["task"] or "none", j["walltime_seconds"], j["checkpoint_step"], j["gpus"], j["cpus"], j["memory"])
            for j in freeze["extractions"]["jobs"]] == [(c[0], c[1], c[2], c[4], 19618, 1, 8, "100G") for c in CELLS]
'''),
    ('''@pytest.mark.parametrize("arm,stage,task", [("T100-G-30M", "extract-pose", "none"),
''', '''@pytest.mark.parametrize("arm,stage,task", [("T100-G-30M", "extract-pose", "none"),
    ("T100-G-30M", "extract-classification", "p1"), ("T100-G-30M", "extract-classification", "p2"),
'''),
    ('''@pytest.mark.parametrize("name", ["wrong_step.json", "legacy.json", "missing_pin.json", "escaped.json", "../../outside.json"])
''', '''@pytest.mark.parametrize("name", ["wrong_step.json", "legacy.json", "missing_pin.json", "control_pin.json", "escaped.json",
                                  "../../outside.json"])
'''),
    ('    if kind == "extra_endpoint_arg": extra = ("19618",)\n', '    if kind == "extra_endpoint_arg": extra = ("5000",)\n'),
]

READERS_LAUNCHER_TEST = [
    ('''"""Execute the CPU launcher guards; intercept only scientific subprocesses.
''', '''"""Execute the 19618 CPU launcher guards; intercept only scientific subprocesses.

''' + NOTE_TEST.format("test_scaling_readers_launcher.py")),
    ("LAUNCHER = ROOT/'scripts/slurm/scaling_readers.sbatch'\n", "LAUNCHER = ROOT/'scripts/slurm/scaling_readers19618.sbatch'\n"),
    ("'scaling-pose-readout-20261104-v1'", "'scaling-pose-readout-19618-20261106-v1'", 2),
    ("'step': 5000, 'recipe': {'test': 'recipe'},", "'step': 19618, 'recipe': {'test': 'recipe'},"),
    ("'scaling-pose-resources-20261104-v1'", "'scaling-pose-resources-19618-20261106-v1'"),
    ("'scaling-probe-resources-20261104-v1'", "'scaling-probe-resources-19618-20261106-v1'"),
    ("inputs = {'protocol': 'scaling-comparison-inputs-20261104-v1', 'checkpoint_step': 5000,",
     "inputs = {'protocol': 'scaling-comparison-inputs-19618-20261106-v1', 'checkpoint_step': 19618,"),
    ("sys.modules['scaling_pose_readout']=module\nsys.modules['scaling_compare']=types.ModuleType('scaling_compare')\n",
     "sys.modules['scaling_pose_readout19618']=module\nsys.modules['scaling_compare19618']=types.ModuleType('scaling_compare19618')\n"),
    ("'import scaling_compare;sys.argv=sys.argv[1:];runpy.run_path' in cmd[2]",
     "'import scaling_compare19618;sys.argv=sys.argv[1:];runpy.run_path' in cmd[2]"),
    ("'scripts/scaling_pose_readout.py'", "'scripts/scaling_pose_readout19618.py'", 2),
    ("'scripts/scaling_probe.py'", "'scripts/scaling_probe19618.py'", 2),
    ("'scripts/scaling_compare.py'", "'scripts/scaling_compare19618.py'", 2),
    ("'probe-benchmark': '01:30:00' if cell == CELLS[1] else '00:45:00', 'probe-fit': '02:00:00',",
     "'probe-benchmark': '00:10:00', 'probe-fit': '02:00:00',"),
    ("                   'probe-benchmark': ['--cell', cell],\n",
     "                   'probe-benchmark': ['--cell', cell, '--dataset', 'uci_emg'],\n"),
    ('''def test_probe_benchmark_all_five_datasets_sequentially(harness, cell):
    h = harness
    got = h['run']('probe-benchmark', cell=cell)
    assert got.returncode == 0, got.stderr
    cmds = [json.loads(line)['argv'] for line in h['record'].read_text().splitlines()]
    assert [cmd[cmd.index('--dataset')+1] for cmd in cmds] == DATASETS
    assert all(cmd[1] == 'benchmark' and cmd[cmd.index('--cell')+1] == cell for cmd in cmds)
    assert len({cmd[cmd.index('--out')+1] for cmd in cmds}) == 5
''', '''def test_probe_benchmark_runs_exactly_one_dataset_per_job(harness, cell):
    h = harness
    got = h['run']('probe-benchmark', cell=cell)
    assert got.returncode == 0, got.stderr
    cmds = [json.loads(line)['argv'] for line in h['record'].read_text().splitlines()]
    assert [cmd[cmd.index('--dataset')+1] for cmd in cmds] == ['uci_emg']
    assert all(cmd[1] == 'benchmark' and cmd[cmd.index('--cell')+1] == cell for cmd in cmds)
    assert Path(cmds[0][cmds[0].index('--out')+1]).name == 'uci_emg'
'''),
    ("    ('compare', ['--task', 'p1']), ('probe-benchmark', ['--cell', CELLS[0], '--datasets', 'epn612'])])\n",
     "    ('compare', ['--task', 'p1']), ('probe-benchmark', ['--cell', CELLS[0], '--datasets', 'epn612']),\n"
     "    ('probe-benchmark', ['--cell', CELLS[0]])])\n"),
    ("    if kind == 'too_long': b['shards'][0]['walltime_seconds'] = 7201\n",
     "    if kind == 'too_long': b['shards'][0]['walltime_seconds'] = 10801\n"),
    ('''def test_probe_total136corehour_cap_is_enforced(harness):
    h = harness
    # Twelve nonoverlapping half-fold shards need192coreh at their7200s caps.
    h['resource']['shards'] = [{**s, 'folds': f} for s in h['resource']['shards'] for f in ([0, 1], [2, 3])]
    h['repin'](h['resource_pin'], h['resource'])
    got = h['run']('probe-fit')
    assert got.returncode != 0 and '136 core-hours' in got.stderr and not h['record'].exists()
''', '''def test_probe_total240corehour_cap_is_enforced(harness):
    h = harness
    # Twelve nonoverlapping half-fold shards at their 10800s caps need 288 core-hours.
    h['resource']['shards'] = [{**s, 'folds': f, 'walltime_seconds': 10800} for s in h['resource']['shards']
                               for f in ([0, 1], [2, 3])]
    h['repin'](h['resource_pin'], h['resource'])
    got = h['run']('probe-fit')
    assert got.returncode != 0 and '240 core-hours' in got.stderr and not h['record'].exists()
'''),
    ('''    # Stage ceilings179coreh plus checkout1/3coreh leave2/3coreh reserve.
    assert 24+4+8+136+2+4+1 == 179
''', '''    # Budget freeze v2: the requested stage ceilings stay inside the 350 core-hour cap.
    freeze = json.loads((ROOT/'results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json').read_text())['readers']
    table = freeze['stages']['probe_benchmark']['seconds_by_cell_and_dataset']
    requested = (3*2*300 + 2*1800 + 3*8*600 + 8*sum(sum(row.values()) for row in table.values())
                 + 8*3600 + 240*3600 + 6*2*600 + 8*1800)/3600
    assert requested <= freeze['budgeted_core_hours_with_scheduler_margin'] <= freeze['cpu_only_core_hours_cap'] == 350
    assert '#SBATCH --time=03:00:00' in text and '#SBATCH --mem=30G' in text and '#SBATCH --cpus-per-task=8' in text
'''),
    ("shell=Path('scripts/slurm/scaling_readers.sbatch').read_text()", "shell=Path('scripts/slurm/scaling_readers19618.sbatch').read_text()"),
    ('import scaling_compare\nimport scaling_pose_readout as P\n', 'import scaling_compare19618\nimport scaling_pose_readout19618 as P\n'),
    ("for script in ('scaling_pose_readout.py','scaling_probe.py','scaling_compare.py'):",
     "for script in ('scaling_pose_readout19618.py','scaling_probe19618.py','scaling_compare19618.py'):", 2),
]

READERS_LIMITS_TEST = [
    ('''"""Recovery changes only dispatch/ceilings; exercise the real shell guards.

Reuse the original launcher's scheduler/git/controller fixture, with the exact
historical code pin. No cluster calls, scientific fits or synthetic readiness.
"""
import copy
''', '''"""Per-dataset benchmarks and fit ceilings of the 19618 CPU launcher; real shell guards.

''' + NOTE_TEST.format("test_scaling_readers_recovery_launcher.py") + '''Reuses the scheduler/git/controller fixture of test_scaling_readers19618_launcher.py. The
launcher has no fixed code pin: it requires the frozen checkout given as EXPECTED_SHA.
No cluster calls, scientific fits or synthetic readiness.
"""
import ast
import copy
'''),
    ('''PIN = 'aef665ba7ea66fc80291846487907bb3ef9107df'
LAUNCHER = ROOT/'scripts/slurm/scaling_readers_recovery.sbatch'
spec = importlib.util.spec_from_file_location('original_reader_fixture',
                                             ROOT/'tests/cpu/test_scaling_readers_launcher.py')
''', '''LAUNCHER = ROOT/'scripts/slurm/scaling_readers19618.sbatch'
spec = importlib.util.spec_from_file_location('reader19618_fixture',
                                             ROOT/'tests/cpu/test_scaling_readers19618_launcher.py')
'''),
    ('''CELLS, TASKS = original.CELLS, original.TASKS
''', '''CELLS, TASKS = original.CELLS, original.TASKS
PIN = original.CODE  # the fixture's frozen checkout; the launcher itself pins no commit
FREEZE = json.loads((ROOT/'results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json').read_text())
SECONDS = FREEZE['readers']['stages']['probe_benchmark']['seconds_by_cell_and_dataset']
GRID = [(cell, dataset) for cell in CELLS for dataset in original.DATASETS]


def hms(seconds):
    return f'{seconds//3600:02}:{seconds%3600//60:02}:{seconds%60:02}'
'''),
    ('''def test_diff_is_only_the_declared_operational_allowlist():
    frozen = subprocess.check_output(['git', 'show', PIN+':scripts/slurm/scaling_readers.sbatch'], cwd=ROOT)
    assert hashlib.sha256(frozen).hexdigest() == '2da66c150fbc238dcf9b211a2eb672bb41adab8de96fc6f3f0c0ea5e716f6355'
    assert (ROOT/'scripts/slurm/scaling_readers.sbatch').read_bytes() == frozen
    edits = [
        ('#SBATCH --time=02:00:00', '#SBATCH --time=03:00:00'),
        ('#SBATCH -J scaling_readers', '#SBATCH -J scaling_readers_recovery'),
        ('logs/scaling_readers_%j.out', 'logs/scaling_readers_recovery_%j.out'),
        ('# pose ridge 8CPU/60min; probe benchmark 8CPU/45min (300M-natural90min),\\n'
         '# all five datasets in one job; probe fit 8CPU/published shard <=120min;',
         '# pose ridge 8CPU/60min; recovery benchmark only300M-natural, one dataset:\\n'
         '# EPN612 8CPU/120min, UCI 8CPU/10min; fit <=15shards, <=180min, <=240coreh;'),
        ("p.add_argument('--cell', choices=CELLS)",
         "p.add_argument('--cell', choices=CELLS)\\np.add_argument('--dataset', choices=TASKS['p2'])"),
        ("'probe-benchmark': ('cell',)", "'probe-benchmark': ('cell', 'dataset')"),
        ("expected_code = os.environ['EXPECTED_SHA']",
         "expected_code = os.environ['EXPECTED_SHA']\\n"
         "require(expected_code == 'aef665ba7ea66fc80291846487907bb3ef9107df', 'recovery requires the original frozen reader code')"),
        ("cpus = 2 if a.stage in ('pose-validate', 'probe-aggregate') else 8",
         "if a.stage == 'probe-benchmark':\\n"
         "    require(a.cell == 'T100-G-300M-natural', 'recovery benchmark only for300M-natural')\\n"
         "cpus = 2 if a.stage in ('pose-validate', 'probe-aggregate') else 8"),
        ("'probe-benchmark': 5400 if a.cell == 'T100-G-300M-natural' else 2700,",
         "'probe-benchmark': 7200 if a.dataset == 'epn612' else 600,"),
        ("0 < s['walltime_seconds'] <= 7200", "0 < s['walltime_seconds'] <= 10800"),
        ("<= 136*3600, 'probe fits exceed136 core-hours'", "<= 240*3600, 'probe fits exceed240 core-hours'"),
        ('# The five calls are sequential; any failure stops the chain, no retry.',
         '# Exactly one missing dataset; any failure stops without retry.'),
        ('for ds in DATASETS]', 'for ds in (a.dataset,)]'),
    ]
    expected = frozen.decode()
    for before, after in edits:
        assert expected.count(before) == (2 if before == 'logs/scaling_readers_%j.out' else 1)
        expected = expected.replace(before, after)
    assert LAUNCHER.read_text() == expected
    subprocess.run(['/bin/bash', '-n', str(LAUNCHER)], check=True)


def benchmark(h, dataset='epn612', **kwargs):
    fields = {'TimeLimit': '02:00:00' if dataset == 'epn612' else '00:10:00', **kwargs.pop('fields', {})}
    flags = kwargs.pop('flags', ['--cell', CELLS[1], '--dataset', dataset])
    return h['run']('probe-benchmark', cell=CELLS[1], flags=flags, fields=fields, **kwargs)


@pytest.mark.parametrize('dataset', ['epn612', 'uci_emg'])
def test_one_missing_dataset_only_and_original_numeric_dispatch(harness, dataset):
    h = harness
    got = benchmark(h, dataset)
''', '''def test_launcher_is_the_declared_derivation_of_the_reviewed_recovery_launcher_with_the_frozen_table():
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    import scaling_derive19618 as D
    target = 'scripts/slurm/scaling_readers19618.sbatch'
    source, pinned, _ = D.DERIVATIONS[target]
    assert source == 'scripts/slurm/scaling_readers_recovery.sbatch'
    assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == pinned
    text = LAUNCHER.read_text()
    assert text == D.derive(target)
    table = ast.literal_eval(text.split('BENCHMARK_SECONDS = ', 1)[1].split('\\ncpus = ', 1)[0])
    assert table == SECONDS and FREEZE['readers']['stages']['probe_benchmark']['jobs'] == len(GRID) == 15
    assert FREEZE['readers']['stages']['probe_benchmark']['datasets_per_job'] == 1
    assert 'aef665ba7ea66fc80291846487907bb3ef9107df' not in text
    subprocess.run(['/bin/bash', '-n', str(LAUNCHER)], check=True)


def benchmark(h, dataset='epn612', cell=CELLS[1], **kwargs):
    fields = {'TimeLimit': hms(SECONDS[cell][dataset]), **kwargs.pop('fields', {})}
    flags = kwargs.pop('flags', ['--cell', cell, '--dataset', dataset])
    return h['run']('probe-benchmark', cell=cell, flags=flags, fields=fields, **kwargs)


@pytest.mark.parametrize('cell,dataset', GRID)
def test_each_of_the_fifteen_benchmarks_runs_one_dataset_with_its_frozen_limit(harness, cell, dataset):
    h = harness
    got = benchmark(h, dataset, cell)
'''),
    ("    assert argv[argv.index('--cell')+1] == CELLS[1]\n", "    assert argv[argv.index('--cell')+1] == cell\n"),
    ("'scripts/scaling_probe.py'", "'scripts/scaling_probe19618.py'", 3),
    ("    assert 'import scaling_compare;sys.argv=sys.argv[1:];runpy.run_path' in row['bootstrap']\n",
     "    assert 'import scaling_compare19618;sys.argv=sys.argv[1:];runpy.run_path' in row['bootstrap']\n"),
    ('''@pytest.mark.parametrize('flags', [
    ['--cell', CELLS[0], '--dataset', 'epn612'],
    ['--cell', CELLS[2], '--dataset', 'epn612'],
    ['--cell', CELLS[1]],
    ['--cell', CELLS[1], '--dataset', 'ninapro_db2'],
    ['--cell', CELLS[1], '--datasets', 'epn612', 'uci_emg'],
    ['--cell', CELLS[1], '--dataset', 'epn612', '--dataset', 'uci_emg', '--max-iter', '100'],
])
def test_recovery_scope_is_not_an_arbitrary_rerun(harness, flags):
    assert benchmark(harness, flags=flags).returncode != 0
    assert not harness['record'].exists()
''', '''@pytest.mark.parametrize('flags', [
    ['--cell', CELLS[1]],
    ['--dataset', 'epn612'],
    ['--cell', 'T100-G-30M-matched', '--dataset', 'epn612'],
    ['--cell', CELLS[1], '--dataset', 'other'],
    ['--cell', CELLS[1], '--datasets', 'epn612', 'uci_emg'],
    ['--cell', CELLS[1], '--dataset', 'epn612', '--dataset', 'uci_emg', '--max-iter', '100'],
])
def test_benchmark_needs_exactly_one_registered_cell_and_dataset(harness, flags):
    assert benchmark(harness, flags=flags).returncode != 0
    assert not harness['record'].exists()


@pytest.mark.parametrize('cell,dataset', GRID)
def test_benchmark_limit_of_another_cell_or_dataset_is_refused(harness, cell, dataset):
    other = next(s for s in sorted({v for row in SECONDS.values() for v in row.values()}) if s != SECONDS[cell][dataset])
    got = benchmark(harness, dataset, cell, fields={'TimeLimit': hms(other)})
    assert got.returncode != 0 and 'actual walltime differs' in got.stderr
    assert not harness['record'].exists()
'''),
    ('''def test_even_matching_other_checkout_and_expected_sha_are_rejected(harness):
    got = benchmark(harness, environment={'EXPECTED_SHA': 'b'*40, 'TEST_GIT_HEAD': 'b'*40})
    assert got.returncode != 0 and 'original frozen reader code' in got.stderr
    assert not harness['record'].exists()
''', '''def test_no_fixed_code_pin_but_expected_sha_must_be_the_frozen_checkout_head(harness):
    got = benchmark(harness, environment={'EXPECTED_SHA': 'b'*40})
    assert got.returncode != 0 and not harness['record'].exists()
    got = benchmark(harness, environment={'EXPECTED_SHA': 'b'*40, 'TEST_GIT_HEAD': 'b'*40})
    assert got.returncode == 0, got.stderr
    row = json.loads(harness['record'].read_text())
    assert row['argv'][row['argv'].index('--expected-code-sha')+1] == 'b'*40
'''),
    ("'scripts/scaling_compare.py'", "'scripts/scaling_compare19618.py'"),
    ("'scripts/scaling_pose_readout.py'", "'scripts/scaling_pose_readout19618.py'"),
]

DERIVATIONS = {
    "scripts/scaling_evaluation19618.py": ("scripts/scaling_evaluation.py",
        "395849450dc24a804a71133e272a2b76dd502a5466d7ca3a2d79c42bb292323e", EVALUATION),
    "scripts/scaling_pose_readout19618.py": ("scripts/scaling_pose_readout.py",
        "238a5cd479e37310235c42dcb0b930d232230ff0d2a8d1d1309b05b7292900cf", POSE),
    "scripts/scaling_probe19618.py": ("scripts/scaling_probe.py",
        "0b3d40994ea7b6ee189a5831e22a5b298cc0cf2ceaa24908c80242642b5d26b0", PROBE),
    "scripts/scaling_compare19618.py": ("scripts/scaling_compare.py",
        "a3610519b453a3d37747f5a4ce926b2164a40dd503546b22b6c45829085eb938", COMPARE),
    "scripts/scaling_reading19618.py": ("scripts/scaling_reading.py",
        "13231e82664cfa6a815a8e87fb6c85380b70838d8746109d8af627b5a4f056d5", READING),
    "scripts/slurm/scaling_evaluation19618.sbatch": ("scripts/slurm/scaling_evaluation.sbatch",
        "a3e880e42d9dc82aaedee2eb85265f750402da9c2492e617b0fd579be0382084", EVALUATION_LAUNCHER),
    "scripts/slurm/scaling_readers19618.sbatch": ("scripts/slurm/scaling_readers_recovery.sbatch",
        "c2bd7979699b2d70150cdfad9d2a6eca657615551ba5073630417c8c46cf14d5", READERS_LAUNCHER),
    "tests/cpu/test_scaling_evaluation19618.py": ("tests/cpu/test_scaling_evaluation.py",
        "2006f66ac82f92d59852ebe5f579f7acf14a6be1abe5941061fc189e10464f08", EVALUATION_TEST),
    "tests/cpu/test_scaling_pose_readout19618.py": ("tests/cpu/test_scaling_pose_readout.py",
        "dd107161a248de6e6d89ee41ed27a4549dbd2778fb24b1485a5bf828e3518ea9", POSE_TEST),
    "tests/cpu/test_scaling_probe19618.py": ("tests/cpu/test_scaling_probe.py",
        "a1dbeb22ab5b73998c15df7e185192948e6c31b29d1809c67ad8ca34877d3146", PROBE_TEST),
    "tests/cpu/test_scaling_compare19618.py": ("tests/cpu/test_scaling_compare.py",
        "331931f3313706fc10747fca6d848bdedc89cdb01cbd820f8534f0e6c6bb0b44", COMPARE_TEST),
    "tests/cpu/test_scaling_reading19618.py": ("tests/cpu/test_scaling_reading.py",
        "60f7978c81b885fdffcd6280c8ad6df12ede45e57eff2da70f9663bf67ce661f", READING_TEST),
    "tests/cpu/test_scaling_evaluation19618_launcher.py": ("tests/cpu/test_scaling_evaluation_launcher.py",
        "c0233415c679d4926e7542f58dced36314acb6e66210e1f8a9f5693db86abbad", EVALUATION_LAUNCHER_TEST),
    "tests/cpu/test_scaling_readers19618_launcher.py": ("tests/cpu/test_scaling_readers_launcher.py",
        "d32e77f65538678d2dea89ce9e7956738baee06f0c8db0cd07e8250f8a2771b7", READERS_LAUNCHER_TEST),
    "tests/cpu/test_scaling_readers19618_limits.py": ("tests/cpu/test_scaling_readers_recovery_launcher.py",
        "5fba59bff8d89460faead328925a48cd2473f10039f094da1edab307e07893ab", READERS_LIMITS_TEST),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def derive(target, root=ROOT):
    """Text of one target, from the pinned frozen source and its ordered replacements."""
    require(target in DERIVATIONS, "unregistered derived file: " + str(target))
    source, expected, replacements = DERIVATIONS[target]
    data = (Path(root) / source).read_bytes()
    require(sha256(data) == expected, "frozen source changed: " + source)
    text = data.decode()
    for index, entry in enumerate(replacements):
        old, new, count = entry if len(entry) == 3 else (*entry, 1)
        require(old and old != new and text.count(old) == count,
                f"{target}: replacement {index} expected {count} occurrence(s), found {text.count(old)}")
        text = text.replace(old, new)
    return text


def check(root=ROOT):
    """Problems found; an empty list means every committed target is exactly its derivation."""
    problems = []
    for target in DERIVATIONS:
        path = Path(root) / target
        try:
            expected = derive(target, root)
        except ValueError as error:
            problems.append(str(error))
            continue
        if not path.is_file() or path.read_text() != expected:
            problems.append("derived file missing or different: " + target)
    return problems


def write(root=ROOT):
    for target in DERIVATIONS:
        (Path(root) / target).write_text(derive(target, root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="compare the committed targets with their derivations")
    group.add_argument("--write", action="store_true", help="regenerate every target; the frozen sources are only read")
    args = parser.parse_args(argv)
    if args.write:
        write()
    problems = check()
    for problem in problems:
        print(problem)
    print("derived files:", len(DERIVATIONS), "problems:", len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
