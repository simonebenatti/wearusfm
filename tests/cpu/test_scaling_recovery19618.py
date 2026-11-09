"""Recovery (v3) of the 300M continuation 5000 -> 19618: one identity, its submission lines, its amended envelope and the
re-read of the cluster before the deploy.

The lines come from the single identity of the recovery and from its plan; the envelope is rebuilt from the frozen budget, the
signed v2 envelope, the job tables and the attested evidence of the jobs already run, with exact arithmetic; the plan builder
refuses fresh readings that differ from the published plan or show outputs of the recovery already there. The failed v1 attempt
and the stopped v2 package are kept, never rewritten. Nothing reaches Leonardo or the common ledger.
"""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_plans19618 as S  # noqa: E402

V3 = ROOT / "results/infra/scaling_300m_20261104/continuation19618_20261109_v3"
V2 = ROOT / "results/infra/scaling_300m_20261104/continuation19618_20261109_v2"
V1 = ROOT / S.PACKAGE
PLAN = json.loads((ROOT / S.RECOVERY["plan"]).read_text())
V1_PLAN_SHA = "eded0922bd1022c417b8d70523c0c66d1ddb1bd8ad273925ec6d53fda5957d5f"
V2_PLAN_SHA = "19441cd663db059840a36651d987d76d4154419258816058950590c1594b6fe0"
SIGNED_V2_ENVELOPE_SHA = "224743dff392bcd07a2a9a69fd8341b9b8f42e09d1176c2e14aefca20199736e"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


envelope = module("recovery_envelope_test", V3 / "make_envelope.py")
builder = module("recovery_plan_builder_test", V3 / "make_training_plan.py")


def submission(line):
    argv = shlex.split(line)
    cut = argv.index("--")
    head = argv[2:cut]
    assert argv[:2] == [S.PYTHON, S.MAC + "/scripts/submit_job.py"] and len(head) % 2 == 0
    return dict(zip(head[::2], head[1::2])), argv[cut + 1:]


def check_lines(lines, commit):
    """Deploy reference, frozen checkout, identity pre-check, training: in this order, from the recovery identity only."""
    campaign, code = S.RECOVERY["campaign"], f"{S.WORK}/wearusfm_campaigns/{S.RECOVERY['campaign']}"
    push, checkout, precheck, training = lines
    assert shlex.split(push) == ["git", "-C", S.MAC, "push", "leonardo", f"{commit}:refs/heads/deploy/{campaign}"]
    common = {"--agente": "claude", "--campagna": S.CAMPAIGN, "--timeout": "120"}
    options, sbatch = submission(checkout)
    assert options == {**common, "--id": S.RECOVERY["checkout"], "--gpu-ore": "0",
                       "--script-stdin": S.MAC + "/scripts/slurm/campaign_checkout.sbatch"}
    assert sbatch == ["--cpus-per-task=2", "--mem=4G", "--time=00:05:00",
                      f"--export=ALL,CAMPAIGN={campaign},CAMPAIGN_SHA={commit},EXPECTED_SHA={commit}", "/dev/stdin"]
    export = f"--export=ALL,CODE_DIR={code},EXPECTED_SHA={commit}"
    options, sbatch = submission(precheck)  # resources from the launcher header, checked again by the wrapper with scontrol
    assert options == {**common, "--id": S.RECOVERY["precheck"], "--gpu-ore": "0"}
    assert sbatch == [export, code + "/scripts/slurm/scaling_identity_precheck.sbatch", S.RECOVERY["plan"],
                      PLAN["identity_precheck"]["receipt_dir"]]
    options, sbatch = submission(training)  # one segment, the frozen cap; no afterok chain, no retry
    assert options == {**common, "--id": S.RECOVERY["training"], "--gpu-ore": "59"}
    assert sbatch == [export, code + "/scripts/slurm/scaling_continue.sbatch", S.RECOVERY["plan"], PLAN["receipt_dir"]]
    v1, v2 = (json.loads(path.read_text()) for path in (V1 / "training_plan.json", V2 / "training_plan.json"))
    for old in ("SCALING-300M-train19618-20261106", "SCALING-300M-cont19618-checkout-20261106",
                "scaling_300m_continuation19618_20261106_v1", v1["run_dir"] + " ", v1["receipt_dir"], "continuation19618_20261106/",
                "73524b63be534f81b3db4ade4b7fc7f47129a0f3", "SCALING-300M-cont19618-checkout-20261109-v2",
                "SCALING-300M-cont19618-idcheck-20261109-v2", "SCALING-300M-train19618-20261109-v2",
                "scaling_300m_continuation19618_20261109_v2", "continuation19618_20261109_v2/", v2["run_dir"] + " ",
                v2["receipt_dir"] + " ", v2["identity_precheck"]["receipt_dir"], "cb48c5b81b2fc4ef07b67b616345836e585c6d02"):
        assert all(old not in line + " " for line in lines), old  # nothing of the failed v1 attempt or of the v2 package


def test_recovery_lines_come_from_the_single_identity_and_the_recovery_plan():
    check_lines(S.recovery_lines("a" * 40), "a" * 40)
    assert S.RECOVERY["plan"].endswith("/continuation19618_20261109_v3/training_plan.json")
    assert all(S.RECOVERY[key].startswith("SCALING-300M-") and S.RECOVERY[key].endswith("-20261109-v3")
               for key in ("checkout", "precheck", "training"))  # line jobs of the common ledger, new logical IDs
    with pytest.raises(ValueError, match="full commit SHA"):
        S.recovery_lines("deadbeef")


def test_envelope_is_rebuilt_exactly_from_attested_evidence_and_keeps_every_attempt_in_the_account(capsys):
    published = json.loads((V3 / "budget_envelope.json").read_text())
    assert published == json.loads(json.dumps(envelope.build())) and envelope.main(["--check"]) == 0
    value = lambda entry: Fraction(entry["exact"])  # noqa: E731
    amended, frozen, F = published["amended"], published["frozen"], Fraction
    # Frozen 66.35 GPU-hours with margin, plus 0.381111 spent by the failed attempt (205 s) and the GPU diagnosis (138 s).
    assert value(amended["gpu_hours_with_margin"]) == F(1327, 20) + F((205 + 138) * 4, 3600) == F(30029, 450)
    # CPU only: 345.166667 of the reviewed table, the v2 and v3 checkouts and pre-checks at their allocation with margin and the
    # CPU diagnosis (2224 core-seconds): the +1.8 core-hours of the v3 asked to Simone on top of the signed 347.584444.
    cpu_only = value(amended["cpu_only_core_hours_with_margin"])
    assert cpu_only == F(2071, 6) + 2 * F(2 * 480, 3600) + 2 * F(4 * 1380, 3600) + F(48 * 4 + 100 * 4 + 204 * 8, 3600) == F(157223, 450)
    signed = json.loads((V2 / "budget_envelope.json").read_text())
    assert cpu_only - value(signed["amended"]["cpu_only_core_hours_with_margin"]) == F(9, 5)
    assert cpu_only <= 350 and amended["within_cpu_only_cap"] is True and amended["cpu_only_core_hours_cap"] == 350
    assert value(frozen["checkout_stage_core_hours_with_margin"]) == F(4, 5) and value(amended["checkout_stage_core_hours_with_margin"]) == F(4, 3)
    assert value(amended["identity_precheck_core_hours_with_margin"]) == F(46, 15)
    assert value(amended["gpu_job_cpu_core_hours_with_margin"]) == 528 + F(14, 5) + F((205 + 138) * 32, 3600)
    spent = {row["slurm_job_id"]: row for row in published["spent"]}
    assert list(spent) == ["63695634", "63696864", "63860499", "63861166", "63861825", "63864244", "63974034", "63974196"]
    assert {job: row["state"] for job, row in spent.items() if row["state"] != "COMPLETED 0:0"} == {"63696864": "FAILED 1:0"}
    assert all((ROOT / row["attested_by"]).is_file() for row in spent.values()) and published["attestation_pending"] == []
    assert [(spent[j]["elapsed_seconds"], spent[j]["cpus"], spent[j]["gpus"]) for j in ("63974034", "63974196")] == [(32, 2, 0), (73, 4, 0)]
    jobs = (S.checkout_job("training"), S.precheck_job(PLAN), S.training_job(PLAN))
    assert [(r["logical_id"], r["cpus"], r["gpus"], r["memory"], r["seconds"]) for r in published["recovery"]] == [
        (j["id"], j["cpus"], j["gpus"], j["memory"], j["seconds"]) for j in jobs]
    assert value(published["recovery"][2]["gpu_hours_cap"]) == 59 and value(published["recovery"][2]["gpu_hours_with_margin"]) == F(296, 5)
    assert published["signed_v2_envelope"]["sha256"] == sha(V2 / "budget_envelope.json") == SIGNED_V2_ENVELOPE_SHA
    assert published["freeze"]["sha256"] == PLAN["budget_freeze"]["sha256"] == sha(ROOT / S.FREEZE)  # the freeze is not rewritten
    assert published["plan"]["sha256"] == sha(ROOT / S.RECOVERY["plan"]) and "«Sì, prepara il v3»" in published["status"]
    assert published["evaluation"]["unchanged"] is True and value(published["evaluation"]["cpu_only_core_hours_with_margin"]) == F(2071, 6)


def readings(tmp_path, receipts_line="", hashes=None):
    """The read-only readings of 09/11 11:43, input of the plan; `receipts_line` adds an entry to the scratch receipts folder,
    `hashes` rewrites the SHA log."""
    listing, digests = tmp_path / "listing.txt", tmp_path / "sha256.txt"
    header = "=== cartella ricevute scaling su scratch\n"
    text = (V3 / "remote_listing.txt").read_text()
    assert text.count(header) == 1
    listing.write_text(text.replace(header, header + receipts_line))
    text = (V3 / "remote_sha256.txt").read_text()
    digests.write_text(hashes(text) if hashes else text)
    return str(listing), str(digests)


def test_fresh_readings_must_rebuild_the_published_plan_with_the_recovery_outputs_absent(tmp_path, capsys):
    assert builder.main(*readings(tmp_path), check=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True and report["identical_to_published_plan"] is True and all(report["absent"].values())
    receipts = Path(PLAN["receipt_dir"]).name
    assert builder.main(*readings(tmp_path, f"drwxr-x--- 2 sbenatti iscrb 4096 2026-11-10 08:00 {receipts}\n"), check=True) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False and report["absent"][PLAN["receipt_dir"]] is False and report["differences"] == []
    old = PLAN["previous"]["files"]["metrics.jsonl"]["sha256"]
    changed = hashlib.sha256(b"metrics.jsonl rewritten").hexdigest()
    assert builder.main(*readings(tmp_path, hashes=lambda text: text.replace(old, changed)), check=True) == 1  # the 5000 run changed
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False and report["differences"] == ["previous"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["listing.txt", "sha256.txt"]  # the check writes nothing


def test_the_failed_v1_attempt_and_the_stopped_v2_package_are_kept_and_superseded_not_rewritten():
    assert sha(V1 / "training_plan.json") == V1_PLAN_SHA == PLAN["superseded_attempt"]["plan_sha256"]
    assert "73524b63be534f81b3db4ade4b7fc7f47129a0f3" in (V1 / "training_submit_lines.txt").read_text()
    assert (ROOT / PLAN["superseded_attempt"]["diagnosis"]).is_file() and PLAN["superseded_attempt"]["slurm_job_id"] == "63696864"
    assert json.loads((V1 / "evidence/checkout-training/training/accounting.json").read_text())["JobIDRaw"] == "63695634"
    assert sha(V2 / "training_plan.json") == V2_PLAN_SHA == PLAN["superseded_package"]["plan_sha256"]
    assert "cb48c5b81b2fc4ef07b67b616345836e585c6d02" in (V2 / "training_submit_lines.txt").read_text()
    stopped = json.loads((V1 / "evidence/identity-precheck/identity_precheck.json").read_text())  # the receipt that stopped v2
    assert stopped["identical"] is True and stopped["import_order"]["torch_before_numpy"] is False and stopped["slurm_job_id"] == "63974196"
    assert json.loads((V1 / "evidence/checkout-training-v2/training/accounting.json").read_text())["JobIDRaw"] == "63974034"
