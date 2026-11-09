"""Build the v2 continuation training plan (09/11/2026) from the read-only readings taken on Leonardo.

Usage (from the repository root, with the project Python):
    python -B results/infra/scaling_300m_20261104/continuation19618_20261109_v2/make_training_plan.py \
        <listing log> <sha256 log>
The two logs are the raw outputs of the read-only ssh sessions (ls -l, sha256sum, scheduler
queries). The script keeps a filtered copy of them next to the plan (no ssh banner, no job of
other projects), writes remote_verification.json and training_plan.json, and never contacts
the cluster. It refuses to overwrite an existing file. With `--check` it writes nothing: the plan
rebuilt from FRESH readings must equal the published one (same pins, the 5000 run above all) and
the outputs, checkout and deploy reference of the recovery must still be absent (re-read before
the deploy, raw outputs kept among the evidence of the chain).

v2 after the FAILED resume of job 63696864 (08/11/2026, fixed-set identity check; cause: import
order torch/numpy in the worker, see failure_63696864_diagnosis_20261108/ of the v1 package):
same arm, dose, recipe, allocation, 5000 run, control and budget freeze; new execution source
pins (wrapper with torch first and the identity pre-check, its launcher), new run and receipt
directories, the CPU identity pre-check gate (`identity_precheck`) and a later retention
horizon. The same readings of 06/11 are the input (ssh to Leonardo unavailable on 09/11: the
wrapper re-verifies every pin at job time and a fresh read-only re-check precedes the deploy).
"""
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("scaling_continue_plan", ROOT / "scripts/scaling_continue.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
R = C.R

SCRATCH = "/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs"
WORK = "/leonardo_work/IscrB_WearUsFM"
BASE = "results/infra/scaling_300m_20261104/"
PREVIOUS_RUN = SCRATCH + "/runs/scaling_300m_20261104/T100-G-300M-5000"
PREVIOUS_RECEIPTS = WORK + "/wearusfm_runs/results/scaling_300m_20261104/scientific5000_v1"
CONTROL_RUN = SCRATCH + "/runs/tokenizer_20261103/T100-G"
V1 = BASE + "continuation19618_20261106/"
SUPERSEDED = {"plan": V1 + "training_plan.json", "plan_sha256": "eded0922bd1022c417b8d70523c0c66d1ddb1bd8ad273925ec6d53fda5957d5f",
              "logical_id": "SCALING-300M-train19618-20261106", "slurm_job_id": "63696864",
              "state": "FAILED 1:0 on 2026-11-08 05:29 CET at the fixed-set identity check of the resume; no training step",
              "run_dir": SCRATCH + "/runs/scaling_300m_20261104/T100-G-300M-19618-from5000",
              "receipt_dir": SCRATCH + "/results/scaling_300m_20261104/continuation19618_v1",
              "diagnosis": V1 + "failure_63696864_diagnosis_20261108/README.md",
              "cause": "worker imported numpy before torch: numpy bound its own OpenBLAS instead of MKL, last-bit differences in "
                       "signals and anchor targets; v2 imports torch first and gates the GPU job with a CPU identity pre-check"}
RUN_DIR = SCRATCH + "/runs/scaling_300m_20261104/T100-G-300M-19618-from5000-v2"
RECEIPT_DIR = SCRATCH + "/results/scaling_300m_20261104/continuation19618_v2"
PRECHECK_DIR = SCRATCH + "/results/scaling_300m_20261104/continuation19618_v2_identity_precheck"
REQUIRED_UNTIL = "2026-12-01T00:00:00Z"  # control needed until the end of the v2 job; earliest scratch expiry of the control 08/12
SECTIONS = {"run 300M a 5000": PREVIOUS_RUN, "ricevute 5000": PREVIOUS_RECEIPTS, "controllo 30M T100-G": CONTROL_RUN}


def pin(relative):
    path = ROOT / relative
    return {"path": relative, "sha256": R.sha256_file(path), "bytes": path.stat().st_size}


def filtered(text):
    """Drop the ssh banner, jobs of other projects and associations of other accounts."""
    keep = []
    for line in text.splitlines():
        if line.startswith("**") or re.match(r"^\d+\|(mona|t1|t2)_", line):
            continue
        if re.match(r"^[a-z0-9_\-]+\|\|", line) and not line.startswith("iscrb_wearusfm||"):
            continue
        keep.append(line)
    return "\n".join(keep) + "\n"


def sections(text):
    out, name = {}, None
    for line in text.splitlines():
        if line.startswith("=== "):
            name = line[4:]
            out[name] = []
        elif name is not None:
            out[name].append(line)
    return out


def listing(lines):
    """name -> {bytes, mtime} from `ls -l --time-style=long-iso`."""
    files = {}
    for line in lines:
        match = re.match(r"^-\S+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(\S+)$", line)
        if match:
            files[match.group(3)] = {"bytes": int(match.group(1)), "mtime_cet": match.group(2)}
    return files


def hashes(lines):
    return {name: sha for sha, name in (line.split("  ", 1) for line in lines if re.match(r"^[0-9a-f]{64}  ", line))}


def recheck(plan, listed, read_at):
    """Fresh read-only readings against the published plan: same pins, outputs, checkout and deploy reference still absent."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import scaling_plans19618 as S
    published = json.loads((HERE / "training_plan.json").read_text())
    rebuilt = json.loads(json.dumps(plan, sort_keys=True))
    differences = sorted(k for k in set(published) | set(rebuilt) if published.get(k) != rebuilt.get(k))
    campaign = S.RECOVERY["campaign"]

    def named(section, name):
        return any(line.split()[-1:] == [name] for line in listed.get(section, []))

    absent = {RUN_DIR: not named("cartella runs scaling", Path(RUN_DIR).name),
              RECEIPT_DIR: not named("cartella ricevute scaling su scratch", Path(RECEIPT_DIR).name),
              PRECHECK_DIR: not named("cartella ricevute scaling su scratch", Path(PRECHECK_DIR).name),
              WORK + "/wearusfm_campaigns/" + campaign: not any(line.strip() == WORK + "/wearusfm_campaigns/" + campaign
                                                                 for line in listed.get("campagne scaling", [])),
              "refs/heads/deploy/" + campaign: not any(line.split()[:1] == ["refs/heads/deploy/" + campaign]
                                                       for line in listed.get("deploy refs scaling", []))}
    missing = sorted({"cartella runs scaling", "cartella ricevute scaling su scratch", "campagne scaling",
                      "deploy refs scaling"} - set(listed))
    passed = not differences and all(absent.values()) and not missing
    print(json.dumps({"protocol": "scaling-continuation-recovery-recheck-20261109-v1", "read_at_cet": read_at,
                      "identical_to_published_plan": not differences, "differences": differences,
                      "absent": absent, "missing_sections": missing, "passed": passed}, indent=1))
    return 0 if passed else 1


def write_new(path, text):
    with path.open("x") as stream:
        stream.write(text)


def main(listing_log, hash_log, check=False):
    raw_listing, raw_hashes = filtered(Path(listing_log).read_text()), filtered(Path(hash_log).read_text())
    listed, hashed = sections(raw_listing), sections(raw_hashes)
    read_at = {"listing": listed["date"][0], "sha256_start": hashed["date inizio"][0], "sha256_end": hashed["date fine"][0]}
    directories = {}
    for title, directory in SECTIONS.items():
        sizes = listing(listed[title])
        digests = hashes(hashed["sha256 " + title.replace(" T100-G", "")])
        directories[directory] = {name: {**sizes[name], "sha256": digests[name]} for name in sorted(digests)}
        if title != "controllo 30M T100-G":  # every listed file was hashed
            assert set(sizes) == set(digests), title
    previous = directories[PREVIOUS_RUN]
    assert set(previous) == C.PREVIOUS_FILES
    cache_root = SCRATCH + "/results/tokenizer_evaluation_20261103/19618/T100-G"
    cache_sizes = {}
    for line in listed["cache controllo 19618"]:
        if line.startswith("--- "):
            folder = line[4:]
        else:
            cache_sizes.update({folder + "/" + k: v for k, v in listing([line]).items()})
    cache_hashes = hashes(hashed["sha256 cache controllo 19618"])
    reference_root = SCRATCH + "/results/pilot_posa/B19618_20261028/B/pose"
    reference_sizes, reference_hashes = listing(listed["riferimenti posa"]), hashes(hashed["sha256 riferimenti posa"])
    remote = {cache_root + "/" + name: {**cache_sizes[name], "sha256": sha} for name, sha in cache_hashes.items()}
    remote.update({reference_root + "/" + name: {**reference_sizes[name], "sha256": sha} for name, sha in reference_hashes.items()})
    inventories = {name: pin("results/infra/tokenizer_evaluation_20261103/" + name + "_plan_19618.json") for name in ("pose_readout", "probe")}
    artifacts = C.control_pins(*(json.loads((ROOT / inventories[name]["path"]).read_text()) for name in ("pose_readout", "probe")))
    for name, value in artifacts.items():  # the reviewed inventories against what is on the cluster today
        assert remote[value["path"]]["sha256"] == value["sha256"], name
        assert value.get("bytes", remote[value["path"]]["bytes"]) == remote[value["path"]]["bytes"], name
    control_checkpoint = CONTROL_RUN + "/checkpoint_step0019618.pt"
    control = directories[CONTROL_RUN]
    # Scratch is purged 40 days after the modification time; the horizon REQUIRED_UNTIL is 00:00 UTC, 01:00 CET.
    scratch_mtimes = [control["checkpoint_step0019618.pt"]["mtime_cet"], *(remote[value["path"]]["mtime_cet"] for value in artifacts.values())]
    earliest_expiry = min(datetime.strptime(m, "%Y-%m-%d %H:%M") for m in scratch_mtimes) + timedelta(days=C.SCRATCH_RETENTION_DAYS)
    assert earliest_expiry > datetime(2026, 12, 1, 1, 0), earliest_expiry
    assert R.sha256_file(ROOT / SUPERSEDED["plan"]) == SUPERSEDED["plan_sha256"]
    assert len({RUN_DIR, RECEIPT_DIR, PRECHECK_DIR, SUPERSEDED["run_dir"], SUPERSEDED["receipt_dir"], PREVIOUS_RUN}) == 6
    receipt_copy = pin(BASE + "scientific5000_completion_20261105/receipt.json")
    assert directories[PREVIOUS_RECEIPTS]["receipt.json"]["sha256"] == receipt_copy["sha256"]
    assert directories[PREVIOUS_RECEIPTS]["receipt.json"]["bytes"] == receipt_copy["bytes"]
    for name in C.IDENTITY_FILES:  # candidate and control were trained on the same fixed diagnostics and sessions
        assert previous[name]["sha256"] == control[name]["sha256"], name
    contract = json.loads((ROOT / BASE / "training_source_contract.json").read_text())
    input_hashes = hashes(hashed["sha256 ingressi del training"])
    for name, value in contract["inputs"].items():
        assert input_hashes[value["path"]] == value["sha256"], name
    scheduler = {key: [line for line in listed[key] if line.strip()] for key in
                 ("qos normal", "assoc", "partition", "reservations", "squeue mio", "saldo", "sinfo boost", "quota work")}
    verification = {
        "protocol": "scaling-continuation-remote-verification-20261106-v1",
        "method": "read-only ssh from the Mac: ls -l, sha256sum, scontrol, sacctmgr, sacct, squeue of this project, saldo, lfs quota",
        "read_at_cet": read_at, "host": listed["date"][1], "raw": ["remote_listing.txt", "remote_sha256.txt"],
        "readings_reused": {"from": [V1 + "remote_listing.txt", V1 + "remote_sha256.txt"],
                            "sha256": {"listing": R.sha256_file(Path(listing_log)), "hashes": R.sha256_file(Path(hash_log))},
                            "why": "ssh to Leonardo unavailable on 2026-11-09 (expired certificate); the 5000 run, the control and the "
                                   "inputs were read untouched on 2026-11-09 09:10 CET before that; every pin is re-verified by the "
                                   "wrapper at job time and a fresh read-only re-check precedes the deploy"},
        "directories": directories, "control_caches_and_pose_references": remote,
        "training_inputs_sha256": input_hashes, "scheduler": scheduler,
        "control_required_until_utc": REQUIRED_UNTIL, "control_earliest_scratch_expiry_cet": earliest_expiry.strftime("%Y-%m-%d %H:%M"),
        "previous_training_job": [line for line in hashed["job training 5000"] if line.strip()],
        "checks": {"previous_run_has_exactly_the_eleven_pinned_files": True,
                   "previous_named_and_resume_checkpoints_byte_identical":
                       previous["checkpoint.pt"]["sha256"] == previous["checkpoint_step0005000.pt"]["sha256"],
                   "previous_receipts_equal_repository_copies": True,
                   "control_checkpoint_equals_freeze_v2": control["checkpoint_step0019618.pt"]["sha256"] ==
                       "a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056",
                   "control_resume_checkpoint_equals_named": control["checkpoint.pt"]["sha256"] == control["checkpoint_step0019618.pt"]["sha256"],
                   "control_two_pose_and_five_classification_caches_equal_inventories": True,
                   "identity_files_equal_between_candidate_and_control": True,
                   "training_inputs_equal_source_contract": True,
                   "control_scratch_retention_covers_required_until": True},
        "not_verified_here": ["content of the caches beyond SHA-256 and size (folds, supports): re-verified by the evaluation stage",
                              "retention is recomputed from the real modification times by the wrapper at job start"]}
    assert all(verification["checks"].values())
    plan = {
        "protocol": C.PROTOCOL, "arm": C.ARM, "start_step": C.START, "end_step": C.END, "additional_steps": C.END - C.START,
        "source_contract": pin(BASE + "training_source_contract.json"),
        "scientific_source_sha256": contract["scientific_source_sha256"],
        "execution_source_sha256": {path: R.sha256_file(ROOT / path) for path in sorted(C.EXECUTION_SOURCES)},
        "allocation": dict(C.ALLOCATION), "memory": dict(C.MEMORY),
        "budget_freeze": pin(BASE + "continuation19618_budget_freeze_20261105_v2.json"),
        "previous": {"run_dir": PREVIOUS_RUN,
                     "files": {name: {"sha256": value["sha256"], "bytes": value["bytes"]} for name, value in sorted(previous.items())},
                     "receipt": {"path": PREVIOUS_RECEIPTS + "/receipt.json", "sha256": receipt_copy["sha256"], "bytes": receipt_copy["bytes"]},
                     "receipt_copy": receipt_copy},
        "control": {"checkpoint": {"path": control_checkpoint, "sha256": control["checkpoint_step0019618.pt"]["sha256"],
                                   "bytes": control["checkpoint_step0019618.pt"]["bytes"]},
                    "receipt": pin("results/infra/tokenizer_execution_20261103/19618/T100-G/receipt.json"),
                    "pose_inventory": inventories["pose_readout"], "classification_inventory": inventories["probe"],
                    "artifacts": artifacts, "required_until_utc": REQUIRED_UNTIL},
        "identity_precheck": {"protocol": C.IDENTITY_PROTOCOL, "receipt_dir": PRECHECK_DIR, "allocation": dict(C.PRECHECK_ALLOCATION)},
        "superseded_attempt": SUPERSEDED,
        "run_dir": RUN_DIR, "receipt_dir": RECEIPT_DIR}
    context = C.M.check_source_contract(ROOT / BASE / "training_source_contract.json")
    _, cfg, argv = C.validate_plan(plan, context)
    C.check_budget_freeze(plan, {})
    if check:
        return recheck(plan, listed, read_at)
    write_new(HERE / "remote_listing.txt", raw_listing)
    write_new(HERE / "remote_sha256.txt", raw_hashes)
    write_new(HERE / "remote_verification.json", json.dumps(verification, indent=1, sort_keys=True) + "\n")
    write_new(HERE / "training_plan.json", json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"plan_sha256": R.sha256_file(HERE / "training_plan.json"), "max_steps": cfg.max_steps,
                      "control_pins": len(artifacts), "trainer_argv_tail": argv[-2:], "precheck_dir": PRECHECK_DIR,
                      "execution_sources": sorted(C.EXECUTION_SOURCES)}, indent=1))
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    sys.exit(main(*[a for a in arguments if a != "--check"], check="--check" in arguments))
