"""r3 review: audit-only release, dispatcher binding checks and release-commit mutations.

argv[1]: temporary clone at 0efa4d45 (modified: commits a synthetic release on a detached HEAD).
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TMP = Path(os.environ["R3TMP"]) / "r3_work"
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_release as C  # noqa: E402
import contrastive_prepare as P  # noqa: E402
import contrastive_job as J  # noqa: E402
import contrastive_run as RUN  # noqa: E402

ENV = dict(os.environ, GIT_AUTHOR_NAME="R3", GIT_AUTHOR_EMAIL="r3@example.invalid",
           GIT_COMMITTER_NAME="R3", GIT_COMMITTER_EMAIL="r3@example.invalid")


def git(*args, inp=None, env=None, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, input=inp, capture_output=True, check=check,
                          env=env or ENV).stdout.decode().strip()


def orphan_commit(path, data: bytes):
    blob = git("hash-object", "-w", "--stdin", inp=data)
    with tempfile.TemporaryDirectory(dir=os.environ["R3TMP"]) as tmp:
        env = dict(ENV, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("update-index", "--add", "--cacheinfo", "100644," + blob + "," + path, env=env)
        tree = git("write-tree", env=env)
    return git("commit-tree", tree, "-m", "claude review probe", inp=b"")


def report(label, fn):
    try:
        value = fn()
        out = "ACCEPTED" + ("" if value is None else " " + str(value)[:110])
    except Exception as error:
        out = "REJECTED " + type(error).__name__ + ": " + str(error)[:140]
    print(f"{label:64s} {out}", flush=True)


FBF = "fbfde797b48644744c36547533acf0e8a7793535"
TIP = git("rev-parse", "HEAD")
SRC, SP, PP = P.source_digest(), C.file_sha(P.PLAN), C.file_sha(C.PREFLIGHT)
good = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": FBF,
        "source_sha256": SRC, "science_plan_sha256": SP, "preflight_plan_sha256": PP}
REVIEW_PATH = "docs/reviews/claude_review_r3_probe_20261119.md"
REL = "results/infra/contrastive_targets_20261118/release_audit_r3"
text = "# Review r3 (probe)\n\nVerdetto: approvato.\n\n```contrastive-release-review\n" + json.dumps(good, indent=2) + "\n```\n"
review_commit = orphan_commit(REVIEW_PATH, text.encode())
(ROOT / REVIEW_PATH).write_text(text)
binding = C.bind_review(review_commit, REVIEW_PATH, ROOT / REVIEW_PATH)
print("binding keys", sorted(binding), "review pin", binding["review"])

# Audit job from make_job: evidence carries config, inputs (science hashes) and roots.
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
preflight = P._read(C.PREFLIGHT)
fresh = Path(preflight["fresh_root"])
audit_cell = [c for c in preflight["cells"] if c["mode"] == "audit"][0]
(ROOT / REL).mkdir(parents=True)
cfg = P.build_review_configs(P._read(P.BASELINE))["calibration"]
(ROOT / REL / "audit_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
science_inputs = P._read(P.PLAN)["input_sha256"]
evidence = {"config": C.pin(ROOT / REL / "audit_config.json", relative=True),
            "inputs": {k: {"path": "/leonardo_scratch/large/userexternal/sbenatti/data/processed/x/" + k + ".json", "sha256": v}
                       for k, v in science_inputs.items()},
            "roots": sorted(map(str, J.DATA_ROOTS))}
job = C.make_job(binding, audit_cell["id"], evidence, str(fresh / "audit" / "audit.json"))
print("job keys", sorted(job))
(ROOT / REL / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
(ROOT / REL / "audit.job.json").write_text(json.dumps(job, indent=2) + "\n")
git("add", REVIEW_PATH, REL)
git("commit", "-qm", "synthetic audit-only release (metadata only)")
RELEASE = git("rev-parse", "HEAD")
print("release", RELEASE[:12], "parent", git("rev-parse", "HEAD^")[:12])
print("diff names:", git("diff", "--name-only", TIP, RELEASE).split())
report("B0 validate_review_binding nel commit di release", lambda: C.validate_review_binding(binding)["reviewed_commit"][:8])

# Positive dispatcher path: physical input files and roots replaced by local stand-ins (binding untouched).
roots = [TMP / "rootA", TMP / "rootB"]
for r in roots:
    r.mkdir(exist_ok=True)
J.DATA_ROOTS = set(r.resolve() for r in roots)
J.validate_input_bindings = lambda pins: None
local_inputs = {}
for k in science_inputs:
    f = TMP / (k + ".json")
    f.write_text(json.dumps({"stand_in": k}))
    local_inputs[k] = {"path": str(f), "sha256": C.file_sha(f)}
job_pos = copy.deepcopy(job)
job_pos["inputs"] = local_inputs
job_pos["roots"] = [str(r) for r in roots]
report("B1 build_command(audit, cpu, 7200) con binding reale", lambda: [Path(x).name if "/" in x else x for x in J.build_command(job_pos, "cpu", 7200)[1:4]])


def mutated(label, change, seconds=7200, kind="cpu"):
    candidate = copy.deepcopy(job_pos)
    change(candidate)
    report(label, lambda: J.build_command(candidate, kind, seconds)[1])


mutated("N1 job senza package_review", lambda j: j.pop("package_review"))
mutated("N2 package_review = None", lambda j: j.update(package_review=None))
mutated("N3 review_payload manomesso nel binding", lambda j: j["package_review"]["review_payload"].update(status="REJECTED"))
mutated("N4 original_review_sha256 diverso", lambda j: j["package_review"].update(original_review_sha256="0" * 64))
mutated("N5 binding che punta al piano storico", lambda j: j["package_review"].update(
    science_plan=C.pin(ROOT / "results/infra/contrastive_comparison_20261116/plan.json", relative=True)))
mutated("N6 review del job diversa da quella del binding", lambda j: j.update(review=C.pin(ROOT / "docs/decisioni.md", relative=True)))
mutated("N7 review_commit falso (non verificato a runtime)", lambda j: j["package_review"].update(review_commit="2" * 40))
mutated("N8 original_review_path falso (non verificato a runtime)", lambda j: j["package_review"].update(original_review_path="nowhere.md"))
mutated("N9 id della cella di calibrazione con mode audit", lambda j: j.update(id="CONTRAST-CALIB-targets-v3-20261118"))
mutated("N10 arm CN sull'audit", lambda j: j.update(arm="CN"))
mutated("N11 repeat 'a' sull'audit", lambda j: j.update(repeat="a"))
mutated("N12 allocazione 3600 s (senza --time=02:00:00)", lambda j: None, seconds=3600)
mutated("N13 kind gpu", lambda j: None, kind="gpu")
mutated("N14 out = fresh_root", lambda j: j.update(out=str(fresh)))
mutated("N15 out fuori da fresh_root (altro results)", lambda j: j.update(out="/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/other/audit.json"))
mutated("N16 source_sha256 della coorte storica", lambda j: j.update(source_sha256="0" * 64))
mutated("N17 status diverso", lambda j: j.update(status="APPROVED_FOR_TRAINING"))
mutated("N18 mode readout con id audit", lambda j: j.update(mode="readout"), seconds=3600)
hist_job = P._read(ROOT / "results/infra/contrastive_comparison_20261116/releases/audit.job.json")
report("N19 job audit storico 17/11 (releases/audit.job.json)", lambda: J.build_command(hist_job, "cpu", hist_job.get("seconds", 3600))[1])
report("N19b job audit storico 17/11 con 7200 s", lambda: J.build_command(hist_job, "cpu", 7200)[1])

# Release-commit mutations: only inventory files and the two plans are compared with the reviewed commit.
def mutation(label, apply, cleanup, extra=None):
    apply()
    try:
        report(label, lambda: C.validate_review_binding(binding)["reviewed_commit"][:8])
        if extra:
            extra()
    finally:
        cleanup()
        assert git("status", "--porcelain", "--untracked-files=all") == "", git("status", "--porcelain")


def add_file(rel, content):
    return (lambda: (ROOT / rel).write_text(content)), (lambda: (ROOT / rel).unlink())


def restore():
    git("checkout", "-q", "--", ".")


a, c = add_file("scripts/contrastive_extra.py", "print('x')\n")
mutation("M1 nuovo scripts/contrastive_extra.py", a, c)
mutation("M2 src/wearusfm/__init__.py modificato", lambda: (ROOT / "src/wearusfm/__init__.py").write_text((ROOT / "src/wearusfm/__init__.py").read_text() + "\n"), restore)
mutation("M3 plan.json modificato (status)", lambda: (ROOT / P.PLAN).write_text((ROOT / P.PLAN).read_text().replace('"REVIEW_REQUIRED"', '"APPROVED"', 1)), restore)
mutation("M4 preflight_plan.json modificato (seconds audit)", lambda: (ROOT / C.PREFLIGHT).write_text((ROOT / C.PREFLIGHT).read_text().replace('"seconds": 7200', '"seconds": 7300', 1)), restore)
victim = ROOT / "src/wearusfm/data/virtual_montage.py"
mutation("M5 sorgente dell'inventario cancellato", lambda: victim.unlink(), restore)
mutation("M6 scripts/slurm/contrastive_preflight_cpu.sbatch modificato", lambda: (ROOT / "scripts/slurm/contrastive_preflight_cpu.sbatch").write_text((ROOT / "scripts/slurm/contrastive_preflight_cpu.sbatch").read_text() + "\n"), restore)


def run_python_startup():
    out = subprocess.run([sys.executable, "-B", "-c", "import json; print('json from', json.__file__)"], cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, text=True)
    print("    python -c con PYTHONPATH=src ->", (out.stdout + out.stderr).strip().replace("\n", " | ")[:200])


a, c = add_file("src/sitecustomize.py", "print('R3 INJECTED sitecustomize: codice non rivisto eseguito')\n")
mutation("M7 nuovo src/sitecustomize.py (fuori inventario)", a, c, extra=run_python_startup)


def run_script_shadow():
    out = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/contrastive_release.py"), "--help"], cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, text=True)
    print("    contrastive_release.py --help ->", (out.stdout + out.stderr).strip().splitlines()[0][:160])


a, c = add_file("scripts/argparse.py", "print('R3 INJECTED scripts/argparse.py eseguito al posto della stdlib')\nraise SystemExit(7)\n")
mutation("M8 nuovo scripts/argparse.py (ombra della stdlib)", a, c, extra=run_script_shadow)
a, c = add_file("src/torch_shadow_marker.py", "x = 1\n")
mutation("M9 file .py nuovo in src/ fuori da src/wearusfm", a, c)

print("ancestry/diff checks in contrastive_release.py:",
      subprocess.run(["grep", "-cE", "merge-base|is-ancestor|rev-list|diff --name", str(ROOT / "scripts/contrastive_release.py")],
                     capture_output=True, text=True).stdout.strip())
print("final status:", git("status", "--porcelain", "--untracked-files=all") or "clean", "HEAD", git("rev-parse", "HEAD")[:12])
