"""Scettico R3-M2: provenienza della review dopo bind-review. Eseguire con cwd = clone temporaneo."""
import copy, json, os, subprocess, sys, hashlib
from pathlib import Path

D = Path(sys.argv[1]).resolve(); W = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(D / "scripts"))
import contrastive_release as C
import contrastive_prepare as P
assert C.ROOT == D, (C.ROOT, D)

def run(cmd, **kw):
    return subprocess.run(cmd, cwd=D, capture_output=True, **kw)

def git(*a, env=None, data=None):
    r = subprocess.run(["git", "-c", "user.name=Probe", "-c", "user.email=probe@example.invalid", *a], cwd=D,
                       capture_output=True, input=data, env=env)
    if r.returncode: raise RuntimeError(r.stderr.decode())
    return r.stdout.decode().strip()

def orphan_commit(files, parent=None, msg="probe"):
    idx = W / "idx"
    if idx.exists(): idx.unlink()
    env = dict(os.environ, GIT_INDEX_FILE=str(idx))
    for path, data in files.items():
        blob = git("hash-object", "-w", "--stdin", data=data)
        git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
    tree = git("write-tree", env=env)
    args = ["commit-tree", tree, "-m", msg] + (["-p", parent] if parent else [])
    return git(*args)

def outcome(label, fn):
    try:
        val = fn()
        print(f"{label}: ACCEPTED" + (f" -> {val}" if val is not None else ""))
        return True
    except Exception as e:
        print(f"{label}: REJECTED {type(e).__name__}: {str(e)[:160]}")
        return False

src, plan, pre = P.source_digest(), C.file_sha(P.PLAN), C.file_sha(C.PREFLIGHT)
print("hashes", src, plan, pre)
CONTENT = "fbfde797b48644744c36547533acf0e8a7793535"
payload = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": CONTENT, "source_sha256": src,
           "science_plan_sha256": plan, "preflight_plan_sha256": pre}
fake = ("# Review NON di Claude: scritta dall'operatore per la prova R3-M2\n\n```contrastive-release-review\n"
        + json.dumps(payload, indent=1) + "\n```\n").encode()

# E1: commit orfano, non raggiungibile da alcun ref, con review falsa
F = orphan_commit({"docs/reviews/vm2_fake_review.md": fake}, msg="orphan fake review")
print("E1 orphan commit", F, "refs containing it:", repr(git("for-each-ref", "--contains", F)))
copy_path = D / "docs/reviews/vm2_fake_review.md"
copy_path.write_bytes(run(["git", "show", F + ":docs/reviews/vm2_fake_review.md"]).stdout)
r = run([sys.executable, "-B", "scripts/contrastive_release.py", "bind-review", "--review-commit", F,
         "--review-path", "docs/reviews/vm2_fake_review.md", "--review-copy", "docs/reviews/vm2_fake_review.md",
         "--out", "results/infra/contrastive_targets_20261118/vm2_release/binding.json"], env=os.environ)
print("E1 bind-review CLI (orphan, review non di Claude) rc", r.returncode, r.stderr.decode()[-200:])
git("add", "docs/reviews/vm2_fake_review.md", "results/infra/contrastive_targets_20261118/vm2_release/binding.json")
git("commit", "-qm", "synthetic release (probe R3-M2)")
REL = git("rev-parse", "HEAD")
print("release commit", REL, "parent", git("rev-parse", "HEAD^"), "status:", repr(git("status", "--porcelain")))
binding = json.loads((D / "results/infra/contrastive_targets_20261118/vm2_release/binding.json").read_text())

def check_cli(path):
    r = run([sys.executable, "-B", "scripts/contrastive_release.py", "check-binding", "--binding", str(path)], env=os.environ)
    return r.returncode, (r.stdout.decode().strip()[:90] or r.stderr.decode().strip()[-160:])

print("E2 check-binding CLI binding reale:", check_cli("results/infra/contrastive_targets_20261118/vm2_release/binding.json"))
forged = copy.deepcopy(binding); forged["review_commit"] = "2" * 40; forged["original_review_path"] = "docs/reviews/non_esiste.md"
(W / "binding_forged.json").write_text(json.dumps(forged, indent=2) + "\n")
print("E3 check-binding CLI review_commit 2*40 + path inesistente:", check_cli(W / "binding_forged.json"))
print("E3 git cat-file -e 2*40 rc", run(["git", "cat-file", "-e", "2" * 40 + "^{commit}"]).returncode)
hand = {"schema": C.BINDING_SCHEMA, "review_commit": "3" * 40, "original_review_path": "qualunque",
        "original_review_sha256": C.file_sha(copy_path), "review": C.pin(copy_path, relative=True),
        "review_payload": payload, "science_plan": C.pin(P.PLAN, relative=True), "preflight_plan": C.pin(C.PREFLIGHT, relative=True)}
outcome("E4 binding scritto a mano senza bind-review", lambda: C.validate_review_binding(hand)["reviewed_commit"])

# E5 controlli negativi: cosa resta verificato
b = copy.deepcopy(forged); b["original_review_sha256"] = "0" * 64
outcome("E5a original_review_sha256 diverso dalla copia", lambda: C.validate_review_binding(b))
b = copy.deepcopy(forged); b["review_commit"] = "2" * 39
outcome("E5b review_commit non pieno", lambda: C.validate_review_binding(b))
bad = dict(payload, reviewed_commit="0" * 40)
nocopy = D / "docs/reviews/vm2_bad_commit.md"
nocopy.write_bytes(fake.replace(CONTENT.encode(), b"0" * 40))
b = copy.deepcopy(hand); b.update(review=C.pin(nocopy, relative=True), review_payload=bad, original_review_sha256=C.file_sha(nocopy))
outcome("E5c blocco con reviewed_commit inesistente", lambda: C.validate_review_binding(b))
stale = D / "docs/reviews/vm2_stale_hash.md"
stale.write_bytes(fake.replace(src.encode(), b"1" * 64))
b = copy.deepcopy(hand); b.update(review=C.pin(stale, relative=True), review_payload=dict(payload, source_sha256="1" * 64), original_review_sha256=C.file_sha(stale))
outcome("E5d blocco con source_sha256 non corrente", lambda: C.validate_review_binding(b))
nocopy.unlink(); stale.unlink()

# E6 versione superata: Ra approva, Rb (figlio) la ritira
v1 = fake.replace(b"# Review NON", b"# v1 Review NON")
v2 = b"# v2: approvazione RITIRATA, nessun blocco di release\n"
Ra = orphan_commit({"docs/reviews/vm2_sup.md": v1}, msg="review v1 approved")
Rb = orphan_commit({"docs/reviews/vm2_sup.md": v2}, parent=Ra, msg="review v2 withdrawn")
sup_copy = D / "docs/reviews/vm2_sup.md"; sup_copy.write_bytes(v1)
sup = None
def bind_sup():
    global sup
    sup = C.bind_review(Ra, "docs/reviews/vm2_sup.md", sup_copy)
    return sup["review_commit"][:12]
outcome(f"E6 bind-review su versione superata Ra={Ra[:12]} (punta Rb={Rb[:12]} ritirata)", bind_sup)

# E7 correzione proposta: riverifica git show review_commit:original_review_path == copia
def recheck(bd):
    original = C.git_bytes(bd["review_commit"], bd["original_review_path"])
    if original != (D / bd["review"]["path"]).read_bytes():
        raise ValueError("copy differs from original")
    C.validate_review_binding(bd)
outcome("E7a fix proposto su binding falsificato (E3)", lambda: recheck(forged))
outcome("E7b fix proposto su review falsa in commit orfano (E1)", lambda: recheck(binding))
outcome("E7c fix proposto su versione superata (E6)", lambda: recheck(sup))
sup_copy.unlink()

# E8 make_job e dispatcher con binding falsificato
cfg = D / "results/infra/contrastive_targets_20261118/review_configs/calibration.config.json"
science = P._read(P.PLAN)
evidence = {"inputs": {k: {"path": f"/leonardo_work/IscrB_WearUsFM/data/processed/vm2_{k}", "sha256": v} for k, v in science["input_sha256"].items()},
            "config": C.pin(cfg, relative=True),
            "roots": sorted(["/leonardo_work/IscrB_WearUsFM/data/processed", "/leonardo_scratch/large/userexternal/sbenatti/data/processed"])}
out = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/vm2_audit_probe"
job = None
def mk():
    global job
    job = C.make_job(forged, "CONTRAST-AUDIT-targets-v3-20261118", evidence, out)
    return (job["package_review"]["review_commit"][:8], job["id"])
outcome("E8 make_job audit con binding falsificato", mk)
os.environ["WORK"] = "/leonardo_work/IscrB_WearUsFM"
import contrastive_job as J
outcome("E9 dispatcher build_command(audit,cpu,7200) con binding falsificato", lambda: J.build_command(job, "cpu", 7200))

# E10 runtime validate_release (calibrazione) con review_pin == review_commit falso
import contrastive_run as RN
RN.validate_input_bindings = lambda pins: None  # non pertinente: input fittizi sul Mac
fake_inputs = {}
for k in ("manifest", "scales", "window_seconds", "pose_stats"):
    p = W / f"input_{k}"; p.write_bytes(k.encode()); fake_inputs[k] = p
release = {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration", "source_sha256": src, "config_sha256": RN.sha(cfg),
           "world_size": 4, "workers_per_rank": 6, "inputs": {k: RN.sha(p) for k, p in fake_inputs.items()},
           "review_pin": "2" * 40, "package_review": forged, "time_limit_s": 1770}
outcome("E10 runtime validate_release, review_pin=review_commit=2*40 (time_limit diverso per fermarsi DOPO le righe 179-184)",
        lambda: RN.validate_release(release, "calibration", cfg, fake_inputs, time_limit_s=1))
r2 = dict(release, review_pin="4" * 40)
outcome("E10b runtime con review_pin diverso dal binding", lambda: RN.validate_release(r2, "calibration", cfg, fake_inputs, time_limit_s=1))
