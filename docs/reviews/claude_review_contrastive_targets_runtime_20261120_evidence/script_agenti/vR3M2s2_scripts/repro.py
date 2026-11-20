"""Independent reproduction for R3-M2 (skeptic 2). Run with cwd = temp clone root."""
import copy, json, os, subprocess, sys, time
from pathlib import Path

D = Path(sys.argv[1]).resolve(); W = Path(sys.argv[2]).resolve()
A, B, O = sys.argv[3], sys.argv[4], sys.argv[5]
PY = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{D}/src:{D}/scripts")
REV = "docs/reviews/claude_review_sim_20261120.md"
REL = "results/infra/contrastive_targets_20261118/release_sim"
FRESH = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118"

def cli(*args):
    t = time.monotonic()
    p = subprocess.run([PY, "-B", "scripts/contrastive_release.py", *args], cwd=D, env=ENV, capture_output=True, text=True)
    tail = (p.stdout.strip() or p.stderr.strip().splitlines()[-1] if (p.stdout.strip() or p.stderr.strip()) else "")
    return p.returncode, tail[:300], round(time.monotonic() - t, 2)

def git(*args):
    return subprocess.run(["git", "-c", "user.name=Sim", "-c", "user.email=sim@example.invalid", *args], cwd=D,
                          capture_output=True, text=True)

# 1. review copy in the release working tree, byte-identical to A
(D / REV).write_bytes(subprocess.check_output(["git", "show", f"{A}:{REV}"], cwd=D))
# 2. bind-review via CLI
print("BIND A (commit su ramo sim-claude-review):", cli("bind-review", "--review-commit", A, "--review-path", REV,
      "--review-copy", REV, "--out", f"{REL}/binding.json"))
# 3. make-job audit via CLI
cell = "CONTRAST-AUDIT-targets-v3-20261118"
(W / "evidence_empty.json").write_text("{}\n")
print("MAKE-JOB reale:", cli("make-job", "--binding", f"{REL}/binding.json", "--cell-id", cell,
      "--evidence", str(W / "evidence_empty.json"), "--result-out", FRESH + "/audit_sim", "--out", f"{REL}/audit.job.json"))
# 4. release commit (child of 0efa4d45)
git("add", REV, REL)
c = git("commit", "-qm", "Sim: release audit (review copy + binding + job)")
print("release commit:", git("rev-parse", "HEAD").stdout.strip(), "parent:", git("rev-parse", "HEAD^").stdout.strip()[:8],
      "status pulito:", git("status", "--porcelain").stdout == "")
# 5. check-binding, real binding
print("CHECK-BINDING reale:", cli("check-binding", "--binding", f"{REL}/binding.json"))
# 6. fake binding outside the repository
real = json.loads((D / REL / "binding.json").read_text())
fake = copy.deepcopy(real); fake["review_commit"] = "2" * 40; fake["original_review_path"] = "docs/reviews/non_esiste.md"
(W / "binding_fake.json").write_text(json.dumps(fake, indent=2) + "\n")
print("CHECK-BINDING falso (review_commit 2*40, path inesistente):", cli("check-binding", "--binding", str(W / "binding_fake.json")))
print("git cat-file -e 2*40^{commit}: rc", subprocess.run(["git", "cat-file", "-e", "2" * 40 + "^{commit}"], cwd=D, capture_output=True).returncode)
print("docs/reviews/non_esiste.md esiste:", (D / "docs/reviews/non_esiste.md").exists())
# controls: tamper original_review_sha256 / review_payload
for name, mut in (("original_review_sha256", lambda b: b.__setitem__("original_review_sha256", "0" * 64)),
                  ("review_payload.status", lambda b: b["review_payload"].__setitem__("status", "REJECTED")),
                  ("review_commit abbreviato", lambda b: b.__setitem__("review_commit", "2" * 39))):
    ctl = copy.deepcopy(real); mut(ctl); p = W / f"binding_ctl_{name.split('.')[0].replace(' ', '_')}.json"
    p.write_text(json.dumps(ctl)); print(f"CONTROLLO {name}:", cli("check-binding", "--binding", str(p)))
# 7. make-job with the fake binding
print("MAKE-JOB falso:", cli("make-job", "--binding", str(W / "binding_fake.json"), "--cell-id", cell,
      "--evidence", str(W / "evidence_empty.json"), "--result-out", FRESH + "/audit_sim2", "--out", str(W / "audit_fake.job.json")))
