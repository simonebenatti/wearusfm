"""r3 review (ripresa 20/11): casi limite aggiuntivi di bind-review e CLI, in un clone temporaneo (argv[1]).

Commit orfani simulano il commit della review di Claude su un altro ramo; la copia locale e' scritta e rimossa.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_release as C  # noqa: E402
import contrastive_prepare as P  # noqa: E402

ENV = dict(os.environ, GIT_AUTHOR_NAME="R3", GIT_AUTHOR_EMAIL="r3@example.invalid",
           GIT_COMMITTER_NAME="R3", GIT_COMMITTER_EMAIL="r3@example.invalid")
TMP = Path(os.environ["R3TMP"])


def git(*args, inp=None, env=None):
    return subprocess.run(["git", *args], cwd=ROOT, input=inp, capture_output=True, check=True,
                          env=env or ENV).stdout.decode().strip()


FBF = "fbfde797b48644744c36547533acf0e8a7793535"
SRC, SP, PP = P.source_digest(), C.file_sha(P.PLAN), C.file_sha(C.PREFLIGHT)
REVIEW_PATH = "docs/reviews/claude_review_r3_probe2_20261120.md"
LOCAL = ROOT / REVIEW_PATH
GOOD = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": FBF,
        "source_sha256": SRC, "science_plan_sha256": SP, "preflight_plan_sha256": PP}
J = json.dumps(GOOD, indent=2)


def orphan_commit(path, data):
    blob = git("hash-object", "-w", "--stdin", inp=data)
    with tempfile.TemporaryDirectory(dir=TMP) as tmp:
        env = dict(ENV, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("update-index", "--add", "--cacheinfo", "100644," + blob + "," + path, env=env)
        tree = git("write-tree", env=env)
    return git("commit-tree", tree, "-m", "review probe", inp=b"")


def attempt(label, data):
    if isinstance(data, str):
        data = data.encode()
    sha = orphan_commit(REVIEW_PATH, data)
    LOCAL.write_bytes(data)
    try:
        binding = C.bind_review(sha, REVIEW_PATH, LOCAL)
        out = "ACCEPTED " + json.dumps({k: binding["review_payload"][k] for k in ("status", "reviewed_commit")})[:90]
    except Exception as error:
        out = "REJECTED " + type(error).__name__ + ": " + str(error)[:120]
    finally:
        LOCAL.unlink()
    print(f"{label:66s} {out}", flush=True)


dup_status = J[:-2] + ',\n  "status": "APPROVED"\n}'
dup_status = dup_status.replace('"status": "APPROVED"', '"status": "REJECTED"', 1)
dup_commit = J[:-2] + ',\n  "reviewed_commit": "' + FBF + '"\n}'
dup_commit = dup_commit.replace(FBF, "0" * 40, 1)
cases = [
    ("E1 apertura a meta' riga ('Testo ```contrastive-release-review')", "Testo ```contrastive-release-review\n" + J + "\n```\n"),
    ("E2 recinto a 4 backtick per il blocco reale", "````contrastive-release-review\n" + J + "\n````\n"),
    ("E3 blocco in citazione '>'", "> ```contrastive-release-review\n> " + J.replace("\n", "\n> ") + "\n> ```\n"),
    ("E3b citazione '>' seguita da un altro blocco ```bash", "> ```contrastive-release-review\n> " + J.replace("\n", "\n> ") + "\n> ```\n\n```bash\nls\n```\n"),
    ("E4 BOM UTF-8 in testa", "﻿# Review\n```contrastive-release-review\n" + J + "\n```\n"),
    ("E5 apertura indentata, chiusura a colonna 0", "  ```contrastive-release-review\n" + J + "\n```\n"),
    ("E6 blocco reale + menzione ```contrastive-release-review``` inline", "```contrastive-release-review\n" + J + "\n```\nIl blocco ```contrastive-release-review``` sopra e' unico.\n"),
    ("E7 blocco reale + esempio del formato in ```text", "```contrastive-release-review\n" + J + "\n```\n\nFormato:\n```text\n```contrastive-release-review\n{}\n```\n"),
    ("E8 chiavi in ordine diverso", "```contrastive-release-review\n" + json.dumps(dict(reversed(list(GOOD.items())))) + "\n```\n"),
    ("E9 righe vuote e spazi nel JSON", "```contrastive-release-review\n\n\n" + J.replace(",", " ,  ") + "\n\n```\n"),
    ("E10 file in latin-1 con accenti", ("# Revisione: è approvata\n```contrastive-release-review\n" + J + "\n```\n").encode("latin-1")),
    ("E12 reviewed_commit duplicato ('0'*40 poi fbfde797)", "```contrastive-release-review\n" + dup_commit + "\n```\n"),
    ("E12b status duplicato (REJECTED poi APPROVED)", "```contrastive-release-review\n" + dup_status + "\n```\n"),
    ("E13 APPROVED scritto con escape \\u0041", "```contrastive-release-review\n" + J.replace('"APPROVED"', '"\\u0041PPROVED"') + "\n```\n"),
    ("E14 chiusura ``` attaccata al JSON ('}```')", "```contrastive-release-review\n" + J + "```\n\nAltro testo\n"),
    ("E15 blocco valido dopo un blocco ```contrastive-release-review vuoto", "```contrastive-release-review\n```\n\n```contrastive-release-review\n" + J + "\n```\n"),
    ("E16 tab dopo il nome del recinto", "```contrastive-release-review\t\n" + J + "\n```\n"),
    ("E17 valori come null (preflight)", "```contrastive-release-review\n" + json.dumps({**GOOD, "preflight_plan_sha256": None}) + "\n```\n"),
    ("E18 JSON con NaN in un campo", "```contrastive-release-review\n" + J[:-2] + ',\n  "x": NaN\n}' + "\n```\n"),
]
for label, text in cases:
    attempt(label, text)

# CLI end-to-end, come lo userebbe l'operatore.
data = ("# Review\n\n```contrastive-release-review\n" + J + "\n```\n").encode()
sha = orphan_commit(REVIEW_PATH, data)
LOCAL.write_bytes(data)
out_dir = Path(tempfile.mkdtemp(dir=TMP))
binding_out = out_dir / "binding.json"
env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}/scripts", PYTHONDONTWRITEBYTECODE="1")
cli = [sys.executable, "-B", str(ROOT / "scripts/contrastive_release.py")]
t0 = time.monotonic()
p1 = subprocess.run(cli + ["bind-review", "--review-commit", sha, "--review-path", REVIEW_PATH, "--review-copy", str(LOCAL), "--out", str(binding_out)],
                    cwd=ROOT, env=env, capture_output=True, text=True)
t1 = time.monotonic()
print("CLI bind-review rc", p1.returncode, "s=%.2f" % (t1 - t0), (p1.stderr.strip().splitlines() or [""])[-1][:120])
p2 = subprocess.run(cli + ["check-binding", "--binding", str(binding_out)], cwd=ROOT, env=env, capture_output=True, text=True)
print("CLI check-binding rc", p2.returncode, p2.stdout.strip()[:100], (p2.stderr.strip().splitlines() or [""])[-1][:120])
p3 = subprocess.run(cli + ["bind-review", "--review-commit", sha, "--review-path", REVIEW_PATH, "--review-copy", str(LOCAL), "--out", str(binding_out)],
                    cwd=ROOT, env=env, capture_output=True, text=True)
print("CLI bind-review su --out esistente rc", p3.returncode, (p3.stderr.strip().splitlines() or [""])[-1][:120])
p4 = subprocess.run(cli + ["bind-review", "--review-commit", sha[:12], "--review-path", REVIEW_PATH, "--review-copy", str(LOCAL), "--out", str(out_dir / "b2.json")],
                    cwd=ROOT, env=env, capture_output=True, text=True)
print("CLI bind-review con SHA abbreviato rc", p4.returncode, (p4.stderr.strip().splitlines() or [""])[-1][:120])
for loc in ("C", "en_US.US-ASCII"):
    accented = ("# Revisione approvata: è tutto\n```contrastive-release-review\n" + J + "\n```\n").encode()
    sha2 = orphan_commit(REVIEW_PATH, accented)
    LOCAL.write_bytes(accented)
    out = out_dir / ("b_" + loc + ".json")
    p5 = subprocess.run(cli + ["bind-review", "--review-commit", sha2, "--review-path", REVIEW_PATH, "--review-copy", str(LOCAL), "--out", str(out)],
                        cwd=ROOT, env=dict(env, LC_ALL=loc, LANG=loc), capture_output=True, text=True)
    print(f"CLI bind-review review con accenti, LC_ALL={loc} rc", p5.returncode, (p5.stderr.strip().splitlines() or [""])[-1][:120])
# Tempo di validate_review_binding (127 git show).
binding = json.loads(binding_out.read_text())
LOCAL.write_bytes(data)
t0 = time.monotonic()
C.validate_review_binding(binding)
print("validate_review_binding s=%.2f" % (time.monotonic() - t0))
t0 = time.monotonic()
C.validate_review_binding(binding, verify_git=False)
print("validate_review_binding(verify_git=False) s=%.2f" % (time.monotonic() - t0))
LOCAL.unlink()
for f in out_dir.iterdir():
    f.unlink()
out_dir.rmdir()
print("status", git("status", "--porcelain", "--untracked-files=all") or "clean")
