"""r3 review: bind-review/review_payload experiments in a temporary clone (argv[1]).

Creates orphan commits (simulating Claude's review commit on another branch) via a
temporary index, writes an untracked local copy and calls the real bind_review.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_release as C  # noqa: E402
import contrastive_prepare as P  # noqa: E402

assert C.ROOT == ROOT, (C.ROOT, ROOT)
ENV = dict(os.environ, GIT_AUTHOR_NAME="R3", GIT_AUTHOR_EMAIL="r3@example.invalid",
           GIT_COMMITTER_NAME="R3", GIT_COMMITTER_EMAIL="r3@example.invalid")


def git(*args, inp=None, env=None):
    return subprocess.run(["git", *args], cwd=ROOT, input=inp, capture_output=True, check=True,
                          env=env or ENV).stdout.decode().strip()


FBF = "fbfde797b48644744c36547533acf0e8a7793535"
TIP = git("rev-parse", "HEAD")
BASE = git("rev-parse", "438c8d65")
SRC, SP, PP = P.source_digest(), C.file_sha(P.PLAN), C.file_sha(C.PREFLIGHT)
REVIEW_PATH = "docs/reviews/claude_review_r3_probe_20261119.md"
LOCAL = ROOT / REVIEW_PATH


def payload(**over):
    value = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": FBF,
             "source_sha256": SRC, "science_plan_sha256": SP, "preflight_plan_sha256": PP}
    for key, item in over.items():
        if item is None:
            value.pop(key)
        else:
            value[key] = item
    return value


def fence(value, info="contrastive-release-review", indent=2, raw=None):
    body = raw if raw is not None else json.dumps(value, indent=indent)
    return "```" + info + "\n" + body + "\n```\n"


def orphan_commit(path, data: bytes):
    blob = git("hash-object", "-w", "--stdin", inp=data)
    with tempfile.TemporaryDirectory(dir=os.environ.get("R3TMP")) as tmp:
        env = dict(ENV, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git("update-index", "--add", "--cacheinfo", "100644," + blob + "," + path, env=env)
        tree = git("write-tree", env=env)
    return git("commit-tree", tree, "-m", "review probe", inp=b"")


def attempt(label, text, *, local=None, review_path=REVIEW_PATH, commit=None, encoding_bytes=None):
    data = encoding_bytes if encoding_bytes is not None else text.encode()
    sha = commit or orphan_commit(review_path, data)
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    LOCAL.write_bytes(data if local is None else local)
    try:
        binding = C.bind_review(sha, review_path, LOCAL)
        out = "ACCEPTED reviewed_commit=" + binding["review_payload"]["reviewed_commit"][:8]
    except Exception as error:  # report the exact refusal
        out = "REJECTED " + type(error).__name__ + ": " + str(error)[:150]
        binding = None
    finally:
        LOCAL.unlink()
    print(f"{label:58s} {out}")
    return binding


good = payload()
cases = [
    ("A1 un blocco multi-riga (come nel prompt), commit fbfde797", "# Review\n\ntesto\n\n" + fence(good) + "\nfine\n"),
    ("A2 reviewed_commit = punta 0efa4d45", fence(payload(reviewed_commit=TIP))),
    ("A3 JSON su una riga", fence(good, raw=json.dumps(good))),
    ("A4 blocco in testa al file, nessun testo prima", fence(good)),
    ("A5 blocco a fine file senza newline finale", "# R\n" + fence(good).rstrip("\n")),
    ("A6 spazi dopo il nome del recinto", "```contrastive-release-review   \n" + json.dumps(good, indent=2) + "\n```\n"),
    ("A7 recinto indentato (dentro un elenco)", "- voce\n  ```contrastive-release-review\n" + json.dumps(good) + "\n  ```\n"),
    ("A8 dentro un recinto ````text (come nel prompt)", "````text\n" + fence(good) + "````\n"),
    ("A9 CRLF in tutto il file", fence(good).replace("\n", "\r\n")),
    ("A10 chiave status duplicata (REJECTED poi APPROVED)", fence(None, raw=json.dumps(good)[:-1] .replace('"status": "APPROVED"', '"status": "REJECTED"') + ', "status": "APPROVED"}')),
    ("A11 recinto di chiusura con info ```json", "```contrastive-release-review\n" + json.dumps(good) + "\n```json\n"),
    ("A12 testo con accenti/simboli (UTF-8)", "# Revisione: è approvata, ≤ 7200 s — ok\n" + fence(good)),
    ("R1 due blocchi (modello con segnaposto + reale)", fence(payload(reviewed_commit="SHA_COMPLETO")) + "\n" + fence(good)),
    ("R2 modello del prompt in ````text + blocco reale", "````text\n" + fence(payload(status="X")) + "````\n\n" + fence(good)),
    ("R3 due blocchi identici", fence(good) + fence(good)),
    ("R4 spazio fra ``` e nome", "``` contrastive-release-review\n" + json.dumps(good) + "\n```\n"),
    ("R5 info string con suffisso json", fence(good, info="contrastive-release-review json")),
    ("R6 nome con maiuscole", fence(good, info="Contrastive-Release-Review")),
    ("R7 recinto ~~~", "~~~contrastive-release-review\n" + json.dumps(good) + "\n~~~\n"),
    ("R8 nessun blocco (solo menzione inline)", "Blocco `contrastive-release-review` APPROVED\n"),
    ("R9 status APPROVATO", fence(payload(status="APPROVATO"))),
    ("R10 status approved (minuscolo)", fence(payload(status="approved"))),
    ("R11 status APPROVED_WITH_NOTES", fence(payload(status="APPROVED_WITH_NOTES"))),
    ("R12 schema v2", fence(payload(schema="contrastive-target-release-review-v2"))),
    ("R13 chiave in piu' (note)", fence(payload(note="ok"))),
    ("R14 chiave mancante (preflight_plan_sha256)", fence(payload(preflight_plan_sha256=None))),
    ("R15 reviewed_commit abbreviato", fence(payload(reviewed_commit=FBF[:12]))),
    ("R16 reviewed_commit maiuscolo", fence(payload(reviewed_commit=FBF.upper()))),
    ("R17 reviewed_commit = base 438c8d65 (sorgenti diversi)", fence(payload(reviewed_commit=BASE))),
    ("R18 reviewed_commit inesistente", fence(payload(reviewed_commit="1" * 40))),
    ("R19 source_sha256 della base", None),
    ("R20 science_plan_sha256 di un altro piano", None),
    ("R21 preflight_plan_sha256 hex maiuscolo", fence(payload(preflight_plan_sha256=PP.upper()))),
    ("R22 JSON con virgola finale", "```contrastive-release-review\n" + json.dumps(good)[:-1] + ",}\n```\n"),
    ("R23 JSON lista invece di oggetto", None),
]

# Values from another package: base source digest and the historical comparison plan.
base_src = None
try:
    out = subprocess.run(["git", "show", BASE + ":scripts/contrastive_prepare.py"], cwd=ROOT, capture_output=True, check=True)
    base_src = "0" * 64
except Exception:
    pass
hist_plan = ROOT / "results/infra/contrastive_comparison_20261116/plan.json"
for index, (label, text) in enumerate(cases):
    if label.startswith("R19"):
        # digest of base sources: compute by checking out nothing; use declared source of base preflight plan
        base_preflight = json.loads(subprocess.run(["git", "show", BASE + ":results/infra/contrastive_targets_20261118/preflight_plan.json"],
                                                   cwd=ROOT, capture_output=True, check=True).stdout)
        text = fence(payload(source_sha256=base_preflight.get("source_sha256", "0" * 64)))
    elif label.startswith("R20"):
        text = fence(payload(science_plan_sha256=C.file_sha(hist_plan)))
    elif label.startswith("R23"):
        text = fence(None, raw=json.dumps(sorted(good)))
    cases[index] = (label, text)

for label, text in cases:
    attempt(label, text)

# Local copy and provenance checks.
attempt("L1 copia locale con newline in piu'", fence(good), local=(fence(good) + "\n").encode())
attempt("L2 copia locale in CRLF", fence(good), local=fence(good).replace("\n", "\r\n").encode())
sha = orphan_commit(REVIEW_PATH, fence(good).encode())
attempt("L3 review_path assente nel commit della review", fence(good), review_path="docs/reviews/altro.md", commit=sha)
attempt("L4 review_path con ..", fence(good), review_path="docs/../docs/reviews/x.md", commit=sha)
# Copy outside the checkout.
outside = Path(os.environ["R3TMP"]) / "outside_review.md"
outside.write_text(fence(good))
try:
    C.bind_review(sha, REVIEW_PATH, outside)
    print(f"{'L5 copia locale fuori dal checkout':58s} ACCEPTED")
except Exception as error:
    print(f"{'L5 copia locale fuori dal checkout':58s} REJECTED {type(error).__name__}: {str(error)[:120]}")
outside.unlink()
# Symlink inside the checkout pointing outside.
outside.write_text(fence(good))
link = ROOT / "docs/reviews/r3_symlink_review.md"
link.symlink_to(outside)
try:
    C.bind_review(sha, REVIEW_PATH, link)
    print(f"{'L6 symlink nel checkout verso file esterno':58s} ACCEPTED")
except Exception as error:
    print(f"{'L6 symlink nel checkout verso file esterno':58s} REJECTED {type(error).__name__}: {str(error)[:120]}")
link.unlink(); outside.unlink()

# Real historical reviews.
for label, commit, path in [
        ("H1 review Claude 94f50942 (file della review preparatoria)", "94f5094224bde9cc076c093ef37e57f3f83dc49c",
         "docs/reviews/claude_review_contrastive_targets_amendment_20261119.md"),
        ("H2 release 17/11 (contrastive_preflight_release_20261117)", TIP, "docs/reviews/contrastive_preflight_release_20261117.md"),
        ("H3 review CUDA/benchmark 17/11", TIP, "docs/reviews/contrastive_cuda_benchmark_release_20261117.md"),
        ("H4 prompt per Claude (contiene il modello del blocco)", TIP, "docs/prompt_claude_contrastive_targets_runtime_20261119.md")]:
    data = subprocess.run(["git", "show", commit + ":" + path], cwd=ROOT, capture_output=True, check=True).stdout
    LOCAL.write_bytes(data)
    try:
        C.bind_review(commit, path, LOCAL)
        print(f"{label:58s} ACCEPTED")
    except Exception as error:
        print(f"{label:58s} REJECTED {type(error).__name__}: {str(error)[:120]}")
    LOCAL.unlink()
print("blocks in prompt:", len(C.REVIEW_BLOCK.findall(subprocess.run(["git", "show", TIP + ":docs/prompt_claude_contrastive_targets_runtime_20261119.md"], cwd=ROOT, capture_output=True, check=True).stdout.decode())))
print("status", git("status", "--porcelain", "--untracked-files=all") or "clean")
