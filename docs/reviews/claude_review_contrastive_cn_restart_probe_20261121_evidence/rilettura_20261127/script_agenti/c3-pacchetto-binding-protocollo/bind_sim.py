"""Simulate the real bind flow in a local copy of GPT's clone (no network, no push)."""
import json, os, subprocess, sys
from pathlib import Path

W = Path(__file__).resolve().parent
COPY = W / "copy"
PY = os.path.expanduser("~/.venvs/wearusfm-tok/bin/python")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src:scripts:scripts/diagnostics",
           GIT_AUTHOR_NAME="sim", GIT_AUTHOR_EMAIL="sim@x", GIT_COMMITTER_NAME="sim", GIT_COMMITTER_EMAIL="sim@x")
TARGET = "8c72655c3d894c82fd4918ed523baf0ac9c69e23"
REVIEW_PATH = "docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md"
REF = "refs/remotes/origin/worktree-claude-training-engine"
BINDING = "results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/release_binding.json"
PKG = "e9310d67b8304c064e09d71b0ec55d37b57abf3b4e2090c2d02d4def586f702d"
PLAN = "4f538cf183f6e3daf7bd30a5612eecdbbd07106f64d5f348a2f3e759e877853b"


def git(*a, cwd=COPY, check=True, inp=None):
    r = subprocess.run(["git", "-C", str(cwd), *a], env=ENV, capture_output=True, input=inp)
    if check and r.returncode:
        raise RuntimeError(r.stderr.decode())
    return r.stdout.decode().strip()


def review_commit(body_bytes, point_ref=True, parent=None):
    """Create a commit (on top of `parent` or orphan tree) containing only the review file."""
    blob = git("hash-object", "-w", "--stdin", inp=body_bytes)
    # tree with docs/reviews/<file>
    t1 = git("mktree", inp=f"100644 blob {blob}\t{Path(REVIEW_PATH).name}\n".encode())
    t2 = git("mktree", inp=f"040000 tree {t1}\treviews\n".encode())
    t3 = git("mktree", inp=f"040000 tree {t2}\tdocs\n".encode())
    args = ["commit-tree", t3, "-m", "review"]
    if parent:
        args += ["-p", parent]
    c = git(*args)
    if point_ref:
        git("update-ref", REF, c)
    return c


def block(**over):
    d = dict(schema="cn-restart-review-v1", status="APPROVED_FOR_DIAGNOSTIC", reviewed_commit=TARGET,
             package_sha256=PKG, plan_sha256=PLAN)
    d.update(over)
    return {k: v for k, v in d.items() if v is not None}


def body(data=None, json_text=None, pre="", post="", nl="\n", fence_open="```cn-restart-review", fence_close="```"):
    js = json_text if json_text is not None else json.dumps(data if data is not None else block())
    text = (f"# Review à è ì ò ù — «virgolette» naïve{nl}{nl}La parola cn-restart-review in prosa non rompe nulla.{nl}"
            f"{pre}{fence_open}{nl}{js}{nl}{fence_close}{nl}{post}fine{nl}")
    return text.encode()


def bind(c):
    git("checkout", "-q", "--detach", TARGET)
    p = COPY / BINDING
    if p.exists():
        p.unlink()
    r = subprocess.run([PY, "-B", "scripts/diagnostics/contrastive_cn_restart.py", "--bind-review", c],
                       cwd=COPY, env=ENV, capture_output=True, text=True)
    last = (r.stderr.strip().splitlines() or [""])[-1]
    ok = r.returncode == 0 and p.exists()
    if p.exists():
        p.unlink()
    return ok, last


def case(name, body_bytes, **kw):
    c = review_commit(body_bytes, **kw)
    ok, msg = bind(c)
    print(f"{name:45s} bind={'ACCEPTED' if ok else 'REJECTED'}  {'' if ok else msg[:150]}")
    return c, ok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "negatives"
    if which == "negatives":
        base_review = review_commit(b"base\n", point_ref=False)
        case("positive single-line JSON", body())
        case("old reviewed_commit 9d34dc69 (full)", body(block(reviewed_commit="9d34dc690cf60fcbfe405279dc17546655fa3f5b")))
        case("old reviewed_commit short 9d34dc69", body(block(reviewed_commit="9d34dc69")))
        case("old package/plan pins b0640ca6/9d5811b6", body(block(
            package_sha256="b0640ca6" + "0" * 56, plan_sha256="9d5811b6" + "0" * 56)))
        case("double block", body(pre="```cn-restart-review\n" + json.dumps(block()) + "\n```\n"))
        case("multi-line JSON (indent=2)", body(json_text=json.dumps(block(), indent=2)))
        case("CRLF line endings", body(nl="\r\n"))
        case("sixth field", body(block(note="x")))
        case("indented fence (2 spaces)", body(fence_open="  ```cn-restart-review", fence_close="  ```"))
        case("trailing spaces after fences", body(fence_open="```cn-restart-review  ", fence_close="```  "))
        case("fence ```json-like tag cn-restart-review-v1", body(fence_open="```cn-restart-review-v1"))
        case("~~~ fence", body(fence_open="~~~cn-restart-review", fence_close="~~~"))
        case("blank line after open fence", body(json_text="\n" + json.dumps(block())))
        case("key order changed", body(json_text=json.dumps(dict(reversed(list(block().items()))))))
        case("uppercase reviewed_commit", body(block(reviewed_commit=TARGET.upper())))
        case("other fenced block after (```text)", body(post="```text\nesempio\n```\n"))
        case("prose fence-like line inside example ```cn-restart-review unclosed", body(post="```cn-restart-review\n"))
        # not ancestor: create review commit not reachable from REF
        c = review_commit(body(), point_ref=False)
        review_commit(b"other\n")  # REF now points elsewhere
        ok, msg = bind(c)
        print(f"{'review commit not ancestor of REF':45s} bind={'ACCEPTED' if ok else 'REJECTED'}  {msg[:150]}")
        # ancestor but older: REF advanced on top of review commit (review file later changed)
        c = review_commit(body(), point_ref=False)
        review_commit(b"changed later\n", parent=c)
        ok, msg = bind(c)
        print(f"{'ancestor, later commit changes the review':45s} bind={'ACCEPTED' if ok else 'REJECTED'}  {msg[:150]}")
        # REF missing
        git("update-ref", "-d", REF)
        c = review_commit(body(), point_ref=False)
        ok, msg = bind(c)
        print(f"{'REF missing entirely':45s} bind={'ACCEPTED' if ok else 'REJECTED'}  {msg[:150]}")
        ok, msg = bind(c[:12])
        print(f"{'abbreviated review commit':45s} bind={'ACCEPTED' if ok else 'REJECTED'}  {msg[:150]}")
