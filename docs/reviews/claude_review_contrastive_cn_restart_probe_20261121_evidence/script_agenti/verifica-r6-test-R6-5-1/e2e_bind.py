"""Exercise bind_review/require_binding on a throwaway git repo (REVIEW_REF faked locally)."""
import importlib.util, json, subprocess, sys, os
from pathlib import Path
R = Path(sys.argv[1]); C = "9d34dc690cf60fcbfe405279dc17546655fa3f5b"
spec = importlib.util.spec_from_file_location("g", R/"scripts/diagnostics/contrastive_cn_restart.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
def sh(*a): return subprocess.check_output(["git","-C",str(R),*a], text=True).strip()
_, pins = g.package()
block = "```cn-restart-review\n" + json.dumps(dict(schema="cn-restart-review-v1", status="APPROVED_FOR_DIAGNOSTIC", reviewed_commit=C, **pins)) + "\n```\n"
def review_commit(text, parent):
    sh("checkout","-q","--detach",parent)
    p = R/g.REVIEW_PATH; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    sh("add",g.REVIEW_PATH); sh("-c","user.name=t","-c","user.email=t@t","commit","-q","-m","rev")
    c = sh("rev-parse","HEAD"); sh("checkout","-q","--detach",C); return c
def case(name, fn):
    if g.BINDING.exists(): g.BINDING.unlink()
    try: fn(); print(name, "ACCEPTED")
    except Exception as e: print(name, "REJECTED", type(e).__name__, str(e)[:90])
good = review_commit("# review\n"+block, C)
sh("update-ref", g.REVIEW_REF, good)
case("valid bind+require", lambda: (g.bind_review(good), g.require_binding()))
two = review_commit("# review\n"+block+block, C)
sh("update-ref", g.REVIEW_REF, two)
case("two blocks", lambda: g.bind_review(two))
sh("update-ref", g.REVIEW_REF, good)
orphan = review_commit("# review\n"+block, C+"~1")  # not ancestor of REVIEW_REF
case("not ancestor", lambda: g.bind_review(orphan))
def tamper():
    g.bind_review(good); b = json.loads(g.BINDING.read_text()); b["review_text"] += "x"
    g.BINDING.write_text(json.dumps(b)); g.require_binding()
case("tampered text", tamper)
def forged():
    b = {"review_commit": good, "original_review_path": g.REVIEW_PATH, "review_text": "nothing"}
    import hashlib; b["review_sha256"] = hashlib.sha256(b"nothing").hexdigest()
    g.BINDING.write_text(json.dumps(b)); g.require_binding()
case("forged consistent sha, no block", forged)
if g.BINDING.exists(): g.BINDING.unlink()
