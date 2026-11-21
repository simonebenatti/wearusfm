import json, hashlib, sys
import contrastive_cn_restart as g
good = json.loads(open(sys.argv[1]).read())
blk = good["review_text"][good["review_text"].index("```"):]
def trial(name, text, commit=good["review_commit"], fix_sha=True):
    b = dict(good, review_text=text, review_commit=commit)
    if fix_sha: b["review_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    g.BINDING.write_text(json.dumps(b))
    try: g.require_binding(); print(name, "ACCEPTED")
    except Exception as e: print(name, "rejected:", type(e).__name__, str(e)[:70])
    g.BINDING.unlink()
trial("good", good["review_text"])
trial("sha-not-fixed", good["review_text"]+"x", fix_sha=False)
trial("two-blocks", good["review_text"]+"\n"+blk)
trial("wrong-plan-pin", good["review_text"].replace("9d5811b6","9d5811b7"))
trial("forged-commit-nonexistent", good["review_text"], commit="0"*40)
trial("forged-text-only-block", blk)
