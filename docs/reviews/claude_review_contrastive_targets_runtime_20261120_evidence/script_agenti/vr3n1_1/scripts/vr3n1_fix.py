"""In-memory checks: key order, list items, and the proposed anchored regex + duplicate-key hook (no file changes)."""
import json, re, runpy, sys
from pathlib import Path
script = Path(__file__).with_name("vr3n1_bind.py")
clone = sys.argv[1]
sys.argv = [str(script), clone, "__none__"]
g = runpy.run_path(str(script), run_name="vr3n1")
C, cases, F, B, REAL, HEAD = g["C"], g["cases"], g["F"], g["B"], g["REAL"], g["HEAD"]
payload = json.loads(B)

def res(fn, text):
    try:
        p = fn(text)
        return "ACCEPT status=" + p["status"]
    except Exception as e:  # noqa: BLE001
        return f"REJECT {type(e).__name__}: {str(e)[:70]}"

extra = {
    "X1_key_order_reversed": HEAD + F + "\n" + json.dumps(dict(reversed(list(payload.items()))), indent=1) + "\n```\n",
    "X2_one_line_json": HEAD + F + "\n" + json.dumps(payload) + "\n```\n",
    "X3_list_item_example_plus_real": HEAD + "- esempio:\n  " + F + "\n" + "\n".join("  " + l for l in B.split("\n")) + "\n  ```\n\n" + REAL,
    "X4_unfav_list_item_close_col0": "Verdetto: NON APPROVATO.\n\n- forma futura: " + F + "\n" + B + "\n```\n",
    "X5_block_at_start_no_final_nl": REAL.rstrip("\n"),
    "X6_nbsp_after_name": HEAD + F + " \n" + B + "\n```\n",
}
orig = C.REVIEW_BLOCK
print("== parser attuale (contrastive_release.py:26) ==")
for k, v in extra.items():
    print(k, "->", res(C.review_payload, v))

FIX = re.compile(r"^```contrastive-release-review[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)
def no_dups(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys")
    return dict(pairs)
real_loads = json.loads
def fixed_payload(text):
    C.REVIEW_BLOCK = FIX
    C.json.loads = lambda s, **kw: real_loads(s, object_pairs_hook=no_dups)
    try:
        return C.review_payload(text)
    finally:
        C.REVIEW_BLOCK = orig
        C.json.loads = real_loads
print("== correzione proposta (^...$ MULTILINE + object_pairs_hook) ==")
for k, (text, _m, _e) in cases.items():
    t = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
    print(k, "now:", res(C.review_payload, t).split(":")[0][:30], "| fix:", res(fixed_payload, t)[:80])
for k, v in extra.items():
    print(k, "now:", res(C.review_payload, v).split(":")[0][:30], "| fix:", res(fixed_payload, v)[:80])
