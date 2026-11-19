import json, sys
root = sys.argv[1]
a = json.load(open(root + "/results/infra/tokenizer_execution_20261103/19618/T100-G/config.json"))
b = json.load(open(root + "/results/infra/contrastive_targets_20261118/review_configs/C0.config.json"))
def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    else:
        out[p] = d
    return out
fa, fb = flat(a), flat(b)
for k in sorted(set(fa) | set(fb)):
    va, vb = fa.get(k, "<MISSING>"), fb.get(k, "<MISSING>")
    if va != vb:
        s = str(vb)
        print(f"{k}: T100-G={va!r} -> C0new={s[:80]!r}")
print("amp keys:", {k: fa.get(k) for k in fa if 'amp' in k or 'bf16' in k or 'precision' in k}, {k: fb.get(k) for k in fb if 'amp' in k or 'bf16' in k or 'precision' in k})
