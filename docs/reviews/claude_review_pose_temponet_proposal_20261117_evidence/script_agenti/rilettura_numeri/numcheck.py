import re, sys, glob, os
draft = open(sys.argv[1]).read().splitlines()
srcs = sys.argv[2:]
corpus = ""
for p in srcs:
    for f in glob.glob(p, recursive=True):
        if os.path.isfile(f):
            try:
                corpus += open(f, errors='ignore').read() + "\n"
            except Exception:
                pass
def variants(tok):
    v = {tok}
    v.add(tok.replace('.', ''))           # 8.468 -> 8468
    v.add(tok.replace(',', '.'))          # 0,786 -> 0.786
    v.add(tok.replace('.', '').replace(',', '.'))
    return v
num_re = re.compile(r'(?<![\w/])[-+−]?\d[\d.,]*\d%?|(?<![\w/])\d%?')
seen = {}
for i, line in enumerate(draft, 1):
    for m in num_re.finditer(line):
        tok = m.group(0).rstrip('%').lstrip('+-−')
        tok = tok.rstrip('.,')
        if len(tok.replace('.', '').replace(',', '')) < 2:
            continue
        found = any(v in corpus for v in variants(tok))
        seen.setdefault(tok, []).append((i, found))
missing = {t: occ for t, occ in seen.items() if not any(f for _, f in occ)}
print("token distinti:", len(seen), "non trovati:", len(missing))
for t, occ in sorted(missing.items(), key=lambda x: x[1][0][0]):
    print(f"{t!r} righe {[i for i,_ in occ]}")
