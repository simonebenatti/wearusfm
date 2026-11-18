# Cerca parole chiave in un file di testo e stampa riga e contesto (sola lettura).
import sys, re
path = sys.argv[1]
pats = sys.argv[2:]
lines = open(path, encoding='utf-8').read().split('\n')
for pat in pats:
    print('==', pat)
    rx = re.compile(pat, re.IGNORECASE)
    for i, line in enumerate(lines, 1):
        for m in rx.finditer(line):
            a = max(0, m.start() - 110); b = min(len(line), m.end() + 110)
            print('  %d: ...%s...' % (i, line[a:b]))
