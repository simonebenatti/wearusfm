# Confronta gli enunciati delle 50 note della risposta GPT con i punti elenco del §7 della review 32e5452.
import re, sys, difflib, unicodedata
REV = sys.argv[1]; RISP = sys.argv[2]
rev = open(REV, encoding='utf-8').read().splitlines()
# §7: righe da "## 7. Note" a "## 8."
i0 = next(i for i,l in enumerate(rev) if l.startswith('## 7. Note'))
i1 = next(i for i,l in enumerate(rev) if l.startswith('## 8.'))
sec = rev[i0:i1]
bullets = []
cur = None
for l in sec:
    if l.startswith('- '):
        if cur is not None: bullets.append(cur)
        cur = l[2:].strip()
    elif cur is not None and l.startswith('  '):
        cur += ' ' + l.strip()
    else:
        if cur is not None: bullets.append(cur); cur = None
if cur is not None: bullets.append(cur)
print('punti elenco nel §7:', len(bullets))
# estrai ID fra parentesi finali
idpat = re.compile(r'\(([^()]*)\)\s*\.?\s*$')
ids_per_bullet = []
for b in bullets:
    m = idpat.search(b)
    ids = []
    if m:
        inner = m.group(1)
        for tok in re.split(r',\s*', inner):
            tok = tok.strip()
            tok = re.sub(r'^verificatore,\s*', '', tok)
            tok = re.sub(r'^vedi .*', '', tok)
            if re.match(r'^[a-z\-]+-(\d+|N\d+)$', tok): ids.append(tok)
    ids_per_bullet.append(ids)
allids = [x for ids in ids_per_bullet for x in ids]
print('ID nel §7:', len(allids), 'unici:', len(set(allids)))
print('punti con piu ID:', [(i+1, ids) for i,ids in enumerate(ids_per_bullet) if len(ids)!=1])
# righe della tabella delle note nella risposta
risp = open(RISP, encoding='utf-8').read().splitlines()
rows = [l for l in risp if re.match(r'^\| \d\d \|', l)]
print('righe note nella risposta:', len(rows))
def norm(s):
    s = unicodedata.normalize('NFC', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s
rid2en = {}
for r in rows:
    parts = [p.strip() for p in r.strip('|').split('|')]
    n, idc, en = parts[0], parts[1].strip('`'), parts[2]
    rid2en[idc] = en
missing = [x for x in allids if x not in rid2en]
extra = [x for x in rid2en if x not in allids]
print('ID della review assenti nella risposta:', missing)
print('ID della risposta assenti nella review:', extra)
# confronto testuale
diffs = 0
for b, ids in zip(bullets, ids_per_bullet):
    body = idpat.sub('', b).strip()
    for x in ids:
        en = rid2en.get(x)
        if en is None: continue
        a = norm(body).rstrip(' .'); c = norm(en).rstrip(' .')
        if a != c:
            diffs += 1
            print('\n== DIFF', x)
            print(' REV :', a)
            print(' RISP:', c)
print('\nenunciati con differenze testuali:', diffs)
