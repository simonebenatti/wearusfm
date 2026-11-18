# Confronta i 13 enunciati dei rilievi (risposta §1) con i titoli ### della review 32e5452 (§4-§6).
import re, sys, unicodedata
rev = open(sys.argv[1], encoding='utf-8').read().splitlines()
risp = open(sys.argv[2], encoding='utf-8').read().splitlines()
pat = re.compile(r'^### ([A-Z]-\d)\. (.*) \(([^;]+); (.*)\)\s*$')
heads = {}
for l in rev:
    m = pat.match(l)
    if m: heads[m.group(1)] = (m.group(2), m.group(3).strip(), m.group(4).strip())
print('rilievi con titolo nella review:', len(heads), sorted(heads))
rows = [l for l in risp if re.match(r'^\| [A-Z]-\d — ', l)]
print('righe rilievi nella risposta:', len(rows))
def norm(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFC',s)).strip().rstrip('.')
for r in rows:
    parts = [p.strip() for p in r.strip('|').split('|')]
    m = re.match(r'^([A-Z]-\d) — `([^`]+)`, (\w+)$', parts[0])
    sid, oid, grav = m.group(1), m.group(2), m.group(3)
    t, g, ids = heads[sid]
    ids_clean = ids.replace('verificatore, ', '')
    ok_t = norm(t) == norm(parts[1]); ok_g = (g == grav); ok_i = (ids_clean == oid)
    print(sid, 'titolo', ok_t, '| gravita', ok_g, g, grav, '| id', ok_i, ids_clean, oid)
    if not ok_t: print('   REV :', t); print('   RISP:', parts[1])
