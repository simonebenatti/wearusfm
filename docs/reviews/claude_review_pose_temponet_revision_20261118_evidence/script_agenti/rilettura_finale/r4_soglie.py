"""Rilettura controllo3: B3:105-107 («la soglia si sposta [...] fino a circa 0,02 in giù o 0,01 in su, e resta fra 0,89 e 0,94»).
Legge le soglie continue e le transizioni P->F della griglia attuale da controllo2/verifica-applicazione/v_soglie.out.txt (formule
di v_scan_lineare.py) e calcola lo spostamento per coppia (K, quota). Solo aritmetica."""
import re
BASE="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2"
IN=BASE+"/controllo2/verifica-applicazione/v_soglie.out.txt"
OUT=BASE+"/controllo3/rilettura/r4_soglie.out.txt"
txt=open(IN).read().split("\n")
L=[]; rows=[]
cur=None
for line in txt:
    m=re.match(r"K=\(([\d.]+),([\d.]+)\) f'=([\d.]+)\s+soglia continua ([\d.]+)",line)
    if m:
        cur=[m.group(1)+"/"+m.group(2),float(m.group(3)),float(m.group(4)),None]; rows.append(cur); continue
    m=re.search(r"transizioni griglia attuale.*",line)
    if m and cur is not None:
        tr=re.findall(r"([\d.]+)->([PFN])",line)
        f=[float(x) for x,s in tr if s=="F"]
        cur[3]=f[0] if f else None
mx_up=0; mx_dn=0; lo=9; hi=0
for k,fq,sc,sg in rows:
    d=sg-sc; mx_up=max(mx_up,d); mx_dn=min(mx_dn,d); lo=min(lo,sg,sc); hi=max(hi,sg,sc)
    L.append(f"K {k} quota {fq}: continua {sc:.4f}, griglia {sg:.4f}, spostamento {d:+.4f}")
L.append(f"spostamento massimo in su {mx_up:+.4f}, in giu' {mx_dn:+.4f}; soglie (continue e di griglia) fra {lo:.4f} e {hi:.4f}")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
