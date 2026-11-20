"""Rilettura controllo3: nel prompt (sezione 10, righe 565-717) elenca gli ID dei rapporti (architettura-N, ricetta-N, controlli-N,
criteri-N, costo-N, consegna-N, V-..., S-X) che stanno fuori da parentesi tonde, per la convenzione di B3:569-571 (P-04)."""
import re
BASE="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2"
B=BASE+"/controllo3/bozza_controllo3.md"
OUT=BASE+"/controllo3/rilettura/r6_id_prompt.out.txt"
lines=open(B,encoding="utf-8").read().split("\n")
pat=re.compile(r"(?<![\w:`-])((?:V-)?(?:architettura|ricetta|controlli|criteri|costo|consegna)(?:-controlli|-criteri|-consegna)?-\d+)")
L=[]
for n in range(565,718):
    s=lines[n-1]
    # profondita' di parentesi a ogni carattere (sulla riga; le parentesi del prompt non vanno a capo nei punti ID)
    depth=0; d=[]
    for ch in s:
        if ch=="(": depth+=1
        d.append(depth)
        if ch==")": depth=max(0,depth-1)
    for m in pat.finditer(s):
        if d[m.start()]==0:
            L.append(f"B3:{n} fuori parentesi: {m.group(1)} | {s.strip()[:120]}")
open(OUT,"w",encoding="utf-8").write("\n".join(L)+"\n"); print("\n".join(L))
