# Cerca nella bozza corrente le frasi legate ai problemi trovati e stampa le righe (sola lettura).
import sys,re,hashlib
B=sys.argv[1]
txt=open(B,encoding='utf-8').read()
lines=txt.split('\n')
print('sha256',hashlib.sha256(txt.encode()).hexdigest(),'righe',len(lines))
pr=[i for i,l in enumerate(lines,1) if l.startswith('## 10.')]
p0=pr[0] if pr else 10**9
checks={
 'P-12 segnaposto':[r'HH:MM'],
 'P-3 sbaglia molto':[r'sbaglia molto'],
 'P-4 prompt plausibile':[r'plausibile'],
 'P-2 prompt catena si ferma':[r"89,8% del margine: la catena", r'la catena si ferma con'],
 'P-6 scatta':[r'scatta sulle teste lineari', r'percorso (probabile|plausibile)', r'qualitativ'],
 'P-7 base L':[r'come la base «L»'],
 'P-8 ricetta non c.entra':[r"c'entra"],
 'P-9 complete e fedeli':[r'[Cc]omplete e fedeli'],
 'P-10 forma chiusa':[r'si decidono in forma chiusa', r'etichetta giusta è «controllo non diagnostico»', r'non diagnostico'],
 'P-13 primo quarto':[r'primo quarto'],
 'P-15 temporale non calibrato':[r'a ≤ 1|a <= 1|a<=1|oltre 1'],
 'P-16 ricetta-9':[r'compreso se la catena prosegue'],
 'P-16b P:104':[r'P:104'],
 'P-17 quanto vuole':[r'quanto vuole', r'vanno corrette'],
 'P-18 ID':[r'nota criteri-6', r'32e5452:'],
 'P-19 prompt':[r'sganciata', r'non qualifica', r'tolleranza', r'ricetta non ammessa"|«ricetta non ammessa»', r'minore'],
 'P-20/P-22 OB-4':[r'assenza di deterioramento', r'letture negative o nulle', r'veto'],
 'P-21 OB-1 P:72':[r'In P:72'],
 'P-23 OB-3 prompt':[r'toglierla', r'non peggiore in modo pratico'],
 'P-24 S-C':[r'S-C'],
 'P-25 alpha':[r'alpha dai soli', r'alpha dagli'],
 'P-26 precedenza':[r'preceden', r'prevale'],
}
for k,pats in checks.items():
    print('==',k)
    for pat in pats:
        hits=[(i,l.strip()[:150]) for i,l in enumerate(lines,1) if re.search(pat,l)]
        for i,l in hits:
            print(f'   [{pat}] {"PROMPT " if i>p0 else ""}{i}: {l}')
        if not hits: print(f'   [{pat}] nessuna riga')
