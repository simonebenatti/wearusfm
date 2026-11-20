"""Rilettura controllo3: OB-3 (opzioni a/b) con OB-4 (conseguenza direzionale) e la frase nuova di T-08, sulle stesse 30
combinazioni di controllo2/tenuta/ob3_ob4_combinazioni.py. Regole lette alla lettera nella bozza congelata:
 OB-3(a) B3:308-310: veto solo con `deterioramento_pratico` nel contrasto 2 o 3; un comparatore risolto migliore (upper99<0) ma con
   stima sotto il 2% -> nome misto «candidata, con MLP/L13 risolti migliori (upper99 < 0) ma con stima sotto il 2%».
 OB-3(b) B3:314: nessuna candidatura se il contrasto 2 o 3 ha upper99 < 0.
 OB-4 B3:327-334: allarme sulla TCN -> sospese solo le conclusioni negative o nulle; la condizione di veto esclude comunque la
   candidatura; allarme su L0/MLP/L13 -> bloccate le letture favorevoli alla TCN contro quella testa, compresa l'assenza del veto;
   la candidata richiede L0, MLP e L13 ammessi; NUOVO (T-08) «con un allarme sulla TCN il nome misto di OB-3(a) resta [...]; si
   sospende solo la sua lettura come conclusione sul valore della TCN».
Per ogni combinazione conta le regole applicabili che danno un nome alla candidatura: deve essere uno solo. Nessun dato."""
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo3/rilettura/r3_ob3_ob4.out.txt"
def lab(g,lo,up):
    if g>=2 and lo>0: return 1
    if g<=-2 and up<0: return 2
    if up<2: return 3
    return 4
casi=[("c2 -1,5 [-1,9;-0,8]",-1.5,-1.9,-0.8),("c2 -3,0 [-4,5;-1,5]",-3.0,-4.5,-1.5),("c2 +0,5 [-0,5;+1,5]",0.5,-0.5,1.5),
      ("c2 +1,0 [-0,5;+2,5]",1.0,-0.5,2.5),("c2 +3,0 [+1,0;+5,0]",3.0,1.0,5.0)]
L=[]; amb=0
for opt in ("a","b"):
  for allarme in ("nessuno","TCN","MLP"):
    for nome,g,lo,up in casi:
        l=lab(g,lo,up)
        veto = (l==2) if opt=="a" else (up<0)
        misto = (opt=="a" and not veto and up<0)
        nomi=set()
        if allarme=="MLP":
            nomi.add("nessuna candidatura (MLP non ammessa)")
        else:
            if veto:
                nomi.add("nessuna candidatura (veto)")
            elif misto:
                # OB-3(a) nome misto; con allarme TCN la frase nuova di OB-4 dice che il nome resta
                nomi.add("candidata, nome misto" + ("; lettura come conclusione sospesa" if allarme=="TCN" else ""))
            else:
                nomi.add("candidata")
        if len(nomi)!=1: amb+=1
        L.append(f"OB-3({opt}) allarme {allarme:7s} | {nome} etichetta {l} | {' / '.join(sorted(nomi))}")
L.append(f"combinazioni con piu' di un nome: {amb} su {len(L)}")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
