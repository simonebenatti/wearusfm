"""OB-3 (opzioni a/b del veto) con OB-4 (conseguenza direzionale), regole lette nella bozza congelata:
 OB-3 (a), bozza:276-278: veto solo con `deterioramento_pratico` nel contrasto 2 o 3; un comparatore risolto migliore (upper99<0)
   ma con stima sotto il 2% -> «candidata, con MLP/L13 risolti migliori (upper99 < 0) ma con stima sotto il 2%», con gain e IC99.
 OB-3 (b), bozza:282: nessuna candidatura se il contrasto 2 o 3 ha upper99 < 0.
 OB-4, bozza:295-300: allarme sulla TCN -> si sospendono solo le conclusioni negative o nulle sul valore della TCN; la condizione di
   veto scelta in OB-3 esclude comunque la candidatura. Allarme su L0, MLP o L13 -> bloccate le letture favorevoli alla TCN contro
   quella testa, compresa l'assenza della condizione di veto; la candidata richiede L0, MLP e L13 ammessi.
Etichette del §5 (P:116-120). Contrasti 1, 3, 4, 5 fissati a `beneficio_pratico_risolto`. Nessun dato."""
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/tenuta/ob3_ob4_combinazioni.out.txt"
def lab(g,lo,up):
    if g>=2 and lo>0: return 1
    if g<=-2 and up<0: return 2
    if up<2: return 3
    return 4
casi=[("c2 -1,5 [-1,9;-0,8]",-1.5,-1.9,-0.8),("c2 -3,0 [-4,5;-1,5]",-3.0,-4.5,-1.5),("c2 +0,5 [-0,5;+1,5]",0.5,-0.5,1.5),
      ("c2 +1,0 [-0,5;+2,5]",1.0,-0.5,2.5),("c2 +3,0 [+1,0;+5,0]",3.0,1.0,5.0)]
L=[]
for opt in ("a","b"):
  for allarme in ("nessuno","TCN","MLP"):
    for nome,g,lo,up in casi:
        l=lab(g,lo,up)
        veto = (l==2) if opt=="a" else (up<0)
        misto = (opt=="a" and not veto and up<0)
        if allarme=="MLP":
            esito="nessuna candidatura (MLP non ammessa; bloccata anche l'assenza del veto)"
        elif veto:
            esito="nessuna candidatura (veto)"+("; etichetta del contrasto 2 sospesa come conclusione" if allarme=="TCN" and l in (2,3,4) else "")
        elif misto and allarme=="TCN":
            esito="AMBIGUO: candidata con il nome misto di OB-3(a) (qualificazione 'MLP risolti migliori'), oppure candidata senza qualificazione perche' l'etichetta 3 del contrasto 2 e' una conclusione negativa o nulla sospesa da OB-4"
        elif misto:
            esito="candidata, con MLP/L13 risolti migliori (upper99<0) ma con stima sotto il 2%"
        else:
            esito="candidata"+("; etichetta %d del contrasto 2 sospesa"%l if allarme=="TCN" and l in (3,4) else "")
        L.append(f"OB-3({opt}) allarme {allarme:7s} | {nome} etichetta {l} | {esito}")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
