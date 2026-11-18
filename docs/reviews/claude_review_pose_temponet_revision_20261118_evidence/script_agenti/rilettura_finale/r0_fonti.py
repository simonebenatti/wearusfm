"""Rilettura controllo3: estrae le righe di P, J, DEC e R citate nei passi cambiati e controlla le frasi che la bozza attribuisce.
Solo lettura della copia gpt_3e007c8. Scrive solo in questa cartella."""
import hashlib, re
BASE="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2"
G=BASE+"/gpt_3e007c8"
F={"P":G+"/docs/pose_temponet_frozen_protocol_20261117.md",
   "J":G+"/results/infra/pose_temponet_proposal_20261117/proposal.json",
   "DEC":G+"/docs/decisioni.md",
   "R":G+"/docs/reviews/pose_temponet_response_20261118.md"}
OUT=BASE+"/controllo3/rilettura/r0_fonti.out.txt"
B=BASE+"/controllo3/bozza_controllo3.md"
T={k:open(v,encoding="utf-8").read().split("\n") for k,v in F.items()}
L=[]
L.append("sha256 bozza: "+hashlib.sha256(open(B,"rb").read()).hexdigest())
# (sigla, riga, frase che deve comparire nella riga, che cosa dice la bozza)
checks=[
 ("P",69,"LR ancora al bordo dopo l'estensione di quel lato","BL-3: allarme del bordo; B3:255 altre condizioni"),
 ("P",69,"minimo MSE o E_dyn ancora nelle ultime5 epoche","OB-1 B3:281-282 caso tipico: minimo di E_dyn nelle ultime 5 epoche"),
 ("P",69,"checkpoint selezionato al primo quarto di epoca","BL-3 B3:255-256"),
 ("P",72,"non automaticamente non convergenza","B3:248"),
 ("P",80,"12 blocchi di soluzioni","B3:472 (P-03)"),
 ("P",80,"RHS-base reale,L,T,rumore","B3:146, B3:615 RHS-base T"),
 ("P",86,"usando solo il fitting outer","P-03 / B3:472"),
 ("P",92,"nessun nuovo seed/griglia automatica","B3:145 'P:92 vieta un nuovo seme'"),
 ("P",92,"poi daE_real","B3:120 spareggio del temporale basato su E_real"),
 ("P",92,"scegliere errore più vicino aE_real","B3:120 scelta di sigma nel lineare basata su E_real"),
 ("P",92,"abs(E_LS,synth−E_real)<=0.05","ancora del temporale (T-04)"),
 ("P",95,"Margini non positivi/non finiti → controllo non diagnostico","B3:84"),
 ("P",96,"margine relativo oracolo in[6%,10%]","B3:82 banda 6-10%"),
 ("P",96,"fuori intervallo → controllo non diagnostico","B3:170 non diagnostico della banda outer"),
 ("P",98,"causa non attribuita automaticamente a ottimizzazione","B3:83"),
 ("P",104,"Nessuna vittoria costruita su un comparatore inadeguatamente allenato","B3:204-205 (citazione)"),
 ("P",104,"Se L0 o TCN-post non è ammissibile, niente promozione primaria","B3:336"),
 ("P",122,"Se MLP o L13 sono risolti migliori, la TCN non è candidata preferita","OB-3"),
 ("J",40,"L0/MLP/L13/TCNpost admitted","OB-3/OB-4 J candidate"),
 ("J",41,"missing_or_unresolved_comparator","OB-4 J:40-41"),
 ("J",104,"practical_readout_recommendation","T-09 J:104"),
 ("J",104,"L0/MLP/L13/TCNpost admitted","T-09 J:104 chiede TCN-post ammessa"),
 ("J",142,"selectedLR still at expanded boundary","BL-3 J:142"),
 ("J",471,"same fit policy as real","BL-3 J:471"),
 ("DEC",5319,"alpha 10**[-2..4], estensioni fino a 3 decadi per lato, parita' alpha minore","T-05 B3:116-117"),
 ("DEC",5320,"bordo irrisolto dichiarato","T-05 B3:117"),
 ("DEC",6747,"Bordo finale irrisolto","T-09(ii) BL-3 DEC:6747"),
 ("DEC",6747,"Allarme S-2","OB-2 DEC:6747"),
 ("DEC",6753,"Candidata iff","T-09 OB-3/OB-4 DEC:6753"),
 ("DEC",6753,"TCN-post ammessi","OB-4 DEC:6753 chiede TCN-post"),
 ("DEC",6753,"S-2 limita attribuzione","T-09(iii) DEC:6753"),
 ("R",20,"Un L13 peggiore di L0","consegna-4: C-2 (R:20, L13 peggiore di L0)"),
]
for sig,n,frase,uso in checks:
    riga=T[sig][n-1]
    ok = frase in riga
    L.append(f"{sig}:{n} {'OK ' if ok else 'NO '} «{frase}» -- {uso}")
    if not ok:
        L.append("   riga: "+riga[:300])
open(OUT,"w",encoding="utf-8").write("\n".join(L)+"\n"); print("\n".join(L))
