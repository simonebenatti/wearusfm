"""BL-2 (clausola MLP come cancello) composta con OB-1, controllo temporale: tutte le 16 combinazioni.
Variabili: T = la TCN supera il proprio 50%; aT = allarme aperto sulla TCN; C = clausola (E_MLP-E_TCN)/(E_LS*-E_or) >= q nei numeri;
aM = allarme aperto sulla MLP. Regole lette nella bozza congelata:
 - bozza:173 «la clausola si valuta solo se la TCN supera il proprio 50%; se non lo supera, valgono gli stati per testa di OB-1»;
 - bozza:174-175 «se la TCN supera il 50% e la clausola non passa, l'etichetta è «ottimizzazione non risolta» quando la TCN o la MLP
   hanno un allarme aperto, altrimenti «controllo non diagnostico»»;
 - bozza:176 «con un allarme aperto sulla MLP la clausola non conta come superata»;
 - bozza:246-248 (OB-1) stati per testa: allarme -> «ottimizzazione non risolta», altrimenti «ricetta non ammessa»; gli allarmi
   delle teste che passano si registrano.
Etichetta «direzionale» (proposta di questo controllo): una clausola fallita nei numeri e' una prova di esistenza (una MLP, anche
poco allenata, legge gia' quasi quanto la TCN): l'allarme della sola MLP non la cambia; conta solo l'allarme della TCN, che
potrebbe allargare il distacco. L'allarme della MLP conta solo quando la clausola passerebbe (bozza:176-177, P:104).
Nessun dato: solo logica delle regole."""
import itertools
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/tenuta/bl2_ob1_combinazioni.out.txt"
def bozza(T,aT,C,aM):
    if not T:
        return "ottimizzazione non risolta" if aT else "ricetta non ammessa"
    clausola = C and not aM
    if clausola:
        return "passa" + (" (allarme TCN registrato)" if aT else "")
    return "ottimizzazione non risolta" if (aT or aM) else "non diagnostico"
def direzionale(T,aT,C,aM):
    if not T:
        return "ottimizzazione non risolta" if aT else "ricetta non ammessa"
    if C:
        if aM: return "ottimizzazione non risolta"
        return "passa" + (" (allarme TCN registrato)" if aT else "")
    return "ottimizzazione non risolta" if aT else "non diagnostico"
L=["T aT C aM | bozza | direzionale | uguali?"]
diff=0
for T,aT,C,aM in itertools.product((True,False),repeat=4):
    b=bozza(T,aT,C,aM); d=direzionale(T,aT,C,aM)
    same = (b==d); diff += (not same)
    L.append(f"{int(T)} {int(aT)} {int(C)} {int(aM)} | {b} | {d} | {'si' if same else 'NO'}")
L.append(f"combinazioni con etichetta diversa: {diff} su 16")
L.append("Ogni combinazione ha un'etichetta unica in entrambe le regole; la sola differenza e' T=1, aT=0, C=0, aM=1.")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
