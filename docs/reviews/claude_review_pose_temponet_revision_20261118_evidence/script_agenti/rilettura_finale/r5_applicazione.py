"""Rilettura controllo3: per ciascuno dei 16 rilievi di controllo2/sintesi_controllo2.txt cerca nella bozza congelata le frasi
chiave della correzione (CORRECTED_FIX per i PARZIALE, FIX per i CONFERMATI e per VA2-N1) e, dove serve, l'assenza della frase
vecchia. Stampa le righe trovate. Il giudizio finale (APPLICATO, IN PARTE, MALE) e' nel rapporto, non qui. Solo lettura."""
import hashlib
BASE="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2"
B=BASE+"/controllo3/bozza_controllo3.md"
OUT=BASE+"/controllo3/rilettura/r5_applicazione.out.txt"
raw=open(B,"rb").read()
lines=raw.decode("utf-8").split("\n")
flat=" ".join(l.strip() for l in lines)
def where(s):
    # riga di inizio di una frase che puo' andare a capo: cerca sulla prima parola-chiave lunga
    out=[]
    for i,l in enumerate(lines,1):
        if s in l: out.append(i)
    return out
def has(s): return s in flat
chk={
 "P-01":[("+","la MLP recupera più del 50% − q del margine"),("+","si legge almeno in parte in modo istantaneo"),("+","L'etichetta segue le regole sotto"),("-","Se non passa, il margine piantato si legge in modo istantaneo")],
 "P-02":[("+","fino a circa 0,02 in giù o 0,01 in su"),("+","resta fra 0,89 e 0,94"),("+","più in basso la griglia ha anche buchi"),("-","Sotto quella soglia l'esito dipende")],
 "P-03":[("+","misurare sul fitting outer la frazione di unità attive"),("+","centrare z per dimensione sul fitting outer prima di T (il centro entra nel primo bias)")],
 "P-04":[("+","T quasi lineare, da misurare prima dei fit di controllo (controlli-7), e P:122 come destinazione di C-2 (consegna-4)"),("-","- controlli-7 e consegna-4;")],
 "P-05":[("-","tolleranza dichiarata ora, cioè"),("+","per esempio lo 0,5% della MSE pooled del LR vicino; sotto quella quota")],
 "VA2-N1":[("+","se la clausola passa nei numeri ma la MLP ha un allarme aperto, non conta come superata e l'etichetta è «ottimizzazione non risolta»"),("+","Se passa nei numeri ma la MLP ha un allarme aperto: non conta come superato, «ottimizzazione non risolta»")],
 "T-01":[("+","se la ridge non abbassa affatto E_real cresce di poco"),("+","(a) aiuta nella misura in cui l'errore di stima pesa su E_real; se E_real nasce soprattutto dallo spostamento fra utenti la stretta resta, e la chiude solo l'alternativa")],
 "T-02":[("+","da 0,27 a 0,37, con a = 1 ed E_real 0,85"),("+","va da 1 caso in meno a 12 in più, su 81"),("+","nel modello a = 3 chiede sigma fino a circa 6"),("+","P:92 vieta un nuovo seme"),("+","rende più probabile «non calibrato»; non è equivalente la banda su y con E_LS* solo nel margine"),("?","In cambio toglie il «non diagnostico»"),("?","toglie il «non diagnostico» dello spostamento")],
 "T-03":[("+","verifica almeno che il margine piantato esista"),("+","il temporale verifica almeno che il margine esista (P:96)"),("+","«ricetta non ammessa» diventa più probabile"),("-","il temporale sì (P:96)")],
 "T-04":[("+","Un'alternativa a (b) chiude il caso"),("+","altrimenti valgono gli stati per testa di OB-1"),("+","La (a) resta per l'ancora"),("+","Se GPT sostituisce anche (a) con l'intervallo di E_oracolo"),("+","se sostituisci anche (a), la (c) deve essere la banda sganciata dall'ancora"),("-","In alternativa ad (a) e (b)")],
 "T-05":[("+","La griglia ha bisogno di una regola di bordo"),("+","con la conseguenza scritta se il minimo resta al bordo"),("+","L'intercetta non si penalizza"),("+","regola di bordo (per esempio quella storica di DEC:5319)"),("+","intercetta non penalizzata"),("?","conseguenza scritta se il minimo resta al bordo"),("+","come si sceglie sigma nel lineare e come si spareggia nel temporale")],
 "T-06":[("+","se la TCN supera il 50% e la clausola non passa nei numeri: «ottimizzazione non risolta» se la TCN ha un allarme aperto"),("+","Un allarme della sola MLP non cambia l'esito"),("+","Se non passa nei numeri: «ottimizzazione non risolta» con un allarme aperto sulla TCN"),("-","con un allarme aperto su TCN o MLP")],
 "T-07":[("+","Sul lato inferiore il bordo è un allarme solo se il LR più piccolo vince di più di una quota relativa fissata ora"),("+","Vale per tutte le teste, nei fit reali e nei controlli"),("-","per L0 e L13 ovunque e per tutte le teste nel controllo lineare")],
 "T-08":[("+","con un allarme sulla TCN il nome misto di OB-3(a) resta, perché descrive i predittori selezionati"),("+","con un allarme sulla TCN il nome misto di OB-3(a) resta; si sospende solo"),("+","La scelta opposta va bene, purché dichiarata")],
 "T-09":[("+","J `candidate` (J:40), J:104, DEC:6753 e §10"),("+","in J:104, in DEC:6753 e nel §10"),("+","J `candidate` (J:40-41), J:104 e DEC:6753"),("+","Correzione proposta (P:69, J:142, J:471, DEC:6747)"),("+","aggiorna anche DEC:6747"),("+","J candidate (J:40), J:104, DEC:6753 e §10"),("+","J candidate, J:104, DEC:6753 e §10"),("+","J candidate, J:104 e DEC:6753")],
 "T-10":[("+","(a) aiuta nella misura in cui"),("+","E_LS*, la LS su y* (stimatore da dichiarare: WLS o ridge) valutata su y"),("+","controlli-7 propone di misurarla in forma chiusa prima dei fit di controllo"),("+","se un allarme di P:69 e S-2 riguardano la stessa testa, vale l'allarme"),("+","Il caso tipico è un minimo di E_dyn nelle ultime 5 epoche dell'orizzonte massimo"),("+","prevale su S-2 sulla stessa testa")],
}
L=["sha256 bozza: "+hashlib.sha256(raw).hexdigest(), "(+ deve esserci, - non deve esserci, ? da controllare a mano; ricerca sul testo con gli a capo uniti)"]
for k,v in chk.items():
    L.append("== "+k)
    for seg,s in v:
        h=has(s)
        tag = ("OK " if h else "MANCA") if seg=="+" else (("OK " if not h else "RESTA") if seg=="-" else ("presente" if h else "assente"))
        # riga: cerca il primo pezzo di 30 caratteri
        pr=where(s[:30])
        L.append(f"  [{seg}] {tag:8s} righe~{pr[:4]} «{s[:110]}»")
open(OUT,"w",encoding="utf-8").write("\n".join(L)+"\n"); print("\n".join(L))
