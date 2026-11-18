"""Rilettura controllo3: le 16 combinazioni di BL-2 (clausola MLP come cancello) con OB-1, nel corpo e nel prompt della bozza.
Variabili: T = la TCN supera il proprio 50%; C = la clausola passa nei numeri; aT = allarme aperto sulla TCN; aM = allarme aperto
sulla MLP. Ogni regola e' scritta come (condizione di applicabilita', etichetta), letta alla lettera:
 corpo  B3:199 «la clausola si valuta solo se la TCN supera il proprio 50%; se non lo supera, valgono gli stati per testa di OB-1»
        B3:200-202 «se la TCN supera il 50% e la clausola non passa nei numeri: «ottimizzazione non risolta» se la TCN ha un allarme
        aperto, altrimenti «controllo non diagnostico». Un allarme della sola MLP non cambia l'esito»
        B3:203-204 «se la clausola passa nei numeri ma la MLP ha un allarme aperto, non conta come superata e l'etichetta è
        «ottimizzazione non risolta»»
        OB-1 B3:277-279 «gli allarmi delle teste che passano si registrano. Se una testa non passa e ha un allarme aperto,
        l'etichetta è "ottimizzazione non risolta", altrimenti "ricetta non ammessa"»; nel temporale la MLP e' descrittiva (P:96,
        J:441), quindi lo stato per testa riguarda solo la TCN.
        Caso residuo (T, C, non aM): nessuna regola lo chiude esplicitamente; la clausola passa e l'allarme TCN di una testa che
        passa si registra (OB-1) -> «passa».
 prompt B3:631-634, stesse regole; OB-1 B3:658-660.
Controlla: (i) ogni combinazione ha una e una sola etichetta; (ii) corpo e prompt coincidono; (iii) B3:192: se la clausola non
passa nei numeri e la TCN e' >= 50%, la MLP recupera piu' del 50% - q (verifica numerica su una griglia). Nessun dato."""
import itertools
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo3/rilettura/r2_bl2_16.out.txt"
def regole_corpo():
    return [
     (lambda T,C,aT,aM: (not T) and aT,            "ottimizzazione non risolta (TCN, OB-1)"),
     (lambda T,C,aT,aM: (not T) and not aT,        "ricetta non ammessa (TCN, OB-1)"),
     (lambda T,C,aT,aM: T and (not C) and aT,      "ottimizzazione non risolta"),
     (lambda T,C,aT,aM: T and (not C) and not aT,  "controllo non diagnostico"),
     (lambda T,C,aT,aM: T and C and aM,            "ottimizzazione non risolta"),
     (lambda T,C,aT,aM: T and C and not aM,        "passa" ),
    ]
def regole_prompt():
    # B3:631 «Si valuta solo se la TCN supera il 50%; altrimenti valgono gli stati per testa di OB-1.»
    # B3:632-633 «Se non passa nei numeri: «ottimizzazione non risolta» con un allarme aperto sulla TCN, altrimenti «non
    #   diagnostico»; un allarme della sola MLP non cambia l'esito.»
    # B3:634 «Se passa nei numeri ma la MLP ha un allarme aperto: non conta come superato, «ottimizzazione non risolta».»
    return [
     (lambda T,C,aT,aM: (not T) and aT,            "ottimizzazione non risolta (TCN, OB-1)"),
     (lambda T,C,aT,aM: (not T) and not aT,        "ricetta non ammessa (TCN, OB-1)"),
     (lambda T,C,aT,aM: T and (not C) and aT,      "ottimizzazione non risolta"),
     (lambda T,C,aT,aM: T and (not C) and not aT,  "controllo non diagnostico"),
     (lambda T,C,aT,aM: T and C and aM,            "ottimizzazione non risolta"),
     (lambda T,C,aT,aM: T and C and not aM,        "passa"),
    ]
L=["T C aT aM | corpo | prompt | n.regole corpo/prompt"]
ok=True
for T,C,aT,aM in itertools.product((1,0),repeat=4):
    rc=[e for f,e in regole_corpo() if f(T,C,aT,aM)]
    rp=[e for f,e in regole_prompt() if f(T,C,aT,aM)]
    uni = len(set(rc))==1 and len(set(rp))==1
    same = rc==rp
    ok &= uni and same
    extra = "" if not (T and C and not aM and aT) else " (allarme TCN registrato)"
    L.append(f"{T} {C} {aT} {aM} | {rc[0] if rc else 'NESSUNA'}{extra} | {rp[0] if rp else 'NESSUNA'} | {len(rc)}/{len(rp)}")
L.append(f"etichetta unica e corpo = prompt in tutte le 16: {ok}")
# (iii) identita' di B3:192 con denominatori uguali (tabella B3:162-163: E_LS* - E_oracolo per il 50% e per la clausola)
viol=0; n=0
for xt in [0.50+0.01*i for i in range(51)]:
    for q in [0.02,0.05,0.10,0.20]:
        for c in [q-0.001-0.01*j for j in range(30)]:
            xm = xt - c                       # frazione recuperata dalla MLP = X_TCN - clausola
            n+=1
            if not (xm > 0.5 - q): viol+=1
L.append(f"B3:192 (MLP > 50% - q quando la clausola non passa e la TCN e' >= 50%): violazioni {viol} su {n}")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
