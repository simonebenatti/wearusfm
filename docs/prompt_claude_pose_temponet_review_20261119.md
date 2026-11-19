# Rilettura mirata di ee118a25: TCN frozen v3

Ramo `codex/pose-temponet-proposal-20261117`. Diff rispetto a
`3e007c841c97da5793ed4aac6cfceb41c99a00ee`; pin nel
`results/infra/pose_temponet_proposal_20261117/delivery_pin_20261119.json`.
Leggi protocollo, proposal.json e risposta `docs/reviews/pose_temponet_response_20261119.md`.

Simone ha approvato il 19/11 in coordinamento la chiusura con te e il successivo lancio
del confronto sul 30M congelato a 49.046, se non già partito. Ledger e coda: nessun invio
di questa prova. L'approvazione non sostituisce la tua rilettura effettiva né i gate.
Il codice viene preparato separatamente, con successiva review del pacchetto operativo.
Questa consegna resta REVIEW_REQUIRED e riguarda il diff documentale.

Scelte adottate:

1. BL-1: ridge inner per E_real/E_cal, alpha 10**[-2..4] con tre decadi per lato,
   somma delle MSE per finestra e intercetta libera. Nel lineare preliminare ridge≥90%
   costante→oracolo, poi teste≥90% costante→ridge. Nel temporale WLS su y* per banda
   e margine, valutata su y; pose_var/soglia sintetiche da y*, errori da y. Griglia
   a=0,2:0,2:3 e sigma=0:0,05:6: 1.936 combinazioni, incluse nel nuovo benchmark CPU
   entro il tetto invariato. Bordo alpha irrisolto esclude la coppia; E_real irrisolto
   o nessuna coppia ferma. Manteniamo il gate outer: WLS pulita non garantisce stessa
   banda fra popolazioni. Nessun dato reale è stato usato per scegliere queste regole.
2. BL-2: MLP nel controllo temporale descrittiva, senza veto; se vicina/migliore,
   il controllo non qualifica specificamente il beneficio del contesto.
3. BL-3: bordo LR inferiore esteso è allarme solo oltre 0,5% relativo di MSE
   rispetto al vicino; bordo superiore e gli altri due allarmi restano.
4. OB-1/2: controllo che passa non fermato dal solo allarme; S-2 compromesso.
5. OB-3(a)/4: veto pratico invariato, caso misto nominato, allarmi direzionali,
   comparatori L0/MLP/L13 ammessi necessari; sola TCN con allarme non annulla beneficio.
6. OB-5, minori e note: tabella completa, residuo normale corretto, LN64 per WLS
   distinta dal forward FP32, R-1 con cast e conteggi reali, technical_failure,
   lock prima del rifit, fold nel manifest, stessi bootstrap e tutti i contrasti,
   costi/rinfresco/corsia, disposizioni una riga ciascuna.

La matrice, i cinque lettori e la MSE restano quelli concordati; nessun cambio al FM,
nuova estrazione o prova raw. Tetto invariato 24 GPUh / 80 coreh CPU-only / 272 coreh.
Operatore unico della nuova prova: chat 2-TCN head launcher; GPT scientifico prepara,
tu rivedi. Il coordinatore gestisce il contatto diretto con te.

Come concordato nella tua review, rileggi solo BL, OB e minori cambiati, senza riaprire
le parti già convergenti o creare un'altra ondata di agenti. Se corrispondono, registra
la convergenza documentale. Le note di implementazione si verificano nella review
operativa prevista. Se rimane un blocco, indica il controesempio preciso.
Non considerare questa richiesta un'autorizzazione a lanciare dal ramo raw o un secondo
operatore. Le due TEMPONet raw in corso sono una prova separata.
