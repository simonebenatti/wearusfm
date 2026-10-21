# Pilot dei rami: review GPT del 21/10/2026

**Esito: da rivedere.** Due difetti riprodotti e un controllo del protocollo ancora
da implementare. Target: snapshot immutabile **134b77c**; i file di modello,
mascheramento, trainer e protocollo sono invariati anche sul ramo Claude a
**58380a3**. Questa review non comprende le nuove modifiche ai planner dei bracci
JEPA-only/ancore-only, ne' integra il pilot su main.

## R1 — P2: metrica normalizzata indefinita interrompe il trainer nel logging

`src/wearusfm/training/run.py:1354-1355`, insieme a `1298-1299`.
L'obiettivo restituisce NaN quando la varianza del riferimento costante e' nulla
(`branch_masking.py:269/282`). `_json_value` lo converte correttamente in None,
ma il messaggio al passo 1 e ogni 50 applica comunque `:.3f`: TypeError, prima
della chiusura ordinata e del summary. Riprodotto usando l'obiettivo reale e
iniettando soltanto questa metrica indefinita; loss e gradienti restano finiti.

Richiesta: rappresentare esplicitamente la misura non definita nel log e nel
verdetto, oppure arrestare ordinatamente il run con una motivazione; non lasciarla
diventare un'eccezione di formattazione. Non assumere che null significhi PASS.

## R2 — P2: i controlli mancanti non interrompono il conteggio dei consecutivi

`src/wearusfm/model/branch_masking.py:186,196-205`.
Sequenza riprodotta: riferimento positivo nei passi 1-10; valori bassi a 110 e
120; nessun target/contesto per quella scala a 130; valore basso a 140.
Il codice dichiara tre controlli consecutivi e ferma il run a 140. Il protocollo
(righe 70-72) specifica controlli ogni 10 passi e tre consecutivi, non tre
osservazioni basse separate da controlli senza misura. Il caso e' pertinente
alle scale lente, per cui l'assenza di contesto e' prevista e conteggiata.

Richiesta: definire e implementare prima del collaudo la gestione delle lacune:
interrompere la sequenza, oppure distinguere uno stop per copertura insufficiente.
Se si vogliono invece tre sole osservazioni valide consecutive, esplicitarlo nel
protocollo: oggi il tempo massimo fra le osservazioni non e' limitato. Chiarire
anche il comportamento con riferimento iniziale assente/nullo, attualmente
ignorato da `update`; questo ultimo caso non e' una quarta regressione eseguita.

## R3 — P2 / completamento: manca il monitor dei rami per PWf

`src/wearusfm/training/run.py:1161-1166,1298-1300`.
Il monitor e le sue metriche si attivano solo con `branch_objective="masked"`.
PWf usa `readout` con `branch_freeze_output=True`: il run di prova termina con
`collapse_reference=None`, senza varianza/diversita'/rango dei token dei rami.
Il monitor generale guarda il raw congelato (`diagnostics`, righe 386-405),
quindi non e' un sostituto. Il primario richiede invece che **entrambi** PWf e
PWm non siano collassati (protocollo, righe 106-109).

Richiesta: misure e regola confrontabili anche per PWf, senza cambiarne la loss,
oppure una revisione esplicita del criterio prima del lancio. L'assenza di STOP
su PWf oggi non dimostra che quel controllo sia stato superato.

## Parti convincenti e limiti di prontezza

- Test confermati: indipendenza bit per bit dello studente dai campioni nascosti,
  refactor tokens/forward invariato, target stop-gradient, strati fissi esclusi
  dall'ottimizzatore/EMA, identita' del primo loss PWf con l'inizializzazione 2C.
- Il confronto PWf/PWm separa meglio l'obiettivo dal congelamento degli strati
  finali; resta un pilot, non una dimostrazione della causa del risultato 2C.
- `scripts/branch_pilot.py`, citato nel protocollo, ancora assente anche a 58380a3.
  Da completare wrapper/launcher, caricamento con verifica dell'identita' dello
  step 0, guardia della prima loss PWl/PWf, controlli finali d'integrita', ibrido,
  estrazione/confronto e verdetto del collaudo. Il test di uguaglianza esistente
  non sostituisce la guardia eseguita dal job. Sono lavori dichiarati incompleti,
  non regressioni di campagne gia' svolte.
- Protocollo ancora PROPOSTA: congelamento dei criteri, preventivo e via al
  pilot restano necessari. Il collaudo JEPA-only/ancore-only e C3 sono campagne
  diverse; questa review non le ferma e non autorizza nuovi job.

## Verifiche

- **47 test esistenti PASS**: branch_masking, branch_pilot_trainer e
  spectral_branches. Uno dei 47 (DDP CPU) inizialmente bloccato dal sandbox su
  torch_shm_manager, poi rieseguito con permesso e PASS. Nessuna prova GPU.
- **3 regressioni nuove FAIL attesi**, una per ciascun punto sopra, in
  `test_branch_pilot_review_20261021.py`. Sono riproduzioni dei rilievi, non tre
  nuovi fallimenti della suite di Claude. Impostare WEARUSFM_REVIEW_SOURCE al
  checkout del pilot, PYTHONPATH al suo src, OMP/MKL/OPENBLAS_NUM_THREADS=1;
  usare il venv wearusfm-tok e pytest con `-p no:cacheprovider`.
- Le due letture delegate si sono interrotte per limite del servizio; non sono
  conteggiate come review indipendenti completate. Riscontri e regressioni sopra
  controllati direttamente da GPT. Nessuna modifica al codice di Claude.

Consegna: Claude corregge sul proprio ramo con test; GPT verifica i fix. Nessun
cambiamento retroattivo ai risultati 2A/2C, C1/C2 o alla decisione sui default.
