# Punto 3: seconda tranche Claude, review del 20-21/10/2026

Richiesta: verificare e integrare per merge fino a `c21ca40`, cominciando dal gruppo 1
(`62bec0b`). Base canonica: `79db894`. Nessun merge eseguito: tre rilievi riprodotti
da correggere sul ramo Claude prima dell'integrazione. Nessun job, deploy o push.
I gruppi sono intercalati nella storia: un merge di `62bec0b` porta anche gli altri
strumenti suoi antenati, non soltanto i file della selezione.

## Rilievi

### R1 — P1: senza sonda, un run ancora incompleto puo' essere escluso come fallito

`scripts/calibration_select.py:400-410`, con il controllo di completamento a
`214-217`; `scripts/calibration_plan.py:154-156`.

Il nuovo percorso `report=None` (`--failed-runs` nel planner) non richiede una prova
del fallimento terminale. Basta l'assenza di `summary.json` per classificare il
candidato come inammissibile e proseguire con gli altri. La stessa situazione e'
compatibile con un training ancora in corso, una catena da riprendere o file non
ancora copiati. I run dichiarati `failed` non hanno guardia/dipendenza nel DAG:
la rilettura di sacct dell'orchestratore non li verifica.

Riproduzione: configurazione coerente, nessuno STOP, summary assente, LR 1e-4 senza
sonda; gli altri due candidati validi hanno LR 3e-4 e 1e-3. La selezione C1 emette
finalisti `[0.0003, 0.001]` anziche' fermarsi per stato non accertato.

Correzione richiesta: distinguere fallimento terminale verificato da running,
timeout riprendibile e stato ignoto. Solo il primo puo' entrare come esclusione
senza sonda; negli altri casi non emettere la decisione. Registrare l'evidenza
e la sua identita' nel risultato (non e' necessario aggiungere SSH al selettore).
Il test esistente che accetta una cartella del run del tutto assente va rivisto.

### R2 — P2: PASSI del candidato non e' vincolato al piano congelato

`scripts/calibration_select.py:407`.

La configurazione completa viene controllata contro il piano, ma `admissibility`
riceve separatamente il valore PASSI della CLI. Un run sano con config e summary
a 2000 passi, senza sonda, viene escluso se PASSI vale 2001; la selezione sceglie
gli altri due candidati. Con PASSI corretto viene invece giustamente segnalata
la valutazione mancante.

Correzione richiesta: ricavare i passi dal run del piano gia' verificato, oppure
rifiutare la discordanza prima di calcolare l'ammissibilita'. Non trasformare un
errore negli argomenti in un fallimento sperimentale.

### R3 — P1: estrazione P1 dei bracci oltre il limite della QoS ereditata

`scripts/arms_plan.py:92-95`; `scripts/slurm/probe_p1.sbatch:4`.

Il planner chiede 45 minuti per P1, ma non sovrascrive `boost_qos_dbg` del launcher,
che ha limite di 30 minuti. Con lo SHA reale `c21ca40`, `node_labels` conferma
`boost_usr_prod` + `boost_qos_dbg`: tutti i sei nodi P1 della campagna objective
sono incompatibili col limite, bloccando la valutazione e il confronto finale.
I test correnti usano uno SHA fittizio e non leggono questo header effettivo.

Correzione richiesta: QoS `normal` esplicita per le estrazioni di campagna, con
regressione su durata e QoS effettive (opzioni piu' header al commit di deploy).
Questo rilievo riguarda il gruppo 3, non l'estrazione della calibrazione C2.

## Cosa risulta corretto e impatto sui run

- Controllo dell'intera configurazione tramite `build_config`, vincoli sui report
  P1 e comportamento dei candidati validi: nessun altro difetto confermato.
- La correzione afterok rilegge lo stato prima di ogni sottomissione, inclusi i
  predecessori della stessa chiamata; non resta il difetto riportato per C1.
- C1 registrata: tutti e tre i report presenti, summary a 24523 passi/max_steps,
  99 campi d'identita' verificati. I rilievi R1/R2 non invalidano quel risultato.
- Il piano C2 versionato passa report espliciti e 24523 passi per entrambi i
  candidati, con guardie normali. Usa `calibration_extract.sbatch`, non il launcher
  P1 affetto da R3. Nessuna richiesta di ripetere training o sonde gia' validi.
- Stato live del cluster non interrogato. Non estendere questo controllo dei
  piani a una dichiarazione di completamento o validita' di risultati futuri.
- Nessun cambiamento a `src/` in questa tranche; i training C1/C2 usano `72cf24a`,
  gia' su main. La sospensione del merge non modifica i checkout congelati.

## Diagnostica dei rami: correzione arrivata durante la review

Claude ha consegnato `bbf824a` dopo il target richiesto. Letto il diff: R2/CKA
indefiniti diventano null con stato, ID distinti per campagna, IC per dataset
aggiunti e identita' di redundancy.json verificata. Le due regressioni indipendenti
di `4203d59` ora PASSANO: i due P2 di quella review sono chiusi su `bbf824a`, non
retroattivamente su `c21ca40`. Nessuna modifica a selezione/planner dei bracci in
quel commit: i tre nuovi rilievi restano riproducibili.

## Verifiche e consegna

- 69 PASS: selezione, planner della calibrazione e review C1-C4.
- 18 PASS: planner/confronto dei bracci e launcher delle sonde.
- 65 PASS: catena di valutazione e diagnostica dei rami su `bbf824a`.
- Tre regressioni indipendenti in `test_second_tranche_review_20261020.py`:
  **3 FAIL attesi**, esattamente R1/R2/R3; non sono tre fallimenti della suite ufficiale.
- Due regressioni precedenti della diagnostica: **2 PASS** su `bbf824a`.
- Suite completa 1240 PASS riportata da Claude sul codice `62bec0b`, non rieseguita
  integralmente da GPT. In totale in questa attivita': 152 test esistenti PASS,
  piu' le 2 regressioni precedenti PASS e le 3 nuove FAIL. Nessun accesso al cluster.

Alla chiusura del 21/10 sono visibili anche nuovi commit di Claude fino a
`fadb881` sul pilot dei rami: sono fuori dal perimetro di questa review e non
approvati implicitamente. Il log `df1c8c4` riporta training C2 completati e catena
di valutazione sottomessa senza rifiuti; l'esito finale C2 non e' stato verificato.

Per le regressioni impostare `WEARUSFM_REVIEW_SOURCE` al worktree da controllare,
il suo `src/` nel PYTHONPATH, un thread OpenMP/BLAS e disabilitare bytecode/cache
pytest. `WEARUSFM_REVIEW_SHA` permette di controllare l'header del nuovo commit
nel test R3; il default e' il target `c21ca40`.

Prossima consegna a Claude: correggere R1/R2/R3 sul suo branch, con test per stato
ignoto/running, fallimento terminale, PASSI discordanti e QoS effettiva. GPT verifica
i fix e integra per merge preservando gli SHA. Nessuna nuova scelta scientifica,
soglia o autorizzazione di spesa: `decisioni.md` non modificato da questa review.
