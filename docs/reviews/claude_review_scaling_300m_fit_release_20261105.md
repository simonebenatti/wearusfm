# Review Claude — rilascio dei fit CPU a 5.000, scaling 30M/300M (05/11/2026)

Review 9 della linea scaling, breve. Richiesta di Simone del 05/11 alle 22:00 circa: «Review 7 e 8 consegnate. Il
rilascio dei fit non è stato rivisto, vuoi controllare? riesci ad avere tutte le info dai log e dai file?». Alla
seconda domanda: sì, tutto il necessario è nei commit di GPT, nel registro comune dei job e nelle letture di Leonardo.
Commit rivisti sul ramo `codex/scaling-300m-20261104`: da `5275b0d` a `ff93003` (nove commit, 20:10–21:51).
Aggiunta delle 22:40 al §4: `ef8db2b` e `25eca17`, il processo locale che accoda i fit rimasti.

## VERDETTO

**Rilascio dei fit: CONFORME. Nessun blocco, niente da fermare o da rifare.**
Tre note operative (N-1, N-2, N-3); nessuna richiede di toccare i job in corso.
**Aggiunta (§4): processo locale dei cinque invii, nessun blocco.** N-1 e N-3 superate; note nuove N-4, N-5, N-6.

| Controllo | Esito |
|---|---|
| Correzione della review 8 (`/dev/stdin`) applicata prima degli invii | SÌ, 20:10:45; primo invio 20:22:48 |
| Dispatcher eseguito = dispatcher rivisto | SÌ: 12 ricevute e lo script registrato dallo scheduler, sha256 `c2bd7979…` |
| Recuperi EPN612 e UCI | COMPLETED 0:0, 2.820 s e 29 s; sei C convergenti in entrambi |
| 15 report | sha256 remoti = pin = copie pubblicate (15/15); 90 fit convergenti |
| Piano misurato dei fit | Formula ricalcolata uguale; 14 shard, copertura 60/60, 205,866667 core-ore; è il minimo |
| Comandi e ricevute | 14 comandi = 14 shard; 10 righe `sbatch` inviate = argv pubblicati |
| Sequenza dei gate (S8 v2, tetto 310, piano pubblicato prima degli invii) | RISPETTATA |
| Budget | Al più 247,75 core-ore contro 310 |

## 0. Perimetro e metodo

- Solo lettura: nessun job, nessuna cancellazione, nessuna modifica a codice, dati, checkout di GPT o file remoti.
  Letture di Leonardo fra le 22:03 e le 22:07 in `~/wearusfm_local/logs/claude_review9_leonardo_{1,2,3}.txt`.
- Controlli riproducibili con `docs/reviews/claude_review_scaling_300m_fit_release_20261105_check.py` (argomenti: un
  checkout della punta di GPT e la lettura di Leonardo con gli sha256).
- Ultimo commit coperto da questo verdetto: `ff93003`. Fin lì, dopo `3d76ae3`, non cambia nessun file in
  `scripts/`, `src/` o `tests/`. Rettifica: la prima stesura lo chiamava «punta di GPT al verdetto», ed era
  sbagliato (vedi §4).
- Non ho aperto gli score. La ridge della posa e i fit ne scrivono già (report della posa, `val_bacc` nell'audit): la
  lettura è di GPT, al confronto.
- Non coperti: aggregati e confronto, non ancora pubblicati né inviati; i cinque invii 10–14, coperti come comandi
  pubblicati e non come esecuzione; l'esito dei fit, che è in corso.

## 1. Verifiche

1. **Dispatcher.** `/dev/stdin` è in tutti i 17 comandi (2 recuperi, ridge, 14 fit). Le 12 ricevute del dispatcher
   hanno `script_stdin_sha256` = `c2bd7979…`. Lo script che lo scheduler ha registrato per il fit in corso (job
   63626931, letto con `scontrol write batch_script`) ha lo stesso sha256, e `Command=/dev/stdin`.
2. **Recuperi.** EPN612 63620780: COMPLETED 0:0, 2.820 s, 6,266667 core-ore. UCI 63625951: COMPLETED 0:0, 29 s,
   0,064444 core-ore. Ordine rispettato: EPN accettato nel commit delle 21:32:02, UCI inviato alle 21:32:50. Per
   EPN612 300M naturale C=100 ha chiesto 1.238 s e 8.933 iterazioni su 12.000: vicino al mio scenario «2 × C=10».
3. **15 report.** Gli sha256 ricalcolati da me su Leonardo sono uguali ai pin del piano risorse (15/15) e alle copie
   pubblicate. 90 fit di benchmark, tutti convergenti, senza limite né eccezioni; stesso codice `aef665b` e stesso
   contesto numerico `1ac8481e…`.
4. **Piano misurato.** Ho ricalcolato la formula congelata shard per shard senza usare lo script di GPT: valori
   uguali, walltime uguale all'arrotondamento al minuto, tutti entro 10.800 s (massimo 10.680). Copertura esatta
   60/60. Totale 92.640 s: 205,866667 core-ore, 25 h 44 min seriali, 211,466667 col margine di scheduler. La mia
   forza bruta indipendente dà lo stesso minimo: 14 shard e 92.640 s. Memoria: picco massimo 3,47 GiB, da
   moltiplicare per 1,3 contro 30.
5. **Comandi e ricevute.** I 14 comandi di fit coprono i 14 shard col walltime pubblicato. Le 10 righe `sbatch`
   inviate sono uguali agli argv pubblicati. Catena `afterok`: ridge → fit 01 → … → fit 09.
6. **Su Leonardo.** Checkout dei budget a `ec221a6`, pulito; `CODE_DIR` a `aef665b`, pulito; piano risorse
   `92726f54…`, budget della posa `342d1576…`, piani `d2cc7da1…` e `a500cc3c…`. Il `main` canonico è pulito e non è
   stato toccato.
7. **Esecuzione fin qui.** Ridge 63626871: COMPLETED 0:0 in 289 s (formula 660 s, allocati 3.600). Fit 01 in corso
   dalle 21:52:31: i primi quattro C hanno lo stesso numero di iterazioni del benchmark (212, 313, 494, 1.135) e
   tempi al 96 % di quelli misurati, con `login08` a carico 235 su 128 CPU. Il margine 1,5 della formula è intatto.
8. **Sequenza dei gate.** S8 v2 pubblicato alle 19:40 e ritoccato solo alle 20:10:45 (chiarimenti della review 8,
   nessun tetto cambiato), prima del primo score (ridge, 21:47:31). Decisione 310 registrata. Piano dei fit
   `ec221a6` (21:42:35), checkout (21:44), verifica `9e1e24e` (21:47:00), invii fra le 21:47:25 e le 21:48:35.
9. **Contabilità.** Consumato alle 22:07: 22,027778 + 6,266667 + 0,064444 + 0,015 + 0,642222 = 29,016111 core-ore,
   più il fit in corso. Tetti residui: fit 211,466667, aggregati 2,6, confronto 4,4, un checkout 0,266667. In tutto al
   più 247,75 core-ore contro 310. Il 255,76 di GPT è lo stesso conto fatto prima della ridge, più prudente.

## 2. Note operative (nessun blocco)

- **N-1 — Tranche 2, fit 10–14.** I cinque comandi rimasti dipendono ciascuno dal fit precedente. Con
  `MinJobAge = 30 sec` un job esce dallo scheduler mezzo minuto dopo la fine e non può più fare da dipendenza: vanno
  accodati mentre il fit 09 è ancora in coda, altrimenti si emenda e si ripubblica l'argv (GPT lo ha già scritto
  nella `dependency_note`). Senza monitor in background la catena si ferma dopo il fit 09 finché GPT non accoda gli
  altri: stima mia fra le 6:30 e le 10:30 del 06/11, attese in coda escluse. Non è un errore, è tempo: restano al
  massimo 13 ore di walltime da accodare.
- **N-2 — Posti e CPU condivisi con la linea 2.** Alle 22:07 la QoS seriale aveva 10 job dell'utente su 10: nove
  dello scaling e `LONG-checkout-extract-20261105` della linea 2, fermo per il limite di 8 CPU per utente. Finché la
  catena gira, ogni job seriale delle altre linee aspetta la fine del fit in corso (fino a 3 ore) e ogni posto
  liberato è conteso. È materia di coordinamento, non un difetto dello scaling. Con
  `DependencyParameters = kill_invalid_depend`, se un fit fallisce lo scheduler cancella da solo il resto della
  catena.
- **N-3 — Spazio in `$WORK`.** La ridge ha scritto 1,3 GB (tre file di predizioni da 439 MB). Alle 22:04 il progetto
  era a 989,3G su 1T, circa 34,7 GiB liberi. La dimensione degli OOF dei fit non è ancora misurabile: il primo shard
  non è finito.

## 3. Testo per GPT

> Review Claude 9 sul ramo `claude/review-scaling-300m-20261104`, file
> `docs/reviews/claude_review_scaling_300m_fit_release_20261105.md`. Copertura: da `5275b0d` a `ff93003`.
>
> Rilascio dei fit CONFORME, nessun blocco. Verificato in modo indipendente: 15 report con sha256 remoto uguale ai
> pin; formula congelata ricalcolata shard per shard, 14 shard, copertura 60/60, 205,866667 core-ore, minimo
> confermato dalla mia forza bruta; 12 ricevute e lo script registrato dallo scheduler per il fit 01 con sha256
> `c2bd7979…`; 10 righe `sbatch` uguali agli argv; sequenza dei gate rispettata; al più 247,75 core-ore contro 310.
> Il fit 01 riproduce le iterazioni del benchmark con tempi al 96 %.
>
> Tre note. N-1: i fit 10–14 vanno accodati mentre il fit 09 è ancora in coda (`MinJobAge = 30 sec`), altrimenti
> serve l'argv emendato; senza interventi la catena si ferma dopo il fit 09, stima fra le 6:30 e le 10:30 del 06/11.
> N-2: la QoS seriale è a 10 job su 10 (nove tuoi e `LONG-checkout-extract-20261105` della linea 2, fermo per le 8
> CPU): i posti liberati sono contesi, da coordinare. N-3: la ridge ha scritto 1,3 GB in `$WORK`, 34,7 GiB liberi
> alle 22:04; misura gli OOF appena finisce il primo shard.
>
> Non ho aperto gli score. Aggregati e confronto non sono coperti: mandami il commit quando li pubblichi, se Simone
> lo chiede.

## 4. Aggiunta delle 22:40 — processo locale che accoda i fit rimasti (GPT `ef8db2b`, `25eca17`)

**Rettifica.** Al §0 avevo scritto che la punta di GPT al verdetto era `ff93003`: era sbagliato. Quando ho
pubblicato `c5efb80` la punta era già `ef8db2b` (22:07:38), e alle 22:10:52 è arrivato `25eca17`. Il verdetto dei
§1–3 copre da `5275b0d` a `ff93003`; questa aggiunta copre i due commit successivi. Punta riletta alle 22:29, prima
di pubblicare: `25eca17`.

**Verdetto dell'aggiunta: nessun blocco, il processo non va fermato.** Tre note nuove (N-4, N-5, N-6); N-1 e N-3
sono superate.

**Che cos'è.** `results/infra/scaling_300m_20261104/cpu_fit_release_20261105/finish_submissions.py` è un processo
locale sul Mac, senza AI. Ogni minuto legge la coda; quando nella QoS seriale si libera un posto invia il prossimo
dei cinque fit rimasti, col comando già pubblicato. GPT lo ha avviato alle 22:08:47 (PID 2321), 69 secondi dopo
averlo pubblicato. Vive al massimo 8 ore (fino alle 06:08 del 06/11), parte una volta sola e non ritenta: a ogni
anomalia scrive `STOP` in `finish_result.json` ed esce.

| Controllo | Esito |
|---|---|
| File in esecuzione = file pubblicato | SÌ, sha256 `30b8b9b9…`; piano fissato `0370cdf8…` |
| Test di GPT rieseguiti da me | 42 passati |
| Cosa può inviare | Solo i job 10–14 del piano, col submitter comune (sha256 fissato), un tentativo per ID |
| Controlli prima di ogni invio | Predecessore ancora in coda, meno di 10 job seriali, al più 8 CPU in corso, inviluppo ≤ 310 |
| Cosa non fa | Cancellazioni, git, scritture remote (a parte l'invio), rilanci, riavvii |
| Raccolta dei job conclusi | Report, audit e log; degli output grandi solo lo sha256 remoto, senza scaricarli |
| Campi verificati sui fit | Gli stessi che scrive il codice congelato `aef665b` (confronto mio, campo per campo) |
| Primi due invii (fit 10 e 11) | Ricevute = argv pubblicati = registro dei job; sullo scheduler dipendenze, limiti e script giusti |

**Osservato fra le 22:22 e le 22:30.**

- Fit 01 (63626931): COMPLETED 0:0 in 1.801 s su 3.540 concessi, 4,002222 core-ore, picco 4,02 GiB. Il processo lo
  ha accettato e ha inviato il fit 10 (63629605, 22:23:08) e il fit 11 (63629606, 22:23:10).
- Sullo scheduler: `afterok:63627052` e `afterok:63629605`, limiti 02:19:00 e 02:21:00, `login08`, 8 CPU, 30G,
  script registrato con sha256 `c2bd7979…`. Ricevute, argv pubblicati e registro coincidono per 12 job su 12
  (`claude_review_scaling_300m_fit_release_20261105_check_tranche2.py`).
- Inviluppo calcolato dal processo: 255,54 core-ore su 310. Conto mio: consumate 33,018333, al più 243,49.
- Fit 02 in corso dalle 22:23:36. Restano da inviare i fit 12, 13 e 14: partiranno alla fine dei fit 02, 03 e 04.
  Stima mia, se il passo del fit 01 si mantiene: verso le 22:55, le 23:30 e le 00:15; coi limiti pieni entro le
  02:00. Poi il processo esce da solo.
- Alle 22:29 il processo è attivo, senza STOP, e la QoS seriale è a 10 su 10. Condizioni per la notte, lette sul
  Mac: certificati SSH validi almeno fino alle 07:35 del 06/11; alimentazione di rete e sospensione disattivata,
  finché il coperchio resta aperto.
- Non ho aperto gli score: né il `report.json` della posa raccolto da GPT, né i `val_bacc` dell'audit.

**Note superate.** N-1: l'accodamento dei fit rimasti lo fa il processo, che ne ha già inviati due. N-3: l'OOF di
un fit pesa 0,4 MB (436.444 byte per un dataset e un fold: etichette predette, non probabilità).

**Note nuove (nessun blocco).**

- **N-4 — Falso STOP possibile alla fine di un fit.** Lo script si ferma con «ambiguous completion» se `sacct` dà
  già COMPLETED mentre `squeue` elenca ancora il job in COMPLETING. È una deduzione mia dal codice e dal
  funzionamento di Slurm, non un fatto osservato; nessuno dei 42 test copre il caso. La finestra è stretta: il
  fit 02 è diventato eleggibile 2 secondi dopo la fine del fit 01, e il processo legge una volta al minuto. Vale
  qualche punto percentuale per ciascuno dei tre fit ancora attesi. Se succede non si perde nulla: i job in coda
  proseguono, e i fit non inviati si accodano a mano col comando pubblicato finché il fit 11 è in coda.
- **N-5 — Decimo posto conteso con la linea 2 (coordinamento).** Alle 22:22 il job della linea 2
  `LONG-checkout-extract-20261105` è partito per primo ed è durato 23 secondi; poi il processo ha preso i due
  posti liberi. Ora la QoS seriale è a 10 su 10, tutti dello scaling. Finché il processo è attivo prende ogni posto
  entro un minuto circa. Un invio seriale della linea 2 a QoS piena viene rifiutato (limite di 10 job sottomessi
  per utente), e uno fatto negli stessi secondi dell'invio di GPT fa rifiutare uno dei due. Se tocca a GPT, il
  processo si ferma con l'ID in stato `rifiutato`, da ripetere a mano con `--dopo-rifiuto`. Proposta mia: la
  linea 2 aspetta `all_15_submitted` in `finish_result.json`; da lì ogni fit concluso libera un posto.
- **N-6 — Avvio senza review indipendente.** Lo script è partito 69 secondi dopo la pubblicazione, con il «PASS
  independent critic» di GPT e senza la mia lettura. Il contenuto dei cinque invii era già rivisto, e le regole in
  vigore non rendono la mia review bloccante per i lanci di GPT (lettura mia). Resta codice nuovo che sottomette
  job senza sorveglianza: decide Simone se strumenti così devono passare da me prima dell'avvio.

### Testo per GPT (sostituisce quello del §3)

> Review Claude 9 con aggiunta, ramo `claude/review-scaling-300m-20261104`, file
> `docs/reviews/claude_review_scaling_300m_fit_release_20261105.md`. Copertura: da `5275b0d` a `25eca17`.
>
> Rilascio dei fit CONFORME e processo locale dei cinque invii senza blocchi: resta com'è. Verificato in modo
> indipendente: 15 report con sha256 remoto uguale ai pin; formula ricalcolata, 14 shard, 205,866667 core-ore,
> minimo confermato; file in esecuzione con sha256 `30b8b9b9…`; 42 test rieseguiti; campi controllati sui fit
> uguali a quelli scritti da `aef665b`; fit 01 accettato; fit 10 (63629605) e 11 (63629606) con ricevute uguali
> agli argv e, sullo scheduler, `afterok` su 63627052 e 63629605, limiti 02:19:00 e 02:21:00, script `c2bd7979…`.
> Inviluppo 255,54 su 310. Le mie N-1 e N-3 sono superate (l'OOF di un fit pesa 0,4 MB); N-2 resta.
>
> Tre note nuove. N-4: `status()` si ferma con «ambiguous completion» se `sacct` dà COMPLETED mentre `squeue`
> mostra ancora COMPLETING; finestra di circa 2 secondi a ogni fine fit, nessun test la copre. Se trovi quello STOP
> non è un fit fallito: controlla `sacct` e accoda a mano i fit rimasti finché il fit 11 è in coda. Per un riuso
> futuro basta trattare quel caso come «non ancora concluso» e rileggere al giro dopo. N-5: finché il processo è
> attivo prende ogni posto della QoS seriale entro un minuto; un invio della linea 2 negli stessi secondi fa
> rifiutare uno dei due, e se tocca al tuo il processo si ferma con l'ID in stato `rifiutato` (`--dopo-rifiuto`).
> Da coordinare con la linea 2. N-6: lo script è partito 69 secondi dopo la pubblicazione, senza una review
> indipendente mia; decide Simone se in futuro strumenti che inviano job da soli passano prima da me.
>
> Non ho aperto gli score. Aggregati e confronto restano non coperti.
