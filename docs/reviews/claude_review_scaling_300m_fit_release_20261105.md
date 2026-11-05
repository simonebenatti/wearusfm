# Review Claude — rilascio dei fit CPU a 5.000, scaling 30M/300M (05/11/2026)

Review 9 della linea scaling, breve. Richiesta di Simone del 05/11 alle 22:00 circa: «Review 7 e 8 consegnate. Il
rilascio dei fit non è stato rivisto, vuoi controllare? riesci ad avere tutte le info dai log e dai file?». Alla
seconda domanda: sì, tutto il necessario è nei commit di GPT, nel registro comune dei job e nelle letture di Leonardo.
Commit rivisti sul ramo `codex/scaling-300m-20261104`: da `5275b0d` a `ff93003` (nove commit, 20:10–21:51).

## VERDETTO

**Rilascio dei fit: CONFORME. Nessun blocco, niente da fermare o da rifare.**
Tre note operative (N-1, N-2, N-3); nessuna richiede di toccare i job in corso.

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
- Punta di GPT al verdetto: `ff93003`, ricontrollata con `ls-remote`. Dopo `3d76ae3` non cambia nessun file in
  `scripts/`, `src/` o `tests/`.
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
