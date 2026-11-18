# Review Claude — dispatcher di recupero CPU e piano degli invii, scaling 30M/300M a 5.000 (05/11/2026)

Review 8 della linea scaling, breve. Richiesta di Simone del 05/11 sera: «mi fai una veloce review sul dispatcher? B-1
chiusa: 310 core-ore complessive approvate.»; a review in corso Simone ha incollato il prompt di GPT con i cinque punti
(testo completo in `docs/prompt_claude_scaling_cpu_dispatcher_review_20261105.md`). Commit rivisti sul ramo
`codex/scaling-300m-20261104`: `4c670bd` (dispatcher, piano, budget, S8 v2) e `3d76ae3` (approvazione di Simone).

## VERDETTO

**Dispatcher: APPROVATO. Riuso dei 13 report, inviluppo 310 e freeze S8 v2: APPROVATI.
Comandi dei due invii: UNA correzione obbligatoria prima del primo invio (B-1).**
Con quella sola correzione il piano vale come approvato: non serve un altro giro di review.

| Punto del prompt | Esito |
|---|---|
| 1. Dispatcher: solo EPN612/UCI, runtime fermo ad `aef665b`, guardie e formula invariate | APPROVATO |
| 2. Fit: al massimo 15 shard, 3 ore, 240 core-ore; piano misurato dopo i 15 benchmark | APPROVATO: limiti fail-closed, reggono nei miei scenari |
| 3. Riuso dei 13 report, padre TIMEOUT conservato, pin, ricevute, accounting, output nuovi | APPROVATO |
| 4. Budget cumulativo 310 con costi sostenuti e margini | APPROVATO, conti riprodotti; B-1 della review 7 chiusa |
| 5. S8 v2: QoS e tetti CPU corretti prima degli score, nessun 19.618 | APPROVATO; B-3 della review 7 chiusa |
| Comandi dei due invii (`submission_plan.json`) | DA CORREGGERE: manca il nome dello script per `sbatch` (B-1) |

## 0. Perimetro e metodo

- Solo review: nessun job, nessuna modifica a codice, dati, checkout di GPT o file remoti. Letture di Leonardo in sola
  lettura alle 19:55 (`~/wearusfm_local/logs/claude_review8_leonardo_{1,2}.txt`).
- Punta di GPT al verdetto: `3d76ae3`, ricontrollata con `ls-remote`.
- Fuori copertura: `6e57e19` e `d2c099b` (test storico B19618); il piano misurato dei fit, che non esiste ancora; i
  wrapper futuri per 19.618.
- L'«approvo» di Simone sul tetto 310 è riportato da GPT in `docs/decisioni.md` (`3d76ae3`); a me Simone ha scritto
  «B-1 chiusa: 310 core-ore complessive approvate». Non ho letto la chat di GPT.

## 1. Verifiche

1. **Dispatcher.** `scripts/slurm/scaling_readers_recovery.sbatch` (sha256 `c2bd7979…14d5`, 17.688 byte, uguale nel
   commit e nel checkout di GPT da cui parte l'invio) è il launcher congelato (`2da66c15…`) più 13 sostituzioni. Le ho
   lette con un `diff` mio; il test `test_diff_is_only_the_declared_operational_allowlist` le riapplica ai byte
   dell'oggetto git `aef665b` e pretende l'uguaglianza dell'intero file. Cambiano: nome del job e dei log; default di
   3 ore; `--dataset` obbligatorio e limitato a EPN612/UCI; sola cella `T100-G-300M-natural`; walltime esatto 7.200 s
   (EPN612) o 600 s (UCI); shard fino a 10.800 s; fit fino a 240 core-ore; `EXPECTED_SHA` accettato solo se
   `aef665ba…`. Restano: moduli, venv, 8 thread, preload di `scaling_compare`, guardie di scheduler (`login08`,
   `lrd_all_serial`, 8 CPU, 30G, zero GPU, walltime uguale a quello pubblicato), pin degli input, output nuovo e fuori
   dal codice, stop al primo errore, al massimo 15 shard, copertura esatta 3×5×4.
2. **Codice numerico.** `git diff aef665b 3d76ae3` è vuoto su `scripts/scaling_probe.py`, `scaling_pose_readout.py`,
   `scaling_compare.py`, `scaling_reading.py`, `scripts/slurm/scaling_readers.sbatch`, `lib_code_dir.sh`,
   `scripts/submit_job.py` e `src/`. Su Leonardo `CODE_DIR` è a `aef665b`, pulito; `probe_plan.json` ha sha256
   `d2cc7da1…` e 42.708 byte.
3. **Test.** 166 passati in 118,10 s sulla punta `3d76ae3` (42 dispatcher, 72 launcher originale, 36 sonde, 16
   lettura), con il comando del prompt e un JUnit mio.
4. **Riuso.** I 13 report di `accepted_benchmarks.json` hanno lo sha256 delle mie copie indipendenti della review 7
   (13/13); ricevute, accounting, `sacct` e riga di completamento nel log corrispondono ai pin (13/13). I tre NinaPro
   del 300M naturale restano figli del job 63593508 in TIMEOUT (`parent_success_required: false`,
   `timeout_parent_not_promoted_to_success: true`, 12,408889 core-ore a consuntivo). `validate_resources` nel codice
   congelato legge ogni report da un pin con percorso proprio e non guarda lo stato del job padre: il riuso da
   directory diverse funziona senza toccare il codice, e pretende lo stesso contesto numerico (`1ac8481e…`) nei report
   nuovi e nei fit. Su Leonardo il parziale di EPN612 è intatto (`audit_progress.jsonl`) e la directory
   `probe_benchmark_recovery_20261105` non esiste ancora.
5. **Tetti nuovi contro i miei scenari** (conto della review 7, 10.800 s per shard, UCI escluso perché vale pochi
   secondi):

   | C=100 di EPN612 300M naturale nel benchmark | Shard | Core-ore dei fit |
   |---|---:|---:|
   | limite inferiore, 559 s | 12 | 186,84 |
   | 1,5 × C=10, 1.012 s | 12 | 197,41 |
   | 2 × C=10, 1.350 s | 14 | 206,96 |
   | 12.000 iterazioni al ritmo di C=10, 1.569 s | 14 | 212,81 |

   più al massimo 1,87 core-ore per l'arrotondamento al minuto. Il benchmark di EPN612 chiederebbe circa 3.000 s
   contro 7.200. Margini nello scenario peggiore: uno shard, circa 25 core-ore, circa 2,4 volte sul benchmark. Non è
   una garanzia: EPN612 non è ancora misurato. Il tetto 240 regge finché C=100 resta sotto circa 2.500 s (a parità
   degli altri cinque C); oltre, la regola pubblicata è fermarsi senza adattare i tetti, ed è quella giusta.
6. **Inviluppo 310.** Voci ricalcolate: 22,027778 + 18,133333 + 246 + 8,4 + 2,6 + 4,4 + 0,533333 = 302,094444; riserva
   7,905556. I margini sono CPU × 180 s per job. Decisione registrata in `docs/decisioni.md` (`3d76ae3`).
7. **S8 v2.** JSON v1 intatto (`3ab57580…`, 13.134 byte) e regola scientifica intatta (`da58ec3a…`); QoS `normal`;
   53.100 s, 59 + 7 = 66 GPU-ore, margine 0,35; lettori 345,166667 + 4,833333 = 350; 878 e 880,8 core-ore
   complessive. Pubblicato alle 19:40, prima di qualunque score: su Leonardo esistono solo `pose_validate`,
   `pose_benchmark` e `probe_benchmark`, e `sacct` non mostra job di ridge o di fit.
8. **Nessun job nuovo.** Nel ledger non ci sono ID `SCALING-probe-recover-*`; l'utente non ha job su
   `lrd_all_serial`.

## 2. Blocco

### B-1 — I due comandi pubblicati verrebbero rifiutati da `sbatch`: manca il nome dello script

- `submission_plan.json` passa il dispatcher con `--script-stdin` e, dopo le opzioni, mette subito gli argomenti del
  dispatcher: `… --export=ALL,CODE_DIR=…,EXPECTED_SHA=… probe-benchmark <piano> <sha> <out> --cell … --dataset …`.
  Il submitter compone `sbatch --parsable --job-name=… --comment=… <argomenti>` e manda lo script su stdin senza
  aggiungere altro (`scripts/submit_job.py`, funzione `submit`).
- Per `sbatch` il primo argomento che non è un'opzione è il file dello script. Letto su Leonardo (Slurm 25.05.7):
  `Usage: sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]`; il manuale dice che lo script si legge
  da stdin solo se sulla riga di comando non c'è un nome di file. `probe-benchmark` verrebbe quindi cercato come file.
- Precedenti nel registro comune dei job: 89 invii con script su stdin; i 12 che hanno argomenti posizionali (14/10,
  `step1_random_*`) mettono tutti `/dev/stdin` come nome dello script e sono stati accettati (per esempio il job
  60563048). Gli invii CPU di oggi passavano invece il launcher per percorso, dal checkout di campagna.
- Conseguenza: nessun costo CPU (rifiuto netto, stato `rifiutato`), ma l'ID logico resterebbe consumato e la
  correzione avverrebbe a mano, fuori dal piano rivisto.
- Correzione: inserire `/dev/stdin` subito prima di `probe-benchmark` nei due `argv` e ripubblicare
  `submission_plan.json`; la stessa forma vale per i comandi successivi (fit, ridge, aggregati, confronto) se passano
  dallo stesso dispatcher su stdin. Alternativa: dispatcher per percorso da un checkout congelato, che però aggiunge
  un job di checkout alle voci del budget. Il dispatcher non cambia: il pin `c2bd7979…` resta valido.
- Limite della verifica: è una deduzione dalla sintassi documentata e dai precedenti; non ho eseguito `sbatch`, perché
  non lancio nulla. I test non possono vederlo: eseguono il dispatcher direttamente, non la riga di `sbatch`.

## 3. Facoltativi

- **F-1 — Legare i report nuovi ai byte del dispatcher.** Il `code_sha` dei report copre `CODE_DIR`, non lo script
  passato su stdin. Il submitter registra `script_stdin_sha256` nella ricevuta ma non lo impone: aggiungere
  all'accettazione dei due report «`script_stdin_sha256` = `c2bd7979…`» e controllare lo sha256 del file prima
  dell'invio.
- **F-2 — Il limite dei 10 job è della QoS.** `MaxSubmitPU=10` vale per i job dell'utente nella QoS `lrd_all_serial`
  (di tutte e tre le linee), non per tutti i job dell'utente: alle 19:55 erano 53 su `boost_usr_prod` e zero su
  `lrd_all_serial`. La frase «<=10 total user jobs» della policy, presa alla lettera, non passerebbe mai. È la mia
  lettura del limite di Slurm; il dato da leggere prima di ogni invio è `squeue -u sbenatti -p lrd_all_serial`.
- **F-3 — S8 v2 non è un budget approvato.** 350 core-ore e 66,35 GPU-ore sono tetti di una proposta futura:
  l'«approvo» di Simone copre le 310 core-ore della valutazione a 5.000, come dice lo `scope` nel JSON di GPT. Va
  ripetuto quando si proporrà il seguito.
- **F-4 — `login08` resta sovraccarico** (carico 175 su 128 CPU alle 19:55): i tempi di EPN612 misurati stasera
  possono differire da quelli del pomeriggio. Nessuna azione: si legge il risultato e si applica la formula.

## 4. Testo per GPT

> Review Claude 8 sul ramo `claude/review-scaling-300m-20261104`, file
> `docs/reviews/claude_review_scaling_300m_cpu_dispatcher_20261105.md`. Copertura: `4c670bd`, `3d76ae3`.
>
> Dispatcher APPROVATO: è il launcher congelato più le 13 sostituzioni dichiarate (diff mio e test dell'elenco
> chiuso), sha256 `c2bd7979…`; codice numerico, submitter e `src/` identici ad `aef665b`; 166 test passati. Riuso dei
> 13 report, inviluppo 310 (302,094444 + 7,905556) e S8 v2 APPROVATI, conti riprodotti; chiuse B-1 e B-3 della
> review 7. I tetti dei fit reggono nei miei quattro scenari (12–14 shard, 186,84–212,81 core-ore).
>
> Una correzione obbligatoria prima del primo invio (B-1): in `submission_plan.json` dopo le opzioni di `sbatch`
> c'è subito `probe-benchmark`. Con lo script su stdin `sbatch` prende il primo argomento non opzione come file dello
> script (`sbatch [OPTIONS…] script(0) [args(0)...]`) e rifiuterebbe l'invio. I 12 precedenti del 14/10 su stdin con
> argomenti mettono `/dev/stdin` come nome dello script. Inserisci `/dev/stdin` subito prima di `probe-benchmark` nei
> due `argv` (e nei comandi successivi su stdin) e ripubblica il piano; il pin del dispatcher non cambia. Con questa
> sola correzione il piano è approvato: non serve rimandarmelo.
>
> Facoltativi: F-1 accetta i report nuovi solo se `script_stdin_sha256` della ricevuta è `c2bd7979…`; F-2 il limite
> dei 10 job vale per la QoS `lrd_all_serial`, non per tutti i job dell'utente (53 ora su `boost_usr_prod`); F-3 i
> tetti di S8 v2 non sono budget approvato da Simone; F-4 `login08` è ancora sovraccarico (175 su 128 CPU).
