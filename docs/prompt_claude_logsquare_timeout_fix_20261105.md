# Prompt per Claude — chiusura della review della correzione T100-LQ

GPT ha revisionato `fd874d75dcc5252076399c3c8d2d5ae97584570c`:
**A (guard) e B (ingresso dei rank) APPROVATI. C approvato per criteri e risorse,
ma il lancio diagnostico è fermo per un solo difetto tecnico nella gestione dei timeout.**

Leggi sul ramo `origin/codex/tokenizer-logcontrast-20261104`:

- `docs/reviews/gpt_review_logsquare_initialization_fix_20261105.md`;
- `docs/reviews/test_logsquare_detached_timeout_20261105.py`;
- `results/infra/logsquare_initialization_20261105/gpt_review_timeout_reproduction.json`.

Correggi sul tuo ramo, preservando i commit precedenti:

1. In `scripts/logsquare_initialization_diagnostic.py`, `run_bounded` uccide solo il
   gruppo del launcher; i rank elastic partono in sessioni separate. È confermato anche
   nel Torch 2.2 installato su Leonardo. Il watchdog esterno di `execute` uccide solo
   il figlio diretto; può scattare prima del timeout interno se la preparazione supera
   il margine 300−270 secondi. Entrambi i casi lasciano worker attivi nei due test GPT.
2. Implementa pulizia completa e limitata nel tempo della sola invocazione: chiusura
   ordinata del launcher, escalation se necessaria anche sui discendenti in sessioni
   separate, gestione della gara con il timeout esterno e conservazione di stdout,
   stderr ed errore nel report. Non usare kill globali o aumenti dei timeout come rimedio.
   Includi la pulizia nei limiti del piano; conserva il tetto di 15 minuti se possibile.
3. Integra regressioni equivalenti ai due test GPT nella suite CPU e verifica anche
   ritorno normale, errore del figlio e worker che ignora SIGTERM. I test GPT sono la
   prova minima: possono essere adattati alle firme nuove, mantenendo i due scenari e
   il controllo che il worker non continui a lavorare. Tutti i processi dei test vanno
   ripuliti anche quando il test fallisce.
4. Rettifica solo il testo descrittivo: lo stato completo è una ricostruzione registrata
   da riferimento e gain storico, non un hash iniziale storico osservato; il dry run
   legge manifest/indici/scale, pur non eseguendo batch o forward sui segnali.

Non modificare guard, ingresso dei rank, controller di training o criterio scientifico
salvo una necessità dimostrata da sottoporre separatamente. Non cambiare epsilon,
ricetta, sorgenti scientifici, vecchia release o artefatti del fallimento. Mantieni
il piano `REVIEW_REQUIRED`: nessun job, training, retry o release da parte tua.

Riesegui i quattro file LQ indicati nella tua consegna, le regressioni dei timeout e
`bash -n` del launcher. Aggiorna pin pertinenti se richiesti dal diagnostico, documento
e registro nelle due sezioni; commit/push e restituisci SHA, diff, test e limite temporale
complessivo. GPT chiude la review e procede con il diagnostico sotto l'autorizzazione
operativa già data da Simone, senza un'altra richiesta di consenso.
