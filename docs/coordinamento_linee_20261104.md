# Coordinamento delle tre linee — 4 novembre 2026

Simone affida alla chat `coordinamento` il controllo delle interferenze tra i tre
fork. Base documentale comune `8ee6603`; nessun cambio scientifico con questa nota.
Worktree sotto `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/`.

| Linea | Codice e responsabilita' scientifica | Gestione operativa attuale | Worktree di sviluppo GPT | Branch di sviluppo |
|---|---|---|---|---|
| **1 — Proponi soluzione / scaling 300M** | GPT scaling | GPT scaling | `wearusfm-scaling-300m` | `codex/scaling-300m-20261104` |
| **2 — Schedule lungo e consolidamento (EPN15 e valutazione lunga)** | GPT consolidamento | **Claude: deployment, invii, stato/log e raccolta** | `wearusfm-consolidation` | `codex/consolidation-20261104` |
| **3 — Log quadratico e contrastive loss** | GPT logcontrast | GPT logcontrast | `wearusfm-logcontrast` | `codex/tokenizer-logcontrast-20261104` |
| Coordinamento | GPT integratore | Coordinamento dei responsabili, nessun invio duplicato | `wearusfm-coordination` | `codex/coordination-20261104` |

**Aggiornamento degli operatori del 05/11 sera:** eccezione della sola linea 2,
fonte Claude3566c8d e messaggio diretto di Simone nella chat2. Le verifiche di stato
successive spettano al suo operatore Claude; i resoconti precedenti sotto sono snapshot storici.
Codice e responsabilita' scientifica restanoGPT. Regola canonica pubblicata su main **fe66e2d**.


Thread locali: **1** `01a177ab-e433-7b33-aa9a-7515e750f20f`;
**2** `01a1a72a-070b-7bc1-8eea-452c4c53cff9`;
**3** `01a1a725-89e9-7300-99f8-8df0975cf7dd`;
**coordinamento** `01a1a72c-1ad9-7cd3-b1ad-0f8b77ac0d52`.

## Verifica

Verificati i worktree effettivi e il checkout science pulito a `8ee6603`.
Il fork consolidamento aveva riattivato un duplicato dell'agente scaling dal
contesto ereditato: l'owner conferma che si è fermato senza scrivere né lanciare.
Il thread originale conferma autore/critico conclusi normalmente, consegna
`31f6d03` (codice `05d4bf0`, 23 test verdi), nessuna modifica altrui rimossa.
Nessun job o invio al ledger dalla linea scaling, in attesa di Claude.

Log-quadratico consegnato a `729093b`, worktree pulito, 112 test dichiarati nella
consegna e nessun job; verificata nella storia Git l'inclusione di `8ee6603`.
Il protocollo pubblicato conferma namespace LOGSQ/CONTRAST, output dedicati e
assenza di agenti aggiuntivi. Anche questa consegna aspetta Claude.

Il ledger globale ha come ultima sottomissione `63519077`,
`TOKENIZER-probe19618-compare-20261104`; nessuna nuova prenotazione delle tre linee
al controllo. Questo non sostituisce una lettura della coda remota.
I tentativi SSH di sola lettura non hanno restituito l'esito di `squeue` e sono
stati interrotti; nessuna conclusione sullo stato SLURM attuale. La verifica di
assenza di nuove sottomissioni si basa sul ledger e sulle dichiarazioni degli
owner, non su una coda remota osservata vuota.

## Confini operativi comunicati

- Un owner orchestra gli agenti della propria linea, senza riattivare agenti
  ereditati delle altre. Autore e revisore si assegnano file distinti.
- Si modifica e pubblica soltanto il proprio worktree. Nessun reset, stash,
  checkout o merge nella cartella di un altro. Coordinamento integra su main
  un ramo revisionato alla volta, preservando commit e testo, senza squash/rebase.
  Nessun merge sperimentale su main è eseguito da questa consegna.
- Preparazione, test e preventivo, poi review Claude e convergenza, poi lanci:
  è l'ultima istruzione di Simone nelle tre chat. Dopo la review il piano già
  approvato non richiede un secondo via. Nessun monitor automatico nuovo.
- Un submitter per linea: GPT per 1/3; Claude per EPN15 e valutazione lunga della 2 (eccezione del 05/11 sera). Coordinamento non duplica lanci.
  Tutti usano `scripts/submit_job.py` e `~/wearusfm_local/jobs/`, prenotazione
  sotto lock, controllo ID esistente e riconciliazione delle risposte ambigue.
- Namespace assegnati: `SCALING-300M-*` (confermato dall'owner), `LOGSQ-*` /
  `CONTRAST-*`, `CONSOL-*` / `FT-*`. Eventuali nomi già scelti vanno comunicati
  prima del lancio. Risultati separati per linea/fase/braccio/passo/seme;
  checkpoint sorgenti e cache storiche soltanto in lettura. Resume e fine-tuning
  hanno destinazioni nuove. Preventivi e consuntivi distinti ma sommabili.
- Owner comunica SHA, review, risorse, output e ID dei job a coordinamento.
  Codice da SHA completo pubblicato in checkout detached immutabile; nessun
  aggiornamento del checkout o dell'ambiente condiviso di job già attivi.

## Componenti comuni

Log-quadratico possiede tokenizer e nuovi campi di configurazione. Consolidamento
possiede `src/wearusfm/harness/backbone_depth.py`, estrattore pre/post backbone
con supporto D384/D896 e test di parità. Scaling lo riuserà dopo review, senza
duplicarlo (confermato dall'owner); possiede il collaudo e i controlli 300M.

`fm.py`, `training/run.py`, `train_jepa.py` e wrapper del tokenizer sono possibili
conflitti futuri: modifiche nei propri branch, integrazione esplicita con review
del diff e test di retrocompatibilità. Nessun merge incrociato automatico.
Registro, decisioni e piano si aggiornano sul proprio ramo; coordinamento conserva
le voci di tutti. Il fork log incorpora l'amendamento `8ee6603` (T25-L/B e T100-G
secondario) preservando il lavoro locale. Questa nota non modifica i protocolli.

## Aggiornamento del 05/11/2026

Numerazione da usare sempre: **1 scaling**, **2 schedule lungo e consolidamento**, **3 log-quadratico e contrastiva**.
La verifica iniziale sopra e' storica: stato e confronto dei rami aggiornati in
[Audit di allineamento GPT / Claude](reviews/allineamento_gpt_claude_20261105.md).
Su conferma diretta di Simone, main `7616f33` assegna formalmente a GPT sia sottomissioni sia controllo operativo dei log,
un owner per linea; semplicione non e' piu' operativo. Nessun cambiamento ai checkout congelati o alle review richieste.

### Seguito serale del 05/11

Audit Claude ef0d987 e stato19:03 recepiti in [seguito](reviews/seguito_audit_claude_20261105.md).
Chat1: stop scalingCPU dopoTIMEOUT; diagnosi/promptClaude03cbd18, nessun recupero lanciato sotto il piano fallito.
Chat2: EPN15in corso; valutazione lungo implementata e consegnata per reviewClaude a6f5d909.
Chat3: LQ5000completo,19618Priority. D-5 corretto nel solo test a d2f2f08 (15verdi), integrazione main ancora distinta.

### Passaggio operativo della linea 2 a Claude — 05/11 sera

Verificato il messaggio diretto di Simone nella chat2: «c'e un cambio di programma. i task di questa chat d'ora in poi li lancia claude».
Citazioni e fonte Claude3566c8d recepite su main **fe66e2d**; copia canonica CLAUDE.md allineata anche in questo ramo.
GPT consolidamento consegna modifica radiceSCRATCH, ownerclaude, N-4 e istruzioni operative; non esegue nuovi deploy/invii
sulla linea2. Claude rivede il diff e prepara pin/deploy secondo il passaggio concordato. Nessun nuovo esperimento implicito.
Riepiloghi futuri della linea2 devono citare gli esiti raccolti dall'operatore Claude, senza attivare un secondo ciclo operativo.

### Consegna operativa della linea 2 pubblicata — 05/11, 21:42 CET

GPT consolidamento ha consegnato **9d195e7bf96e8d8bc4b602434a8c0fcfc75ac7d5**, verificato su origin,
con il prontuario unico `docs/handoff_line2_claude_20261105.md` nel ramo `codex/consolidation-20261104`.
Piani verificati: `status=awaiting_claude_rereview`, `review.commit=null`, `owner=claude`.
Nuova radice lunga su SCRATCH grande; owner, percorsi e retention aggiornati. N-4 corretta nel solo test.
Ricetta, endpoint, 82 job di calcolo e 4 checkout invariati; test dichiarati:252 passati con MKL_NUM_THREADS assente.

| Parte della linea 2 | Responsabile operativo | Stato della consegna |
|---|---|---|
| EPN15 gia' in corso | Claude | Inventario, istruzioni di raccolta e aggregazione nel prontuario; nessun nuovo controllo SLURM da coordinamento |
| Valutazione lunga | Claude | Attende quarta review sul diff da51b4574; dopo convergenza Claude prepara pin, checkout e lanci |

Questa registrazione verifica pubblicazione e metadati del passaggio; non sostituisce la quarta review e non dichiara
nuovi invii o nuove misure di coda. GPT consolidamento ha dichiarato nessuna prenotazione aperta nel ledger.
