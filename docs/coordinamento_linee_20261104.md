# Coordinamento delle tre linee — 4 novembre 2026

Simone affida alla chat `coordinamento` il controllo delle interferenze tra i tre
fork. Base documentale comune `8ee6603`; nessun cambio scientifico con questa nota.
Worktree sotto `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/`.

| Owner | Perimetro | Worktree | Branch |
|---|---|---|---|
| Proponi soluzione per parte scientifica | Scaling 300M, collaudo e screening | `wearusfm-scaling-300m` | `codex/scaling-300m-20261104` |
| log quadratico e contrastive loss | Log-quadratico su T100-L; contrastiva successiva separata | `wearusfm-logcontrast` | `codex/tokenizer-logcontrast-20261104` |
| schedule lungo e consolidamento | T100-G 30M lungo, sonde a due momenti/livelli, fine-tuning | `wearusfm-consolidation` | `codex/consolidation-20261104` |
| coordinamento | Responsabilità e integrazione dei rami revisionati | `wearusfm-coordination` | `codex/coordination-20261104` |

Thread locali, rispettivamente: `01a177ab-e433-7b33-aa9a-7515e750f20f`,
`01a1a725-89e9-7300-99f8-8df0975cf7dd`, `01a1a72a-070b-7bc1-8eea-452c4c53cff9`,
`01a1a72c-1ad9-7cd3-b1ad-0f8b77ac0d52`.

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
- Un submitter per linea: il rispettivo owner. Coordinamento non duplica lanci.
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
