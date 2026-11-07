# Prompt per Claude — prima proposta contrastiva, linea 3

Simone chiede di passare alla contrastiva, preparando la proposta e facendola
revisionare prima dei lanci. **Qui devi fare soltanto review: niente codice,
job, collaudo, deploy o modifiche delle altre linee.**

Ramo GPT: `origin/codex/tokenizer-logcontrast-20261104`, commit di consegna
indicato da Simone nel messaggio. La base precedente è `8c3a4fa`, con lettura
finale e archivio LQ: la tua N-2 sui rapporti mancanti è già chiusa in quel commit.

Leggi:

1. `docs/contrastive_proposal_20261107.md` — proposta completa, stato solo review.
2. `results/infra/contrastive_proposal_20261107/proposal.json` — scelte numeriche
   della proposta, non un piano runtime.
3. `docs/reviews/risposta_claude_lq_final_20261107.md` — N-1/N-2/N-3 e precisazione
   sul confronto T25-L/T100-L, che non è una replica a ricetta identica.
4. `docs/reviews/contrastive_proposal_critic_20261107.md` — controllo interno della proposta.
5. Per verificare la fattibilità: `src/wearusfm/training/jepa.py`,
   `training/run.py`, `data/pretraining_loader.py`, `model/fm.py`, `model/backbone.py`
   e `model/spectral_step1.py` sotto `src/wearusfm/`.

Domanda: **su T100-G 30M, InfoNCE aggiunta ad ancore + posa migliora le sonde di
classificazione senza deterioramento rilevante della regressione?** Due nuovi
training da stessa inizializzazione, C0 peso 0 e C1 peso calibrato sul solo train,
stessa infrastruttura, maschere, FIFO, dati e dose. Nessun warm-start, LQ o JEPA.

Punti su cui chiedo un verdetto separato:

- **Base e controllo.** Scelta adattiva T100-G; due nuovi bracci, storico solo
  contestuale; checkpoint 5.000/19.618, stesso WSD totale 49.046.
- **Positivi e livello.** Student con maschera storica, teacher EMA 0,996 sulla
  stessa finestra completa; media di z su slot e patch fisicamente valide,
  incluse query slab; proiettore 384→384 GELU→128, L2 solo nell'uscita NCE.
  Conferma che entrambi gli encode già esistono con JEPA peso zero e che il
  pool768 delle sonde non viene impropriamente riusato sulla vista mascherata.
- **Negativi.** FIFO per dataset 1.024, anzianità 256, minimo 32 negativi;
  duplicati fisici e vicinanza 4 s esclusi, stesso soggetto consentito;
  conservatorismo per trial senza tempo comune. Rischio residuo identità,
  montaggio, gesto ripetuto e sottopeso dei dataset con bassa copertura.
- **Peso.** τ=0,2; una calibrazione dei gradienti globali a inizializzazione,
  target 10% e reset completo. La testa posa a zero rende il riferimento
  iniziale soprattutto ancore: valuta se questa regola è un punto di partenza
  ragionevole o richiede una modifica motivata prima del collaudo. Nessuna
  ricerca sulla validation o adattamento del peso durante il training.
- **Correttezza.** Denominatore globale 32 con query invalide a zero, chiavi
  detached all-gather, riduzioni DDP, FIFO/EMA una volta per update, ripresa
  esatta e percorso C0 paragonabile al trainer di base. I test sono requisiti
  futuri, non risultati già dichiarati.
- **Lettura.** P1/P2 nuovi primari con soglia pratica +0,5 pp e IC97,5 per i due
  confronti; NI 1 pp sull'altro aggregato, guardia posa con IC95 superiore del
  peggioramento relativo sotto +2%; EPN secondario. Valuta soglie e limiti
  della rappresentazione media, senza promettere recupero della dinamica.
- **Costi.** Due training hanno riferimento storico complessivo 33,89 GPU-ore,
  prima dell'overhead. Tempi reali, tetti, estrazioni, CPU e checkout devono
  essere congelati dopo un collaudo distinto, non sono già approvati qui.

Riporta APPROVATO / APPROVATO CON CORREZIONI / DA RIVEDERE per il disegno,
distinguendo blocchi scientifici da dettagli da verificare nell'implementazione.
Per ogni cambio suggerisci la minima alternativa concreta; niente griglia
implicita di loss o ampliamento della campagna. Pubblica review e voci del
registro sul tuo ramo, con commit/push. Non sottomettere nulla.
