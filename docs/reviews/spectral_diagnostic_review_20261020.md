# Review GPT: diagnostica dei rami, attivita' 1 del seguito 2A/2C

Data: 20/10/2026. Consegna esaminata: branch Claude, snapshot `6086df2` (codice di deploy `62bec0b`).
Perimetro: `docs/rami_diagnostica_v1.md`, `scripts/spectral_diagnostic.py`, launcher, test e dipendenze dirette di
calibrazione/confronto/catena. Non e' una review completa della nuova tranche Claude e non e' un merge.

## Verdetto

Impostazione scientifica favorevole; due P2 di robustezza riprodotti, da correggere sul ramo Claude. Nessun P0/P1 confermato.
Non trovata contaminazione dei test ufficiali nel percorso esaminato. Nessun risultato reale della diagnostica disponibile nei log
letti: questa review riguarda protocollo e implementazione. Nessun accesso al cluster, lancio o modifica del codice di produzione.

## P2: R2 indefinito blocca la diagnostica quando il ramo e' costante

`scripts/spectral_diagnostic.py:276`, consumato alle righe 286-287 e serializzato alla riga 311.

`ridge_r2` restituisce NaN quando la varianza del target e' nulla. Le feature costanti sono finite e accettate dal loader; sono proprio
un caso che la diagnostica del rango contempla. Con `--redundancy`, `json.dumps(..., allow_nan=False)` solleva quindi ValueError,
facendo fallire il job dopo le sonde e impedendo il confronto dipendente. Riprodotto con feature dei rami allenati tutte zero,
raw e rami iniziali non costanti: `Out of range float values are not JSON compliant: nan`.

Correzione richiesta: rappresentare l'R2 non definito con null e motivo/stato esplicito, conservando rango e sonde; non attribuire
un R2 fittizio di zero o uno. La riproduzione non dimostra che le cache reali del 2C siano collassate.

## P2: campagne diverse condividono gli ID globali dei job

`scripts/spectral_diagnostic.py:429` e `:433`; `--campaign` entra solo nei metadati alla riga 437.

Il planner genera sempre `RAMI-<teacher>-<job>` e `RAMI-compare`. Il registro comune indicizza gli stati per ID, senza namespace
automatico della campagna. Generando due piani con campaign e out-dir differenti, e applicando al secondo gli stati sottomessi
del primo, tutti i suoi 17 nodi risultano gia' sottomessi: non parte nessun job nuovo e i nuovi output non vengono prodotti.

Correzione richiesta: ID univoci per campagna, stabili nelle tranche/riprese della stessa campagna; aggiornare anche le dipendenze.
Non cancellare/rinominare gli stati di job eventualmente gia' sottomessi. Il difetto non blocca il primo uso con ledger privo di tali ID.

## Osservazioni minori

- Il protocollo promette gli intervalli della differenza dei delta anche per dataset; l'implementazione alle righe 370-374 salva
  per dataset solo stima e delta per teacher, con IC per il solo aggregato. Aggiungere gli IC per dataset o precisare il testo;
  non e' un difetto del calcolo dell'IC aggregato.
- `compare` legge un eventuale `redundancy.json` senza controllarne protocollo/task/arm (righe 390-392). In una cartella nuova il
  job standard lo riscrive correttamente; per riuso/copie manuali conviene verificarne identita' e collegamento alle sorgenti.
- Il tetto CPU e' consistente: 16 job da 8 core, 60 ore complessive di allocazione, piu' confronto e checkout = circa 481 core-ore.
  Resta una stima di durata e l'ordine dopo C2 e' una regola operativa, non un'automatica autorizzazione a sottomettere in questa review.

## Controlli positivi e verifiche

- P1 legge solo train_/val_; P2 filtra i soggetti fuori dai tre test ufficiali prima delle metriche. Scaler sul train del fold,
  C/alpha sulla validation del fold, valutazione sui soggetti held-out.
- Cache verificate per SHA, identita', allineamento delle righe e coppia di checkpoint dello stesso run completo del 2C.
- Confronti accoppiati, bootstrap per soggetti e differenza dei delta calcolata sulle stesse estrazioni. Le conclusioni restano
  osservative; non superare la regola non implica equivalenza. Nessuna promozione retroattiva del risultato descrittivo del passo 0.
- **128 test esistenti PASS**: 31 fra diagnostica/launcher; 97 fra selezione, review della calibrazione, confronto dei bracci e catena.
  Uno dei 97 richiede `.git`: falliva nel solo archivio temporaneo per assenza del repository, poi PASS nel worktree Claude
  allo stesso HEAD. Il primo comando aveva anche indicato un file di test inesistente: nessun test eseguito da quel comando.
- **Due nuove regressioni FAIL attese sul codice esaminato**, nel file adiacente `test_spectral_diagnostic_review_20261020.py`.
  Non eseguita la suite completa e nessuna conferma sul cluster. Controlli indipendenti paralleli incorporati nei rilievi sopra.

Comando delle regressioni, sostituendo il checkout da verificare:

```sh
WEARUSFM_REVIEW_SOURCE=/percorso/checkout PYTHONPATH=/percorso/checkout/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider docs/reviews/test_spectral_diagnostic_review_20261020.py
```

Le quattro attivita' sui rami appartengono al punto 5 della tabella comune; il punto 1 storico (2A/2C) resta chiuso.
