# Integrazione delle consegne scientifiche — 28/10/2026

Richiesta di Simone: dopo i lanci autorizzati, integrare tutto su main. Le tre consegne sono GPT science,
Claude `7a4221d` e GPT evaluation `da0e803`, su main `4680617`. Conservare i commit originali mediante merge,
le voci testuali dei due registri e i risultati originali. Nessun deploy verso i checkout di campagna.

## Review prima del merge

Due review indipendenti non trovano rilievi sostanziali aperti che impediscano l'integrazione. Il codice di Claude
fino al pannello posa e' gia' incluso nel ramo science; i delta esclusivi degli altri due rami riguardano documenti,
review e riproduzioni storiche. `src/`, `scripts/train_jepa.py` e `scripts/slurm/train_ddp.sbatch` sono identici fra
il freeze `24139ab` e la consegna `8405a0b`. I job B19618 restano sul freeze, indipendenti dal successivo main.

Il delta rispetto al precedente main comprende frontend con finestra opzionale, target JEPA per regioni,
ingest/target/testa e pannello posa, rango assoluto opzionale, readout per canale, baseline TD e valutazioni posa,
submitter con stdin isolato. Copertura: default/readout bit-identici, inizializzazione e firma dati senza posa,
contratti loss e microbatch, resume/DDP, guardie cache/provenienza e bootstrap appaiato. Le riproduzioni
storiche `docs/reviews/repro_*` non sono una suite di regressione corrente e non vanno eseguite indiscriminatamente.

Conservare esplicitamente firma di B, responsabilita' GPT per sviluppo/lanci e Claude revisore, via specifico
B19618, copia indipendente per il resume e nessuna continuazione automatica a 49.046. La proposta Claude `a2753a6`
resta storica: soglie sul plateau, obbligo di crescita E2 e resume nella stessa cartella sono superati dal protocollo
attuale. Conservare la rettifica M3 accanto alla diagnosi iniziale: E3 legacy non fittava giunti invalidi; un IC che
include 0 non prova equivalenza. E2 e' disallineato meno allineato, in radianti, e non isola da solo la dinamica.

## Rettifiche descrittive del sintetico

Senza ricalcolare o modificare i risultati storici: nelle 32 configurazioni × 3 semi del JSON
`pose_head_synthetic_20261027.json`, la massima differenza fra i ratio finali dei semi e' 0,0198646646 (circa 0,02),
per schedule r3, q=0,5, B=28, P=1; non «al piu' 0,01». I tre ratio sono 0,5288286213, 0,5089639567 e 0,5203535567.

Il valore 1−q, salvato storicamente come `optimum_ratio_at_fixed_features`, e' la MSE irriducibile di popolazione
del rumore per giunto del generatore. Non e' il rapporto rispetto alla MSE iniziale del pannello finito, ne' un minimo
empirico garantito su quel pannello. q e' una frazione spiegabile nominale: la normalizzazione euclidea di W non
corregge esattamente la covarianza di LayerNorm. Queste rettifiche non cambiano curve, run o diagnosi.
La vecchia duplicazione del caricatore importlib resta una pulizia non bloccante distinta; non e' dichiarata risolta.

## Verifica tecnica

Eseguiti i 34 moduli CPU modificati rispetto a main: 477 passati, 1 saltato e 2 impediti dalla sandbox
(Gloo non poteva risolvere il loopback locale). I due test del pannello DDP rieseguiti fuori dalla sandbox passano
entrambi in 16,98 s: esito complessivo dei moduli cambiati 479 passati e 1 saltato. Nessuna correzione al codice
necessaria. I 64 test del rapporto e delle guardie eseguiti prima del lancio si sovrappongono a questa copertura:
i conteggi non vanno sommati come test distinti. Altri 22 test distinti stream/resume e training DDP passati: totale 501 passati e 1 saltato. Anche qui i test con worker/processi richiedevano il permesso fuori sandbox; nessuna modifica al codice.

## Esito operativo

Training 62582916 e dodici valutazioni sottomessi prima dei merge. Merge 2b8e352 da GPT evaluation e 2cb1644 da Claude,
senza squash/rebase. Conflitti solo nei registri/documenti, risolti per conservare le aggiunte di entrambi. Verificate
628/472/605 voci cronologiche dei tre snapshot: tutte presenti testualmente; le vecchie tabelle operative sono superate
dagli aggiornamenti correnti. `src/scripts/tests` restano identici a 8405a0b, quindi i test restano pertinenti.
Main viene avanzato alla consegna integrata e pubblicato su origin; nessun deploy verso Leonardo. Lo SHA del commit
che pubblica questa voce si ricava dalla storia Git, senza riferimento autoreferenziale.

