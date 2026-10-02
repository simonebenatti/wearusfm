# Review esterna del repo — istruzioni per il revisore (02/10/2026)

Sei un revisore indipendente del repo `simonebenatti/wearusfm` (foundation model per sEMG; documentazione in italiano).
**SOLA LETTURA:** non modificare file, non creare branch, non fare commit ne' push, non aprire PR.

Leggi prima `CLAUDE.md`, `docs/fm_emg_reference_v10.md`, `docs/piano_operativo_v10.md` e `docs/decisioni.md`. Poi fai la review di tutto il lavoro, con
attenzione particolare agli ultimi giorni (`git log 62d523f..HEAD`): ingest (`src/wearusfm/ingest/`, `scripts/ingest_*.py`), gate D8
(`src/wearusfm/gate/`, `results/step3/`), misure RVQ (`results/step1bis/`), manifest e split (`src/wearusfm/data/pretraining_manifest.py`,
`scripts/build_manifest.py`, `scripts/make_splits_draft.py`, `splits/`), `docs/proposta_manifest_d9.md`, `docs/fatti_da_verificare.md`,
`docs/dataset_access.md`.

Cerca:
1. difetti reali nel codice: dati falsati o persi, errori silenziosi, test che non verificano cio' che dicono;
2. numeri nei documenti che non coincidono con i JSON in `results/` e `splits/`;
3. deduzioni scritte come fatti, decisioni registrate come prese quando erano proposte;
4. contraddizioni con `CLAUDE.md` o con decisioni precedenti;
5. lacune che rendono prematura la firma di D9 (manifest).

Puoi eseguire i test CPU (`python3 -m pytest tests/cpu -q`); il codice GPU non gira. Per ogni rilievo: file:riga, gravita' (alta/media/bassa), cosa
non va, prova (riga di codice, numero a confronto, output di un test), e se e' verificato o solo sospetto. Ordina per gravita'. Rispondi in italiano.

## Gia' trovato e corretto (non ripeterlo; segnala solo se la correzione e' sbagliata o incompleta)

Dalle review interne del 01-02/10/2026 (commit fra parentesi):
- **quantizzazione int16:** un solo NaN dava scala 1 e tutti i canali a zero, validi per il QC; ora `to_int16` rifiuta NaN e infiniti (543edc6);
- **rinfresco di `$SCRATCH`** (`scripts/slurm/refresh_scratch.sbatch`): non toccava un bersaglio-file, le voci nascoste al primo livello e il contenuto
  dietro i link; ora si', con confronti numerici (543edc6);
- **costruttore del manifest:** split per sessione (`test_sessions`), controllo che gli split siano disgiunti, soggetti senza sessioni e sessioni di
  test non trovate segnalati, `manifest-v1` rifiutato con split in bozza o incompleti, ordine canonico prima dei pesi (l'hash dipendeva dall'ordine di
  scoperta), percorso del file di split fuori dall'hash (543edc6); salti dell'asse dei tempi contati per sessione (7153585);
- **emg2pose:** le 3.539 registrazioni del test ufficiale per fasi nuove escluse dal pretraining per sessione (decisione di Simone, 543edc6);
- **Zhang:** test di allineamento campione per campione fra canali a 4 kHz ricampionati e a 2 kHz (543edc6); una sequenza con durate incoerenti si
  esclude invece dell'intera sessione, 1-2 campioni di differenza fra canali dello stesso tasso si tagliano (8d02f5b); riconoscimento
  dell'intestazione nello script di ispezione (2b429fa);
- **gate D8:** filtro a muro al posto del Butterworth (firmato, 954cc3f); 20 finestre per soggetto e non per segmento, un soggetto alla volta in
  memoria (954cc3f); finestre copiate e non viste (fuori memoria, c730b18); i test usano `G.CASES` e controllano il taglio sotto la Nyquist (543edc6);
- **ingest di emg2pose:** il tar si scorre una volta sola (6e73759) ed estraendo col TarInfo, non col nome (3f374fd);
- **ingest di DB10:** budget di tempo controllato prima di ogni soggetto, report parziale a ogni soggetto (543edc6);
- **downloader di Dataverse:** nome e dimensione dell'originale per i file tabulari (543edc6);
- **`manifest_table.py`:** Kaifosh fuori di default, come nella bozza (543edc6);
- **documenti:** citazioni, numeri e interpretazioni corretti in `decisioni.md`, `fatti_da_verificare.md`, `dataset_access.md` e
  `results/step3/RIEPILOGO.md` (dabe4f8 e commit di questa nota); CapgMyo: la seconda applicazione registrata come senza effetto (b36819a); la frase
  «i dati difficili da riottenere restano anche su $WORK» corretta (08f4b6e).

## Gia' noto e ancora aperto (non e' un rilievo nuovo)

- **Bozza D9 in revisione:** gli 8 punti elencati in testa a `docs/proposta_manifest_d9.md` (ore contate con i soggetti di test, sovrapposizione fra i
  dataset Meta, tetto rispetto a 2D, origine/presentata, interpretazioni da marcare, salti dell'asse dei tempi, fatti da firmare, numeri piccoli). La
  bozza si riscrive coi numeri del costruttore sui dati veri.
- **In corso sul cluster:** misura dei tratti costanti di DB10 ed emg2pose; poi V2 su DB10 e prima bozza vera del manifest.
- **Fatti raccolti ma non firmati:** 1 (DB8), 2, 10b, 10c, 10d.
