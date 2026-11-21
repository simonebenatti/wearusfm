# Verifica avversaria di R4-I1 (test di compare deboli)

Esito: il rilievo REGGE. Gravita' giusta: **importante**, non bloccante. Non sono riuscito a confutarlo.

## Che cosa ho eseguito
- Ho rilanciato lo script del revisore `agents/r4-confronto/mutate.py` scrivendo le copie mutate in
  `agents/verifica-r4-confronto-R4-I1-2/mut/`. Ogni mutazione viene applicata una sola volta (lo script fa assert `count == 1`) a una copia di
  `scripts/diagnostics/contrastive_cn_restart_compare.py`. I 13 test originali vengono importati da quella copia.
  Ho ottenuto lo stesso risultato del revisore: KILLED M01, M02, M11, M12, M14, M16, M19, M21, M22, M23 (10 mutanti); SURVIVED M03-M10,
  M13, M15, M17, M18, M20, M24-M30 (20 mutanti su 30). Tempo: 64 s.
- Ho riletto il test (`tests/cpu/test_cn_restart_compare.py`, 128 righe). `compare()` (compare.py:214-304) non viene mai chiamata con
  osservazioni valide. L'unico percorso che passa da `main()` (test:72-80) usa directory inesistenti. Lancia quindi l'eccezione alla prima
  require (compare.py:220 o 221) e verifica soltanto che l'esito sia INCONCLUSIVE. Le guardie da 230 a 300 non vengono mai raggiunte da un
  test. La soglia `LIMIT` dentro `compare()` (riga 289), `first_exact` (riga 300) e il cablaggio `any_local`/`any_reduced` (righe 287-288)
  non sono coperti. `classify()` invece viene provata in isolamento.

## Perche' non e' bloccante
- Il rilievo riguarda la suite di test, non il codice che verra' eseguito. Ho confrontato le guardie presenti nel commit candidato con le
  30 mutazioni: le guardie ci sono tutte e la soglia e' 1e-6 relativo (compare.py:24 e 289). Un mutante sopravvissuto non e' quindi un
  controesempio sul run vero. Manca un input concreto che renda il diagnostico INCONCLUSIVE, o che faccia attribuire male l'esito, per
  effetto della debolezza dei test in se'.
- Il danno e' indiretto. La copertura scarsa non scopre i difetti di integrazione di `compare()`, che in un run a tentativo unico portano a
  INCONCLUSIVE dopo l'uso delle GPU. Due casi sono gia' stati trovati e trattati a parte come bloccanti:
  - B-1 (mmap con un Path, observe.py:144);
  - R4-B1 (`cuda_rng` di primo livello nel confronto del payload, compare.py:241,248), confermato da due verifiche indipendenti.
- Attenuante, dedotta dal codice e non eseguita. `contrastive_cn_restart.py:113-115,165-176` scrive tutto sotto `$WORK/wearusfm_runs/...`
  in una directory nuova e non cancella nulla:
  - `ordinary/` e `warm/` con checkpoint, tracce e `vectors.rank*.pt`;
  - `allocation.json`;
  - `execution_failure.json`.
  Se il confronto fallisce per un difetto di `compare.py`, le misure GPU restano. Il confronto, che gira solo su CPU, si puo' rifare dopo
  la correzione senza consumare di nuovo le GPU. Serve pero' una decisione registrata, perche' il piano prevede un solo tentativo.

## Precisazione sulla correzione
- Il test end-to-end proposto avrebbe preso R4-B1 soltanto con payload che riproducono la divergenza reale di `cuda_rng`: dispositivi 1-3
  del rank 0 a `seed*1_000_003+1` nel continuo, a `seed` nelle riprese. Su questo punto la proposta e' giusta, ma il test va scritto
  sapendolo.
- B-1 non lo avrebbe preso nessun test sul Mac. Con torch 2.14 `torch.load(Path, mmap=True)` funziona. Sul torch 2.2 del cluster solleva
  ValueError (serialization.py:1012-1014). Al test end-to-end va quindi aggiunto uno shim che riproduce quel controllo, oppure va eseguito
  il `--validate-only` sul cluster su file veri.

## Correzione minima proposta
Un test di `compare()` su quattro directory sintetiche: 4 rank e pochi parametri, con patch di `PARAMETERS`, `ARCHIVE`, `ENDPOINT`,
`FORK_SHA`, `REFERENCE_REPORT` e dell'identita' del modello.
- I file `vectors`/`observation` vanno scritti con l'osservatore vero, non costruiti a mano.
- `torch.load` va avvolto in uno shim che rifiuta `mmap=True` quando f non e' una stringa.
- Il payload contiene `rank_rng` e `cuda_rng` realistici.
- Serve un caso per ciascun esito:
  - SUPPORTED;
  - PARTIAL_SUPPORT, una volta per il primo update e una per l'endpoint;
  - HYPOTHESIS_UNSUPPORTED, con differenza locale (anche solo di zero con segno), con eccesso appena sopra 1e-6 e senza scarto;
  - INCONCLUSIVE per la fedelta' dell'ordinaria.
- Il test deve fallire almeno sui mutanti M03-M06, M15 e M25.
