# Verifica avversaria R6-5 (guardie di processo senza test)

Verdetto: **il rilievo regge come constatazione, ma la gravita' scende da importante a minore.**

## Cio' che ho eseguito

1. Ho riletto il log delle mutazioni del revisore (`agents/r6-test/mutation_log.txt`). G1 e G4-G16 sopravvivono con 32 test passati, G2 e G3 vengono uccise. La lacuna nei test e' reale.
2. Ho fatto girare le guardie vere in un clone usa e getta (`git clone --shared`, detached a 9d34dc69, poi cancellato), con REVIEW_REF finto creato da `update-ref`. Script: `agents/verifica-r6-test-R6-5-1/e2e_bind.py`.
   - `--package` dal clone: b0640ca6...dcfa3 / 9d5811b6...50d6, uguali ai pin.
   - bind_review su un commit di review valido, poi require_binding: ACCEPTED.
   - Review con due blocchi: REJECTED ("precisely one"), quindi G1 funziona.
   - Commit non antenato di REVIEW_REF: REJECTED (CalledProcessError da `merge-base --is-ancestor`, check=True), quindi G4 funziona.
   - review_text manomesso nel binding: REJECTED ("Invalid review provenance"), quindi G5 funziona.
   - Binding falsificato con sha coerente ma senza blocco: REJECTED ("precisely one"), quindi G6 funziona.
3. Ho letto il codice delle altre guardie (contrastive_cn_restart.py:43-59, 105-177). G7/G16 (stato e schema del piano), G8 (source_digest), G9 (`git diff --quiet`, check=True), G10 (arm/repeat/fork), G11 (deadline = min(start+875, fine-25)), G12, G13 (cartella fresca), G14 (LAUNCHER in FILES) e G15 sono scritte correttamente e falliscono in modo chiuso.

## Gravita'

Oggi nessuna guardia lascia passare un pacchetto. Il codice resta congelato dalla review fino al checkout. bind_review, `--validate-only` e `--run` eseguono queste guardie per davvero prima del job GPU, e ogni loro errore blocca il lancio prima che le GPU vengano usate. Manca un controesempio che produca un esito INCONCLUSIVE o attribuito male. Il rischio riguarda solo una regressione futura, quindi la gravita' e' minore.

## Rilievo collaterale trovato provando lo scenario di run() proposto (minore)

contrastive_cn_restart.py:163-168 con contrastive_split_benchmark.py:31. Quando la deadline scade a meta' di una fase, `child.wait(timeout=...)` solleva `subprocess.TimeoutExpired`, che deriva da SubprocessError e non da TimeoutError. La tupla `except (RuntimeError, OSError, ValueError, TimeoutError)` di run() non lo intercetta, quindi `execution_failure.json` non viene scritto. Il job termina comunque con traceback e codice diverso da zero, e non c'e' alcun report: nessun esito viene attribuito male. Manca solo il record INCONCLUSIVE.
Prova eseguita: `checked_command(['sleep','5'], monotonic()+1)` -> "NOT CAUGHT by run() tuple: subprocess.TimeoutExpired".
Correzione minima: aggiungere `subprocess.TimeoutExpired` (oppure `subprocess.SubprocessError`) alla tupla di run().

## Correzione minima di R6-5

Basta un solo test con un repository git temporaneo, che riprenda e2e_bind.py: binding valido, due blocchi, commit non antenato, testo manomesso. Vanno aggiunte un'asserzione `LAUNCHER in FILES` e un test di run() con checked_command che supera la deadline e verifica `execution_failure.json`, che oggi fallirebbe per il rilievo collaterale. Non deve bloccare la release.
