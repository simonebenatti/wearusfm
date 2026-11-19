# Review Claude: adeguamento dei target della linea 3 a RMS100/fast125 (`0efdc29c`)

19/11/2026. Revisore: Claude. Linea 3: codice, protocollo e lettura spettano a GPT scientifico, i lanci a 3-launcher.
Claude non lancia job.

## Richiesta

Prompt di GPT passato da Simone. Ramo `codex/contrastive-target-windows-20261118`, con questa precisazione: «qui verificare
soltanto l'adeguamento contrastivo». Diff scientifico da `e388a1de` a `0efdc29c`. Si chiede: «Verifica in un unico passaggio» i
sei punti e «Riporta prima eventuali blocchi dimostrati; note non bloccanti insieme, senza chiedere giri ripetuti su parti
invariate». Mandato di Simone per la linea 3, da `CLAUDE.md`: «sul punto 3 passiamo a 100/100».

Perimetro: il diff `e388a1de..0efdc29c` e i tre commit successivi. Su questi ho verificato quattro cose:
- **Merge di main `afb89b45`:** non tocca `src`, `scripts` e `tests`.
- **Merge della diagnostica `65bc6022`:** porta i suoi cinque file identici a `365cb1e`.
- **`438c8d65`:** non tocca codice ne' test. Aggiunge:
  - le evidenze della suite;
  - il prompt per Claude;
  - voci di registro e di `decisioni.md`.

  Inoltre riordina il registro e aggiorna `local_validation.json`.
- **Registro:** quello della punta conserva testualmente ogni voce dei commit precedenti. Rispetto a prima sposta un blocco,
  toglie il titolo doppio «## 2) Problemi trovati» e aggiunge due voci nuove.

Mio codice delle finestre: `0d91297` e' antenato di main e della punta.
- **File identici.** Dei 30 file che tocca, 26 sono identici byte per byte in main e alla punta.
- **`run.py`.** In main e' identico. Alla punta contiene le sue modifiche piu' quelle contrastive del ramo: il merge
  `e388a1de` aggiunge a main esattamente le righe del ramo contrastivo, e la consegna non tocca il file.
- **Documenti.** `CLAUDE.md`, registro e `decisioni.md` hanno voci successive.

Metodo:
- **Workflow di 38 agenti:**
  - un revisore per ciascuno dei sei punti e uno per test e ricevuta della suite;
  - da uno a tre scettici indipendenti per rilievo, secondo la gravita';
  - un critico di completezza.
- **Controlli miei:**
  - quattro lacune del critico: pin della review, matrice invariata, checkout, stato di Leonardo;
  - il rilievo P6-N3;
  - i test riprodotti, il ledger e due letture di Leonardo.
- **Rilettura della bozza.** L'hanno controllata due verificatori indipendenti, su circa 255 affermazioni: 19 punti da
  correggere, tutti corretti qui.
- **Prove.** Nella cartella `claude_review_contrastive_targets_amendment_20261119_evidence/`.

## Esito

**Nessun blocco dimostrato.** La preparazione e' approvata come tale: piano `REVIEW_REQUIRED`, non eseguibile, nessun invio.
Le condizioni R-1...R-4 vanno soddisfatte nella release, prima del primo invio della nuova coorte. M-1 e le note si correggono
senza un nuovo giro di review.

| Punto | Esito | Rilievi |
|---|---|---|
| 1. Storico non reinterpretato, 100/125 esplicito | conforme | N-2, N-6, N-7 |
| 2. Nessuna accettazione di evidenze vecchie | conforme per le ricevute; le autorizzazioni restano procedurali | R-2, R-4, M-1, N-2 |
| 3. Confronti appaiati, C0−T100-G a tre fattori | conforme nei config; parita' numerica ancora aperta | R-1, N-1 |
| 4. Coefficiente di λ come prior, λ da ricalcolare | conforme | nessuno |
| 5. Preventivo e audit da 2 h | numeri esatti; checkout e presentazione da completare | R-3, R-4 |
| 6. Diagnostica prima del rilascio | conforme come cancello; diagnostica conclusa alle 11:39, da leggere | R-1, N-3 |

## Condizioni per il rilascio

**R-1 (punti 3 e 6). Decidere la matrice dopo la lettura della diagnostica.**
- **Matrice invariata.** Il nuovo preflight ha le stesse nove celle (modo, braccio) e le stesse dipendenze di quello fallito.
  Cambiano solo i tre prerequisiti procedurali dell'audit.
- **Ripresa e parita' invariate:**
  - `scripts/contrastive_split_benchmark.py` e' invariato da `8905e346`;
  - il cancello di parita' a peso zero resta invariato: `scripts/contrastive_benchmark_readout.py:395`, `rel_tol` 1e-4 e
    `abs_tol` 1e-6 su `total` e `grad_norm`, con `amp_bf16` come prima.
- **Esito storico.** Quel cancello e' fallito su 146 norme per CN e 135 per CR. GPT ha scritto nel registro del 18/11 che il
  test di ripresa «confronta due training indipendenti fin dall'inizio».
- **Appaiamento solo nei config.** I tre bracci nuovi differiscono solo per `contrastive.arm`, quindi CN−C0 e CR−C0 sono
  appaiati nella ricetta, non ancora sul piano numerico.
- **Diagnostica da stato comune:**
  - **Prima.** Alle 10:07 non era ancora deployata.
  - **Invii.** Alle 11:32 GPT ha inviato il checkout e le tre celle: 64835236, 64835286, 64835297 e 64835304.
  - **Esito SLURM.** Tutti `COMPLETED 0:0`, l'ultimo alle 11:39:57. Il ref `deploy/contrastive-common-20261118` e il checkout
    sono a `365cb1e`.
  - **Stati dei rapporti,** letti alle 11:47 senza interpretarli:
    - CUDA `MEASURED_FOR_REVIEW`, con i quattro profili completi e nessun fallimento;
    - riprese CN e CR `CONTROL_INPUTS_VERIFIED`, con struttura finale esatta e differenze numeriche presenti.
  - **Impronte SHA-256:** `06a2be6b...` (CUDA), `cce7b198...` (CN), `899d3ff9...` (CR).
- **Richiesta:**
  - la lettura spetta a GPT scientifico, con le regole pre-registrate in `db66b507`;
  - la release dica poi se cancello di parita' e disegno split restano come sono o cambiano, e con quale regola scritta prima.

  Senza questa decisione le cinque celle di benchmark (tre complete e due di ripresa, 10 GPU-ore) ripeterebbero una matrice
  negativa prevedibile: e' il retry cieco escluso dal punto 6.

**R-2 (punto 2). Il pin della review non e' legato alla nuova coorte.**
- **Cosa controlla il dispatcher.** `contrastive_job.build_command` (`scripts/contrastive_job.py:62-69`) verifica:
  - lo stato `APPROVED_FOR_PREFLIGHT`;
  - modo e tipo;
  - il digest dei sorgenti;
  - i secondi allocati;
  - lo SHA del file di review indicato nel job, qualunque esso sia (`checked_file`, righe 22-31).

  L'audit, prima cella, non ha release.
- **Sonda del critico.** Un job di audit nuovo che pinna la review del 17/11
  (`docs/reviews/contrastive_preflight_release_20261117.md`) supera i controlli di review e config. Si ferma solo sugli input,
  che sul Mac mancano.
- **Conseguenza.** Le ricevute vecchie sono rifiutate dal codice. Le autorizzazioni vecchie, invece, solo dalla procedura.
- **Correzione**, una delle due:
  - la review pinnata al rilascio e' quella del pacchetto nuovo e cita i sorgenti `643a8937...` e il piano scientifico
    `plan.json` `6081832...`, che `preflight_plan.json` (impronta `ea2cef0f...`) pinna come `science_plan`;
  - il dispatcher fa questo controllo. In questo caso cambia il digest e serve una nuova review del codice.

**R-3 (punto 5). Preventivo da completare prima della presentazione a Simone.**
- **Checkout:**
  - **Dichiarati e storici.** Dichiarati 4 CPU, 30 GiB e 240 s. Il launcher ha 1 CPU, 2G e 10 min di default
    (`scripts/slurm/campaign_checkout.sbatch:5-7`). Tutti i checkout della linea sono partiti con `--cpus-per-task=4 --mem=2G`,
    e tutti con `--time=00:04:00` tranne il primo (64652044, 5 minuti).
  - **Override mancanti.** Il piano rende obbligatorio solo l'override dell'audit. Con 4 CPU e il tempo di default il totale
    diventa 138,666667 core-ore, sopra il tetto di 137,066667.
  - **Correzione.** Rendere obbligatori anche gli override dei checkout e dichiarare 2 GiB. Altrimenti va motivato perche' 30
    GiB, che occupano tutta la memoria per utente della corsia seriale (30.800 MB).
- **Ore locali e residuo.** Mancano le ore locali e il residuo del budget, che `CLAUDE.md` chiede in ogni richiesta di
  conferma. Nella contabilita' storica il billing coincide con le CPU: celle GPU billing=32, audit 8, checkout 4 con 2G. Quindi
  137,066667 core-ore sono anche le ore locali solo con i checkout a 2 GiB.
- **Cumulato del confronto contrastivo** (campagna `CONTRASTIVE-v2-20261116`, training escluso):
  - GPU-ore: 5,284444 + 6 + 14 = 25,284444;
  - core-ore: 51,371111 + 48,266667 + 137,066667 = 236,704444.

  Gli addendi sono la catena storica, il tetto della diagnostica e il nuovo preflight. Il consuntivo reale della diagnostica
  e' molto sotto il suo tetto: tre celle da 2-5 minuti.
- **Margine di SLURM.** Il «totale massimo» somma i limiti richiesti. Va scritto che non comprende il margine di SLURM:
  OverTimeLimit 3 min e KillWait 180 s.
- **Durata dell'audit.** Stima del revisore: 3.900-4.000 s, con un limite superiore di 6.032 s. I 7.200 s bastano.
- **Manutenzione.** Leonardo e' fermo dal 23/11 alle 08:00 al 25/11 alle 08:00, e la catena ha dipendenze: va programmata
  tenendone conto.

**R-4 (punti 2 e 5). Stato e durata dell'audit da verificare alla release.**
- **Eccezione solo dichiarata.** La non eredita' del TIMEOUT e' scritta nel piano, ma nessun codice la impone:
  - l'audit non ha un limite di tempo interno e scrive il rapporto solo alla fine (`scripts/contrastive_audit.py`);
  - nessun lettore controlla stato SLURM o durata (`contrastive_run.validate_audit`).
- **Caso storico.** Un audit oltre 7.200 s ma dentro il margine di SLURM (OverTimeLimit piu' KillWait) verrebbe accettato. E'
  il caso di 64652559: 3.787 s su 3.600, quindi 7 s oltre i tre minuti di OverTimeLimit.
- **Correzione.** Nella release della calibrazione richiedere, per l'audit, sacct `State=COMPLETED`, `ExitCode 0:0` ed
  `Elapsed` non oltre 7.200 s.

## Rilievo minore

**M-1 (punto 2, test). Tre barriere di secondo livello senza test sul codice corrente.**
- **Quali sono:**
  - `config_sha256` in `validate_audit` (`scripts/contrastive_run.py:100`);
  - lo stesso controllo in `read_preflight` (`scripts/contrastive_benchmark_readout.py:333`);
  - la guardia della ricevuta in `authenticate_calibration` (`scripts/contrastive_prepare.py:384-385`).
- **Prova.** Il revisore le ha disattivate tutte e tre insieme e i 337 test contrastivi passano ancora. Due scettici hanno
  riprodotto le due barriere dell'audit, un terzo la guardia della ricevuta da sola.
- **Perche' oggi regge.** Le evidenze vecchie sono fermate prima da barriere testate:
  - l'audit, dalla guardia delle sorgenti;
  - la calibrazione, dalla barriera di ricetta (`scripts/contrastive_prepare.py:273-274`).
- **Test troppo larghi:**
  - `test_historical_audit_does_not_qualify_new_sources` accetta qualunque `ValueError`, perche' manca `match`;
  - `test_pinned_historical_receipt_still_requires_authenticated_recalculation` gira solo nel lettore congelato `8905e346`.
    Il suo commento «Current 100/125 sources must not inherit that exception» non e' quindi verificato sul codice corrente.
- **Correzione**, tutta nei test, che non entrano nel digest:
  - l'audit storico rietichettato al digest corrente, con `match` su «audit stream/config differs» e su «different
    inputs/config/source»;
  - un test sintetico per la guardia della ricevuta;
  - `match="current execution source inventory"`.

## Note

- **N-1 (punto 3).** Il protocollo, par. 6 (`docs/contrastive_comparison_protocol_20261116.md:132`), chiama ancora C0−T100-G
  «controllo descrittivo di compatibilità». Per la nuova coorte e' solo la lettura descrittiva di una differenza a tre fattori:
  impalcatura contrastiva, RMS 100 e spettro rapido 125/v3. Basta una frase nell'emendamento.
- **N-2 (punti 1 e 2).** Tre documenti operativi storici non rimandano all'emendamento:
  - `docs/contrastive_preflight_execution_20261116.md`, con l'audit da 1 h e l'eccezione `audit_recovery`;
  - `docs/contrastive_handoff_20261117.md`;
  - `docs/prompt_3_launcher_contrastive_readout_20261118.md`.

  3-launcher lavora da questi documenti: conviene la stessa intestazione del protocollo. Il rischio e' basso, perche' i job
  vecchi sono rifiutati per sorgenti e secondi.
- **N-3 (punto 6, ramo della diagnostica).** `db66b507` ha sostituito la regola generale «Le coppie che richiedono il profilo
  mancante restano inconclusive» con una valida solo quando manca il profilo deterministico
  (`docs/contrastive_common_diagnostics_20261118.md:22`). In questa esecuzione non ha effetto, perche' i quattro profili sono
  completi. Da ripristinare solo se il testo viene riusato.
- **N-4 (test). Valori e campi non protetti:**
  - Il puntatore `PLAN` al piano nuovo (`scripts/contrastive_prepare.py:30`) e il testo descrittivo del readout
    (`scripts/contrastive_benchmark_readout.py:111`) non sono protetti da test. Riportati ai valori vecchi, i test contrastivi
    e mirati restano verdi (fino a 322 casi); nessun altro test li legge.
  - I campi target di `plan.json` non sono confrontati con `build_review_configs`.
  - `test_modified_historical_receipt_is_rejected[...]` alla punta si fermano gia' alla barriera di ricetta. Non provano piu'
    la logica di manomissione, che resta coperta dal test sintetico.
- **N-5 (ricevute). Prove non riproducibili:**
  - `local_validation.json` non ha uno script che lo generi;
  - la ricevuta della suite non pinna i log originali (`cpu_part0-3.log`, `cpu_ddp.log`, `recovery_remaining.log`);
  - i 121 test mirati non hanno un comando.
- **N-6 (punto 1).** `validate_recipe` accetta un config nuovo senza `target_version`, che per default diventa v3. Il
  comportamento esisteva gia' e a valle lo fermano i pin esatti di audit, CUDA, coorte e release.
- **N-7 (punto 1).** `run_frozen` legge le evidenze da `8905e346`, non dalla punta. L'integrita' delle ricevute storiche alla
  punta resta affidata a `git diff`, che oggi mostra solo aggiunte.

## Verifiche positive, per punto

**1.**
- **Lettura storica:**
  - `config_from_dict` applica `with_legacy_target_windows` (`scripts/contrastive_prepare.py:46`). Aggiunge solo le due
    finestre mancanti, non tocca `target_version` e non modifica l'ingresso.
  - Sono letti come 25/50/v1: il T100-G (`f9d03184...`), i 4 config di review storici, i 36 config delle coorti del 17 e del
    18/11 e 6 config di run.
- **Config nuovi:**
  - I quattro config nuovi differiscono dagli storici solo per le due finestre e per `target_version`.
  - `build_review_configs` li rigenera byte per byte dal baseline, che verifica 25/50/v1 (`scripts/contrastive_prepare.py:117`).
- **Uso a valle:**
  - Ogni modalita' del dispatcher usa il config pinnato: audit, calibrazione, CUDA, full, split e lettura.
  - La fixture del CUDA e i config calibrati a 544, 576, 5.000 e 19.618 restano 100/125/v3.
  - Nessun `_config_with_defaults` o `LoaderConfig()` di default sul percorso contrastivo.

**2.**
- **Riferimenti alla coorte storica.** Nei file nuovi ce ne sono solo due, entrambi di politica: copertura dal manifest e
  prior dei gradienti. Nessuna λ, ID, stato o SHA di ricevute vecchie nella cartella nuova.
- **Evidenze vecchie rifiutate sul codice corrente:**
  - l'audit, alla guardia delle sorgenti; rietichettato, alla barriera di ricetta;
  - il run e la ricevuta della calibrazione 64701482 (λ 0,1194 e 0,3304), alla barriera di ricetta
    (`scripts/contrastive_prepare.py:273-274`), e dalla guardia anche se chiamata direttamente;
  - 21 job e 16 release storici;
  - benchmark, CUDA e ricevuta preflight bf16.
- **Eccezione del TIMEOUT.** Esiste solo come documento: nessun codice la legge.

**3.**
- La sezione `evaluation` dei due piani e' identica: coppie, endpoint, pesi, bootstrap, margini e regola Monte Carlo.
- `contrastive_compare.py` e' invariato. Ogni differenza fra i piani e' dichiarata: aggiunte, piu' la riscrittura del testo
  di `scope`.
- C0−T100-G non e' calcolato da nessun codice. Compare solo come lettura descrittiva dichiarata:
  - limitazioni e `target_recipe_amendment` del piano;
  - l'emendamento;
  - il protocollo, par. 6.

**4.**
- **Coefficiente.** Il coefficiente 0,1179564123141922 viene dal file pinnato ed e' dichiarato prior.
- **Formula.** 0,1 × coefficiente × mediana sui primi 32 batch informativi. Nessun clipping. Una λ fuori da [1e-4, 1] ferma la
  coorte (`scripts/contrastive_prepare.py:349`).
- **Ingresso delle λ.** Le λ entrano solo da `calibrated_configs`, che ricalcola tutto da una calibrazione della ricetta
  nuova.
- **Cancelli invariati.** Regola dei rapporti invariata; il confronto dei primi otto update e' solo descrittivo.

**5.**
- **Totali.** 9 celle + 4 checkout = 13 allocazioni, 14 GPU-ore, 137,066667 core-ore.
- **Allocazioni ammesse** (`ALLOCATED_SECONDS`, `scripts/contrastive_job.py:17`): 7.200 s per l'audit, 3.600 per la lettura,
  1.800 per le celle GPU. Un audit senza override, una lettura da 2 h o una cella GPU con un altro tempo falliscono prima di
  leggere dati.
- **Limiti delle code.** 30 GiB = 30.720 MB, entro i 30.800 della corsia seriale.

**6.**
- **Diagnostica intatta.** Il merge conserva identici i file della diagnostica e i loro pin.
- **Niente misure miste.** Il digest della punta (`643a8937...`) e' diverso da quello atteso dalla diagnostica
  (`46938f31...`). Lanciata dal nuovo albero, la diagnostica si ferma subito; nel proprio checkout passa il controllo. Cosi'
  e' andata oggi: le celle sono girate dal checkout a `365cb1e`.
- **Soglie.** Nessuna soglia nuova o allentata: `contrastive_cuda_check.py` e' byte per byte identico, e nel readout cambia
  solo la stringa descrittiva.
- **db66b507.** Tocca solo documenti.

**Test e suite.**
- **Test eseguiti da me sulla punta:** 173 diretti e 32 indiretti, tutti passati.
- **Ricevuta della suite di GPT:**
  - le impronte dichiarate corrispondono ai file;
  - 3.723 test distinti su 233 file, gli stessi raccolti oggi alla punta;
  - i 131 nodi non passati nel primo giro stanno tutti nei 13 file rieseguiti, dove passano 263 casi su 263;
  - i fallimenti erano dovuti al disco pieno (ENOSPC);
  - fra `65bc6022` e `438c8d65` non cambiano ne' sorgenti ne' test.

## Limiti di questa review

- **Esecuzione.** Niente CUDA, NCCL o Leonardo: il percorso GPU e la lettura con i checkpoint sono verificati solo leggendo il
  codice.
- **Release futura.** Il builder della release non esiste ancora: una copia futura di autorizzazioni vecchie non e'
  verificabile oggi (da qui R-2).
- **Suite.** Non ho rieseguito la suite completa per lo spazio disco. La ricevuta di GPT l'ho controllata per coerenza.
- **Lacune del critico.** Entrano tutte e nove nella review:
  - 1 in R-2; 2, 3 e 8 in R-1; 4 e 5 in R-3;
  - 6 in N-5; 7 in N-2; 9 in M-1.

  Ne ho verificate da solo quattro: pin della review, matrice invariata, checkout e stato di Leonardo. Le altre poggiano sulle
  prove del critico, senza un secondo giro di scettici.
- **Rapporti dei diagnostici.** Ho letto solo stati e impronte, non i contenuti: la lettura scientifica spetta a GPT.
- **Prove degli agenti.** Sono nella cartella delle evidenze (`rapporti_agenti/`, `script_agenti/`, `sintesi_workflow.json`,
  `verifica_bozza/`) e in `~/wearusfm_local/reviews/contrastive_targets_review_20261119/`.

## Prompt per GPT

Per la chat GPT dell'adeguamento target (worktree `wearusfm-contrastive-targets`):

```text
Review Claude dell'adeguamento target della linea 3 (0efdc29c, punta 438c8d65): nessun blocco dimostrato. La preparazione e' approvata come tale (REVIEW_REQUIRED, nessun invio). Ramo origin/worktree-claude-training-engine, file docs/reviews/claude_review_contrastive_targets_amendment_20261119.md, prove nella cartella _evidence/ accanto (38 agenti con verifica avversaria, bozza riletta da due verificatori, 205 test riprodotti, due letture di Leonardo del 19/11).

Prima del primo invio della nuova coorte, nella release:
- R-1: la diagnostica da stato comune e' conclusa (64835236, 64835286, 64835297, 64835304 COMPLETED 0:0 alle 11:39; CUDA MEASURED_FOR_REVIEW con quattro profili, riprese CONTROL_INPUTS_VERIFIED). Dopo la lettura di GPT scientifico, dire se il cancello di parita' a peso zero (rel 1e-4, abs 1e-6, bf16) e il disegno split, entrambi invariati, restano o cambiano, con una regola scritta prima. Altrimenti la matrice negativa si ripete.
- R-2: legare il pin review dei job al pacchetto nuovo (sorgenti 643a8937..., plan.json 6081832..., preflight_plan.json ea2cef0f...). Oggi build_command accetta qualunque review pinnata.
- R-3, preventivo:
  - override obbligatori anche per i checkout (4 CPU, 2 GiB, 4 min): con il tempo di default si arriva a 138,666667 core-ore, sopra il tetto;
  - ore locali e residuo del budget;
  - cumulato della campagna (25,284444 GPU-ore, 236,704444 core-ore);
  - margine di SLURM;
  - manutenzione dal 23 al 25/11.
- R-4: per l'audit, sacct COMPLETED 0:0 ed Elapsed <= 7200 prima della release della calibrazione.

Senza nuovo giro:
- M-1: tre barriere senza test sul codice corrente (config_sha256 di validate_audit e read_preflight, guardia della ricevuta in authenticate_calibration) e match nei test storici;
- N-1: C0-T100-G non e' un controllo di compatibilita';
- N-2: intestazione nei documenti operativi storici;
- N-4: test di PLAN, del testo del readout e dei campi target del piano;
- N-5: produttore di local_validation e pin dei log della suite;
- N-6 e N-7: note.
```

Per la chat di GPT scientifico della linea 3 (worktree `wearusfm-logcontrast`), se e' un'altra chat:

```text
Diagnostici da stato comune conclusi su Leonardo:
- job: checkout 64835236, CUDA 64835286, ripresa CN 64835297, ripresa CR 64835304, tutti COMPLETED 0:0 fra le 11:32 e le 11:39 del 19/11, dal checkout a 365cb1e;
- stati dei rapporti: CUDA MEASURED_FOR_REVIEW con i quattro profili completi; riprese CN e CR CONTROL_INPUTS_VERIFIED, struttura esatta, differenze numeriche presenti;
- report.json SHA-256: CUDA 06a2be6b43f2c6ccf734ef5536d3351df30b321846344a82687155a45852dc5f, CN cce7b19850a1705c3c60175fd6d12d29ce61cb132ac2d491c6b592352b678b19, CR 899d3ff9a012bf24cf64d4d5a52bad6481d192ef704b8ab2146bf9651c3585c7.

Claude ha letto solo stati e impronte, non i contenuti. La lettura e' tua, con le regole pre-registrate in db66b507.
Nota N-3 della review Claude del 19/11: in db66b507 la regola generale «Le coppie che richiedono il profilo mancante restano inconclusive» e' stata sostituita da una valida solo per il profilo deterministico. Qui non ha effetto (quattro profili completi): ripristinala solo se il testo viene riusato.
Dalla tua lettura dipende la condizione R-1 della nuova coorte RMS100/fast125: dire se il cancello di parita' a peso zero e il disegno split restano o cambiano, con una regola scritta prima della release.
```
