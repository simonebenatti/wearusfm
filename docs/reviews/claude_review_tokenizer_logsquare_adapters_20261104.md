# Review di Claude — adattatori di valutazione T100-LQ (GPT c8f5c12, diff da 6d5b353) — 04/11/2026

Richiesta di Simone: «Rivedi la consegna GPT c8f5c12 [...] Pubblica verdetto e registro sul tuo ramo. Se approvi, indica
se sono soddisfatti i prerequisiti per il commit dedicato APPROVED_FOR_TRAINING». Prompt completo:
`docs/prompt_claude_logsquare_adapters_20261104.md`. Solo review: nessun lancio, nessuna contrastiva.

**Verdetto: APPROVATO.** Nessun blocco reale. Gli adattatori e i loro gate di risorse sono approvati con le note A1-A5
(operative, non bloccanti). **I prerequisiti per il commit dedicato `APPROVED_FOR_TRAINING` sono soddisfatti:**
R1, R2, controller e disegno erano gia' approvati (1d6f96a); con questa consegna gli adattatori sono implementati e
rivisti. Il commit di rilascio resta da fare ed e' distinto da questa consegna preparatoria; il training non e' ne'
rilasciato ne' lanciato. Le estrazioni restano senza tetto finche' rho_infer non e' misurato.

| Componente | Verdetto |
|---|---|
| R1-a, R1-b, N1-N4 | **RECEPITI** (readout, piano esecutivo, `training_plan.execution_notes`) |
| `logsquare_evaluation.py` + launcher (D3, pin, ricevute) | **APPROVATO** |
| `logsquare_infer_benchmark.py` + launcher (D1, D2) | **APPROVATO**, note A1-A3 |
| `logsquare_pose_readout.py` + launcher (fit solo LQ, quattro celle, R1-b) | **APPROVATO** |
| `logsquare_probe.py` + launcher (fit solo LQ, otto aggregati, provenienze) | **APPROVATO** |
| Gate di risorse: benchmark 0,5 GPU-ore / 4 core-ore; posa 16 core-ore; sonde 34,3333 core-ore; checkout 0,3333 ciascuno | **APPROVATI**; riuso dei tempi storici delle sonde **ACCETTATO** con le condizioni A5 |

## Cosa ho eseguito

- Checkout temporaneo detached di c8f5c12 (un solo commit dopo 6d5b353, 33 file, +5.249/−40), poi rimosso. Letti piano
  esecutivo, critico interno, readout e protocollo aggiornati, handoff, i quattro adattatori, i quattro launcher, i quattro
  file di test, i piani (`extraction`, `inference`, `pose_readout_5000/19618`, `probe_19618`), le tre politiche di risorse,
  `component_review.json`, `local_validation.json`.
- **Test rieseguiti con i due comandi di GPT:** 276 regressioni passate in 135,76 s e 141 test degli adattatori in 15,85 s
  (**417**, come dichiarato). `bash -n` dei sei launcher LOGSQ pulito.
- **Invarianza:** `git diff --stat 6d5b353 c8f5c12` e' vuoto su controller e launcher del training, collaudo, analisi
  offline, `src/`, `tokenizer_evaluation.py`, `tokenizer_pose_readout.py`, `tokenizer_probe.py`, `pose_eval.py`,
  `channel_readout.py`, `calibration_probe.py`, `train_jepa.py`, `pose_temporal_readout.py`, `pose_e3_b.py`,
  `pose_dynamics.py`. La copia della mia review 1d6f96a e' identica (diff vuoto); `component_review.json` la pinna
  (2baf4431…). I 102 sorgenti scientifici del piano di training hanno gli stessi hash a c8f5c12.
- **Pin locali:** 301 pin a file del repo verificati (sorgenti dei piani di estrazione e inferenza, 110 identici fra i due;
  `sources` dei piani di posa e sonde; piani e report storici; archivi locali dei sei aggregati storici; politica delle
  risorse storica; `training_plan`, `resource_proposal`, `component_review`; `local_validation.source_hashes`): 0
  discrepanze. I pin remoti (checkpoint, cache, ricevute su Leonardo) non li ho aperti.
- **Piani storici con i sorgenti attuali:** `logsquare_probe.historical_plan` accetta il piano storico delle sonde
  (`T.validate_plan`, sorgenti numerici e ricetta identici, `jhash` uguale al `plan_content_sha256` dei sei report);
  `logsquare_pose_readout.validate_plan` + `load_historical` accettano i report storici a 5.000 (70fd080) e 19.618
  (b315dad) e il bootstrap appaiato ricalcolato coincide.
- **Percorsi reali verificati nel codice condiviso** (`src/wearusfm/harness`): `pose_eval.extract` apre una registrazione
  alla volta con `default_rng([seed, ri])`, un solo batch ≤ 10 finestre, `encode → latent_mean → pose_head` sotto
  `no_grad` senza autocast (FP32): la selezione «prime due registrazioni» riproduce esattamente le finestre della cache
  storica e il `PoseProxy` segue la stessa sequenza. `extract_channel_readouts` calcola la scala dalle prime 10 finestre
  di ogni chiave (sessione per P1, soggetto per P2), applica l'autocast bf16 intorno a `readouts` (encoder e decoder via
  `forward_layers`) e ritaglia a 4 s: EPN 5 s → due ritagli, UCI e P1 1 s → uno. Il benchmark strumenta proprio `readouts`
  dentro quel contesto, con lo stesso `inp` per i due bracci; `first64_with_scale_completion` conserva le prime 10 righe
  di ogni soggetto, come la scala di produzione. I lettori train-only coincidono con la produzione: `load_epn612`
  (`fix_length` a 5 s, soggetti `<root>/<user>`), `load_uci_emg` (1 s / 1 s, classi 1-6, codici Myo), P1 con chiave
  `soggetto/sessione` e `session_windows(window_s=1)`.
- **Identita' del controllo nel benchmark:** `canonical_config(config T100-L a 5.000) == canonical_config(candidato con
  flag False)` e' vero sui file locali; il config storico a 19.618 differisce dal 5.000 solo in `max_steps`;
  `calibration_signature` dipende da loader, dataset, manifest, scale e finestre, non dal modello: la firma dei dati e'
  condivisa fra candidato e controllo. Walltime storici nel codice (2.700/1.800/2.700 s) uguali al launcher storico;
  le ricevute storiche scrivono `elapsed_seconds`, `which` e le provenienze con `n_windows`.

## 1. R1-a, R1-b, N1-N4 — recepiti

R1-a: la frase sul limite analitico e' sostituita dalla rettifica esplicita. R1-b: `log_weight_diagnostics` calcola le 64
norme L2 per colonna di `tokenizer.logsquare_weight` per student e teacher, con famiglie Fourier/spline/MLP/non-MLP
(n, min, mediana, media, massimo, somma dei quadrati), `descriptive_only`; entra nella ricevuta di estrazione a entrambi
gli endpoint e il lettore di posa la riporta nel candidato e nel confronto dopo averla ricontrollata colonna per colonna
(`load_log_weight_diagnostics`), senza soglie. N1-N4 sono nel piano esecutivo e in `training_plan.execution_notes`
(override espliciti 01:30:00 e 04:00:00). Epsilon 0,001 fissato; un cambio dopo i risultati e' un candidato nuovo.

## 2. Estrattore e benchmark (D1, D2, D3) — APPROVATI

- **D3.** `validate_checkpoint`: endpoint nominato, STOP, ricevuta di training (protocollo LQ, `release_sha256` e
  `plan_sha256` pinnati per il candidato), config completo uguale al byte salvo i due soli default LQ
  (`canonical_config` rifiuta qualunque altro campo mancante: test `two legacy`), `FF.load_model` stretto, flag ed epsilon
  nel config e nel tokenizer, pesi uguali con `torch.equal`, `fe_gain` dal checkpoint senza ricalibrazione, `W_log`
  (384, 64) uguale al checkpoint; il controllo lineare non deve avere `W_log`. I test D3 passano dai percorsi reali
  (`PE.extract` con loader vero; `extract_channel_readouts` a 200 e 2000 Hz con tutti gli strati, decoder compreso):
  `W_log` nullo = lineare, non nullo ≠ lineare. Nessuna certificazione CUDA, come dichiarato.
- **D1.** `resource_proposal`: tetto = walltime storico × max(1, massimo dei rapporti dei gruppi del job), arrotondato
  per eccesso a 300 s; stima = durata storica × media pesata con i conteggi storici di finestre dalle provenienze pinnate;
  gruppo o conteggio mancante, rapporto non finito, piu' forme per gruppo → stop. Ordine L/LQ alternato per round, 2
  warmup + 6 ripetizioni, sincronizzazione CUDA fuori dalla chiamata misurata, ingressi invariati (`input_identity`),
  output non finiti rifiutati. Tutto ricostruito a valle in `allocation_for` prima di ogni estrazione: regola, codice e
  piano del benchmark, contesto degli ingressi, sorgenti, tempi grezzi e preventivo; il contesto esclude stato, ricevuta,
  allocazioni e pin del 19.618 (test), quindi nessun ciclo fra gli hash.
- **D2.** Il controller del training non dipende dal benchmark (invariato); `continuation_waits_for_benchmark: false`
  nella politica; il benchmark richiede `APPROVED_FOR_TRAINING`, release pinnata e checkpoint 5.000 completo.
- **Launcher:** 1 GPU, 8 CPU, 100G; il benchmark esige esattamente 00:30:00; checkout congelato e piano contenuto;
  output nuovo fuori dal codice (`new_output`); nessuna sottomissione.

## 3. Lettori (posa e sonde) — APPROVATI

- **Posa.** `run_candidate` fitta solo LQ con `Q.fit_reader`/`Q.predict`/`Q.summarize` (test: bit-identico alle funzioni
  storiche; `val` aperto solo dopo il fit). `compare` importa le tre celle storiche dal report pinnato dopo averne
  riverificato piano, sorgenti, contesto numerico, selezione dell'alpha, supporti e utenti, e ricontrolla coefficienti,
  predizioni e statistiche sufficienti dai file (`verify_scientific_artifacts`); bootstrap a quattro celle con gli stessi
  draw storici (test: `draws_sha256` e intervalli dei controlli identici a `Q.paired_comparison`), tre contrasti LQ−
  controllo dichiarati nel piano, nessuna intersezione silenziosa (12 mutazioni rifiutate). Alpha irrisolto ferma il
  confronto. Gate del fit: formula storica con un solo candidato, `ceil_60s(1,5·(2V + 11S + 13E) + 120) ≤ 1.800 s`
  (lo storico aveva il fattore 3 dei tre bracci), ricevute e accounting pinnati, contesto numerico a 8 thread.
- **Sonde.** `fit_fold` e `AuditLog` sono gli oggetti storici (identita', test); `load_cache` richiede protocollo LQ,
  flag, epsilon, finestra, attivazione, dim e K uguali in ricevuta e provenienza; `fit_gate` confronta i conteggi dei fold
  del candidato con il report storico di T100-L e il contesto numerico con il pin; `compare` importa i sei aggregati
  storici ricalcolando punteggi OOF e identita' degli audit (`verify_aggregate`), conserva `original_provenance`, usa
  `compare_arrays` con gli stessi draw (test) e i tre contrasti LQ; solver irrisolto → stime conservate, NI sospesa.
- **Contratto ricevute → lettori:** `logsquare_evaluation.execute` scrive gli stessi campi della ricevuta storica
  (`protocol`, `arm`, `which`, `checkpoint_step`, `checkpoint_sha256`, `code_sha`, `plan_sha256`, `postload_recipe_passed`,
  `model_config`, `outputs[nome] = {path, sha256, bytes, provenance_path, provenance_sha256}`, `checkpoint_unchanged`,
  `complete`) e le provenienze con i `details` delle stesse `verify_*` storiche: `Q.validate_artifact` e `T.load_cache`
  li leggono senza adattamenti; le validazioni LQ aggiunte sono simmetriche fra ricevuta e provenienza.

## 4. Risorse — APPROVATE

- Benchmark 1 GPU × 1.800 s = 0,5 GPU-ore, 8 × 1.800 s = 4 core-ore; checkout 2 × 600 s = 0,3333 core-ore.
- Posa per endpoint: validate 1 + benchmark 2 + ridge 4 + compare 1 = 8 core-ore; 16 per i due endpoint; stessi tetti
  della politica storica, piu' il confronto separato dichiarato.
- Sonde: 8 + 8 + 4 + 4 + 8 (shard) + 2 × 0,6667 (aggregati) + 1 (confronto) = 34,3333 core-ore. Lettori CPU totali
  50,3333. Estrazioni: tetti nulli fino a rho_infer, come deve essere.
- **A5, riuso dei tempi storici per shard come gate CPU: ACCETTATO** con queste condizioni, gia' in codice o da scrivere
  nel registro al momento del rilascio dei piani: (i) i margini storici si pubblicano con il piano: massimo
  elapsed/tetto per shard db2-01 0,49, db2-23 0,47, db3 0,33, db6 0,57, p2 0,45 (tre bracci, tutti COMPLETED 0:0,
  RSS ≤ 3,1 GiB), T100-L 13,26 core-ore di fit contro 34,33 di tetto; (ii) il runtime esige contesto numerico identico
  (`07b776ba…`) e conteggi dei fold uguali a T100-L (gia' in `fit_gate`); (iii) un shard che tocca il walltime si ferma
  con `failed.json`, senza rilancio ne' aumento del tetto: si rivede il piano; (iv) resta scritto che e' un proxy: feature
  diverse possono cambiare le iterazioni di lbfgs (max_iter 12.000) e la misura storica non garantisce la durata LQ.

## Note non bloccanti

- **A1 (controllo prima del benchmark).** Il benchmark esige esattamente 10 finestre da ciascuna delle prime due
  registrazioni della selezione train congelata (`Meter.measure` e `len(meta) == 20`). La cache storica ha 1.907 finestre
  su 200 registrazioni (media 9,5): alcune ne hanno meno di 10. Prima di sottomettere, leggere dalla `meta` della cache
  storica `train_subset.npz` (sola lettura) i conteggi delle registrazioni 0 e 1; se una e' sotto 10 il job fallisce
  subito senza riselezione e il piano va rivisto prima del lancio, dichiarando la nuova selezione.
- **A2 (submitter).** Il launcher delle estrazioni ha `--time=00:45:00` di default, ma `allocation_for` esige
  l'allocazione uguale al tetto misurato della cella: il submitter passa `--time` pari al tetto pubblicato, come per N4.
- **A3 (sorgenti congelati dal benchmark).** La ricevuta del benchmark pinna i 110 sorgenti (adattatori, launcher e
  `src/wearusfm`): qualunque modifica successiva, anche una correzione degli adattatori, invalida la ricevuta per le
  estrazioni e richiede un nuovo benchmark. Scriverlo nel piano esecutivo.
- **A4 (ordine dei pin dopo il rilascio).** Il commit di rilascio cambia `training_plan.json`: vanno aggiornati nell'ordine
  `extraction_plan.training_plan` (sha e byte) e poi `inference_plan.extraction_plan`; il contesto del benchmark si fissa
  dopo, quindi nessun ciclo. Il checkout nuovo deve contenere i piani aggiornati.

## Prerequisiti per il commit dedicato `APPROVED_FOR_TRAINING`

Soddisfatti da questa consegna: adattatori implementati e rivisti (questa review), controller e sorgenti scientifici
invariati, R1-a/R1-b e N1-N4 recepiti. Il commit di rilascio, di GPT, deve contenere: stato `APPROVED_FOR_TRAINING`;
`training_release` con `training_approved`, epsilon 0,001, sha256 correnti di report, analisi e preventivo, hash dei due
file del controller (invariati da 6d5b353) e pin di una review di Claude (questa o 1d6f96a); nessun'altra modifica alla
ricetta o ai 102 sorgenti. Poi checkout congelato nuovo allo SHA di rilascio pubblicato, sottomissione con gli override
di tempo di N4 sotto l'autorizzazione operativa gia' data da Simone. Dopo il 5.000: pin dell'endpoint, A1, benchmark,
pubblicazione di rho_infer e dei tetti delle estrazioni fra le due fasi; la continuazione non li aspetta.
Nessun lancio da parte mia.
