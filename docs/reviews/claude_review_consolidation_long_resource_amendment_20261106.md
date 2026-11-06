# Review di Claude: emendamento delle risorse della valutazione lunga (linea 2), 06/11/2026

Oggetto: proposta di GPT consolidamento su `origin/codex/consolidation-20261104`, commit
`4bb39d254c26bd3b9666dff0e1b1464f96bfe4db` (punta del ramo riletta alle 13:07, 13:16 e 13:21: invariata).
Documento `docs/consolidation_long_resource_amendment_proposal_20261106.md`, dati
`results/infra/consolidation_long_20261105/resource_amendment_proposal/options.json`, script
`docs/reviews/check_long_resource_options_20261106.py`. Misure: le mie ricevute nel commit
`2a20316a0bf77c93e354d4ce3f5acee94db459df` del ramo di esecuzione.
Richiesta: prompt di GPT «Review dell'emendamento delle risorse — linea 2, valutazione lunga», incollato da Simone il 06/11.

**VERDETTO SUL DISEGNO: APPROVATO.** Nessun difetto nel disegno. Otto verifiche da soddisfare nell'implementazione
(sezione 3). Una condizione di governo prima di pin e deployment (sezione 5): la decisione di Simone sull'eccezione a
R-2(i), che il piano congelato riserva a lui.

Review di sola lettura: nessun file di codice o di piano toccato, nessun job, nessun accesso al cluster.

## 1. Come ho verificato

- **Ricalcolo indipendente.** Script `claude_review_consolidation_long_resource_amendment_20261106_recompute.py`, uscita
  nel file `.json` omonimo (`problems` vuoto). I costi per fold vengono dal gate del runtime congelato
  (`CR.resource_gate`, codice `166fb86`) applicato alle 15 ricevute; suddivisione, tempi e conti li ricostruisco dalla
  tabella del documento, non dal JSON di GPT, e solo dopo li confronto col JSON. Differenza massima sulle 69
  proiezioni: 0,0 secondi.
- **Lettura del codice congelato** (`scripts/consolidation_long*.py`, launcher, test) per i punti che i numeri non
  provano: indipendenza dei fold, unione degli shard nel confronto, che cosa copre l'identita' della ricetta, dove
  entra l'identita' dei sorgenti.
- **Ricevute e pin.** I 28 pin del JSON confrontati con i miei due audit e con i file raccolti.

| Voce | Proposta | Ricalcolo |
|---|---:|---:|
| Gate originale: shard respinti / celle respinte | 15 su 48 / 10 su 40 | uguale |
| Fit di classificazione emendati | 69 | 69 |
| Shard invariati (stessa proiezione) | 33 | 33 |
| Celle scientifiche e fold | 40 celle, 4 fold | 40 celle, ogni fold una volta sola |
| Tempo richiesto per shard | da 30 a 225 minuti | uguale |
| Proiezione dei 69 fit, core-ore | 1.115,658874 | 1.115,658874 |
| Allocazione massima dei 69 fit, core-ore | 1.276 | 1.276 |
| Altre allocazioni future, core-ore | 41 | 41 |
| Gia' consumato, core-ore (GPU-ore) | 42,218889 (3,373611) | uguale, dai 30 record `sacct` |
| Consumato + massimo futuro, core-ore | 1.359,218889 | uguale |
| Totale conservativo, core-ore | 1.451,666667 | uguale |
| Tetto approvato, core-ore | 1.711,666667 | margine 260 |
| Job di calcolo / allocazioni con i checkout / restanti | 103 / 107 / 77 | uguale |

## 2. Esito sui cinque punti richiesti

### 2.1 Suddivisione, copertura e margini: confermati

P e' la proiezione congelata, W il tempo richiesto, «scarto» e' W − 600 − P (quanto resta oltre la riserva di 600 s).

| Dataset | Vista | Fold per job | Dosi | P (s) | W (min) | Scarto (s) |
|---|---|---|---:|---:|---:|---:|
| ninapro_db2 | pre | 0 / 1 / 2 / 3 | 3 | 10.631,8 / 11.484,9 / 10.429,6 / 10.625,6 | 195 / 210 / 195 / 195 | 468 / 515 / 670 / 474 |
| ninapro_db2 | post | 0-1 / 2-3 | 3 | 8.191,6 / 7.806,3 | 150 / 150 | 208 / 594 |
| ninapro_db2 | pool | 0 / 1 / 2 / 3 | 2 | 8.066,4 / 8.701,3 / 7.915,9 / 8.061,8 | 150 / 165 / 150 / 150 | 334 / 599 / 484 / 338 |
| ninapro_db3 | pre | 0 / 1 / 2 / 3 | 3 | 7.047,2 / 6.942,3 / 7.818,0 / 6.516,4 | 135 / 135 / 150 / 120 | 453 / 558 / 582 / 84 |
| ninapro_db3 | post | 0-1-2-3 | 3 | 12.680,0 | 225 | 220 |
| ninapro_db3 | pool | 0-1-2-3 | 2 | 11.940,9 | 210 | 59 |
| ninapro_db6 | pre | 0-1-2-3 | 3 | 4.706,3 | 90 | 94 |
| ninapro_db6 | post | 0-1-2-3 | 3 | 2.059,5 | 45 | 40 |
| ninapro_db6 | pool | 0-1 / 2-3 | 2 | 8.811,9 / 8.648,0 | 165 / 165 | 488 / 652 |
| epn612 | pre | 0-1-2-3 | 3 | 6.087,0 | 120 | 513 |
| epn612 | post | 0-1-2-3 | 3 | 3.012,2 | 75 | 888 |
| epn612 | pool | 0-1-2-3 | 2 | 12.386,4 | 225 | 514 |
| uci_emg | pre / post / pool | 0-1-2-3 | 3 / 3 / 2 | 608,1 / 604,2 / 614,3 | 30 | 592 / 596 / 586 |

- 12 + 6 + 8 + 12 + 3 + 2 + 3 + 3 + 4 + 8 + 8 = 69 job; 69 ID e 69 uscite distinti; nessuno coincide con i 48 ID
  vecchi, che non hanno job (`slurm_job_id` nullo) e non compaiono nel ledger. Il piu' lungo e' di 50 caratteri (il
  submitter ne ammette 64).
- Ogni shard rispetta P ≤ W − 600 e W ≤ 14.400. Lo shard piu' stretto (ninapro_db6 post) ha 640,5 s fra proiezione e
  limite. Tutti i 69 passerebbero anche il limite originale: P massimo 12.680,0 contro 13.800.
- Memoria: massimo proiettato 4,08 GiB contro 25,5. Invariata.
- La proiezione di uno shard invariato e' identica a quella del gate originale (33 confronti).

### 2.2 Costi e numero di job: confermati

Conti nella tabella della sezione 1. Le 41 core-ore sono: tre fit della posa 24, due confronti 16, sintesi 0,33, due
checkout 0,67. Il totale conservativo e' 134,67 (tetti delle fasi concluse) + 1.276 + 41. Con 69 fit da quattro ore
si arriverebbe a 2.383,67, oltre il tetto: il tempo per shard e' necessario, non una rifinitura. Nessuna GPU nuova.

### 2.3 Compatibilita' limitata ai 28 input: fattibile; sufficiente alle condizioni V-1...V-6

- **Dove entra oggi l'identita' dei sorgenti** (sei confronti con `ctx["source_sha256"]`): manifest delle cache
  (`consolidation_long_common.py:127`, `consolidation_long_pose_runtime.py:69`), ricevute di estrazione
  (`common.py:235`), ricevute dei benchmark di classificazione (`classification_runtime.py:277`), configurazione
  incorporata nelle NPZ della posa (`pose_runtime.py:98`), ricevuta del benchmark della posa (`pose_runtime.py:226`).
  Basta distinguere in questi sei punti la source degli input da quella dell'eseguibile; gli output nuovi continuano
  a prendere la source nuova da `begin_output`.
- **Un solo valore da ammettere.** Le 28 ricevute (12 estrazioni con codice `7d276c9` e piano `e24d3fae...`; 16
  benchmark con codice `166fb86` e piano `22975a40...`), il manifest delle cache (21 payload da 12 ricevute) e le sei
  NPZ della posa portano tutti `1ca7aead...` e la ricetta `b2356a8e...`. I 28 pin del JSON coincidono con i miei audit
  e con i file.
- **La ricetta non cambia.** `recipe_identity` copre solo protocollo, dosi, viste, teacher, ricette di classificazione e
  posa, regola di lettura, politica del baseline e batch di estrazione (`common.py:84-86`). Risorse e politica dei
  benchmark ne sono fuori: cambiare shard e tempi lascia `b2356a8e...`.
- **La source originale e' ricostruibile.** La mappa percorso → sha256 dei 227 file coperti (100 in `src/`, 125 in
  `scripts/`, due del launcher) ai commit `7d276c9`, `166fb86` e `2a20316` e' la stessa e il suo digest da'
  `1ca7aead...`. Quindi «estrattore e calcoli invariati» si puo' provare file per file, non solo affermare.
- **Limite della sola lista.** La lista dei 28 pin da sola non basta: dice quali input, non che il codice nuovo li
  legga come il vecchio. La sufficienza dipende dalle verifiche V-1 e V-2.

### 2.4 Nessuna nuova misura: confermato

- I fold sono indipendenti nel codice congelato: `fit_fold` crea per ogni fold uno scaler e una regressione logistica
  nuovi (`random_state=0`, `lbfgs`, nessun avvio a caldo); `fit_cache` accetta gia' qualunque sottoinsieme ordinato di
  fold (`consolidation_long_classification.py:311-406`). Il risultato di un fold non dipende dagli altri fold del job.
- Il confronto unisce gia' un numero qualunque di shard per cella, controlla supporto e fit per fold, esige i
  quattro fold una volta sola e allinea per riga (`:685-712`, `:492-501`): la lettura scientifica non dipende dalla
  partizione. Un OOF a fold singolo supera `validate_oof`.
- La misura e' per dataset e vista sul train completo del fold 0; il costo e' gia' calcolato fold per fold e
  l'overhead per job viene contato per ciascun job nuovo. Ambiente numerico invariato: il gate esige nei fit la stessa
  firma dei benchmark.

### 2.5 Posa prima e regola di corsia: confermate

- Il gate della posa e' gia' separato nel codice (`_benchmark_gate`); ricalcolo: 583,269 s per dose contro 3.300 e
  12,53 GiB contro 25,5, ammessa. Corretto non anticiparla col vecchio codice: avremmo risultati nuovi con due source.
- «Al massimo due fit della linea 2» coincide con la decisione di Simone «Due fit per volta». La applico contando
  anche il checkout di fase (lettura mia, la piu' stretta), finche' nel ledger non compare sottomesso un ID
  `SCALING-compare-`.

## 3. Verifiche necessarie sull'implementazione

Non sono difetti del disegno: sono le prove senza le quali l'implementazione non va rilasciata.

- **V-1. Mappa dei sorgenti originali nel manifest di compatibilita'.** Il manifest contiene la mappa completa dei
  227 file; il suo digest deve dare `1ca7aead...`. Il codice nuovo la confronta con i propri file: differenze ammesse
  solo in un elenco chiuso (adattatori di esecuzione, controller, launcher); tutti gli altri byte-identici. Devono
  esserlo in particolare `scripts/consolidation_long_classification.py`, `scripts/consolidation_long_pose.py` e tutto
  `src/`.
- **V-2. Estrazione invariata dentro i file che cambiano.** `load_teacher`, `verify_training_receipt`, il ramo di
  estrazione di `execute` e `_extract` della posa: testo identico a `166fb86`, provato da un test e dal diff.
- **V-3. Fasi chiuse.** Col codice nuovo gli stadi `extract-*` e `benchmark-*` vengono rifiutati prima di creare
  uscite: non possono nascere ricevute di input con la source nuova.
- **V-4. Elenco chiuso e pin esatti.** Ogni ricevuta di input letta deve stare nel manifest con sha256, byte, codice e
  piano originali, source `1ca7aead...` e ricetta `b2356a8e...`. Test negativi: source, ricetta o pin alterati;
  ricevuta non elencata; ricevuta di input con la source nuova; vecchio report di fit presentato come nuovo.
- **V-5. Posa.** Source originale nel manifest delle cache e nella configurazione delle sei NPZ; source nuova solo
  negli output. Politica e risorse della posa invariate (la ricevuta del benchmark le confronta per uguaglianza).
- **V-6. Politica originale dei benchmark intatta.** `BENCHMARK_POLICY` resta identica: le 15 ricevute la portano. La
  nuova ammissione sta in un oggetto separato. Test sulle ricevute reali: il gate originale resta 15 respinti su 48; il
  nuovo da' i 69 P e W del JSON.
- **V-7. Gate, tempi e confronto.** Gate globale sui 69 shard registrati; W ricalcolato dalle ricevute e uguale alla
  tabella congelata nel piano; il launcher confronta il limite del job col W del suo shard; il confronto accetta solo
  le forme registrate per dataset e vista, 69 report.
- **V-8. Piani.** 69 righe nuove, le 48 vecchie marcate sostituite e non cancellate, 103 job piu' 4 checkout, tetti
  invariati, `recipe_sha256` invariata, source nuova. `phase_source_policy` va riscritta in modo esplicito: oggi dice
  «no receipt rewrite, accepted-source whitelist or automatic resubmission» e «all twelve extractions repeated».
  L'emendamento va dichiarato post hoc rispetto ai tempi e anteriore a qualunque fit o punteggio.

## 4. Note non bloccanti

- **N-1. Scarto di tempo.** La riserva garantita resta quella del gate originale: 600 s oltre una proiezione gia'
  moltiplicata per 4,5. Si perde solo lo scarto in piu' che gli shard veloci avevano con quattro ore fisse. Un TIMEOUT
  fermerebbe la fase (nessun requeue). Il tetto lascia 260 core-ore: se GPT o Simone vogliono piu' scarto c'e' spazio.
  Io implemento la regola proposta, senza modificarla.
- **N-2. DB3 pre a fold singoli.** Scelta coerente: a coppie uno shard avrebbe 65,9 s di margine. Costa 24 core-ore
  allocate e 6 job in piu'.
- **N-3. Natura dell'emendamento.** E' deciso dopo i tempi train-only e prima di qualunque fit o punteggio: post hoc
  sulle risorse. Il documento non lo chiama pre-registrato, ed e' giusto cosi'. La regola di lettura non cambia.
- **N-4. Corsia.** Con un job da 8 CPU per volta i 69 fit chiedono al massimo 159,5 ore di tempo. Stima mia, non una
  garanzia: circa 30 ore se i tempi misurati a 19.618 valgono per le altre dosi (la proiezione contiene il fattore 4,5).

## 5. Che cosa manca prima di pin e deployment

- Il piano congelato riserva la scelta a Simone. `launch_plan.json`, `phase_source_policy`: «action: stop chain;
  preserve all artifacts; no receipt rewrite, accepted-source whitelist or automatic resubmission» e «repeat_requires:
  new Claude review, new IDs/outputs, complete new estimate and Simone decision; all twelve extractions repeated».
  Il manifest dei 28 input e' una lista di source accettate, cioe' l'eccezione a quella regola. Il documento di GPT
  lo dice: «proposta per la decisione di Simone».
- Una review favorevole non e' quella decisione. Prima di pin, checkout e invii mi serve la risposta di Simone su:
  eccezione a R-2(i) limitata ai 28 input; catena emendata di 69 fit di classificazione, 3 fit della posa, 2
  confronti, 1 sintesi e 2 checkout; al massimo 1.317 core-ore future e 0 GPU-ore, dentro il tetto di 1.711,67.
- L'alternativa secondo la regola congelata e' ripetere le 12 estrazioni (3,37 GPU-ore l'ultima volta, tetto
  aggiuntivo previsto 9, fuori dal tetto attuale) e i 16 benchmark. La mia valutazione: l'eccezione e' la scelta
  migliore, a patto che V-1 e V-2 siano provate.
- Implementazione: la faccio io sul ramo di esecuzione, come chiede il prompt di GPT; GPT la verifica dopo. Pin,
  deployment e invii restano fermi fino alla decisione.
