# Compito C (c-decorso): decorso temporale dello scarto, dagli artefatti esistenti

Clone in sola lettura a `918245b6`. Cartella degli artefatti: `results/infra/contrastive_targets_20261118/` (abbreviata in `E/`).
Script e uscite intermedie in `agents/c-decorso/`: `metrics_diff.py` → `reldiff.json`, `tensors.py` → `queue_{CN,CR}.npy`,
`queue_map.py` → `queue_by_update_{CN,CR}.json`, `trend.py`, `params.py`.
Legenda: **[V]** verificato con file, comando o campo; **[D]** dedotto; **[?]** non dimostrato.

## Sintesi

Lo scarto nasce **in un solo punto: la riduzione DDP dell'update 545**. È l'unico update in cui la ripresa usa un raggruppamento dei
gradienti diverso da quello del continuo. Da lì lo scarto si propaga in modo deterministico e cresce in salti quantizzati dall'aritmetica bf16.
Gli artefatti escludono quasi del tutto un difetto di ripristino (H1). Le chiavi della coda inserite agli update 545-546 (CN anche 547) sono
**identiche bit per bit**. La norma del gradiente CN al 546 è identica al continuo in tutte le 16 cifre.
Lo scarto è globale e ha lo stesso ordine in CN e in CR (spostamento dello studente da 544: 0,645% contro 0,616%).
I quattro tensori 266-269 superano l'1% perché sono la coda più sensibile della distribuzione: è H3, sopra un meccanismo H2.
Correzione a GPT: secondo questa lettura il layout dei gruppi differisce **solo al 545, non al 545-546**. Le tracce DDP si leggono con un passo di ritardo.

## 1. Metriche per update (545-576), continuo contro A, continuo contro B, A contro B

**[V] Continuo/A e continuo/B coincidono in tutte le metriche di gradiente.** Lo stesso vale per A/B.
Fra A e B differiscono solo i campi di log delle loss e i tempi: `band_shape, contrast_rnc, jepa, log_env, log_rms, ms_fast, ms_slow, total,
t_data_s, t_data_s_mean, t_step_s` (uscita di `metrics_diff.py`, identica per CN e CR). Restano identici bit per bit fra A e B:
`grad_norm`, `grad_clip_factor`, `contrast_nce`, `pose`, `contrast_projector_grad_norm`, `contrast_global_*` e i coseni.
Lo scarto massimo A/B nei log è di ordine float32: jepa 2,7e-7 (CN) e 5,0e-7 (CR), total 1,1e-7 e 8,4e-8.
- **[D] La natura dei campi A/B è un rumore del solo log.** Le medie per campione usano `index_add` (`src/wearusfm/model/anchors.py:47-48`).
  Il commento a `src/wearusfm/training/jepa.py:25-26` lo dice: «su GPU l'ordine delle somme non e' deterministico».
  Il valore in avanti cambia, ma il gradiente di `index_add` è un `index_select` deterministico: i checkpoint A e B restano identici.
  **Conseguenza:** al 545-546 le differenze continuo/A di jepa, log_env, ms_* e total (1e-8 … 4e-7) cadono sotto la soglia di rumore A/B.
  Non provano una divergenza.

**[V] Primo scarto e ordine di grandezza (relativo al continuo):**

| campo | CN 545 | CN 546 | CN 547 | CN 548 | CR 545 | CR 546 | CR 547 |
|---|---|---|---|---|---|---|---|
| grad_norm | 2,2e-12 | **0 (identico)** | 2,2e-5 | 9,1e-5 | 3,4e-13 | 3,4e-6 | 1,3e-4 |
| contrast_projector_grad_norm | 1,0e-13 | 0 | 6,5e-5 | 1,1e-3 | (0, assente) | | |
| contrast_global_grad_base | 0 | 0 | 2,1e-5 | 1,7e-5 | 0 | 4,0e-6 | 1,4e-4 |
| contrast_nce | 0 | 0 | 2,4e-6 | 6,0e-5 | 0 | 2,2e-6 | 2,5e-5 |
| pose | 0 | 0 | 0 | 1,0e-6 | 0 | 0 | 2,7e-7 |

Valori esatti di `grad_norm` (`metrics.jsonl`).
- CN 545: continuo 4.336711902204087, A e B 4.336711902194546 (assoluto 9,5e-12; relativo 2,2e-12).
- CN 546: **9.023137809933427 in tutti e tre**.
- CR 545: 4.618657714258972 contro 4.618657714257402.
Il «9,5 × 10⁻¹²» di GPT è lo scarto assoluto; il relativo è 2,2e-12.

Il passo di partenza (`start.json`, `source.step = 544`) e batch, origini e classi (`origin` e `classes` uguali in ogni update) sono verificati.

**[V] Crescita (`trend.py`, regressione di log10 dello scarto relativo sugli update):**

| campo | CN mediana 548-556 / 557-566 / 567-576 | CN pendenza 548-576 (548-561 / 562-576), decadi/update | CR mediane | CR pendenza |
|---|---|---|---|---|
| grad_norm | 1,4e-4 / 2,8e-4 / 8,7e-4 | +0,037 (+0,060 / +0,030) | 7,0e-5 / 3,0e-4 / 3,2e-4 | +0,040 (+0,071 / −0,039) |
| contrast_projector_grad_norm | 1,1e-3 / 1,8e-3 / 8,2e-4 | −0,004 (+0,029 / −0,025) | — | — |
| total | 1,3e-5 / 1,9e-5 / 4,1e-5 | +0,018 | 6,1e-6 / 9,9e-6 / 1,4e-5 | +0,020 |
| jepa | 6,0e-5 / 9,1e-5 / 3,7e-4 | +0,036 | 3,4e-5 / 8,3e-5 / 2,1e-4 | +0,039 |
| pose | 2,3e-6 / 1,1e-5 / 1,6e-5 | +0,030 | 2,7e-6 / 5,6e-6 / 4,8e-6 | +0,015 |

Lettura [D]. Il decorso ha tre fasi.
1. Al 545 c'è un seme di ordine 1e-12.
2. Al primo update visibile (CN 547, CR 546) c'è un **salto quantizzato** di sette ordini, fino a 1e-6…1e-5.
3. Segue una crescita lenta di 0,02-0,04 decadi per update (×10 ogni 25-50 update).

Nella seconda metà la crescita rallenta (grad_norm CN +0,030; CR −0,039; il proiettore CN −0,025). A 576 non è chiaro se lo scarto
abbia già raggiunto un plateau. CN e CR hanno lo **stesso decorso**. CR parte un update prima (gradiente diverso dal 546 invece che dal 547).
I livelli finali sono simili (grad_norm 567-576: 8,7e-4 CN contro 3,2e-4 CR).

## 2. Distribuzione degli errori relativi nei tensori (JSON, `arms.<braccio>.resume.pairs.<coppia>.floating_tensor_relative_to_left`)

**[V] Composizione:** 6.520 tensori per coppia = 270 `exp_avg` + 270 `exp_avg_sq` + 5.980 chiavi della coda. Fra A e B sono tutti 0.

| | CN continuo/A (uguale a continuo/B) | CR continuo/A (uguale a continuo/B) |
|---|---|---|
| exp_avg: min / mediana / p90 / p99 / max | 6,2e-4 / 4,6e-3 / 5,9e-3 / 1,06e-2 / 1,32e-2 | 0 / 2,9e-3 / 4,0e-3 / 5,1e-3 / 5,9e-3 |
| exp_avg oltre 1% / oltre 0,5% | 4 (266-269) / 85 | 0 / 3 |
| exp_avg_sq: mediana / max | 9,6e-5 / 1,68e-3 (266-269 in testa) | 5,8e-5 / 3,4e-4 |
| zeri esatti fra i momenti | nessuno | **solo 266-269** (exp_avg e exp_avg_sq) |
| chiavi: zeri / non zero / max | 5.070 / 910 / 9,2e-3 | 5.040 / 940 / 6,0e-3 |

- **[V] Mappatura 266-269.** `params.py` ricostruisce il modello dalla `config.json` di CN, con la stessa sequenza di `run.py:1387-1390` e
  l'AdamW su `requires_grad` di `run.py:1429`.
  - Totale: 270 parametri e **28.559.133** elementi.
  - 266 = `contrastive_head.layers.0.weight` (384×384); 267 = `layers.0.bias` (384); 268 = `layers.2.weight` (128×384); 269 = `layers.2.bias` (128).
  - I tensori 0-7 del tokenizer sono float64.
  - Prova indipendente: in CR proprio 266-269 hanno errore zero. Il proiettore non riceve gradiente: `contrast_projector_grad_norm = 0.0` e
    `by_block…student.contrastive_head = 0.0` nel JSON. La mappatura di GPT è confermata.
- **[V] Posizione di 266-269 in CN.** Occupano i ranghi 1-4 su 270. Il quinto tensore (194) è a 7,1e-3. I quattro valgono 2,3-2,9 volte
  la mediana CN (4,6e-3). Non sono un caso isolato: sono la coda di una distribuzione **globale**.
  Gli altri exp_avg di CN stanno 1,6 volte sopra quelli di CR (mediana 4,6e-3 contro 2,9e-3).
- **[V] Nello spazio dei parametri i due bracci divergono allo stesso modo.** Lo spostamento dello studente da 544 vale 0,6454% in CN e
  0,6164% in CR (`relative_to_left_displacement_from_544.student`).
- **[D] Perché exp_avg_sq resta 10-100 volte sotto exp_avg.** Con β1 = 0,9, dopo 32 update il peso della storia anteriore al fork è
  0,9³² ≈ 3,4%: exp_avg misura quasi solo i gradienti degli ultimi ~10 update. Con β2 = 0,999 il peso è 0,999³² ≈ 96,8%: exp_avg_sq è
  dominato dalla storia comune. La soglia dell'1% su exp_avg è quindi una soglia sulla differenza dei gradienti recenti, normalizzata per
  ‖m‖. È molto più severa della soglia sui pesi, normalizzata per lo spostamento dall'inizializzazione (H3).

**[V] Le chiavi della coda per update di inserimento (`queue_map.py`).**
La coda è ordinata per (step, global_index): `contrastive_memory.py:236-245` e il controllo d'ordine a `:286`.
Ho ricostruito le voci superstiti degli update 545-576 simulando inserimenti, duplicati del batch e sostituzioni per identità dalle `origin`
di `metrics.jsonl`. Risultano 1.001 voci (≈1.000: all'update 560 una finestra non è valida, `contrast_nce_eligible = 31`).

| update di inserimento | CN | CR |
|---|---|---|
| 545 (29 chiavi) | **tutte 0** | **tutte 0** |
| 546 (32) | **tutte 0** | **tutte 0** |
| 547 (30) | **tutte 0** | mediana 3,3e-3 |
| 548 | mediana 3,6e-3 | 4,0e-3 |
| 550-576 | mediana 4,6-5,9e-3, quasi piatta (+0,0014 decadi/update) | 3,6-4,4e-3, piatta (+0,0005) |

Il confine fra zeri e non zeri cade esattamente su un confine di update. In CN la prima chiave non nulla è l'indice 5.070 = 4.979 + 91 =
29 + 32 + 30. In CR è 5.040 = 4.979 + 61. Prima di 4.979 non c'è nessuna chiave non nulla.
**[V] Le chiavi di 545 e 546 (CN anche 547) sono identiche fra continuo e ripresa.**
- **[D]** Le chiavi dell'update u le calcola il teacher prima dell'update u (`run.py:1722`, «pre-update teacher keys»).
  Al 545 il teacher è quello ripristinato dal fork. Teacher, coda e forward sono quindi ripristinati e deterministici bit per bit.
  In CN il teacher resta identico fino a dopo il 546; in CR solo fino a dopo il 545. È lo stesso sfasamento di un update visto sulla norma del gradiente.
- **[D] Il plateau delle chiavi è a scala bf16.** Le chiavi saltano subito a 3-4e-3 e poi restano quasi piatte. È l'ordine dell'ulp bf16
  (2⁻⁸ ≈ 3,9e-3 relativo): basta una minima perturbazione perché una frazione degli elementi bf16 cambi arrotondamento.
  Il livello dello scarto delle chiavi non dipende quindi dall'ampiezza del seme.

## 3. Conti sull'ordine di grandezza

**[V] Elementi:** 28.559.133 in totale, di cui 8.560 float64. Le dimensioni dei bucket nella traccia DDP sono in byte:
114.202.292/4 + 68.480/8 = 28.550.573 + 8.560.

**[D] Seme al 545.** Supponiamo che k elementi del gradiente ridotto differiscano di 1 ulp float32 (relativo ≈ 2⁻²³ ≈ 1,19e-7) con segno
casuale e grandezza tipica (g_i² ≈ ‖g‖²/N). Allora Δ‖g‖/‖g‖ ≈ 2⁻²³·√k/N.
- Con N = 2,86e7, il valore osservato 2,2e-12 (CN) dà √k ≈ 530, cioè **k ≈ 3e5 (~1% degli elementi)**.
- 3,4e-13 (CR) dà k ≈ 7e3.
- Se la distribuzione dei gradienti ha code pesanti (N_eff < N), bastano meno elementi.
- **Compatibile con H2:** uno scarto di 1e-12 sulla norma è sotto la risoluzione di un singolo elemento (6e-8). Può nascere solo da molti
  arrotondamenti di 1 ulp con segni diversi. Un difetto di ripristino (scala del gradiente, lr, contatore di Adam, momenti) darebbe ≥1e-7 già al 545.
- **Cosa non torna del tutto.** Se il layout a un solo bucket cambiasse l'ordine di somma di *tutti* gli elementi, ci si aspetterebbero
  ~10-30% di elementi diversi e uno scarto ~1e-11. L'osservato è 5-100 volte più piccolo: cambia solo una minoranza degli elementi.
- **Proiettore al 545.** `contrast_projector_grad_norm` ha scarto 1,0e-13 su 197.120 elementi: √k = 1e-13·197.120/1,19e-7 ≈ 0,17.
  Corrisponde a **circa un solo elemento** (piccolo) diverso, non a una riduzione riordinata su tutto il tensore.
  - [?] Spiegazione plausibile ma non verificata. In entrambi i layout il proiettore (788.480 byte, parametri 268, 269, 266, 267) sta all'inizio
    del buffer: nel bucket unico da 114 MB e nel primo bucket ricostruito da 3.241.700 byte. Dovrebbe quindi cadere nello stesso primo blocco
    dell'anello NCCL (~810 KB per rank nel bucket ricostruito), con lo stesso ordine di somma.
  - Dipende da canali e protocollo di NCCL 2.19 e va verificato con la strumentazione.
  - Conseguenza: il proiettore **non è l'origine** dello scarto. Lo riceve dopo e lo amplifica di più (InfoNCE: softmax con temperatura sulle chiavi).
- **[D] Il salto da ~1e-12 a 1e-5…1e-4 (546-548) è compatibile con il bf16.** Con l'autocast (`amp_bf16: true` nella config) pesi e
  attivazioni passano in bf16. Una differenza di pochi ulp float32 cambia il valore bf16 solo se attraversa un confine di arrotondamento
  (probabilità ~2⁻¹⁶ per ulp). Quando lo attraversa, il valore cambia di 2⁻⁸.
  Questo spiega perché CN al 546 resti identico (nessun attraversamento visibile) e poi salti di sette ordini invece di crescere in modo esponenziale.
  Dopo il salto la crescita è lenta (0,03 decadi/update): è un regime di rumore bf16 che si accumula, non una divergenza caotica rapida.
  [?] Non ho un conto quantitativo del valore 2e-5: dipende da quanti pesi e attivazioni bf16 cambiano, e gli artefatti non lo dicono.
- **[V+D] Correzione sui passi 545-546.**
  - Prova: il trace DDP si legge dopo `_accumulate_terms` (`scripts/contrastive_target_resume.py:189-201`). `_get_ddp_logging_data()`
    restituisce le statistiche fissate da `logger.set_runtime_stats_and_log()` all'**inizio** del forward, prima di `reducer._rebuild_buckets()`.
    Nel torch locale lo si legge in `torch/nn/parallel/distributed.py:1758` e poi `:1776`.
  - Quindi `has_rebuilt_buckets = 0` alla riga dell'update 546 descrive lo stato *prima* della ricostruzione, che avviene nello stesso forward del 546.
    La riduzione del 546 usa già i bucket ricostruiti.
  - Prova empirica: la norma del gradiente CN al 546 è identica bit per bit al continuo. Con un ordine di riduzione diverso sarebbe quasi impossibile.
  - [?] La sequenza è verificata sul torch locale; quella del cluster (2.2.0a0) non l'ho letta.
  - **Il layout diverso riguarda solo l'update 545**, non 545-546 come scrive GPT.
- **[D] Perché il proiettore supera l'1% mentre gli scalari restano a 1e-3.** `contrast_projector_grad_norm` confronta norme: |‖a‖−‖b‖| è
  di secondo ordine rispetto a ‖a−b‖. La soglia guarda invece la norma della differenza di m. Gli scalari dei log sottostimano lo scarto vettoriale.

## 4. Nodi, durate, GPU-ore

**[V] Tempi da sacct** (`benchmark_execution_20261121/allocation.sacct.tsv`, `reloads_execution_20261121/allocation.sacct.tsv`).
Abbinamento ID-braccio da `*_submission.txt`. Tutti `COMPLETED 0:0`, 4 GPU, 32 CPU, limite 30 min.

| run | job | ElapsedRaw | GPU-ore (×4/3600) | elapsed_s del trainer |
|---|---|---|---|---|
| C0 continuo | 65017240 | 795 s | 0,883 | 762,0 |
| CN continuo | 65017310 | **784 s** | 0,871 | 753,0 |
| CR continuo | 65020151 | 773 s | 0,859 | 737,9 |
| CN_A | 65033016 | **265 s** | 0,294 | 222,8 |
| CN_B | 65033756 | **213 s** | 0,237 | 164,3 |
| CR_A | 65033804 | 205 s | 0,228 | 160,5 |
| CR_B | 65034107 | 210 s | 0,233 | 163,8 |

- **[V] Le cifre di GPT sono corrette.** 784 + 213 = 997 s e 784 + 265 = 1.049 s, cioè 1,108 e 1,166 GPU-ore (1,11-1,17).
  Il tetto di 30 min × 4 GPU = 2 GPU-ore e 16 core-ore.
  Margine sul tetto: 0,83-0,89 GPU-ore (43-45%) per strumentazione, scrittura dei gradienti e avvio.
  Nota: CN_A ha 42 s di overhead fuori dal trainer contro 49 s di CN_B. I continui hanno circa 31 s di avvio.
- **[V] I nodi non sono registrati da nessuna parte.** I file sacct non hanno `NodeList`. I `.out` (`torch.distributed.run --standalone`) non
  stampano l'host. `summary.json` riporta solo `device: cuda:0`. Le righe `Nodes=lrdn3075` negli `*_operational_snapshot.txt`
  (CN: riga 97) appartengono alla **prenotazione `s_int_check_boost`** di `scontrol show reservation`, non al job.
  Non si può quindi dire se continuo, A e B abbiano girato sullo stesso nodo.
- **[D]** A e B sono job distinti (avvii alle 14:49 e alle 14:58), forse su nodi diversi, eppure sono identici bit per bit. Il nodo
  probabilmente non è un fattore, ma non è dimostrato. Per la prova nuova conviene registrare `SLURMD_NODENAME`/`hostname` e gli UUID delle GPU.

## 5. Implicazioni per la prova proposta da GPT

- La prova va centrata sull'**update 545**, con 546 come controllo, e su **tutti i 270 parametri**, non solo sui quattro del proiettore.
  - Il seme sta fuori dal proiettore (circa un elemento del proiettore diverso al 545). Guardare solo 266-269 rischia di vedere quasi zero
    e di concludere «causa aperta».
  - Bastano, per parametro e rank: sha256 e norma float64 del gradiente locale prima della riduzione e del gradiente ridotto prima del clip,
    più il numero di elementi diversi. I vettori completi servono solo per 266-269 e per qualche tensore di controllo.
- **Previsioni di H2 (falsificabili).**
  1. Al 545 i gradienti locali sono identici bit per bit su ogni rank e quelli ridotti differiscono di ≤1-2 ulp in una minoranza di elementi.
  2. Al 546 i gradienti ridotti sono identici in CN.
  3. Al 547 locale e ridotto differiscono, perché i pesi dopo il 546 sono già diversi.
  - **Smentite:** gradienti locali diversi al 545 (causa a monte: forward o ripristino); scarti di più ulp al 545; ridotti diversi al 546 con locali
    uguali (la lettura dei bucket sarebbe sbagliata).
- **Inerzia della strumentazione: un criterio esatto gratuito.** La strumentazione solo in lettura (hash e norme dopo il backward, nessuna
  modifica dei `.grad`) deve riprodurre gli sha256 finali dei checkpoint a 576.
  - CN continuo: `4743abaff631d43c0f9267b886087fdcbefdfc156332bf34d43b668297668dec`.
  - CN ripresa: `bb39f33998df27c2a0b7b6ca7554c77afccf08bb7405831a17a28c4ef6594301`.
  - In alternativa, a parità almeno della ripresa: le riprese hanno già dimostrato determinismo fra job.
  - Se il continuo rifatto da zero non riproduce `4743ab…`, questo è già un risultato: non-determinismo fra run del continuo.

## Riferimenti principali

- `src/wearusfm/training/run.py:1429` (AdamW), `:1691-1700` (norma del proiettore e clip), `:1722` (commit delle chiavi), `:436-438` (DDP di default).
- `src/wearusfm/training/contrastive_memory.py:182-253` (coda).
- `src/wearusfm/model/anchors.py:47-48` e `src/wearusfm/training/jepa.py:25-26` (`index_add` non deterministico).
- `src/wearusfm/model/contrastive.py:15-27` (proiettore).
- `scripts/contrastive_target_resume.py:153-201` (tracce).
- Profilo numerico (`E/benchmark_execution_20261121/CN/numeric_profile.rank0.json`): `deterministic_algorithms: false`, `cudnn_deterministic: false`,
  `bf16_reduced_precision_reduction: true`, torch `2.2.0a0+git6c8c5ad`.
