# Norme dei gradienti contrastivi: diagnosi del 18/11/2026

Richiesta di Simone: distinguere variabilità numerica bf16 da un errore di codice.
Analisi descrittiva successiva ai risultati, sui cinque benchmark già archiviati;
nessuna nuova soglia, modifica del runtime o sottomissione. I lanci sono affidati
a 3-launcher; Simone riferisce di avergli consegnato il prompt della lettura CPU.

## Conclusione

**Non è emerso un errore dimostrato nel calcolo della norma o nella selezione del
braccio a peso zero. La deriva numerica è plausibile, ma la causa bf16 non è
dimostrata. È invece dimostrato che il confronto delle riprese mescola due effetti:
variabilità fra esecuzioni indipendenti ed eventuale effetto del riavvio.**

Il gate di parità resta negativo. Non si rilasciano training lunghi e non si
allargano le tolleranze per far passare questi risultati. Il superamento del CUDA
a due update non garantisce parità fra traiettorie indipendenti di 256–576 update.

## Misure riproducibili

Script e risultati in
`results/infra/contrastive_comparison_20261116/gradient_parity_diagnostic_20261118/`.
`analyze.py` legge i cinque gzip originali, conserva hash dei dati e confronta tutte
le dieci coppie nei primi 256 passi. Scarto relativo qui definito come
`abs(a-b)/max(abs(a),abs(b))`; nelle tabelle è espresso in percentuale.
Il gate originale usa `math.isclose(rel_tol=1e-4, abs_tol=1e-6)`: circa 0,01% alle scale
delle norme osservate. Nessun test di significatività su queste traiettorie.

| Confronto, passi 1–256 | Norme oltre il limite | Primo superamento | Scarto massimo norma | Loss totale oltre il limite |
|---|---:|---:|---:|---:|
| C0 / CN continuo | 146 / 256 | 83 | 0,725% | 0 / 256 |
| C0 / CR continuo | 135 / 256 | 79 | 0,314% | 0 / 256 |
| CN continuo / CN destinato alla ripresa | 144 / 256 | 88 | 1,183% | 1 / 256 |
| CR continuo / CR destinato alla ripresa | 142 / 256 | 79 | 0,846% | 0 / 256 |

Le ultime due righe sono controlli descrittivi aggiuntivi, non gate nuovi.
I config dei tre bracci differiscono solo in `contrastive.arm`; quelli delle due
esecuzioni finali dello stesso braccio coincidono. Il config della prima fase del
run interrotto differisce dal continuo solo per `max_steps=544` invece di 576.
I cinque record di inizializzazione coincidono; tutte le origini delle finestre
e le mappe delle sessioni coincidono per 576 update.

La prima differenza non esattamente nulla è già all'update 1: scarto relativo della
norma CN/C0 circa 9,42e-7, CR/C0 circa 1,30e-6, ancora sotto la tolleranza.
L'avvio precede sia le diagnostiche globali (prima al passo 100) sia l'attivazione
delle contrastive (passo 257). Queste ultime non spiegano l'inizio della deriva.

## La prova di ripresa non isola il riavvio

`scripts/contrastive_split_benchmark.py` addestra autonomamente fino a 544, salva e
riavvia per 32 update. Il confronto in `contrastive_benchmark_readout.py` usa però
come riferimento un'altra esecuzione da zero fino a 576. Non sono due continuazioni
del medesimo stato a 544. Le differenze sono già ampie prima della separazione:

| Stesso braccio, continuo / interrotto | Mediana scarto norma 513–544 | Massimo 513–544 | Mediana 545–576 | Massimo 545–576 |
|---|---:|---:|---:|---:|
| CN | 3,22% | 12,27% | 2,43% | 10,85% |
| CR | 12,97% | 48,82% | 13,14% | 47,42% |

Anche le componenti globali divergono prima del riavvio: in CR, ai passi 513–544,
mediane dello scarto 20,93% per la base e 17,59% per RnC; la loss totale differisce
in mediana dello 0,295%. Non è quindi soltanto una discrepanza del numero stampato
dal clipping, né basta chiamarla rumore innocuo inferiore all'1%.
Non si confrontano direttamente CN e CR dopo 256, quando gli obiettivi differiscono.
Il confronto dei checkpoint rimane utile e i suoi limiti restano vincolanti, ma un
eventuale fallimento non dimostra da solo un bug di ripresa. Viceversa questi dati
non dimostrano che la ripresa sia corretta.

La figura `gradient_divergence.png` mostra i due confronti lungo tutti i 576 passi,
con fine del peso zero a 256 e riavvio dopo 544; non aggiunge criteri di superamento.

### Lettura CPU completata durante questa analisi

Job di 3-launcher **64775571**, FAILED 2:0 dopo 124 s: uscita prevista per un rapporto
completo negativo, non un crash del lettore. `preflight_readout.json` riporta
`FAILED_REQUIRES_REVIEW`, `training_approved=false`; SHA256
`deb6b8178362e23512f105d95b3859c28945d9ffa5356a653f1c8cad67c929fc`.
Estratto remoto e hash in `cpu_readout_remote_summary.json`; l'archiviazione del
rapporto integrale e del consuntivo operativo resta alla chat 3-launcher.

| Confronto finale continuo / ripreso | Student | Teacher | Limite congelato |
|---|---:|---:|---:|
| CN | 3,248% | 2,047% | 1% |
| CR | 6,012% | 2,337% | 1% |

Questi valori sono `norm(p_continuo-p_ripreso)/norm(p_continuo-p_iniziale)`,
**non** percentuali rispetto al valore totale dei pesi, né la misura simmetrica
delle norme usata sopra. Anche optimizer e chiavi della coda hanno tensori oltre
il limite (1.688 violazioni CN, 1.375 CR). I controlli esatti su struttura, buffer,
RNG finale per rank, stream, config e provenienza non hanno sollevato eccezioni;
il lettore li esegue prima di produrre questo rapporto. Questo non attesta gli
input e i generatori di ogni passo. Entrambi i rapporti delle componenti **attive**
sono invece entro i limiti: mediane CN 0,2261, CR 0,0834, rapporto 2,7106.
Il rapporto CPU conferma differenze effettive degli stati, senza risolverne la
causa né isolare il contributo della ripresa.

## Verifica del codice e prova locale indipendente

- `contrastive_objectives.py:102`: i pesi sono entrambi zero fino all'update 256
  per tutti i bracci. Loss, proiettore, memoria e raccolte distribuite percorrono
  lo stesso codice; i termini a peso zero restano nel grafo.
- `run.py:1677`: `grad_norm` è la norma globale dei gradienti dei parametri restituita
  da `clip_grad_norm_`, **prima del clipping**, dopo il backward distribuito.
  Non è una norma bf16 semplicemente stampata con poche cifre. Le norme delle tre
  componenti (`global_gradients`, encoder condiviso, riduzione finale FP64) sono
  misure diverse dalla norma su tutti i parametri: non vanno confuse.
- Stato iniziale, seed per rank e percorso di dropout sono comuni. Nei cinque log
  il gradiente del proiettore è zero nei primi 256 passi. Il weight decay eventuale
  del proiettore è anch'esso comune.
- Review indipendente: prova locale di **12 update per quattro varianti** (C0, CN,
  CR e C0 senza diagnostiche), vero loader/trainer su dati sintetici, modello piccolo,
  CPU senza autocast (prevalentemente FP32, frontend FP64), dropout 0,4,
  checkpointing, 100 ms e GELU. Student/teacher, optimizer,
  RNG, coda, coverage, loss, norme e clipping sono bit-identici. Diagnostiche ai passi
  4/8/12 senza cambi dell'aggiornamento. Script originale e ricevuta archiviati come
  `zero_weight_cpu_probe.py` e `zero_weight_cpu_result.json` (Torch 2.14, 46,82 s).

La prova locale non replica il 30M CUDA bf16/NCCL e non supera il warmup della coda.
I 117 sorgenti congelati sono invariati: SHA256
`46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320`.

## Cosa manca per attribuire la causa

Il bf16 riduce la precisione; **non introduce casualità da solo**. Con gli stessi
input e identica sequenza di operazioni deterministiche si può ripetere esattamente
anche un calcolo bf16. Ordine delle riduzioni, kernel CUDA e calcolo distribuito
possono introdurre differenze, poi amplificate da arrotondamenti, clipping e update.
È un'ipotesi coerente con il piccolo scarto iniziale, non una misura della causa.

PyTorch documenta l'effetto dell'ordine delle operazioni e delle riduzioni a
precisione ridotta, e distingue il seed dal determinismo degli algoritmi:
[accuratezza numerica](https://docs.pytorch.org/docs/2.2/notes/numerical_accuracy.html),
[riproducibilità](https://docs.pytorch.org/docs/2.2/notes/randomness.html).
Il modello usa SDPA con scelta automatica del backend; i backend possono differire
numericamente ([documentazione SDPA](https://docs.pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html)).
Non abbiamo una traccia del kernel effettivamente scelto né attestazioni runtime
di determinismo: nessuna attribuzione specifica a FlashAttention, cuDNN o NCCL.
L'ambiente reale è Torch `2.2.0a0+git6c8c5ad`, CUDA 12.1, NCCL 2.19.1.

Inoltre `origin` contiene sessione/trial/inizio, non un hash dei campioni, delle
maschere o del RNG di ogni passo. `pairing_audit` e `prefetch_audit` sono spenti.
Origini e seed uguali non certificano da soli tutti i tensori effettivi uguali.
I metadati non contengono vettori gradiente per confrontare direzione e scala.

## Seguito proposto, senza lancio

1. Conservare la lettura CPU negativa ora completata da 3-launcher: stati, RNG,
   coda, optimizer e vincoli delle riprese. Nessun retry del lettore riuscito.
2. Preparare un piccolo diagnostico CUDA a **stato e batch comuni**: attestare
   input/maschere/RNG; ripetere prima lo stesso C0, poi C0/CN/CR a peso zero,
   ripristinando l'intero stato prima di ogni confronto. Registrare gradienti per
   blocco e norma ricalcolata FP64. Confrontare bf16 ordinario, bf16 deterministico
   e riferimento FP32 con backend espliciti e controlli compatibili con la memoria.
   Un'operazione non deterministica non supportata deve essere riportata, non aggirata.
3. Per isolare la ripresa, salvare **un unico stato al punto di separazione** e
   confrontare il seguito nello stesso processo con quello caricato in un processo
   nuovo, accanto a una replica dello stesso seguito per misurare la variabilità.

Prima di nuovi job: piano concreto, risorse e criteri pubblicati; nessun cambio di
lambda, ricetta scientifica o tolleranza del rapporto attuale. Non serve ripetere
alla cieca i cinque benchmark né lanciare ora i training lunghi.
