export const meta = {
  name: 'review-prova-mirata-cn',
  description: 'Review del diagnostico a due riprese InfoNCE (GPT 9d34dc69): fedelta, osservatore, confronto, pacchetto, test, catena CPU, critico; poi verifica avversaria dei bloccanti',
  phases: [
    { title: 'Revisione', detail: 'sette revisori indipendenti su codice e artefatti' },
    { title: 'Verifica', detail: 'due scettici per ogni rilievo bloccante o importante' },
  ],
}

const BASE = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe'
const CLONE = BASE + '/clone'

const COMMON = `Contesto. Progetto WearUsFM, linea 3. Claude e' il revisore; GPT ha scritto un diagnostico GPU ("prova mirata") e chiede a Claude di approvarlo prima del lancio su Leonardo (lo lancia un altro operatore, 3-launcher). Tu lavori per Claude come revisore indipendente di una parte. Vincoli assoluti: NON lanciare nulla su Leonardo (niente ssh), niente rete, niente git commit o push, nessuna modifica al clone. Lavora in sola lettura sul clone; scrivi i tuoi file solo nelle cartelle indicate.

Che cosa fa il diagnostico. Un training InfoNCE ("CN") a 4 GPU con DDP e autocast bf16 e' stato salvato al passo 544 ("fork") e portato a 576 ("continuo"). Due riprese dal fork (CN_A, CN_B) danno a 576 checkpoint identici fra loro ma diversi dal continuo. Ipotesi: nel processo riavviato la PRIMA riduzione DDP (update 545) usa i bucket di costruzione (1 bucket fp32 da 114.202.292 byte + 1 float64 da 68.480 byte) invece dei 7 bucket ricostruiti del continuo, e questo cambia il gradiente ridotto di un arrotondamento. Il diagnostico esegue due riprese dal fork nella stessa allocazione: "ordinary" (ripresa normale, strumentata) e "warm" (prima del vero 545 fa un backward di riscaldamento a effetto nullo, cosi' il 545 riduce con i 7 bucket). Al 545 registra, per rank e per tensore (270 parametri), il gradiente locale prima della riduzione e quello ridotto, tramite un comm hook DDP; poi confronta le due riprese fra loro e con gli artefatti storici (checkpoint CN_A bb39f339..., continuo 4743abaf..., tracce e metriche archiviate). Esiti possibili del report: SUPPORTED, PARTIAL_SUPPORT, HYPOTHESIS_UNSUPPORTED, INCONCLUSIVE. Il diagnostico ha UN SOLO tentativo (tetto 15 minuti, 4 GPU): un difetto che lo rende INCONCLUSIVE per ragioni non scientifiche, o che fa attribuire male l'esito, e' un difetto bloccante.

Percorsi.
- Clone del ramo di GPT al commit candidato 9d34dc690cf60fcbfe405279dc17546655fa3f5b (detached, sola lettura): ${CLONE}
- Diff da rivedere: git -C ${CLONE} diff 918245b6 9d34dc69 (14 file: scripts/diagnostics/contrastive_cn_restart{,_leg,_observe,_compare}.py, scripts/slurm/cn_restart_probe.sbatch, results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/{plan.json,local_validation.json}, docs/contrastive_cn_restart_probe_20261121.md, docs/decisioni.md (ultima voce), tests/cpu/test_cn_restart_{package,leg,observe,compare}.py).
- Codice di produzione usato dal diagnostico (non modificato dal diff): src/wearusfm/training/run.py (trainer: _accumulate_terms, DDP, payload del checkpoint, ripresa), scripts/contrastive_run.py, scripts/contrastive_target_resume.py (traccia della ripresa, rows, checkpoint, TRACE_EQUAL), scripts/contrastive_job.py, scripts/contrastive_release.py, scripts/contrastive_split_benchmark.py (checked_command), src/wearusfm/training/contrastive_objectives.py, contrastive_memory.py, pairing.py (state_sha256).
- Artefatti storici veri nel clone: results/infra/contrastive_targets_20261118/benchmark_execution_20261121/CN/ (continuo: metrics.jsonl, trace-<rank>.jsonl, trace-<rank>.ddp.jsonl, config.json, summary.json, numeric_profile.rank<r>.json), .../reloads_execution_20261121/CN_A/ e CN_B/ (riprese: anche start.json), .../readout_execution_20261121/preflight_readout.json.gz, .../releases/reloads_20261121/reload_CN_A.job.json e .release.json. I checkpoint .pt NON sono sul Mac.
- Letture del torch del cluster (torch 2.2.0a0+git6c8c5ad, Python 3.11.6), fatte da Claude in sola lettura: ${BASE}/leonardo/torch22_sorgente_20261121.out.txt e ${BASE}/leonardo/torch22_api_20261121.out.txt. Fatti accertati li': (1) torch.load con mmap=True esige isinstance(f, str), altrimenti ValueError "f must be a string filename in order to use mmap argument" (serialization.py:1012-1014); (2) _CudaDeviceProperties NON ha l'attributo uuid; (3) GradBucket ha buffer, gradients, index, is_last, parameters, set_buffer; (4) allreduce_hook fa tensor.div_(size) e poi all_reduce asincrono; (5) _check_comm_hook richiede un parametro di nome "bucket".
- La tua cartella di lavoro (creala): ${BASE}/agents/<chiave> . Il tuo rapporto: ${BASE}/reports/<chiave>.md (aggiornalo man mano: se vieni interrotto deve restare utile).

Strumenti. Python: ~/.venvs/wearusfm-tok/bin/python (torch 2.14 solo CPU, Python 3.11). Comando dei test: cd ${CLONE} && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:scripts/diagnostics ~/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=<una cartella nuova sotto la tua cartella di lavoro> <file di test> . I test Gloo aprono solo socket locali. Shell zsh: per elenchi di file usa bash -c; scrivi \${VAR} quando una variabile e' seguita da due punti. Il Mac ha poca RAM (18 GB) e poco disco: niente run oltre qualche minuto, e cancella i file grandi che crei. Per provare il comportamento del torch del cluster sul Mac puoi usare uno shim nel tuo script (per esempio avvolgere torch.load in modo che sollevi lo stesso ValueError quando mmap=True e f non e' str): dichiaralo come simulazione.

Difetto gia' trovato da Claude (non rifarlo, ma tienine conto): B-1, scripts/diagnostics/contrastive_cn_restart_observe.py:144 chiama torch.load(tensor_path, ..., mmap=True) con tensor_path di tipo pathlib.Path: sul torch 2.2 del cluster solleva ValueError, compare.py lo cattura e il report esce INCONCLUSIVE dopo aver consumato le GPU.

Regole di consegna. Ogni rilievo deve avere: gravita' ("bloccante" = puo' rendere il diagnostico INCONCLUSIVE o fallito per ragioni non scientifiche, o far attribuire male l'esito, con un controesempio concreto; "importante" = rischio reale ma mitigabile o di probabilita' incerta; "minore"; "nota"), file:riga, scenario concreto (input o stato -> esito sbagliato), prova (comando e output, o citazione del codice), e correzione proposta minima. Distingui cio' che hai eseguito da cio' che deduci dal codice. Non gonfiare: un rilievo senza controesempio non e' bloccante. Elenca anche le verifiche positive importanti (che cosa hai controllato e risulta corretto), in breve. Rispondi con l'oggetto strutturato richiesto.

`

const TASKS = [
  { key: 'r1-fedelta-ripresa', prompt: `Compito R1 (chiave: r1-fedelta-ripresa) - Le due celle eseguono la vera ripresa CN_A? Il riscaldamento e' davvero a effetto nullo?
1. Confronta la riga di comando ricostruita dal diagnostico (contrastive_cn_restart.py:commands e contrastive_cn_restart_leg.py:validate_argv e main) con quella della ripresa storica CN_A (scripts/contrastive_job.py build_command per mode 'resume', il job archiviato reload_CN_A.job.json e la sua release). Manca o cambia qualche argomento? Il runner di produzione (scripts/contrastive_run.py main) riconosce la ripresa (fase resume_a, fork a 544) con questi argomenti? Qualche controllo del runner dipende dalla cartella di uscita, che qui e' diversa da quella storica? Il payload del checkpoint (run.py: payload e save) contiene qualcosa che dipende dalla cartella o dal job, tale da cambiare i byte del file rispetto a bb39f339...? (Nota: CN_A e CN_B, con cartelle diverse, hanno lo stesso sha del file.)
2. Catena delle patch: instrument_leg sostituisce trainer._accumulate_terms e AdamW.load_state_dict; poi il runner di produzione installa la sua traccia (contrastive_target_resume.trace_resume_updates) che sostituisce di nuovo _accumulate_terms e AdamW.__init__. Ricostruisci l'ordine esatto delle chiamate al backward del 545 nelle due varianti: quando viene registrato il comm hook, quando parte il riscaldamento, quando il trainer fa il primo clipping reale e il primo opt.step. La traccia di produzione registra i suoi RNG "before"/"after" e il confine nel punto giusto rispetto al riscaldamento?
3. null_backward: leggi tutto cio' che trainer._accumulate_terms e le funzioni che chiama (forward di studente e teacher, ContrastiveRuntime.attach e global_gradients, loss, backward) possono MODIFICARE: argomenti e kwargs condivisi (oltre a "contrastive", che viene isolato con deepcopy), micros (per esempio attributi calcolati e messi in cache come anchor_targets), oggetti monitor o contatori, attributi Python dei moduli, buffer, generatori casuali (torch globale CPU e CUDA, random, numpy, eventuali torch.Generator dedicati), stato del runtime vivo. Per ciascuno di' se snapshot() lo copre, se null_backward lo ripristina, e che cosa succederebbe altrimenti: (a) before != after -> la gamba warm fallisce (INCONCLUSIVE falso); (b) stato alterato NON rilevato -> gradienti locali diversi al 545 -> esito HYPOTHESIS_UNSUPPORTED attribuito male; (c) metadati del checkpoint a 576 diversi dal continuo (per esempio alarm_counts) -> INCONCLUSIVE falso. Cerca in particolare generatori o stati non coperti.
4. Esegui davvero: lancia tests/cpu/test_cn_restart_leg.py e leggi che cosa coprono; poi scrivi tuoi esperimenti che chiamino le funzioni vere (snapshot, null_backward, instrument_leg con il vero trainer._accumulate_terms su un modello WearUsFM piccolo con testa contrastiva e runtime preparato, come fanno i test) per verificare: snapshot prima e dopo un riscaldamento coincidono; un backward vero dopo il riscaldamento da' gli stessi gradienti locali di un backward senza riscaldamento; deepcopy del runtime funziona con la coda vera.
5. device_identity: sul cluster props.uuid non esiste, quindi gira il ramo ctypes (libcuda.so.1, cuDeviceGet, cuDeviceGetUuid). Controlla firme e tipi rispetto all'API del driver CUDA; di' che cosa succede se fallisce e se c'e' un modo piu' robusto senza perdere il controllo "stesse GPU per rank nelle due gambe".` },
  { key: 'r2-catena-cpu', prompt: `Compito R2 (chiave: r2-catena-cpu) - Prova della catena intera su CPU con le funzioni vere (4 processi Gloo).
Obiettivo: eseguire sul Mac, con modelli e configurazioni piccoli ma con le funzioni VERE del diagnostico e del trainer, la sequenza completa: (a) un training continuo DDP a 4 rank Gloo che salva un fork e prosegue; (b) una ripresa dal fork senza strumentazione; (c) una ripresa "ordinary" con ReductionObservation e gli hook di record_optimizer_step; (d) una ripresa "warm" con null_backward; (e) il confronto. Usa quanto piu' possibile il codice vero: contrastive_cn_restart_leg.instrument_leg (o le sue parti), contrastive_cn_restart_observe.ReductionObservation e load_observation, contrastive_cn_restart_compare.compare_vector, layouts, checkpoint_differences, classify. Guarda come i test esistenti (tests/cpu/test_cn_restart_leg.py, test_cn_restart_observe.py, test_cn_restart_compare.py) costruiscono i loro fixture a 4 rank e riusali o estendili nella tua cartella.
Domande a cui rispondere con esecuzioni:
1. Hook contro percorso nativo: con il comm hook registrato (allreduce_hook di riferimento), una ripresa da' gradienti ridotti, parametri e momenti identici bit per bit a quelli della stessa ripresa senza hook? Provalo con dtype misti (float32 e float64) e piu' update.
2. Effetto del riavvio su Gloo/CPU: il continuo e la ripresa non riscaldata differiscono? A quale update compare la prima differenza e di quanto? (In una prova precedente su CPU la differenza esisteva.)
3. Intervento: la ripresa "warm" (un backward a vuoto prima del primo update vero, poi zero_grad e ripristino di RNG e buffer) ridiventa identica bit per bit al continuo? I bucket osservati dall'hook al primo update vero sono quelli ricostruiti? Se la warm NON recupera il continuo su CPU, capisci perche' (ordine di prontezza diverso, altro stato) e di' se lo stesso puo' succedere sul cluster.
4. Con lo shim del torch 2.2 (torch.load che rifiuta Path con mmap=True; get_device_properties senza uuid), quali funzioni del diagnostico falliscono? Costruisci il controesempio riproducibile di B-1 (load_observation su un'osservazione vera prodotta dal tuo fixture) e controlla se ci sono altri punti che falliscono con lo shim.
5. Quale status produce il confronto sulle tue esecuzioni? Se per ragioni di fixture non puoi chiamare compare() intero (richiede gli artefatti storici e i checkpoint), chiama le sue funzioni sui tuoi dati e segnala ogni eccezione inattesa.
Riporta comandi, output essenziali e tempi. Cancella i file grandi alla fine.` },
  { key: 'r3-osservatore', prompt: `Compito R3 (chiave: r3-osservatore) - L'osservatore della riduzione (scripts/diagnostics/contrastive_cn_restart_observe.py) e gli hook dell'ottimizzatore (record_optimizer_step in contrastive_cn_restart_leg.py).
1. Correttezza: il comm hook cattura il buffer del bucket PRIMA che allreduce_hook lo divida e lo riduca? "local" e' quindi il gradiente locale non scalato e "reduced" il risultato della riduzione? Il controllo "reduced == param.grad" in finish regge in tutti i casi (gradient_as_bucket_view, parametri float64, parametri con grad None)? La copertura dei 270 parametri, una volta per rank, e' garantita? Offset, stride e dtype sono ricostruiti correttamente (as_strided sul buffer CPU)? L'identita' id(param) fra bucket.parameters() e student.named_parameters() regge?
2. Equivalenza con il percorso nativo: nel Reducer C++ senza hook il gradiente viene moltiplicato per 1/world_size mentre viene copiato nel bucket e poi ridotto; con l'hook viene copiato non scalato e allreduce_hook fa div_(world_size) e all_reduce asincrono sullo stesso buffer. Argomenta se i bit del risultato coincidono (potenze di due, subnormali, ordine di lancio dei bucket, copia finale nei .grad). Verificalo empiricamente su CPU/Gloo a 4 processi con dtype misti: stessa ripresa con e senza hook, piu' update, confronto bit per bit di gradienti ridotti e parametri. Indica che cosa NON si puo' verificare senza NCCL.
3. Compatibilita' con il torch 2.2 del cluster di OGNI chiamata usata nei quattro script nuovi (torch, torch.distributed, hashlib, zip strict, ecc.): elenca quelle che non esistono o si comportano diversamente in 2.2 (usa le due letture del cluster indicate sopra e la tua conoscenza del sorgente 2.2; segnala cio' che non puoi verificare). B-1 e' gia' noto: cercane altri dello stesso tipo (Path dove serve str, argomenti aggiunti dopo la 2.2, attributi mancanti).
4. Costi dell'osservatore sul run vero: memoria GPU in piu' al 545 (clone dei bucket: 114 MB per rank), tempo, dimensione dei file vectors.rank<r>.pt (114 MB x 2 stadi x 4 rank x 2 gambe), scritture su Lustre dentro il backward o dopo; rischi di sincronizzazione o di deadlock nell'hook (GIL, thread dell'autograd).
5. record_optimizer_step: gli hook pre e post del passo leggono davvero il gradiente dopo il clipping e lo stato subito dopo lo stesso step? Esistono in torch 2.2? Vengono rimossi dopo il 545? Alterano qualcosa?
6. Lancia tests/cpu/test_cn_restart_observe.py e di' che cosa copre e che cosa no.` },
  { key: 'r4-confronto', prompt: `Compito R4 (chiave: r4-confronto) - Le regole di lettura (scripts/diagnostics/contrastive_cn_restart_compare.py) contro i criteri registrati (docs/decisioni.md, ultima voce del 21/11, e docs/contrastive_cn_restart_probe_20261121.md, tabella "Lettura fissata prima della misura").
1. Per ogni criterio registrato, trova il codice che lo attua e verifica che l'esito sia quello dichiarato: errore di fedelta' dell'ordinaria o del confine -> INCONCLUSIVE; gradienti locali diversi al 545 -> HYPOTHESIS_UNSUPPORTED; scarto ridotto oltre 1e-6 relativo per tensore -> HYPOTHESIS_UNSUPPORTED; nessuno scarto ridotto -> HYPOTHESIS_UNSUPPORTED; localizzazione senza recupero delle metriche del continuo al 545 o dei tensori a 576 -> PARTIAL_SUPPORT; recupero completo -> SUPPORTED; nessun esito autorizza training; il vecchio FAILED_REQUIRES_REVIEW resta. Segnala criteri registrati che il codice non attua, e controlli del codice non registrati che possono cambiare l'esito.
2. Prova le funzioni sui VERI artefatti storici del clone, simulando le gambe con gli artefatti archiviati: per esempio branch_traces(folder=archivio CN_A, reference=archivio del continuo CN, mode="warm", rank) deve passare se una ripresa soddisfa le uguaglianze di traccia contro il continuo; metrics_evidence(archivio CN_A, archivio continuo, "warm") deve dare differenze dal 545 in poi; metrics_evidence(archivio CN_B, archivio CN_A, "ordinary") deve dare zero differenze (CN_A e CN_B sono identiche); layouts() sul vero trace-<rank>.ddp.jsonl del continuo con osservazioni sintetiche costruite dal layout loggato; history() fin dove possibile senza i checkpoint. Riporta ogni eccezione o esito inatteso: sono esattamente i casi che renderebbero INCONCLUSIVE il run vero.
3. checkpoint_differences: quali tipi di foglie contiene il payload vero del checkpoint (run.py: payload e save: student, teacher, optimizer, step, alarm_counts, config, torch_rng, cuda_rng, stream, data_signature, pose_stats_sha256, contrastive_state, contrastive_initial_sha256, contrastive_provenance, rank_rng, world_size)? Ci sono foglie numpy (per esempio nello stato dello stream o negli RNG), per le quali "a != b" solleva ValueError o da' un array? Ci sono campi che fra una ripresa e il continuo differiscono per costruzione (per esempio RNG CUDA di dispositivi non usati, contatori di sessione) e che finirebbero in metadata_or_schema_different rendendo il confronto INCONCLUSIVE anche con tensori identici? Il confronto storico (contrastive_target_resume.historical_numeric_pair e contrastive_benchmark_readout._nested_equal) confrontava solo alcune parti e gestiva gli ndarray: usa questa differenza come traccia. Costruisci payload realistici (anche con il vero stream del trainer, se riesci a istanziarlo) e prova la funzione.
4. compare_vector e la soglia 1e-6: il denominatore e' max delle due norme L2 del gradiente ridotto. Se fra i rank c'e' forte cancellazione (norma del ridotto molto piu' piccola delle norme locali), uno scarto da solo arrotondamento puo' superare 1e-6? Stima con numeri e, se possibile, con una simulazione fp32 a 4 addendi. Che cosa registra il report per permettere di distinguere a posteriori?
5. Conteggi e identita': differenze di segno dello zero, NaN, tensori vuoti, parametri float64, uguaglianza fra rank dei ridotti e dei dati dopo AdamW, controllo dei posizionamenti (stesso nodo, 4 GPU distinte, stesse GPU per rank nelle due gambe).
6. Lancia tests/cpu/test_cn_restart_compare.py; poi fai test di mutazione mirati sulle guardie principali di compare.py (disattiva o inverti una condizione alla volta in una COPIA nella tua cartella) e di' quali mutazioni i test non rilevano.` },
  { key: 'r5-pacchetto-operativo', prompt: `Compito R5 (chiave: r5-pacchetto-operativo) - Pacchetto, binding della review, piano, lanciatore, tempi e costi.
1. contrastive_cn_restart.py: package(), review_fields(), bind_review(), require_binding(), commands(), run(). Verifica: quali file entrano nell'impronta del pacchetto e quali restano fuori (i test? i moduli di produzione importati?); che il blocco di review richiesto (schema cn-restart-review-v1, status APPROVED_FOR_DIAGNOSTIC, reviewed_commit, package_sha256, plan_sha256) sia accettato solo se unico e con valori esatti; che cosa succede con un blocco in piu' citato come esempio nello stesso file, con righe CRLF, con spazi finali, con caratteri non ASCII nel resto del file (bind_review decodifica in UTF-8?). Simula il flusso vero in un clone temporaneo tuo (copia del clone): crea un ramo che faccia da refs/remotes/origin/worktree-claude-training-engine con un file docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md che contiene il blocco corretto, esegui --bind-review <commit> e poi require_binding(); prova i casi negativi (commit non antenato del ramo, blocco con hash sbagliato, blocco doppio, file del pacchetto modificato dopo la review, vecchio blocco contrastive-release-review). Il vecchio binding del pacchetto di produzione (job['package_review']) viene ancora validato?
2. plan.json e pin storici: i percorsi e gli sha del fork, del riferimento ordinario e del continuo coincidono con quelli degli artefatti storici (rapporto preflight_readout.json.gz, start.json di CN_A, releases/reloads_20261121)? Il job storico pinnato e' quello di CN_A? La cartella di uscita sta fuori dal repository e deve essere nuova?
3. scripts/slurm/cn_restart_probe.sbatch: risorse, controlli, calcolo di SLURM_JOB_END_TIME, variabili richieste, coerenza con i comandi di invio nel documento (submit_job.py con --script-stdin: gli argomenti dopo "--" e l'export arrivano a sbatch? "$#" == 0 regge quando lo script e' passato su stdin?). Confronta con scripts/slurm/contrastive_preflight_gpu.sbatch e con lib_code_dir.sh. Controlla "bash -n".
4. Tempo: il tetto e' una sola allocazione da 900 s per due gambe piu' il confronto. Stima il tempo di ogni fase dai dati storici (metrics.jsonl: t_step_s dei passi 545-576 e del primo passo; elapsed dei job storici 213-265 s; avvio di torchrun; verifica dei pin con sha256 di file da 463 MB; require_binding in ogni processo con git show di molti file; scrittura di 1,83 GB di vettori; confronto con lettura di 4 checkpoint e 8 file di vettori). Il margine basta? Che cosa succede allo scadere (checked_command e deadline in run(); execution_failure.json; i file gia' scritti restano utilizzabili)?
5. Costi: 1 GPU-ora e 8 ore locali per il job, 0,2667 per il checkout, totale 8,2667; inviluppo con i margini SLURM; residuo sul tetto firmato (18 GPU-ore e 169,6 core-ore, consumate 4,74 e 47,3322). Rifai i conti. Spazio su disco richiesto in $WORK (due checkpoint da 463 MB, copie del fork, vettori).
6. --validate-only sul login node: che cosa esegue davvero (hash di file grandi, subprocess del runner di produzione con --validate-only)? Puo' superare limiti del login node o fallire per ragioni note?
Riporta difetti operativi con controesempio.` },
  { key: 'r6-test', prompt: `Compito R6 (chiave: r6-test) - Test e loro forza.
1. Riproduci la ricevuta locale: results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/local_validation.json dichiara 170 test passati negli 11 file elencati. Esegui prima i quattro file nuovi (tests/cpu/test_cn_restart_{package,leg,observe,compare}.py) e poi gli altri sette, con il comando indicato sopra; riporta conteggi e tempi, ed eventuali test instabili (rilancia due volte quelli Gloo).
2. Per ciascuno dei quattro file nuovi: che cosa prova davvero (funzioni vere o sostituti?), quali rami critici NON sono coperti. In particolare: load_observation su torch 2.2 (Path con mmap), il ramo ctypes di device_identity, la scadenza in run(), bind_review e require_binding su un repository vero, la variante warm con il vero _accumulate_terms, il confronto completo compare() con artefatti realistici, i casi INCONCLUSIVE/UNSUPPORTED/PARTIAL/SUPPORTED.
3. Test di mutazione sui quattro script nuovi: copia gli script nella tua cartella, applica una mutazione alla volta (almeno 30 in tutto, scelte sulle guardie che proteggono la validita' scientifica: controlli di confine, di copertura dei 270 parametri, di uguaglianza fra rank, soglia 1e-6, classificazione, pin, binding) e lancia i test contro la copia mutata (PYTHONPATH con la tua copia PRIMA di scripts/diagnostics). Riporta quante mutazioni sopravvivono e quali; per le piu' gravi proponi il test mancante.
4. Segnala test che passano sul Mac ma non proverebbero nulla sul cluster (differenze di versione di torch, assenza di CUDA).` },
  { key: 'r7-critico', prompt: `Compito R7 (chiave: r7-critico) - Critico di completezza e coerenza.
Leggi tutto il diff (i quattro script, il lanciatore, il piano, il documento di protocollo, la voce di decisioni.md, i test) e rispondi:
1. Coerenza: il codice attua esattamente cio' che protocollo e decisioni.md registrano? Elenca ogni scostamento (soglie, denominatori, numero di gruppi attesi, metriche confrontate, stati controllati al confine, regole per INCONCLUSIVE, cosa conta come "identita' a 576"). Il piano (plan.json) e' coerente con codice e documento (per esempio relative_gradient_roundoff_limit e gradient_norm_floor sono letti dal piano o ricodificati nel codice)?
2. Il diagnostico ha un solo tentativo. Fai l'elenco, in ordine di probabilita' per impatto, di tutto cio' che puo' farlo fallire o renderlo INCONCLUSIVE per ragioni non scientifiche (versione di torch del cluster, CUDA, NCCL, Lustre, tempi, SLURM, percorsi, permessi, ambiente, concorrenza fra i 4 rank che scrivono file, git nel checkout di campagna, variabili d'ambiente mancanti). Per ciascuno: prova nel codice, probabilita' stimata, come ridurla senza cambiare il disegno.
3. Attribuzione dell'esito: in quali scenari il report direbbe SUPPORTED, PARTIAL_SUPPORT o HYPOTHESIS_UNSUPPORTED per la ragione sbagliata? Per esempio: il riscaldamento altera uno stato non controllato; la strumentazione cambia la numerica in modo uguale nelle due gambe; i bucket della warm coincidono per composizione ma non per contenuto; l'ordinaria riproduce bb39f339 ma la warm no per un effetto dell'hook.
4. Che cosa resta nei file dopo un esito INCONCLUSIVE o un timeout: i dati grezzi (vettori, checkpoint, tracce) permettono di rifare il confronto dopo su CPU senza rifare le GPU? Che cosa manca per poterlo fare?
5. Governance: la voce di decisioni.md riporta la scelta di Simone alla lettera, congela i criteri prima della misura, non autorizza training, mantiene FAILED_REQUIRES_REVIEW? Preventivo e residui sono corretti? C'e' qualcosa che il revisore dovrebbe chiedere di registrare PRIMA del lancio?
Non ripetere B-1. Sii concreto: file:riga e scenario.` },
]

const FINDING = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    titolo: { type: 'string' },
    gravita: { type: 'string', enum: ['bloccante', 'importante', 'minore', 'nota'] },
    file_riga: { type: 'string' },
    scenario: { type: 'string' },
    prova: { type: 'string' },
    eseguito: { type: 'boolean' },
    correzione: { type: 'string' },
  },
  required: ['id', 'titolo', 'gravita', 'file_riga', 'scenario', 'prova', 'correzione'],
}
const SCHEMA = {
  type: 'object',
  properties: {
    sintesi: { type: 'string' },
    rilievi: { type: 'array', items: FINDING },
    verifiche_positive: { type: 'array', items: { type: 'string' } },
    punti_aperti: { type: 'array', items: { type: 'string' } },
    rapporto: { type: 'string' },
  },
  required: ['sintesi', 'rilievi', 'verifiche_positive', 'rapporto'],
}
const VERDICT = {
  type: 'object',
  properties: {
    regge: { type: 'boolean' },
    gravita_corretta: { type: 'string', enum: ['bloccante', 'importante', 'minore', 'nota', 'infondato'] },
    motivazione: { type: 'string' },
    prova: { type: 'string' },
    correzione_minima: { type: 'string' },
  },
  required: ['regge', 'gravita_corretta', 'motivazione', 'prova'],
}

const results = await pipeline(
  TASKS,
  t => agent(COMMON + t.prompt, { label: t.key, phase: 'Revisione', schema: SCHEMA, model: 'opus' }),
  async (review, task) => {
    if (!review) return { key: task.key, review: null, verified: [] }
    const serious = (review.rilievi || []).filter(f => f.gravita === 'bloccante' || f.gravita === 'importante')
    const verified = await parallel(serious.map(f => () =>
      parallel([1, 2].map(i => () =>
        agent(COMMON + `Compito di verifica avversaria (chiave: verifica-${task.key}-${f.id}-${i}). Un revisore ha segnalato il rilievo qui sotto sul diagnostico. Il tuo compito e' CONFUTARLO o ridimensionarlo: riproduci tu lo scenario con il codice vero (o dimostra dal codice, con file:riga, perche' non puo' verificarsi), controlla se un altro controllo del diagnostico lo intercetta gia', e valuta la gravita' giusta secondo le regole sopra (bloccante solo con controesempio concreto che rende il run vero INCONCLUSIVE o fallito per ragioni non scientifiche, o che fa attribuire male l'esito). Se il rilievo regge, dillo e proponi la correzione minima. Nel dubbio, non confermare un bloccante senza prova.

Rilievo ${f.id} (gravita' dichiarata: ${f.gravita}) - ${f.titolo}
Dove: ${f.file_riga}
Scenario: ${f.scenario}
Prova del revisore: ${f.prova}
Correzione proposta: ${f.correzione}

Scrivi il tuo rapporto in ${BASE}/reports/verifica-${task.key}-${f.id}-${i}.md e i tuoi file in ${BASE}/agents/verifica-${task.key}-${f.id}-${i}/ .`,
          { label: `verifica:${task.key}:${f.id}:${i}`, phase: 'Verifica', schema: VERDICT, model: 'opus' })
      )).then(votes => ({ finding: f, votes: votes.filter(Boolean) }))
    ))
    return { key: task.key, review, verified: verified.filter(Boolean) }
  }
)
log(`revisori completati: ${results.filter(r => r && r.review).length} su ${TASKS.length}`)
return results
