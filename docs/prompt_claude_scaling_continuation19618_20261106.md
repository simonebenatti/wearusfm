# Claude: piano e continuazione 300M a 19.618 passi

Simone ha approvato il 06/11/2026: «va bene, approvo la continuazione del 300M a
19.618 passi. Preparami il prompt per claude, cosi' prepara il piano e poi claude lancia».

La decisione comprende la preparazione e l'esecuzione della catena prevista:
ripresa del 300M, estrazioni teacher, valutazioni posa/P1/P2 e confronto col 30M
allo stesso passo. Claude resta responsabile di codice, deploy, invii, stato/log
e raccolta della linea 1; GPT fa la review. Dopo review favorevole dei rilasci
pertinenti, procedi senza richiedere un altro consenso a Simone per questa catena.

## 1. Riferimenti e domanda scientifica

Leggi la terza review GPT e l'analisi scientifica, pubblicate in `5e15009` sul
ramo `codex/scaling-300m-20261104`:

- `docs/reviews/gpt_review_scaling_final_outcome_20261106.md`;
- `docs/reviews/gpt_scaling_scientific_final_20261106.md`;
- `docs/scaling_300m_protocol_20261104.md`;
- `docs/scaling_300m_continuation_budget_freeze_20261105_v2.md` e il relativo
  `results/infra/scaling_300m_20261104/continuation19618_budget_freeze_20261105_v2.json`.

Il risultato a 5.000 e' accettato: EPN +1,917 pp, posa senza vantaggio,
non inferiorita' P1/P2 sostenuta. La domanda e' se il vantaggio classificativo
persiste o cresce con la dose e se cambia la dinamica della posa. Non aggiungere
loss, log-quadratico, fine-tuning, semi, architetture o altri endpoint di training.
Le linee 2 e 3 restano separate.

## 2. Ripresa esatta, senza ricominciare

Prepara la continuazione **5.000 -> 19.618: 14.618 update aggiuntivi** del
checkpoint completo T100-G-300M gia' accettato. Verifica hash, completezza e
identita' del checkpoint; conserva immutabili checkpoint e output precedenti,
scrivendo la nuova tranche in una cartella distinta.

Conserva student, teacher EMA, AdamW e corrispondenza dei momenti coi parametri,
RNG per rank, stato del loader e schedule originale **49.046 passi, warmup 1.962**.
Nessun reset dell'ottimizzatore o nuovo warmup. Batch globale 32, 4 GPU su un
nodo, microbatch 8, accum_steps 1, stessa ricetta e stesso seme.

Attenzione: `scaling_train.py` e vari valutatori/validatori correnti vincolano
esplicitamente l'endpoint a 5.000; il wrapper di training rifiuta `--resume`.
Serve una versione di continuazione/endpoint rivista: non basta sostituire
globalmente 5000 o cambiare max_steps. Mantieni intatti i checkout congelati
precedenti e verifica con test CPU la continuita' della ripresa contro una
sequenza ininterrotta e la compatibilita' del percorso a 5.000.

## 3. Valutazione a 19.618

Controllo: T100-G-30M **a 19.618**, gia' esistente. Prima del training riverifica
disponibilita', hash, identita', ricevute e conservazione del checkpoint e delle
due cache posa piu' cinque cache classificazione. Se manca un controllo, segnala
il problema: nessun nuovo training 30M o riestrazione fuori piano impliciti.

Mantieni teacher, popolazioni, fold, supporti, griglie e selezione degli
iperparametri solo sul train. Tre viste: 30M naturale, 300M naturale, 300M matched
con le stesse proiezioni fisse. La vista naturale resta primaria; matched resta
secondaria e non decide a posteriori quale risultato presentare.

Pubblica prima dei lanci la lettura dell'endpoint 19.618, mantenendo le definizioni:
EPN612 e E_dyn dinamico primari; P1/P2 con margine di non inferiorita' -1 pp;
posa praticamente migliorata se riduzione puntuale >=2% e IC del delta sotto zero;
bootstrap appaiato per soggetto, 2.000 repliche, seme 20261102. Non scegliere il
checkpoint migliore fra 5.000 e 19.618 dopo i risultati. S8 storica resta immutata:
la nuova lettura descrive il 19.618, non deve raccomandare di nuovo di arrivarci
ne' autorizzare automaticamente ulteriore training. Resta un solo seme.

## 4. Piano e risorse

Usa il freeze v2 come limite della pianificazione, senza ampliamenti automatici:

- training: un nodo, 4 GPU/32 CPU/480G, QoS `normal` da riverificare,
  massimo 14h45, 59 GPU-ore;
- estrazioni: 7 GPU-ore; totale 66 richieste, 66,35 contando le tolleranze;
- lettori: campagna distinta, massimo 350 core-ore CPU-only; rendiconta
  separatamente anche le CPU allocate nei job GPU;
- memoria di produzione di riferimento: 37,751831 GiB allocati; registra
  allocata e riservata per ogni rank e mantieni i gate previsti.

Non usare `boost_qos_bprod`, che nel controllo precedente richiedeva almeno
65 nodi. Fit CPU solo dopo i benchmark reali delle feature a 19.618, massimo
8 CPU/30 GiB, con formula, limiti per fase e shard del freeze v2; niente riuso
automatico dei tempi a 5.000. Valuta il riuso delle cache del controllo solo
dopo verifica; non presumere compatibilita' di vecchi OOF per evitare fit.

## 5. Consegna a GPT e lanci

Pubblica sul tuo ramo di esecuzione un pacchetto concreto: protocollo di endpoint,
wrapper e test, pin reali disponibili, piano dell'intera catena, costi distinti,
comandi esatti, ID univoci, percorsi e condizioni di rilascio. Commit e push con
verifica dello SHA remoto. Consegna il riferimento a GPT per la review prima del
deploy e del primo invio dei nuovi wrapper.

Dopo il via tecnico di GPT, esegui autonomamente i passaggi approvati tramite
`scripts/submit_job.py` e ledger comune, con checkout congelati. I pin delle
future cache e i piani dipendenti dai benchmark si pubblicano quando reali e
passano dalla review pertinente prima dei relativi invii, come nella catena a
5.000. Non serve un altro via di Simone per quei passaggi conformi.

Coordina la corsia con la linea 2, rileggi coda/ledger e controlla il successo
delle letture prima di contarne l'output. Nessun duplicato, runner nuovo non
rivisto o monitor AI in background. Usa gli strumenti gia' rivisti quando
applicabili. Stop al primo gate fallito, errore, TIMEOUT o sforamento: raccolta
e diagnosi, nessun retry cieco. Un recupero tecnico richiede piano/pin e review;
un ampliamento di budget o perimetro richiede una nuova decisione.

## 6. Chiusura

Raccogli ricevute, log, raw `sacct -P`, hash, report e consuntivi; aggiorna registro
in entrambe le sezioni e decisioni, commit/push verificato. Registra questa nuova
approvazione senza riscrivere i freeze storici che correttamente dicevano
«non autorizzato» prima del 06/11.

Correggi separatamente la piccola descrittiva EPN segnalata dalla terza review:
naturale 199 meglio/100 peggio/15 pari; matched 148/153/13. E' una rettifica di
documento e script descrittivo, senza toccare il report congelato o rifare job.
Alla fine consegna a GPT esito tecnico e analisi scientifica in documenti separati.
