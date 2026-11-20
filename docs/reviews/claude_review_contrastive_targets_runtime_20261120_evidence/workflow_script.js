export const meta = {
  name: 'review-linea3-runtime-release',
  description: 'Review del pacchetto operativo RMS100/fast125 della linea 3 (GPT fbfde797): parita, ripresa, release, risorse, test, lettori, governance',
  phases: [
    { title: 'Revisione', detail: 'sette revisori sul diff 438c8d65..fbfde797' },
    { title: 'Verifica', detail: 'scettici indipendenti su ogni rilievo, a maggioranza' },
    { title: 'Completezza', detail: 'critico finale' },
  ],
}

const A = args
const CTX = `CONTESTO COMUNE (leggilo tutto prima di iniziare):
- Repository: copia git di sola lettura in ${A.clone}, HEAD detached = 0efa4d45 (punta di origin/codex/contrastive-target-windows-20261118). Commit utili: base 438c8d65 (preparazione gia' rivista da Claude in 94f5094), contenuti fbfde797 (codice da rivedere), 0efa4d45 (aggiunge solo results/infra/contrastive_targets_20261118/amendment_20261119/delivery_pin.json), decisione scientifica faa4a8d7 (ramo origin/codex/tokenizer-logcontrast-20261104, copiata in docs/reviews/contrastive_common_diagnostics_science_20261119.md), diagnostici storici 365cb1e2.
- DIFF DA RIVEDERE: git -C ${A.clone} diff 438c8d65 fbfde797 (46 file, circa 5.500 righe). Non riaprire le parti storiche invariate. Moduli nuovi: scripts/contrastive_numeric_profile.py, contrastive_target_parity.py, contrastive_target_resume.py, contrastive_release.py, contrastive_targets_plan.py, contrastive_local_validation.py. Modificati: scripts/contrastive_run.py, contrastive_job.py, contrastive_cuda_check.py, contrastive_benchmark_readout.py, contrastive_prepare.py e i test in tests/cpu/test_contrastive_*.py. Documenti: docs/contrastive_targets_execution_20261119.md (prontuario), docs/reviews/contrastive_targets_response_20261119.md (risposta alla review 94f5094: R-1...R-4, M-1, N-1...N-7), piani results/infra/contrastive_targets_20261118/{plan,preflight_plan}.json, ricevuta results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json, pin amendment_20261119/delivery_pin.json.
- Ricetta della nuova coorte: target RMS 100 ms e spettro rapido 125 ms (target_version step1_psd_power_shape_energy_fast125_v3), tre bracci C0/CN/CR su T100-G 30M, profilo numerico candidato bf16 autocast con SDPA solo math e TF32 spento. Sul cluster Leonardo torch e' 2.2.0a0+git6c8c5ad (build della v2.2.1), CUDA 12.1, NCCL 2.19, 4 A100 per nodo, celle GPU da 30 minuti su boost_qos_dbg (al piu' 2 job in esecuzione per utente), corsia seriale lrd_all_serial (8 CPU, 30.800 MB, 4 h). Storico: i benchmark full da 576 passi in bf16 automatico sono durati 661-775 s; la diagnostica da stato comune ha mostrato repliche bf16 automatiche diverse (circa 0,03%) e identiche nei profili math; le riprese reload A e B divergevano gia' al primo passo in bf16 automatico.
- Ruoli: linea 3, codice e scienza di GPT, lanci di 3-launcher; Claude e' revisore. La review di Claude, se favorevole, conterra' un blocco contrastive-release-review che i job useranno come legame di release: va quindi rivista anche la logica che lo legge (scripts/contrastive_release.py bind-review).
- Regole: SOLO LETTURA. Non modificare file nella copia git (file temporanei solo sotto ${A.tmp}), nessun commit o push, non toccare /Users/simonebenatti/Documents/ChatGPT/ ne' /Users/simonebenatti/dev/wearusfm, niente ssh o Leonardo, nessun job. Disco stretto (circa 23 GiB liberi): mai l'intera suite tests/cpu; per pytest usa --basetemp=${A.tmp}/<nome-unico> e cancellalo alla fine.
- Comando pytest (dalla radice della copia): OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${A.clone}/src:${A.clone}/tests/cpu:${A.clone}/scripts ${A.python} -B -m pytest -q -p no:cacheprovider --basetemp=${A.tmp}/<nome> <file di test>. Per script di verifica usa ${A.python} -B con lo stesso PYTHONPATH.
- Mutazioni: sempre in un clone temporaneo (git clone -q --shared ${A.clone} ${A.tmp}/<dir> e poi git -C <dir> checkout -q --detach 0efa4d45), mai nella copia ${A.clone}; i test che usano frozen_campaign_support.run_frozen richiedono un clone git. Cancella i cloni alla fine.
- Rapporto incrementale: scrivi progressi e rilievi man mano nel file ${A.reports}/<la tua etichetta>.md (in coda), cosi' nulla va perso se l'agente si interrompe.
- Gravita': BLOCCANTE = dimostrato che la catena girerebbe in modo scorretto, accetterebbe evidenze o review non valide, attribuirebbe male un esito, falsificherebbe o renderebbe impossibile un cancello per un difetto del codice (non per la fisica del problema), supererebbe un tetto o un limite, o fallirebbe per un difetto certo prima di produrre misure; MINORE = difetto reale senza quegli effetti; NOTA = chiarezza o documentazione. Ogni rilievo con prova riproducibile (file:riga a fbfde797, comando e output essenziale). Riporta anche le verifiche positive in verified_ok con la prova, e in not_checked cio' che non hai potuto verificare. Scrivi in italiano, conciso.`

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    point: { type: 'string' },
    verified_ok: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, evidence: { type: 'string' } }, required: ['claim', 'evidence'] } },
    findings: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, severity: { type: 'string', enum: ['BLOCCANTE', 'MINORE', 'NOTA'] }, title: { type: 'string' },
      detail: { type: 'string' }, evidence: { type: 'string' }, reproduction: { type: 'string' }, fix: { type: 'string' } },
      required: ['id', 'severity', 'title', 'detail', 'evidence', 'reproduction', 'fix'] } },
    not_checked: { type: 'array', items: { type: 'string' } },
  },
  required: ['point', 'verified_ok', 'findings', 'not_checked'],
}
const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    severity_if_real: { type: 'string', enum: ['BLOCCANTE', 'MINORE', 'NOTA', 'NESSUNO'] },
    reasoning: { type: 'string' },
    reproduction: { type: 'string' },
  },
  required: ['refuted', 'severity_if_real', 'reasoning', 'reproduction'],
}
const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    gaps: { type: 'array', items: { type: 'object', properties: {
      point: { type: 'string' }, gap: { type: 'string' }, why_it_matters: { type: 'string' }, how_to_check: { type: 'string' } },
      required: ['point', 'gap', 'why_it_matters', 'how_to_check'] } },
    overall: { type: 'string' },
  },
  required: ['gaps', 'overall'],
}

const DIMENSIONS = [
  { key: 'r1-parita', point: '1', prompt: `PUNTO 1 — PARITA' a peso zero (scripts/contrastive_target_parity.py, contrastive_numeric_profile.py, le modifiche a contrastive_run.py e al trainer se presenti, i test test_contrastive_target_parity.py e test_contrastive_numeric_profile.py).
Verifica:
a) Il punto di intercettazione: stato vivo del full C0 subito prima degli update 128 e 256 (127 e 255 update completati), con batch misto globale 32, teacher EMA, coda FIFO, batch, microbatch e maschere reali. Trova dove il codice si aggancia al trainer e controlla gli indici (zero-based contro numero di update).
b) I sei backward (due per C0, CN, CR): ripristino di RNG CPU e CUDA per rank (e Python/NumPy registrati), pesi e coda identici, reducer nuovi uguali fra loro. ATTENZIONE: se i probe creano nuovi DistributedDataParallel sugli stessi parametri del modello vivo, gli hook del DDP vivo potrebbero scattare durante i backward dei probe e lanciare riduzioni del training o lasciare stato sporco; verifica come sono costruiti i probe (copie del modulo? stessi parametri?), cosa succede a .grad, ai bucket e agli hook, e che l'osservatore dimostri che pesi, teacher, coda, gradienti e RNG del processo continuo restano intatti dopo i probe.
c) Il criterio: gradienti ridotti byte-identici entro e fra bracci e fra rank; supporto di entrambe le loss; pesi effettivi zero; batch globale 32; provenienza e legami (sorgenti, config canonico, summary del C0, inizializzazione). Profilo richiesto, punto o supporto mancante = inconcluso; disuguaglianza con evidenza completa = fallimento; nessun inconcluso rilascia training. Le 256 loss e norme restano descrittive con rtol 1e-4 e atol 1e-6.
d) Il profilo numerico: contrastive_numeric_profile.py imposta e registra le opzioni su tutti i rank; compatibilita' con torch 2.2.0a0 (getattr/hasattr, firme di is_autocast_enabled e get_autocast_gpu_dtype in 2.2, cudnn_sdp in 2.2), ripristino delle opzioni, confronto delle 28 ricevute (7 run x 4 rank) e blocco su flag diversi. Il profilo e' applicato anche ai backward dei probe e a tutti i run della coorte, calibrazione compresa?
e) Costo: stima se i sei backward a 128 e 256 piu' il full C0 in bf16 math entrano nei 1.770 s interni; tempo e memoria misurati a batch pieno come richiesto.
Esegui i test pertinenti e, dove serve, piccoli esperimenti CPU (anche DDP gloo a piu' processi) per provare i punti critici, in particolare b).` },
  { key: 'r2-ripresa', point: '2', prompt: `PUNTO 2 — RIPRESA a loss attive (scripts/contrastive_target_resume.py, il runner, le modifiche a contrastive_run.py e al trainer, test test_contrastive_target_resume.py).
Verifica:
a) Un solo fork a 544 per CN e CR, salvato dal full continuo che prosegue nello stesso processo fino a 576; keep_checkpoints con 544 solo nei config del collaudo (la ricetta lunga resta 5000/19618); due job A/B con quattro processi nuovi che copiano gli stessi byte del checkpoint 544 e fanno solo 545-576; controllo di config, hash dei byte iniziali, mapping delle origini e stato al confine.
b) Tracce per tutti i rank e i 32 update: impronte separate RNG CPU e CUDA prima e dopo il backward, batch e microbatch, supporto delle loss, contatori, ordine e identita' della coda; confronto al confine di studente, teacher, momenti e gruppi Adam al prossimo learning rate e stato contrastivo.
c) Le tre coppie (continuo/A, continuo/B, A/B) con la regola storica: parametri studente e teacher con errore aggregato <= 1% dello spostamento dall'inizializzazione originale del primo membro (denominatore e pavimento originali); buffer, contatori, metadati, stream, RNG e ordine della coda esatti; ogni singolo tensore mobile di Adam e delle chiavi della coda con errore relativo <= 1%. Confronta riga per riga con la regola storica in scripts/contrastive_benchmark_readout.py a 438c8d65 (RESUME_RELATIVE_DISPLACEMENT, _nested_equal, _tensor_distance) e verifica che la definizione sia davvero la stessa. Le distanze dal movimento dopo 544 e per blocco devono essere pubblicate separatamente e non entrare nel cancello.
d) Fattibilita': i job A/B (32 update) e il full con il salvataggio a 544 entrano nei limiti di tempo? Il checkpoint del fork e le copie entrano nello spazio e nella quota (prontuario: almeno 12 GiB)?
e) Coerenza con la diagnostica: il disegno risolve il confondimento del vecchio test (due training indipendenti)?
Esegui i test e, dove serve, esperimenti CPU (trainer reale, piu' processi gloo) per provare i punti critici.` },
  { key: 'r3-release', point: '3', prompt: `PUNTO 3 — RELEASE (scripts/contrastive_release.py, le modifiche a contrastive_job.py, contrastive_run.py e contrastive_benchmark_readout.py, test test_contrastive_release.py e test_contrastive_targets_guards.py).
Verifica:
a) bind-review: cosa legge esattamente (byte originali della review nel commit della review, sorgenti e piani nel commit rivisto, copia nel checkout), come estrae il blocco contrastive-release-review (formato del recinto, unicita', schema contrastive-target-release-review-v1, campi status, reviewed_commit, source_sha256, science_plan_sha256, preflight_plan_sha256), cosa rifiuta (review del 17/11, 94f5094, review di un altro pacchetto, blocchi doppi, valori diversi, status diverso da APPROVED). Prova con review sintetiche in un clone temporaneo. Documenta con precisione il formato che il blocco di Claude deve avere per essere accettato (spazi, recinto, chiavi in piu' o in meno, posizione nel file).
b) Il legame nei job e nel runtime: dispatcher (build_command) e runtime rifiutano job senza binding, con binding di un altro pacchetto o con impronte diverse; nessun hash circolare (i piani non incorporano il proprio hash ne' quello della review).
c) Il cancello dell'audit: validate_audit_accounting (schema contrastive-audit-accounting-v1; TSV sacct JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw con unica riga dell'allocazione; COMPLETED, 0:0, ElapsedRaw <= 7200, 8 CPU, 120 min; pin del report, dei sorgenti, dell'ID logico e SLURM). E' applicato nella release, nel dispatcher, nel runtime della calibrazione e nella lettura finale? Nessuna eccezione TIMEOUT ereditata. Prova i casi negativi (TIMEOUT, exit code, durata, pin, ID, duplicati, righe .batch/.extern del sacct) e anche casi limite (ElapsedRaw esattamente 7200, sacct con piu' righe, TimelimitRaw in minuti).
d) La release solo audit: checkout figlio del commit rivisto con soli metadati operativi, source e piani identici; nomi di deploy e checkout; cosa impedisce di aggiungere alla release codice o piani diversi.
Esegui i test e mutazioni mirate delle guardie in un clone temporaneo.` },
  { key: 'r4-risorse', point: '4', prompt: `PUNTO 4 — RISORSE e PIANI (results/infra/contrastive_targets_20261118/{plan,preflight_plan}.json, scripts/contrastive_targets_plan.py con --check, contrastive_job.py ALLOCATED_SECONDS, launcher sbatch).
Verifica:
a) Rigenera i piani con contrastive_targets_plan.py --check (e in una copia temporanea senza --check) e confronta byte per byte; verifica che gli sha256 di plan.json (b03c0bc6...) e preflight_plan.json (e4b39591...) e il source digest (40d39f56...) del delivery_pin corrispondano ai file a fbfde797 e a 0efa4d45.
b) 17 allocazioni (11 celle + 6 checkout), 18 GPU-ore, 169,6 core-ore: ricalcola cella per cella (GPU x secondi, CPU x secondi), le dipendenze, i sei checkout da 4 CPU, 2 GiB, 240 s con override obbligatori --cpus-per-task=4 --mem=2G --time=00:04:00, l'audit --time=02:00:00, le celle GPU 1.800 s richiesti e 1.770 s interni, la lettura 1 h. Perche' sei checkout (CUDA e full separati)? I reload A/B dipendono dal full del proprio braccio e dal checkout con i pin di 544?
c) Contabilita': consumo storico effettivo 6,086667 GPU-ore e 57,827778 core-ore (5,284444 + 0,802222; 51,371111 + 6,456667), residuo aritmetico del vecchio tetto 7,913333 e 71,505556, incremento 4 e 32,533333, totale 24,086667 e 227,427778, margine SLURM 360 s x 17 = 3,6 GPU-ore e 32,8 core-ore, inviluppo 21,6 e 202,4: ricalcola tutto e trova incoerenze fra prontuario, piani, decisioni.md e risposta. Verifica le fonti dei consuntivi (accounting nel repository, historical_diagnostics_accounting.txt).
d) Limiti di Leonardo: memoria e CPU per cella rispetto ai launcher e alle code; manutenzione 23/11 08:00 - 25/11 08:00; saldo da acquisire (non inventato); nessun budget di training lungo.
e) I dispatcher accettano esattamente le allocazioni del piano (nessuna cella con tempo diverso).` },
  { key: 'r5-test', point: '5', prompt: `PUNTO 5 — TEST E RIPRODUCIBILITA'.
Verifica:
a) Esegui l'insieme di test della ricevuta validation_final/local_validation.json (ricostruisci l'elenco esatto dalla ricevuta, 465 test attesi) e confronta conteggio, durata e hash. Esegui scripts/contrastive_local_validation.py --verify (o l'equivalente documentato) e controlla che ogni pin corrisponda: comandi completi, ambiente, hash dei test e dei log, codici di uscita, source digest. I log del 18/11 pinnati senza dichiararli rieseguiti.
b) M-1: i test negativi isolano davvero config_sha256 in validate_audit e in read_preflight e la guardia della ricevuta in authenticate_calibration? Prova con mutazioni mirate in un clone temporaneo che ciascuna barriera, disattivata da sola, faccia fallire almeno un test. Stessa cosa per la protezione del puntatore PLAN, del testo descrittivo del readout e dei campi target del piano contro build_review_configs.
c) Mutazioni sulle guardie nuove piu' importanti: byte-identita' dei gradienti nella parita', regola dell'1% e per-tensore nella ripresa, legame della review in contrastive_release, cancello sacct dell'audit, confronto delle 28 ricevute del profilo numerico. Per ognuna: quale test fallisce? Una guardia nuova senza test e' un rilievo.
d) Integrazione fra produttori e lettori: i rapporti prodotti da contrastive_target_parity.py, contrastive_target_resume.py, contrastive_numeric_profile.py e contrastive_release.py hanno lo stesso schema che leggono contrastive_benchmark_readout.py, contrastive_run.py e contrastive_job.py? C'e' almeno un test che passi l'uscita vera di un produttore al lettore vero?
e) Le prove CPU non qualificano CUDA/NCCL: i documenti lo dichiarano? Cosa resta da provare solo sul cluster?` },
  { key: 'r6-lettori', point: 'lettori', prompt: `LETTORI E INTEGRAZIONE (scripts/contrastive_benchmark_readout.py, contrastive_run.py, contrastive_cuda_check.py, contrastive_prepare.py, contrastive_job.py: tutte le modifiche del diff 438c8d65..fbfde797 non coperte dai moduli nuovi; test test_contrastive_targets_readout.py, test_contrastive_run.py, test_contrastive_job.py, test_contrastive_benchmark_readout.py, test_contrastive_mmap_compatibility.py, test_contrastive_historical_*).
Verifica:
a) Ogni modifica ai lettori e al dispatcher: correttezza, che non allenti cancelli storici o nuovi, che non cambi la lettura delle evidenze storiche (i lettori congelati restano il modo di riprodurle), che i rifiuti avvengano prima di consumare dati o GPU.
b) La lettura finale: richiede tutte le sette esecuzioni complete, tracce, 28 ricevute numeriche, report di parita' del C0 e cancello dell'audit; un rapporto mancante lascia inconclusi i confronti che lo richiedono, mai un esito negativo della loss; nessun training rilasciato da un esito favorevole del codice.
c) contrastive_cuda_check.py: cosa cambia e perche'; il controllo CUDA usa il profilo numerico nuovo? I criteri bf16 storici sono invariati o cambiati?
d) Coerenza con il piano: i lettori leggono i parametri dal piano nuovo o li codificano? Un piano modificato verrebbe notato?
Esegui i test pertinenti.` },
  { key: 'r7-governance', point: 'governance', prompt: `GOVERNANCE, DOCUMENTI E COERENZA.
Verifica:
a) La decisione scientifica faa4a8d7 (ramo origin/codex/tokenizer-logcontrast-20261104; copia in docs/reviews/contrastive_common_diagnostics_science_20261119.md): e' davvero copiata byte per byte? E' intestata come proposta per Simone: cerca in docs/decisioni.md, docs/claude_GPT_shared e nei documenti dei due rami (a fbfde797 e alla punta di origin/codex/tokenizer-logcontrast-20261104) una firma o approvazione letterale di Simone della proposta (profilo bf16 math con TF32 spento per tutti i bracci, nuova prova di parita', nuova ripresa da 544, cancelli). Riporta testualmente cosa c'e' e cosa manca. Le soglie nuove sono scritte in docs/decisioni.md prima di qualsiasi misura della coorte?
b) La modifica a CLAUDE.md in fbfde797 (blocco «Linea3, incarico del19/11/2026» in testa): «questa chat» e' ambiguo in un file letto da tutti gli agenti? «Questo blocco prevale sulle assegnazioni operative storiche sottostanti» e' limitato alla linea 3? Confronta con il chiarimento d9020c20 (che aveva tolto un «questa chat» ambiguo).
c) La risposta docs/reviews/contrastive_targets_response_20261119.md: per ognuna delle condizioni R-1...R-4, M-1 e N-1...N-7 della review di Claude (docs/reviews/claude_review_contrastive_targets_amendment_20261119.md) la risposta corrisponde al codice? Segnala ogni affermazione della risposta o del prontuario non vera o non verificabile nel codice e nei test.
d) Coerenza fra prontuario, piani, decisioni.md, registro, delivery_pin e codice: numeri, nomi di ID e checkout, deploy, sequenza delle fasi, dipendenze, profilo numerico.
e) La sezione evaluation del piano scientifico e le soglie downstream: invariate come dichiarato?` },
]

const votesFor = sev => (sev === 'BLOCCANTE' ? 3 : sev === 'MINORE' ? 2 : 1)

const reviewed = await pipeline(
  DIMENSIONS,
  d => agent(`${CTX}\n\nIl tuo rapporto incrementale: ${A.reports}/${d.key}.md\n\n${d.prompt}`,
             { label: `rev:${d.key}`, phase: 'Revisione', schema: REVIEW_SCHEMA }),
  (review, d) => {
    if (!review) return { dim: d.key, review: null, verified: [] }
    return parallel(review.findings.map(f => () =>
      parallel(Array.from({ length: votesFor(f.severity) }, (_, i) => () =>
        agent(`${CTX}\n\nSei lo scettico indipendente n. ${i + 1} per un rilievo del revisore «${d.key}». Compito: CONFUTARE il rilievo se possibile. Riproduci da solo la prova (comandi tuoi, non fidarti dell'output citato), cerca nel codice e nei documenti motivi per cui non e' un difetto (un controllo altrove, un percorso non usato in produzione, una dichiarazione esplicita nel prontuario o nei piani), e valuta la gravita' reale secondo la scala comune. Se non riesci a riprodurlo, o resta incerto, rispondi refuted=true. Scrivi il tuo esito anche in ${A.reports}/verify-${d.key}-${f.id}-${i + 1}.md.\n\nRILIEVO:\n${JSON.stringify(f, null, 1)}`,
              { label: `verify:${d.key}:${f.id}:${i + 1}`, phase: 'Verifica', schema: VERDICT_SCHEMA })))
        .then(votes => {
          const vs = votes.filter(Boolean)
          const upheld = vs.filter(v => !v.refuted).length
          return { ...f, votes: vs, upheld, survives: vs.length > 0 && upheld >= Math.floor(vs.length / 2) + 1 }
        })))
      .then(verified => ({ dim: d.key, point: d.point, review, verified }))
  },
)

const results = reviewed.filter(Boolean)
const missing = DIMENSIONS.filter(d => !results.find(r => r && r.dim === d.key && r.review)).map(d => d.key)
if (missing.length) log(`Revisori senza esito: ${missing.join(', ')}`)

phase('Completezza')
const summary = results.map(r => ({
  dim: r.dim, point: r.point,
  verified_ok: r.review ? r.review.verified_ok.map(v => v.claim) : [],
  not_checked: r.review ? r.review.not_checked : ['REVISORE SENZA ESITO'],
  findings: (r.verified || []).map(f => ({ id: f.id, severity: f.severity, title: f.title, survives: f.survives, upheld: f.upheld, of: (f.votes || []).length })),
}))
const critic = await agent(`${CTX}\n\nSei il critico di completezza. Sotto trovi la sintesi di sette revisori (verifiche positive, rilievi con esito della verifica avversaria, cose non verificate) sul pacchetto. Confrontala con la richiesta di GPT, che chiede di concentrarsi su: 1) parita' (stato vivo di C0 prima di 128 e 256, batch misto 32, due backward per braccio, veri teacher/coda/batch/maschere/RNG, gradienti ridotti identici per byte, reducer nuovi, inconcluso se manca supporto o profilo, 256 loss e norme descrittive, flag effettivi fra i sette run, tempo e memoria al batch pieno); 2) ripresa (fork unico a 544, A/B dagli stessi byte con processi nuovi, tracce CPU/CUDA per rank e passo, contatori e ordine, tre coppie con denominatori storici e 1%, buffer esatti, limiti per tensore Adam/coda, distanze dopo 544 separate); 3) release (review legata a commit e hash di sorgenti e due piani, review storica insufficiente, calibrazione bloccata senza audit nuovo e sacct COMPLETED 0:0 entro 7.200 s, nessuna eccezione TIMEOUT); 4) risorse (17 allocazioni, 18 GPU-ore, 169,6 core-ore, sei checkout, consumo effettivo distinto dai massimi, margini SLURM e manutenzione, saldo da acquisire, nessun budget di training lungo); 5) test (prove negative M-1, protezione di PLAN/readout/target, riproducibilita' dei log, integrazione fra produttori e lettori; le prove CPU non qualificano CUDA/NCCL). Elenca le lacune concrete: verifiche non fatte, affermazioni non provate, file non letti, percorsi di codice non coperti, rischi sul cluster (torch 2.2.0a0, NCCL, memoria, tempo) non valutati. Per ciascuna indica come verificarla in pochi comandi; puoi fare tu stesso controlli rapidi. Scrivi l'esito anche in ${A.reports}/critico-completezza.md.\n\nSINTESI:\n${JSON.stringify(summary, null, 1)}`,
  { label: 'critico-completezza', phase: 'Completezza', schema: CRITIC_SCHEMA })

return { results, critic, missing }
