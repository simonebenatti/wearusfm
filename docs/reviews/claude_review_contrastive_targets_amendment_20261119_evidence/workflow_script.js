export const meta = {
  name: 'review-linea3-target-100-125',
  description: 'Review dell adeguamento target RMS100/fast125 della linea 3 (GPT 0efdc29c): sei punti, test, verifica avversaria, completezza',
  phases: [
    { title: 'Revisione', detail: 'sette revisori: i sei punti di GPT e test/ricevuta della suite' },
    { title: 'Verifica', detail: 'scettici indipendenti su ogni rilievo, a maggioranza' },
    { title: 'Completezza', detail: 'critico finale su verifiche mancanti' },
  ],
}

const A = args
const CTX = `CONTESTO COMUNE (leggilo tutto prima di iniziare):
- Repository: copia git di sola lettura in ${A.clone}, HEAD detached = 438c8d65 (punta di origin/codex/contrastive-target-windows-20261118). Tutti i commit citati sono disponibili li' (git -C ${A.clone} show <sha>:<path>, git diff, git log).
- Consegna da rivedere: diff scientifico e388a1de..0efdc29c (scripts/contrastive_prepare.py, scripts/contrastive_job.py, scripts/contrastive_benchmark_readout.py, quattro file di test, docs/contrastive_targets_amendment_20261118.md, results/infra/contrastive_targets_20261118/{plan,preflight_plan,local_validation}.json e review_configs/*.json). e388a1de e' il merge di main 7364f7c (correzioni delle finestre target di Claude, commit 0d91297, gia' integrate) nel ramo della linea 3. Dopo la consegna: 65bc6022 (merge dei diagnostici da stato comune, sorgenti identici a 365cb1e), afb89b45 (merge di main f546c9e4, solo review ed evidenze), 438c8d65 (evidenze della suite CPU completa in results/infra/contrastive_targets_20261118/full_suite/).
- Ricetta nuova (CLAUDE.md, firmata da Simone): nuovi training con target log-RMS su 100 ms e spettro rapido su 125 ms (loader.rms_window_ms=100, loader.ms_fast_window_ms=125, target_version step1_psd_power_shape_energy_fast125_v3). Storico: 25/50 e target_version step1_psd_power_shape_energy_v1; un config senza i due campi e' storico 25/50. Strumenti storici: train_jepa.historical_args, run.with_historical_targets, run.with_legacy_target_windows (aggiunge solo i due campi del 18/11 se mancano); run._config_with_defaults serve solo alla ripresa e NON va usato nei confronti esatti.
- Linea 3 (confronto contrastivo C0/CN/CR su T100-G 30M): codice e scienza di GPT, lanci della chat 3-launcher; Claude e' revisore. Simone: «sul punto 3 passiamo a 100/100»: la linea 3 adotta la ricetta nuova e rifa' audit, calibrazione, benchmark e lambda.
- Coorte storica e ricevute: results/infra/contrastive_comparison_20261116/ (piano storico plan.json, calibrated_cohort_20261118_bf16/, releases/bf16_20261118/, compatibility_20261118/, review_response/). Non devono qualificare la nuova coorte.
- Regole: SOLO LETTURA. Non modificare file nella copia git (file temporanei solo sotto ${A.tmp}), nessun commit o push, non toccare /Users/simonebenatti/Documents/ChatGPT/ ne' /Users/simonebenatti/dev/wearusfm, niente ssh o Leonardo, nessun job. Disco del Mac stretto (circa 23 GiB liberi): mai la suite completa; per pytest usa --basetemp=${A.tmp}/<nome-unico> e cancellalo alla fine.
- Comando pytest (dalla radice della copia): OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${A.clone}/src:${A.clone}/tests/cpu:${A.clone}/scripts ${A.python} -B -m pytest -q -p no:cacheprovider --basetemp=${A.tmp}/<nome> <file di test>. Per script di verifica usa ${A.python} -B con lo stesso PYTHONPATH, oppure python3 per leggere JSON.
- I test che usano tests/cpu/frozen_campaign_support.run_frozen fanno git archive dalla radice del repository: girano solo in una copia git (la copia ${A.clone} va bene; per mutazioni usa un clone temporaneo: git clone -q --shared ${A.clone} ${A.tmp}/<dir> e poi git -C <dir> checkout -q --detach 438c8d65).
- Rapporto incrementale: scrivi progressi e rilievi man mano nel file ${A.reports}/<la tua etichetta>.md (aggiungi in coda, non riscrivere da zero), cosi' nulla va perso se l'agente si interrompe.
- Gravita': BLOCCANTE = dimostrato che la nuova coorte girerebbe con la ricetta sbagliata, accetterebbe evidenze vecchie, attribuirebbe male un confronto, supererebbe un tetto o un limite, oppure fallirebbe per un difetto certo prima di produrre misure; MINORE = difetto reale senza quegli effetti; NOTA = chiarezza o documentazione. Ogni rilievo con prova riproducibile (file:riga al commit, comando e output essenziale). Riporta anche le verifiche positive in verified_ok con la loro prova. Elenca in not_checked cio' che non hai potuto verificare. Scrivi in italiano, conciso.`

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
  { key: 'p1-storico-celle', point: '1', prompt: `PUNTO 1 — Nessuna reinterpretazione del T100-G storico; 100/125 esplicito su tutte le celle nuove.
Verifica:
a) La modifica a contrastive_prepare.config_from_dict (ora applica R.with_legacy_target_windows prima di ricostruire il config). Elenca TUTTI i chiamanti di config_from_dict (scripts/*.py, scripts/diagnostics/*.py, tests) e per ciascuno stabilisci se legge un config storico o nuovo, e se il risultato e' corretto: storico -> 25/50/v1; nuovo esplicito -> 100/125/v3 invariato. Leggi in src/wearusfm/training/run.py la definizione di with_legacy_target_windows e il trattamento di target_version: un config storico senza target_version diventa v1 o il default nuovo? E un config con uno solo dei due campi? Prova con piccoli script sui config veri: T100-G (results/infra/tokenizer_execution_20261103/19618/T100-G/config.json), coorte storica (results/infra/contrastive_comparison_20261116/calibrated_cohort_20261118_bf16/*.config.json) e config nuovi (results/infra/contrastive_targets_20261118/review_configs/*.json).
b) I quattro config nuovi (C0, CN, CR, calibration): loader.rms_window_ms=100, loader.ms_fast_window_ms=125, target_version v3. Confrontali campo per campo con i corrispondenti della coorte storica (per la calibrazione trova il config di calibrazione storico) e classifica ogni differenza come dichiarata (le due finestre, target_version, lambda nulle) o NON dichiarata (per esempio campi nuovi o default cambiati portati dal merge di main). Rigenera i config con build_review_configs dal baseline e confronta con i file salvati.
c) Ogni modalita' del dispatcher (scripts/contrastive_job.py, scripts/contrastive_run.py, scripts/contrastive_cuda_check.py, audit, calibrazione, full, split/ripresa, readout) usa il config JSON pinnato e non default di CLI: trova dove si costruiscono i config di split/ripresa e la fixture del CUDA e verifica che conservino 100/125/v3. Cerca qualunque punto in cui un config nuovo passi da _config_with_defaults o da LoaderConfig() con default, e qualunque punto in cui un artefatto storico (checkpoint, config, metriche T100-G) venga letto senza la lettura storica.
d) I lettori storici congelati (test che usano frozen_campaign_support.run_frozen con FROZEN_READER 8905e346) riproducono le analisi vecchie senza toccare le ricevute.` },
  { key: 'p2-nessuna-eredita', point: '2', prompt: `PUNTO 2 — Nessuna accettazione automatica di vecchi audit, lambda, calibrazioni o benchmark per la nuova coorte.
Verifica:
a) In results/infra/contrastive_targets_20261118/{plan,preflight_plan}.json e review_configs/preparation.json elenca ogni riferimento a file della coorte storica (results/infra/contrastive_comparison_20261116/...). Per ciascuno: riferimento di politica indipendente dai target (per esempio la quota di copertura dal manifest) oppure evidenza che qualifica la nuova coorte? Il gradient_reference.json storico e' usato solo come prior o descrizione, o entra in un cancello?
b) Il codice che accetta evidenze: scripts/contrastive_prepare.py (validate_release, authenticate_calibration, calibration_report o simili), scripts/contrastive_source_compat.py (validate_historical_evidence, HISTORICAL_SOURCE_SHA256, ALLOWED_CHANGED_SOURCES, manifest compatibility_20261118), scripts/contrastive_job.py, scripts/contrastive_run.py (validate_release), scripts/contrastive_audit*.py. Dimostra con esecuzioni concrete (script o test) che l'audit storico, la calibrazione storica (lambda 0,1194 e 0,3304), le release e i benchmark bf16_20261118 e la ricevuta canonica NON sono accettati per un job della nuova coorte (nuovo digest dei sorgenti 643a8937..., nuova target_version). Cerca eccezioni speciali (per esempio il TIMEOUT dell'audit 64652559) che potrebbero applicarsi da sole a un audit nuovo.
c) I test nuovi test_unmodified_historical_calibration_does_not_qualify_new_targets, test_historical_audit_does_not_qualify_new_sources, test_old_calibration_cannot_authorize_new_target_recipe e test_pinned_historical_receipt_still_requires_authenticated_recalculation: esercitano il percorso di produzione che rifiuta, o sono tautologici? Eseguili e prova almeno una mutazione mirata del codice di produzione (in un clone temporaneo sotto ${A.tmp}, mai nella copia git) che renderebbe accettabile un'evidenza vecchia: un test fallisce?
d) Nessuna autorizzazione o ricevuta vecchia copiata nella nuova cartella (status APPROVED_FOR_PREFLIGHT, job ID, lambda, sha di ricevute vecchie dentro results/infra/contrastive_targets_20261118).` },
  { key: 'p3-confronti', point: '3', prompt: `PUNTO 3 — Confronti CN-C0 e CR-C0 appaiati nella nuova ricetta; C0-T100-G con due fattori aggiuntivi dichiarati; protocollo statistico invariato.
Verifica:
a) Confronta campo per campo results/infra/contrastive_targets_20261118/plan.json con il piano storico results/infra/contrastive_comparison_20261116/plan.json (stesso protocollo contrastive-comparison-20261116-v2): elenca OGNI differenza e classificala come dichiarata nell'emendamento (docs/contrastive_targets_amendment_20261118.md) oppure no. Contrasti, coppie, bootstrap (repliche, seme, famiglia, alfa), margini, endpoint primari e descrittivi, regole di lettura e di confine Monte Carlo devono restare invariati.
b) Il codice dei confronti (scripts/contrastive_compare.py, scripts/contrastive_benchmark_readout.py e ogni lettore del piano): legge il piano nuovo? I tre bracci della nuova coorte hanno target identici (stessi due campi e target_version)? Esiste un controllo che rifiuti un confronto appaiato fra bracci con ricette target diverse (per esempio un C0 storico 25/50 con un CN nuovo)? Se manca, valuta se serve e con quale gravita'.
c) Il confronto con il T100-G storico: dove compare (lettori, descrittive, documenti)? E' dichiarato con i due fattori aggiuntivi (RMS e spettro rapido) e mai usato come cancello o come stima dell'effetto contrastivo? Cerca nel plan.json (limitations) e nel codice.` },
  { key: 'p4-lambda', point: '4', prompt: `PUNTO 4 — Coefficiente storico di lambda dichiarato come prior; entrambe le lambda da ricalcolare; cancelli invariati.
Verifica:
a) La formula di lambda nel codice di calibrazione (scripts/contrastive_prepare.py o il modulo che la calcola): 0,1 x 0,1179564123141922 x mediana(norma g_base / norma g_term) sui 32 batch informativi; intervallo ammesso [1e-4, 1]; nessun clipping; fuori intervallo si ferma l'intera coorte. Verifica che il coefficiente sia una costante prespecificata (da dove viene: decisione D-3, gradient_reference.json storico) e che il codice nuovo non lo ricalcoli ne' lo sostituisca.
b) I config nuovi hanno lambda nulle (frozen_weights CN/CR null): individua il passo che inserisce le lambda misurate nei config CN/CR al rilascio e verifica che accetti solo una ricevuta di calibrazione della NUOVA coorte (digest dei sorgenti, sha del config di calibrazione nuovo, target_version v3, radice d'uscita nuova), non la calibrazione storica 64701482.
c) La regola dei rapporti nei benchmark (pannello 513-576, mediana tra 0,005 e 0,5, p95 <= 1, rapporto attivo CN/CR in [0,25, 4]) e' invariata rispetto al piano storico e al codice del readout.
d) La modifica a contrastive_benchmark_readout.initial_calibration_description (ora solo descrittiva anche all'update 1): verifica che nessun cancello (readout, calibrazione, preflight) richieda l'uguaglianza dei primi valori con lo storico (historical_first8, historical_means); trova tutti gli usi.` },
  { key: 'p5-preventivo', point: '5', prompt: `PUNTO 5 — Preventivo: 13 allocazioni, 14 GPU-ore e 137,066667 core-ore; audit di 2 h coerente col dispatcher.
Verifica:
a) Ricalcola da results/infra/contrastive_targets_20261118/preflight_plan.json allocazioni (9 celle + 4 checkout), GPU-ore (GPU x secondi), core-ore (CPU x secondi) e totali, e confronta con la tabella dell'emendamento.
b) Coerenza dell'audit da 2 h: scripts/contrastive_job.py (ALLOCATED_SECONDS: audit 7200, readout 3600, GPU 1800, e build_command); il launcher CPU scripts/slurm/contrastive_preflight_cpu.sbatch (#SBATCH --time=01:00:00, secondi letti da scontrol) con l'override obbligatorio --time=02:00:00; i test test_stage_commands_use_frozen_absolute_scripts_and_fixed_resources e test_stage_allocations_cannot_be_swapped_or_extended. Un audit inviato senza override fallisce subito? Una lettura inviata con 2 h fallisce? Le celle GPU con un altro tempo falliscono? Esegui i test pertinenti.
c) Il tempo interno dell'audit: l'audit riceve un limite di tempo derivato dai secondi allocati (time_limit_s o simili) con un margine per scrivere il rapporto prima dello stop di SLURM? Il vecchio audit (64652559) duro' 3787 s contro 3600 (TIMEOUT con contenuto valido): cerca nel repository log e ricevute dell'audit storico per stimare il tempo per batch, e valuta se 7200 s bastano anche con i target 100/125 (finestre piu' lunghe: cambia il costo del calcolo dei target?). Guarda come il codice dell'audit e del loader calcola i target.
d) Limiti di Leonardo (CLAUDE.md e memoria del progetto): lrd_all_serial 8 CPU, 30.800 MB, 4 h; boost_qos_dbg 30 min e 2 job in esecuzione per utente. 30 GiB = 30.720 MB. Verifica richieste di memoria e CPU di ogni cella e dei checkout rispetto ai launcher.` },
  { key: 'p6-diagnostica', point: '6', prompt: `PUNTO 6 — Il cambio dei target non sana la parita' e la ripresa bf16: la diagnostica gia' preparata va conclusa nel proprio checkout prima del nuovo rilascio; nessun retry cieco ne' soglia nuova implicita.
Verifica:
a) preflight_plan.json (required_before_first_submission, depends dell'audit "diagnostic_resolution", failure_policy) e l'emendamento: il cancello della diagnostica e' esplicito? E' solo procedurale o anche meccanico (qualche codice impedisce il rilascio senza la risoluzione)? Valuta se un cancello procedurale basta, visto che il piano ha executable=false e submission_authorized=false.
b) Il merge 65bc6022: i file della diagnostica (scripts/diagnostics/*.py, scripts/slurm/diagnose_contrastive_common.sbatch, i due test, results/infra/contrastive_comparison_20261116/common_state_diagnostics_20261118/*) sono identici a 365cb1e e 35ea94b? Il digest dei sorgenti atteso dalla diagnostica (SOURCE 46938f31... nel driver) corrisponde o no al nuovo albero? Se no, verifica che lanciarla dal nuovo albero fallirebbe in modo pulito e non produrrebbe misure miste.
c) Nessuna soglia nuova o allentata: confronta i criteri numerici fra il ramo storico (09eeaf93 e 8905e346) e la punta 438c8d65 in scripts/contrastive_cuda_check.py (CRITERIA), scripts/contrastive_benchmark_readout.py (soglia dell'1% della ripresa, parita' a peso zero, regole dei rapporti), scripts/contrastive_job.py e scripts/contrastive_prepare.py. Elenca ogni costante o soglia cambiata.
d) Il commit db66b507 sul ramo origin/codex/tokenizer-logcontrast-20261104 (registra l'approvazione di Claude e pre-registra la lettura dei diagnostici): verifica che non cambi codice o pin dei diagnostici e che le regole di lettura siano coerenti con le note N-1...N-4 della review di Claude (commit 0afd086, file docs/reviews/claude_review_contrastive_common_diagnostics_20261118.md del ramo origin/worktree-claude-training-engine).` },
  { key: 'p7-test-suite', point: 'test', prompt: `TEST E RICEVUTA DELLA SUITE.
Verifica:
a) Esegui i test della consegna e dei chiamanti diretti dei tre script modificati: tests/cpu/test_contrastive_prepare.py, test_contrastive_job.py, test_contrastive_historical_calibration.py, test_contrastive_historical_callsites.py, piu' i test che importano contrastive_prepare, contrastive_job, contrastive_benchmark_readout o contrastive_source_compat (trovali con grep) e tests/cpu/test_target_windows_fix.py. Riporta conteggi esatti di passati e falliti e la durata.
b) Mutazioni mirate (su un clone git temporaneo sotto ${A.tmp}, MAI nella copia ${A.clone}): per ciascuna guardia nuova (controllo della baseline 25/50/v1 in build_review_configs; assegnazione 100/125/v3; with_legacy_target_windows in config_from_dict; ALLOCATED_SECONDS dell'audit 7200 e della lettura 3600; testo descrittivo del readout) applica una mutazione che la disattivi ed esegui i test pertinenti: quale test fallisce? Una guardia non protetta da alcun test e' un rilievo MINORE.
c) La ricevuta results/infra/contrastive_targets_20261118/full_suite/verification.json e cpu_suite.json: gli sha256 delle evidenze corrispondono ai file? Ogni FAILED o ERROR dei log originali (cpu_part*.log) e' fra i file rieseguiti e poi passato? Il conteggio di 3.723 test distinti su 233 file e' coerente con i log, per quanto ricostruibile? La ricevuta dichiara il codice 65bc6022: verifica che 65bc6022..438c8d65 non cambi sorgenti o test.
Cancella le cartelle temporanee alla fine.` },
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
        agent(`${CTX}\n\nSei lo scettico indipendente n. ${i + 1} per un rilievo del revisore del punto ${d.point}. Compito: CONFUTARE il rilievo se possibile. Riproduci da solo la prova (comandi tuoi, non fidarti dell'output citato), cerca nel codice e nei documenti motivi per cui non e' un difetto (un controllo altrove, un percorso non usato in produzione, una dichiarazione esplicita nell'emendamento o nel piano), e valuta la gravita' reale secondo la scala comune. Se non riesci a riprodurlo, o resta incerto, rispondi refuted=true. Scrivi il tuo esito anche in ${A.reports}/verify-${d.key}-${f.id}-${i + 1}.md.\n\nRILIEVO:\n${JSON.stringify(f, null, 1)}`,
              { label: `verify:${d.key}:${f.id}:${i + 1}`, phase: 'Verifica', schema: VERDICT_SCHEMA })))
        .then(votes => {
          const vs = votes.filter(Boolean)
          const upheld = vs.filter(v => !v.refuted).length
          return { ...f, votes: vs, upheld, needed: Math.floor(vs.length / 2) + 1, survives: vs.length > 0 && upheld >= Math.floor(vs.length / 2) + 1 }
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
const critic = await agent(`${CTX}\n\nSei il critico di completezza. Sotto trovi la sintesi di sette revisori (verifiche positive, rilievi con esito della verifica avversaria, cose non verificate) sulla consegna. Confrontala con la richiesta originale di GPT, che chiede di verificare in un unico passaggio: 1) nessuna reinterpretazione del T100-G storico e 100/125 esplicito su tutte le celle nuove; 2) nessuna accettazione automatica di vecchi audit, lambda, calibrazioni o benchmark; 3) confronti CN-C0/CR-C0 appaiati nella nuova ricetta e C0-T100-G con due fattori aggiuntivi dichiarati; 4) coefficiente storico di lambda come prior, entrambe le lambda da ricalcolare, cancelli invariati; 5) preventivo di 13 allocazioni, 14 GPU-ore e 137,066667 core-ore, audit di 2 h coerente col dispatcher; 6) il cambio dei target non sana parita' e ripresa bf16, la diagnostica va conclusa nel proprio checkout prima del nuovo rilascio, nessun retry cieco o soglia nuova implicita. Elenca le lacune concrete: verifiche non fatte, affermazioni non provate, file non letti, percorsi di codice non coperti. Per ciascuna indica come verificarla in pochi comandi. Puoi fare tu stesso controlli rapidi per confermare che una lacuna e' reale. Scrivi l'esito anche in ${A.reports}/critico-completezza.md.\n\nSINTESI:\n${JSON.stringify(summary, null, 1)}`,
  { label: 'critico-completezza', phase: 'Completezza', schema: CRITIC_SCHEMA })

return { results, critic, missing }
