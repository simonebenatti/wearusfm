export const meta = {
  name: 'rilettura-diff-prova-mirata',
  description: 'Rilettura del diff correttivo della prova mirata InfoNCE (GPT 8c72655c): confronto, orchestrazione, pacchetto e binding, test e mutazioni, regressioni; poi verifica avversaria',
  phases: [
    { title: 'Revisione', detail: 'cinque revisori indipendenti sul diff correttivo' },
    { title: 'Verifica', detail: 'due scettici per ogni rilievo bloccante o importante' },
  ],
}

const SP = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad'
const BASE = SP + '/rev_probe2'
const OLD = SP + '/rev_probe'
const CLONE = BASE + '/clone'

const COMMON = `Contesto. Progetto WearUsFM, linea 3. GPT ha scritto un diagnostico GPU a tentativo unico ("prova mirata": due riprese InfoNCE dallo stesso checkpoint a 544, una ordinaria e una "warm" con un backward di riscaldamento DDP a effetto nullo; al passo 545 si registrano per rank e per tensore i gradienti locali e ridotti tramite un comm hook; poi un confronto su CPU classifica l'esito in SUPPORTED, PARTIAL_SUPPORT, HYPOTHESIS_UNSUPPORTED o INCONCLUSIVE). Claude, il revisore, il 21/11 ha chiuso la release per due difetti bloccanti e nove correzioni (review fc3eef6b). GPT ha pubblicato il commit correttivo 8c72655c. Tu lavori per Claude: devi rileggere una parte del SOLO diff correttivo. Un difetto e' bloccante solo se, con un controesempio concreto, rende il run vero INCONCLUSIVE o fallito per ragioni non scientifiche, oppure fa attribuire male l'esito. Vincoli assoluti: niente ssh, niente rete, niente git commit o push, nessuna modifica al clone; scrivi solo nelle cartelle indicate.

Percorsi.
- Clone del ramo di GPT al commit correttivo 8c72655c3d894c82fd4918ed523baf0ac9c69e23 (detached, sola lettura): ${CLONE}
- Diff da rileggere: git -C ${CLONE} diff 9a2ce7e7 8c72655c (13 file). Codice toccato: scripts/diagnostics/contrastive_cn_restart.py (ciclo delle fasi in run(), tempi, eccezioni), contrastive_cn_restart_compare.py (validate_rng_schema, warm_endpoint_differences, screen_reduced_gradients, ritorno INCONCLUSIVE, classificazione), contrastive_cn_restart_observe.py (load_observation: percorso str, norme descrittive). NON toccati: contrastive_cn_restart_leg.py, scripts/slurm/cn_restart_probe.sbatch, il codice di produzione. Altri file: results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/{plan.json,local_validation_20261122.json}, docs/contrastive_cn_restart_probe_20261121.md (protocollo), docs/decisioni.md (voce del 22/11), docs/prompt_claude_cn_restart_probe_20261121.md, tests/cpu/test_cn_restart_{compare,observe,package,end_to_end}.py.
- Review precedente di Claude con i due bloccanti (B-1, B-2) e i nove punti: /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md (sola lettura). Materiale della review precedente (rapporti, script, mutazioni, prova della catena su CPU): ${OLD}/reports/*.md e ${OLD}/agents/<chiave>/ (per esempio r2-catena-cpu, r4-confronto/mutate.py, r6-test).
- Formato VERO dei checkpoint storici, letto da Claude su Leonardo il 21/11: ${OLD}/leonardo/confronto_checkpoint_veri_20261121.out.txt (chiavi del payload; cuda_rng e' una lista di 4 tensori uint8; rank_rng una lista di dict con chiavi cuda_rng e torch_rng; fra continuo e ripresa CN_A differiscono solo cuda_rng.1, .2, .3 fra le voci non in virgola mobile; CN_A e CN_B identici). Letture del torch 2.2 del cluster: ${OLD}/leonardo/torch22_sorgente_20261121.out.txt e torch22_api_20261121.out.txt (torch.load con mmap=True esige str; niente uuid nelle proprieta' CUDA).
- Oggi Leonardo non e' raggiungibile da Claude (certificato scaduto): nessuna prova sui dati veri in questo giro.
- La tua cartella di lavoro (creala): ${BASE}/agents/<chiave> . Il tuo rapporto: ${BASE}/reports/<chiave>.md (aggiornalo man mano).

Strumenti. Python: ~/.venvs/wearusfm-tok/bin/python (torch 2.14 solo CPU, Python 3.11). Test: cd ${CLONE} && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:scripts/diagnostics ~/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=<cartella nuova sotto la tua cartella di lavoro> <file>. Shell zsh: per elenchi di file usa bash -c; scrivi \${VAR} quando una variabile e' seguita da due punti. Il Mac ha poca RAM e poco disco: niente run oltre qualche minuto; cancella i file grandi che crei. Per simulare il torch del cluster puoi avvolgere torch.load in modo che sollevi ValueError quando mmap=True e f non e' str: dichiaralo come simulazione.

Regole di consegna. Ogni rilievo: gravita' (bloccante / importante / minore / nota), file:riga, scenario concreto, prova (comando e output, o citazione del codice), correzione minima. Distingui eseguito da dedotto. Non gonfiare: senza controesempio non e' bloccante. Elenca anche le verifiche positive importanti. Rispondi con l'oggetto strutturato richiesto.

`

const TASKS = [
  { key: 'c1-confronto', prompt: `Compito C1 (chiave: c1-confronto) - Le correzioni al confronto (contrastive_cn_restart_compare.py).
1. B-2: leggi validate_rng_schema e warm_endpoint_differences e l'uso in compare(). Regola registrata (docs/decisioni.md, voce del 22/11, e protocollo): nella sola coppia riscaldata/continuo cuda_rng.1, .2, .3 sono riportati a parte e devono coincidere byte per byte con quelli della ripresa ordinaria; restano esatti il file intero dell'ordinaria, cuda_rng.0, torch_rng e rank_rng di tutti i rank; ogni altra differenza non in virgola mobile a 576, struttura alterata o copia inattiva diversa dall'ordinaria da' INCONCLUSIVE. Verifica che il codice faccia esattamente questo, costruendo payload con il formato vero (vedi il file della lettura su Leonardo): caso atteso (riscaldata con cuda_rng.1-3 uguali all'ordinaria e diversi dal continuo, tutto il resto uguale al continuo) -> nessuna voce scientifica, exact_scientific_payload vero; copia inattiva diversa dall'ordinaria; cuda_rng.0 diverso; rank_rng diverso; torch_rng diverso; lista di lunghezza sbagliata; tensore non uint8 o non 1-D; chiave mancante; altra voce non float diversa (step, alarm_counts, config). validate_rng_schema puo' rifiutare un payload vero (controlla forma e tipo attesi di torch.get_rng_state e torch.cuda.get_rng_state, e come run.py salva torch_rng, cuda_rng e rank_rng)? Il ritorno anticipato con status INCONCLUSIVE contiene i campi che main() e i lettori a valle si aspettano?
2. Soglia e scalari: leggi screen_reduced_gradients e l'accumulo di scalar_addend_scales. Regola registrata: soglia 1e-6 su tutti i 270 tensori; solo per gli ID 253 (anchor_heads.rms.bias) e 257 (anchor_heads.env.bias) il denominatore e' il massimo fra i moduli dei due gradienti ridotti, 1e-12 e la media dei moduli dei quattro gradienti locali ordinari; si mostrano entrambi i rapporti. Verifica: che "local" sia il gradiente locale NON scalato (leggi contrastive_cn_restart_observe.py e l'hook) e che quindi la media dei moduli sia la scala giusta degli addendi della media DDP; che la somma copra tutti e quattro i rank prima dello screening; che per gli altri 268 tensori nulla sia cambiato; che il rapporto originale resti nel report. Dimostra con un conto e con una simulazione fp32 a 4 addendi (due ordini di somma) quale sia lo scarto massimo relativo alla nuova scala per un tensore di un elemento, e se 1e-6 puo' ancora essere superato dal solo arrotondamento. Controlla che un eccesso vero (per esempio 1e-3) venga ancora rilevato anche sui due scalari.
3. Classificazione: classify ora usa exact_scientific_payload. Verifica tutti i rami con dati sintetici coerenti e che SUPPORTED sia raggiungibile solo quando deve. Verifica che il report contenga sempre training_approved falso e historical_status FAILED_REQUIRES_REVIEW.
4. Cerca regressioni introdotte dal diff in compare.py: ordine delle gambe (ordinary prima di warm), uso di ordinary_cuda_rng quando l'ordinaria e' gia' invalida, memoria (clone dei quattro stati), eccezioni non catturate da main().
5. Lancia tests/cpu/test_cn_restart_compare.py e test_cn_restart_end_to_end.py e di' se i nove casi coprono davvero i rami sopra.` },
  { key: 'c2-orchestrazione-lettore', prompt: `Compito C2 (chiave: c2-orchestrazione-lettore) - Orchestrazione del job (run() in contrastive_cn_restart.py), lettore dei vettori (load_observation) e prova della catena su CPU.
1. run(): rileggi il nuovo ciclo delle fasi (ordinary, warm, compare) con phase_timings.jsonl, execution_failure.json e la tupla delle eccezioni (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired). Che cosa solleva checked_command (scripts/contrastive_split_benchmark.py) in ogni caso: scadenza gia' passata, timeout del figlio, codice di uscita diverso da zero? Tutti i casi finiscono nell'except e scrivono il file? Il file phase_timings.jsonl viene scritto e chiuso correttamente anche in caso di errore? Un errore nello scrivere i tempi puo' far fallire una fase riuscita? Il codice gira dentro l'allocazione GPU: cerca qualunque errore che farebbe fallire il job prima o dopo le gambe (nomi di variabili, zip strict, apertura con "x", json.dumps di tipi non serializzabili). Simula run() con comandi finti (monkeypatch di commands, checked_command, validate_allocation, nvidia-smi, variabili d'ambiente) per i casi: tutto riuscito; timeout in ciascuna delle tre fasi; errore nel confronto; report mancante.
2. load_observation: B-1 corretto con str(tensor_path)? Con lo shim del torch 2.2 il lettore ora carica osservazioni vere prodotte dall'osservatore (riusa i fixture dei test o quelli di ${OLD}/agents/r2-catena-cpu)? Il controllo della norma a 1e-12 e' stato sostituito da un controllo di validita': puo' rifiutare una ricevuta vera (norma int? norma zero per un gradiente nullo? float di tipo numpy?) e quindi rendere INCONCLUSIVE il run? Guarda come finish() scrive le norme. Controlla che l'impronta sha256 resti il controllo di identita'.
3. Prova della catena su CPU con il codice corretto: riusa l'harness di ${OLD}/agents/r2-catena-cpu (4 processi Gloo, trainer vero su un modello piccolo, gambe ordinary e warm con le funzioni vere) puntando PYTHONPATH al clone corretto ${CLONE}; rifai le gambe e applica le funzioni corrette del confronto (compare_vector, screen_reduced_gradients, warm_endpoint_differences con payload che abbiano cuda_rng e rank_rng nel formato vero, classify). L'esito resta SUPPORTED? Qualche funzione nuova solleva eccezioni inattese sui dati della catena? Se l'harness non e' riutilizzabile in pochi minuti, usa il test end-to-end di GPT (tests/cpu/test_cn_restart_end_to_end.py) e descrivi che cosa esegue davvero (funzioni vere o sostituti).
4. Conferma con git diff che contrastive_cn_restart_leg.py, lo sbatch e la parte di cattura dell'osservatore (ReductionObservation, _average_observed, finish) NON sono cambiati fra 9d34dc69 e 8c72655c: le due gambe GPU sono le stesse gia' riviste.` },
  { key: 'c3-pacchetto-binding-protocollo', prompt: `Compito C3 (chiave: c3-pacchetto-binding-protocollo) - Pacchetto, binding con le nuove impronte, piano e protocollo operativo.
1. Esegui nel clone: PYTHONPATH=src python scripts/diagnostics/contrastive_cn_restart.py --package (con la venv e le variabili indicate). Atteso: package_sha256 e9310d67b8304c064e09d71b0ec55d37b57abf3b4e2090c2d02d4def586f702d e plan_sha256 4f538cf183f6e3daf7bd30a5612eecdbbd07106f64d5f348a2f3e759e877853b. Verifica anche che il digest dei sorgenti di produzione sia ancora 40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0.
2. Simula il flusso vero del binding in una COPIA del clone nella tua cartella (git clone locale del clone): crea un ramo locale e fallo puntare da refs/remotes/origin/worktree-claude-training-engine; in quel ramo metti il file docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md con un testo qualunque piu' UNA sezione di codice con linguaggio cn-restart-review e il JSON a cinque campi (schema cn-restart-review-v1, status APPROVED_FOR_DIAGNOSTIC, reviewed_commit 8c72655c3d894c82fd4918ed523baf0ac9c69e23, package_sha256 e plan_sha256 come sopra); committa; nella copia a 8c72655c esegui --bind-review <commit>; committa il release_binding.json in un commit figlio di 8c72655c; da un checkout pulito di quel commit chiama require_binding(). Deve passare. Prova i casi negativi: reviewed_commit 9d34dc69 (il vecchio), impronte vecchie (b0640ca6..., 9d5811b6...), blocco doppio, JSON su piu' righe (deve essere accettato o no? riportalo), righe CRLF, un sesto campo, commit della review non antenato del ramo. Riporta esattamente quale forma del blocco viene accettata (apertura e chiusura a colonna 0, JSON su una riga o piu' righe), perche' Claude dovra' scriverlo in quel modo. Verifica che il testo della review possa contenere caratteri non ASCII e la parola cn-restart-review in prosa senza rompere il parser.
3. plan.json: le chiavi nuove (review_request_commit, readout_amendment) sono coerenti con il codice (costanti SCALAR_ADDEND_PARAMETERS, PROCESS_ONLY_CUDA_RNG, LIMIT, DENOM_FLOOR) e c'e' un test che le lega? Il piano resta REVIEW_REQUIRED con training_approved falso?
4. Protocollo (docs/contrastive_cn_restart_probe_20261121.md) e voce del 22/11 di decisioni.md: verifica che contengano, prima della misura, la regola su cuda_rng.1-3, quella sui due scalari e quella sul recupero del solo lettore; il comando esatto di --validate-only sul login; il controllo quota con lfs; la raccolta dei due log e di phase_timings.jsonl; il ledger a 1,0 GPU-ora. Rifai i conti: 1 GPU-ora e 8,2667 ore locali al massimo, inviluppo 1,4 e 11,8667, residui sul tetto 18 e 169,6 partendo da 4,74 e 47,3322. Segnala affermazioni ormai superate (oggi e' il 27/11: la manutenzione del 23-25/11 e' passata).
5. I comandi di invio usano ora python3 /Users/simonebenatti/dev/wearusfm/scripts/mac_launcher_command.py "$PWD" "$RELEASE_SHA" -- python3 scripts/submit_job.py ... : quello script esiste nel repository canonico (/Users/simonebenatti/dev/wearusfm/scripts/, sola lettura) o nel clone? Che cosa fa? Gli argomenti di submit_job.py (--id, --agente, --campagna, --gpu-ore, --script-stdin, argomenti dopo "--") sono validi per scripts/submit_job.py del clone? Elenca, come fatti e senza giudizi, che cosa dovrebbe cambiare nei comandi se a inviare fosse un operatore diverso da 3-launcher (per esempio l'etichetta --agente), e che cosa invece e' vincolato dal codice (ID logici del piano, nome della campagna cn-restart545-20261121, CODE_DIR, EXPECTED_SHA, TORCH_DIST_INIT_BARRIER, --time 00:15:00, risorse).` },
  { key: 'c4-test-mutazioni', prompt: `Compito C4 (chiave: c4-test-mutazioni) - Test e loro forza dopo le correzioni.
1. Riproduci la ricevuta results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/local_validation_20261122.json: 211 test passati nei 12 file elencati. Riporta i conteggi per file e rilancia due volte i file con Gloo.
2. Leggi tests/cpu/test_cn_restart_end_to_end.py: i "nove casi completi" passano davvero da compare.main() (CLI) con osservazioni scritte dall'osservatore vero, lo shim del torch 2.2 per mmap, payload con rank_rng e cuda_rng realistici e inventario 4 x 270? Quali esiti coprono (SUPPORTED, PARTIAL_SUPPORT per 545 e per 576, HYPOTHESIS_UNSUPPORTED per locale, eccesso e assenza di scarto, INCONCLUSIVE per fedelta' dell'ordinaria e per metadati)? Che cosa e' sostituito da finzioni (history, checkpoint storici, tracce)? I test di rifiuto di load_observation hanno ora un match specifico? I timeout sono provati nelle tre fasi?
3. Rifai i test di mutazione della review precedente sul codice corretto: gli script sono in ${OLD}/agents/r6-test (84 mutazioni) e ${OLD}/agents/r4-confronto/mutate.py (30 mutazioni). Adattali al codice nuovo dove le righe sono cambiate (lavora su COPIE nella tua cartella, con la copia mutata prima di scripts/diagnostics nel PYTHONPATH) e aggiungi almeno 15 mutazioni sulle parti nuove: warm_endpoint_differences (per esempio non controllare matches_ordinary, ignorare same_shape, rimuovere anche altri percorsi), validate_rng_schema, screen_reduced_gradients (usare la scala per tutti i tensori, non usarla per gli scalari, soglia x10), ritorno INCONCLUSIVE, ciclo delle fasi in run(), load_observation con Path. Riporta quante mutazioni sopravvivono in tutto e quali, distinguendo quelle che toccano guardie di validita' scientifica. Confronta con il risultato precedente (54 su 84).
4. Segnala test che passano sul Mac ma non proverebbero nulla sul cluster.` },
  { key: 'c5-critico-regressioni', prompt: `Compito C5 (chiave: c5-critico-regressioni) - Critico: i due bloccanti e i nove punti sono davvero chiusi? Il diff introduce rischi nuovi?
1. Costruisci la tabella dei due bloccanti (B-1 lettura dei vettori con Path e mmap; B-2 cuda_rng.1-3) e dei nove punti della review di Claude (1 soglia e parametri scalari; 2 regola per rieseguire il solo confronto; 3 test di compare() e match nei rifiuti; 4 TimeoutExpired; 5 --gpu-ore 1.0; 6 controllo della norma a 1e-12; 7 costanti del piano legate al codice; 8 comando di --validate-only; 9 tempi per gamba, file .err, quota), piu' la richiesta di scrivere nella regola di B-2 che ogni altra differenza non float resta causa di INCONCLUSIVE. Per ciascuno: dove e' stato corretto (file:riga), se la correzione e' completa, e che cosa eventualmente manca.
2. Regole registrate prima della misura: la voce del 22/11 di docs/decisioni.md, il protocollo e plan.json (readout_amendment) dicono la stessa cosa del codice? Cerca scostamenti: per esempio "media dei moduli dei quattro gradienti locali ordinari" contro cio' che il codice accumula; "devono coincidere byte per byte con quelle della ripresa ordinaria" contro il codice; il recupero del solo lettore (condizioni: gambe complete a 576, dati grezzi integri, criteri invariati, rapporto distinto). C'e' qualcosa nelle regole nuove che possa essere contestato come scelta fatta dopo i dati? (I dati della prova non esistono ancora; i checkpoint storici si'.)
3. Rischi nuovi introdotti dal diff per il tentativo unico: il ciclo delle fasi in run() gira dentro l'allocazione; validate_rng_schema gira sul checkpoint vero; il ritorno anticipato. Elenca in ordine di probabilita' per impatto che cosa puo' ancora rendere il run INCONCLUSIVE per ragioni non scientifiche, con prova nel codice. Tieni conto di cio' che la review precedente lasciava aperto: comm hook su NCCL, ramo ctypes per l'UUID della GPU, riproduzione dei sette bucket, tempo (stima 450-620 s su circa 830 utili).
4. Attribuzione: con le regole nuove, in quali scenari il report direbbe SUPPORTED, PARTIAL_SUPPORT o HYPOTHESIS_UNSUPPORTED per la ragione sbagliata?
5. Il prompt di GPT chiede a Claude di "mantenere riconoscibile l'esito storico fc3eef6b" e di inserire una singola sezione di codice cn-restart-review: c'e' qualcosa nel pacchetto che renderebbe invalido il binding se Claude aggiorna lo stesso file della review (per esempio un secondo blocco, o il testo storico)?
Sii concreto: file:riga e scenario.` },
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
        agent(COMMON + `Compito di verifica avversaria (chiave: verifica-${task.key}-${f.id}-${i}). Un revisore ha segnalato il rilievo qui sotto sul commit correttivo. Il tuo compito e' CONFUTARLO o ridimensionarlo: riproduci tu lo scenario con il codice vero (o dimostra dal codice, con file:riga, perche' non puo' verificarsi), controlla se un altro controllo lo intercetta gia', e valuta la gravita' giusta secondo le regole sopra. Se regge, dillo e proponi la correzione minima. Nel dubbio non confermare un bloccante senza prova.

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
