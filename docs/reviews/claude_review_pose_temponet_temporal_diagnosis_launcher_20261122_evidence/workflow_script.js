export const meta = {
  name: 'tcn-temporal-gate-audit',
  description: 'Audit indipendente della catena del controllo temporale TCN, verifica dei numeri di GPT, valutazione del candidato residuale e mappe per i diagnostici',
  phases: [
    { title: 'Audit', detail: 'sette lettori indipendenti con lenti diverse sul codice congelato e sui JSON' },
    { title: 'Verifica', detail: 'uno scettico per ogni difetto serio dichiarato' },
  ],
}

const D = '/Users/simonebenatti/wearusfm_local/tcn_diag_20261122'
const TREE = D + '/tree_e8c98244'
const CTX = D + '/context'
const EVID = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/docs/reviews/claude_review_pose_temponet_temporal_diagnosis_20261121_evidence'
const PY = 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=' + TREE + '/scripts:' + TREE + '/src /Users/simonebenatti/.venvs/wearusfm-tok/bin/python'

const PRE = `Sei un revisore indipendente del progetto WearUsFM (foundation model per sEMG). Lavori in italiano.

CONTESTO SCIENTIFICO
Teste di lettura (L0 lineare, MLP, TCN_post da 72.904 parametri) vengono allenate sui latenti congelati di un encoder (finestre da 160 patch x 384 dimensioni, cache FP16; train: 1.907 finestre di 40 utenti, 53 senza posa valida; due fold esterni da 20 utenti: fitting 920 e 934 finestre). Prima dei dati reali un "controllo positivo temporale" sintetico qualifica la ricetta: sui latenti veri del train si costruisce un bersaglio noto y* = L(z) + a*T(z) e y = y* + sigma*eps, con z = LayerNorm(latente), L affine, T rete temporale congelata larga 8 (cinque ReLU, campo ricettivo 13 patch), eps rumore AR(1) con rho = exp(-1/4). La calibrazione ha scelto a = 0,4 e sigma = 1,3 in entrambi i fold. Il cancello: la TCN deve recuperare almeno il 50% del margine fra il lettore lineare stimato sul bersaglio pulito (clean_ls, valutato su y) e l'oracolo, sulla metrica E_dyn (errore centrato nel tempo sulle finestre "dinamiche"), nei due fold e nel pooled: R = (E_cleanLS - E_TCN)/(E_cleanLS - E_oracle).
Esito del 21/11 (job Slurm 65035631): R = 24,17% / 18,88% / 21,71% (fold0 / fold1 / pooled) -> stato recipe_not_admitted, campagna "A1" ferma. Ricerca: 5 learning rate (1e-4 fino a 300 epoche; 3e-4, 1e-3, 3e-3, 1e-2 fino a 150), scelto LR 0,003 all'esposizione 31 (minimo della MSE pooled sugli esclusi interni).
GPT (responsabile del codice) ha scritto una diagnosi e propone: (1) diagnostici sui checkpoint esistenti senza nuovi fit; (2) una sola prova TCN sul bersaglio pulito y*; (3) eventualmente un candidato "residuale" (LayerNorm fissa, base affine/ridge stimata sul solo fitting con alpha da CV interna e poi congelata, TCN allenata sul residuo, somma a inferenza).

MATERIALE (sola lettura)
- Copia del commit di GPT e8c9824461b5e18494ff462f1be8abda71b76d5e (ramo codex/pose-temponet-launcher-20261120): ${TREE}
  - codice scientifico congelato: ${TREE}/scripts/pose_temponet/{controls,train,selection,metrics,heads,data,linear,workflow,runner,execution,provenance,resources,benchmark,report,__main__}.py
  - launcher: ${TREE}/scripts/slurm/{pose_temponet.sbatch,pose_temponet_checkout.sbatch,pose_temponet_dispatch.sbatch,lib_code_dir.sh,campaign_checkout.sbatch} e ${TREE}/scripts/submit_job.py
  - protocollo congelato: ${TREE}/docs/pose_temponet_frozen_protocol_20261117.md
  - diagnosi di GPT da rivedere: ${TREE}/docs/reviews/pose_temponet_temporal_diagnosis_20261122.md ; richiesta: ${TREE}/docs/prompt_claude_tcn_temporal_diagnosis_20261122.md ; esiti dei controlli: ${TREE}/docs/reviews/pose_temponet_controls_results_20261121.md
  - risultati archiviati (JSON, niente array): ${TREE}/results/infra/pose_temponet_execution_20261121_A1/launcher/ (positive_controls_gate_20261121.json, selection_{linear,temporal}_{L0,MLP,TCN_post}.json, calibration_{0,1}.json, ricevute TCNH-*.json, sacct) e ${TREE}/results/infra/pose_temponet_execution_20261121_A1/analysis_20261122/ (prova di rappresentabilita' di GPT)
  - test CPU: ${TREE}/tests/cpu/test_pose_temponet_*.py
- Contesto gia' discusso (un'altra chat di GPT e la review precedente di Claude sullo stesso fallimento): ${CTX}/gpt_chat2_diagnosi_D0_D1_8fac3a82.md, ${CTX}/gpt_chat2_risposta_8fac3a82.md, ${CTX}/claude_review_f6669115.md, ${CTX}/claude_convergenza_4a48b814.md
- Simulazioni giocattolo di ieri (latenti sintetici, codice vero): ${EVID}/revisori_note_e_script/

COME ESEGUIRE CODICE
- Solo CPU, sul Mac. Comando Python (obbligatorio cosi', con un solo thread): ${PY}
  Il pacchetto si importa come "from pose_temponet import controls, train, ..." (PYTHONPATH punta a ${TREE}/scripts e ${TREE}/src).
- pytest: aggiungi "-p no:cacheprovider" e lancia dalla tua cartella di note, indicando i file di test per percorso assoluto.
- Il Mac ha 18 GB di RAM e poco disco: esperimenti piccoli (sotto ~2 GB di RAM, ogni esecuzione sotto ~5 minuti).

REGOLE
- Non modificare NIENTE sotto ${TREE}, ne' in alcun repository git; nessun comando git che cambi stato. Scrivi solo nella tua cartella di note.
- Nessun accesso al cluster (niente ssh), nessuna rete. I dati veri (latenti, checkpoint) NON sono sul Mac: non fingere di averli letti.
- Non lanciare altri agenti o workflow.
- Tempo: circa 30 minuti di orologio in tutto. Scrivi le note man mano nel file notes.md della tua cartella (aggiungi dopo ogni risultato, non alla fine): se vieni interrotto, deve restare cio' che hai trovato.
- Distingui sempre "dimostrato" (con file:riga e comando riproducibile, o numero ricalcolato dai JSON) da "ipotesi". Un difetto senza prova riproducibile va dichiarato come sospetto, non come difetto. Conteggi e numeri esatti, mai arrotondati a senso.
- La risposta finale e' l'oggetto strutturato richiesto (dati, non una lettera): sii conciso e preciso.
`

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'tre-sei frasi: che cosa hai verificato e con quale esito' },
    verdict: { type: 'string', description: 'una riga: nessun difetto / difetto dimostrato / sospetti aperti' },
    defects: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['bloccante', 'importante', 'minore', 'nota'] },
          status: { type: 'string', enum: ['dimostrato', 'sospetto'] },
          title: { type: 'string' },
          file_line: { type: 'string' },
          claim: { type: 'string' },
          evidence: { type: 'string' },
          repro_command: { type: 'string' },
          would_explain_failed_gate: { type: 'string', description: 'si / no / in parte, con una frase' },
        },
        required: ['id', 'severity', 'status', 'title', 'claim', 'evidence'],
      },
    },
    verified_facts: { type: 'array', items: { type: 'string' } },
    cannot_conclude: { type: 'array', items: { type: 'string' } },
    recommendations: { type: 'array', items: { type: 'string' } },
    notes_path: { type: 'string' },
  },
  required: ['summary', 'verdict', 'defects', 'verified_facts', 'cannot_conclude', 'recommendations', 'notes_path'],
}

const MAP_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    key_points: { type: 'array', items: { type: 'string' } },
    pitfalls: { type: 'array', items: { type: 'string' } },
    open_questions: { type: 'array', items: { type: 'string' } },
    defects: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['bloccante', 'importante', 'minore', 'nota'] },
          status: { type: 'string', enum: ['dimostrato', 'sospetto'] },
          title: { type: 'string' },
          file_line: { type: 'string' },
          claim: { type: 'string' },
          evidence: { type: 'string' },
        },
        required: ['id', 'severity', 'status', 'title', 'claim', 'evidence'],
      },
    },
    notes_path: { type: 'string', description: 'percorso del documento principale scritto su disco' },
  },
  required: ['summary', 'key_points', 'pitfalls', 'open_questions', 'defects', 'notes_path'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    refuted: { type: 'boolean' },
    confirmed_severity: { type: 'string', enum: ['bloccante', 'importante', 'minore', 'nota', 'nessuna'] },
    reasoning: { type: 'string' },
    evidence: { type: 'string' },
    notes_path: { type: 'string' },
  },
  required: ['id', 'refuted', 'confirmed_severity', 'reasoning', 'evidence'],
}

const AGENTS = [
  {
    key: 'target_chain', schema: AUDIT_SCHEMA,
    task: `IL TUO COMPITO: audit della catena generazione -> bersaglio. Cartella note: ${D}/review/target_chain/
Cerca un difetto concreto (file:riga e prova riproducibile) in: controls.generate_components (semi per fold, L e T, layer_norm eps 1e-5, FP16->FP32), normalize_component (media/scala per giunto stimate sul solo fitting? crop? supporto?), window_noise (AR(1): varianza stazionaria unitaria? rho? dipendenza da utente/finestra/giunto; rumore sui bordi), synthetic_target (y* = L + a*T, y = y* + sigma*eps), calibrate_grid e build_positive_control_fold (come sono scelti a e sigma sul solo inner-train; che cosa garantisce e che cosa no), e soprattutto la COERENZA fra tre luoghi: (i) il bersaglio su cui la calibrazione misura il margine cleanLS-oracolo, (ii) il bersaglio su cui le teste sono allenate nel job dei controlli (segui workflow.py/runner.py: quali array arrivano a train.py come target, con quale normalizzazione, e se il rumore e' lo stesso array), (iii) il bersaglio e i pannelli usati da evaluate_outer_references e dal cancello. Controlla anche se la LayerNorm della testa (apprendibile) e quella del generatore (fissa, eps 1e-5) partono identiche, e se lo z visto dalla testa e' lo stesso z del generatore.
Costruisci una piccola prova CPU con latenti sintetici (poche finestre, 160x384, 4-6 utenti finti) che esegua le funzioni VERE e verifichi numericamente: varianza e autocorrelazione di eps, normalizzazione di L e T sul fitting, uguaglianza y - y* = sigma*eps, e che un "allievo" con i pesi veri riproduca y* (puoi partire dallo script di GPT in results/infra/pose_temponet_execution_20261121_A1/analysis_20261122/).
Leggi anche calibration_0.json e calibration_1.json: la griglia provata, i valori scelti, i margini; di' se il margine cleanLS->oracolo sugli esclusi esterni (circa 8%) e' coerente con quello della calibrazione interna.
Rispondi con lo schema: defects (anche vuoto), verified_facts, cannot_conclude, recommendations.`,
  },
  {
    key: 'support_panels', schema: AUDIT_SCHEMA,
    task: `IL TUO COMPITO: audit di supporto, crop, maschere, loss e uguaglianza dei pannelli. Cartella note: ${D}/review/support_panels/
(1) Segui il supporto dall'inizio alla fine: data.py (central_support, lengths_for, key_time_valid), il crop di 6 patch per lato (160 -> 148), metrics.per_window_mse (la loss di training), metrics.evaluate/_cropped/selection_statistics, controls._metric_panel. Domande precise: la loss di training usa esattamente lo stesso supporto (stesse patch, stessi giunti, stesse finestre) della metrica di selezione e del cancello? Le finestre senza posa valida sono escluse allo stesso modo nei tre luoghi? Il crop e' applicato a predizione e bersaglio in modo coerente (nessuno sfasamento di una patch)? Il rumore eps e i componenti L/T sono definiti su 160 patch e ritagliati allo stesso modo?
(2) E_dyn: definizione esatta (window_joint_moments e campi "centered"/"vary" nel modulo storico importato), soglia dinamica (training_threshold = mediana sul fitting della varianza del bersaglio PULITO), selezione delle finestre dinamiche (pose_var dal bersaglio pulito). Verifica che per ogni pannello (fold0, fold1, pooled) clean_ls, oracolo e TCN siano valutati sullo STESSO insieme di finestre, stesso supporto, stessa soglia e stesso denominatore: confrontalo sia nel codice sia nei JSON archiviati (window_count, dynamic_target_ss, support_sha256/evaluation_identity, soglie in calibration_*.json e positive_controls_gate_20261121.json, selection_temporal_*.json). Riporta i numeri esatti a confronto.
(3) Verifica con una prova CPU su dati sintetici l'identita' SSE_y = SSE_y* + SSE_oracolo - 2<e,n> con le funzioni vere (stesso supporto per y e y*), cioe' che valutare contro y e contro y* usa lo stesso supporto dinamico.
(4) Il "pooled" somma gli accumulatori o media i rapporti? E' lo stesso nel cancello e nella selezione?
Rispondi con lo schema: defects (anche vuoto), verified_facts, cannot_conclude, recommendations.`,
  },
  {
    key: 'training_loop', schema: AUDIT_SCHEMA,
    task: `IL TUO COMPITO: audit del ciclo di training e delle sue scelte, con prove giocattolo. Cartella note: ${D}/review/training_loop/
(1) Leggi train.py e heads.py riga per riga: costruzione della testa TCN_post (ramo affine diretto, LayerNorm apprendibile, strati 64/32, dilatazioni, dropout 0,1, inizializzazione: quali strati partono a zero?), ottimizzatore (AdamW: weight decay 0,01 applicato a quali parametri? anche a LayerNorm e bias?), learning rate costante?, clip della norma del gradiente a 1,0, batch 32 con ultimo batch parziale, ordine dei batch e RNG isolati per fold, model.train()/eval() al punto giusto, cattura dello stato migliore (copia profonda? prima o dopo l'update?), sincronizzazione dei due fold per esposizione, valutazione sugli esclusi interni. Cerca un difetto concreto (file:riga e prova riproducibile): per esempio stato migliore che punta a tensori poi modificati, dropout attivo in valutazione, gradiente che non raggiunge il ramo profondo, weight decay o clip che di fatto bloccano il ramo temporale, bersaglio o maschera sfasati nel batch.
(2) Il clip a norma 1,0: stima quanto spesso scatterebbe con bersagli di varianza circa 1 + 0,16 + 1,69 e loss per-window MSE; se scatta quasi sempre, il passo effettivo non e' quello nominale. Misuralo in un giocattolo.
(3) Giocattolo CPU con la testa, la loss e la ricetta VERE su latenti sintetici (riusa gli script di ieri in ${EVID}/revisori_note_e_script/, un fold, LR 0,003, orizzonte troncato a 40-60 epoche, 1-2 semi): misura il recupero sugli esclusi (scala del cancello) all'epoca scelta sulla MSE rumorosa, per la ricetta di base e per UNA ablazione alla volta: senza clip, senza dropout, weight decay 0, LayerNorm congelata, uscita non inizializzata a zero. Scopo: dire quali scelte della ricetta spostano il risultato nel giocattolo e quali no. E' evidenza indicativa (latenti sintetici), dichiaralo.
(4) "Nessun optimizer_alarm" che cosa certifica davvero e che cosa no (selection.assessment)?
Rispondi con lo schema: defects (anche vuoto), verified_facts (con i numeri del giocattolo), cannot_conclude, recommendations.`,
  },
  {
    key: 'numbers_repr', schema: AUDIT_SCHEMA,
    task: `IL TUO COMPITO: verifica indipendente dei numeri della diagnosi di GPT e della prova di rappresentabilita'. Cartella note: ${D}/review/numbers_repr/
(1) Ricalcola DAI JSON GREZZI (accumulatori dynamic_sse, dynamic_target_ss, mse_sum, window_count; mai dai valori gia' arrotondati) ogni numero di ${TREE}/docs/reviews/pose_temponet_temporal_diagnosis_20261122.md e dei punti 1-4 di ${TREE}/docs/prompt_claude_tcn_temporal_diagnosis_20261122.md: recuperi 24,17/18,88/21,71%; pooled clean_ls 0,728479, oracolo 0,670659, soglia 0,699569, TCN 0,715926; miglior E_dyn di tutte le curve 0,715826 (a quale LR ed esposizione? recupero 21,88%?); MLP 0,735130 e "+2,61%"; fitting pooled E_dyn 0,685897 e MSE 1,698865 contro esclusi 0,715926 e 1,800145; E_dyn sul bersaglio pulito degli esclusi TCN 0,139236 contro MLP 0,196055; orizzonti 300/150 completi per ogni LR; LR ed esposizione scelti; assenza di optimizer_alarm; sei controlli COMPLETED 0:0 con le durate (1451, 1287, 3710, 1105, 860, 4427 s) e "1,23 GPUh" per la ricerca TCN temporale; source_sha256 9aaf1c8b... e contesto gpu8 nelle ricevute; i tre controlli lineari passano (recuperi esatti contro il 90%). Per ogni numero: confermato / diverso (valore tuo e origine).
(2) Riesegui lo script di GPT ${TREE}/results/infra/pose_temponet_execution_20261121_A1/analysis_20261122/teacher_embedding_diagnostic.py (copialo nella tua cartella e lancialo col comando Python indicato) e confronta con teacher_embedding_result.json. Poi controlla la costruzione: assorbimento di media/scala e del coefficiente 0,4 nell'ultimo strato e nel ramo affine; LayerNorm della testa lasciata all'identita' affine e con lo stesso eps del generatore?; canali in piu' azzerati; dropout spento. Di' con precisione che cosa la prova dimostra e che cosa NON puo' dimostrare. Estendila tu in due modi piccoli: (a) con il dropout attivo (model.train()) quanto vale l'errore atteso della stessa rete: cioe' quei pesi non sono un punto fisso della ricetta di training; (b) il gradiente della loss pulita in quei pesi e' nullo in eval e quanto vale in train.
(3) Confronta questa diagnosi con quella dell'altra chat di GPT (${CTX}/gpt_chat2_diagnosi_D0_D1_8fac3a82.md): elenca le differenze di fatto (numeri diversi per la stessa grandezza, letture a esposizioni diverse, regole di lettura) e di' quali affermazioni di una sono piu' deboli o piu' forti dell'altra.
Rispondi con lo schema: defects = errori di fatto o numeri sbagliati nella diagnosi (anche vuoto), verified_facts, cannot_conclude, recommendations.`,
  },
  {
    key: 'residual_candidate', schema: AUDIT_SCHEMA,
    task: `IL TUO COMPITO: valutare il candidato "residuale" proposto da GPT e cercare un'alternativa piu' piccola se meglio motivata. Cartella note: ${D}/review/residual_candidate/
Il candidato: LayerNorm fissa; base affine/ridge stimata sul solo fitting con alpha scelto nella CV interna; base congelata; ramo temporale (TCN) allenato sul residuo y - base(z); a inferenza si sommano base e correzione. Vincoli posti da GPT: non mescolare altre modifiche (loss, dropout, larghezza, semi); soglie originali 90% e 50%; stessa selezione MSE.
(1) Ragionamento: che cosa cambia rispetto alla testa attuale, che ha gia' un ramo affine diretto allenato insieme? Tieni conto dei numeri archiviati (ricalcolali dai JSON in ${TREE}/results/infra/pose_temponet_execution_20261121_A1/launcher/): sulla scala del cancello il lettore lineare in forma chiusa stimato su y sta a circa -19,1/-16,4% (fold0/fold1), L0 neurale con arresto anticipato a -13,6/-10,8%, MLP a -12,2/-10,7%, la TCN a +24,2/+18,9%; il riferimento clean_ls e' stimato sul bersaglio PULITO. Una base ridge stimata su y rumoroso e poi congelata parte quindi sotto zero sulla scala del cancello: quanto dovrebbe recuperare il ramo temporale sul residuo per arrivare al 50%? Il residuo contiene a*T + sigma*eps + errore della base: e' piu' facile o piu' difficile da imparare del bersaglio intero? La base congelata toglie alla TCN la possibilita' di correggere l'errore lineare della base?
(2) Giocattolo CPU con testa, loss e ricetta VERE su latenti sintetici (riusa gli script di ieri in ${EVID}/revisori_note_e_script/; un fold o due, LR 0,003, 40-60 epoche, 2-3 semi se il tempo basta). Confronta sul bersaglio rumoroso, all'epoca scelta sulla MSE rumorosa degli esclusi, il recupero sulla scala del cancello di: (a) ricetta attuale; (b) candidato residuale di GPT (ridge con alpha da CV interna, base congelata, LN fissa, TCN sul residuo); (c) le due modifiche separate: solo LayerNorm fissa; solo base ridge congelata; (d) una o due alternative a modifica singola che ritieni meglio motivate (per esempio weight decay molto piu' forte, oppure dropout piu' alto, oppure la stessa testa senza altre modifiche ma con piu' finestre: quest'ultima solo come ragionamento). Riporta medie e dispersione fra semi; di' onestamente se le differenze superano il rumore fra semi.
(3) Conseguenze di protocollo: se il candidato diventasse la ricetta, che cosa andrebbe riqualificato (controlli lineari e temporali di quali teste), quali comparatori (L0, MLP) cambiano significato, quali ricevute di A1 restano riusabili e quali no (leggi ${TREE}/docs/pose_temponet_frozen_protocol_20261117.md e ${TREE}/scripts/pose_temponet/{provenance,execution,budget_amendment}.py).
(4) Verdetto: il candidato e' giustificato dai dati disponibili? condizionato a quale esito dei diagnostici? Esiste una modifica singola piu' piccola e meglio motivata? Che cosa dovrebbe mostrare il controllo sul bersaglio pulito per giustificare l'una o l'altra?
Il giocattolo e' evidenza indicativa (latenti sintetici, non quelli veri): dichiaralo in ogni conclusione.
Rispondi con lo schema: defects = obiezioni concrete al candidato (severity secondo quanto pesano), verified_facts (numeri del giocattolo), cannot_conclude, recommendations.`,
  },
  {
    key: 'ops_costs', schema: MAP_SCHEMA,
    task: `IL TUO COMPITO: mappa operativa per lanciare una piccola catena diagnostica NUOVA e separata (inferenza sui checkpoint esistenti; poi due fit della TCN sul bersaglio pulito), piu' costi e identita'. Cartella note: ${D}/review/ops_costs/ ; scrivi il documento principale in ${D}/review/ops_costs/OPS_MAP.md
Vincoli del progetto (da ${TREE}/CLAUDE.md): i job si inviano solo con scripts/submit_job.py dal Mac (prenotazione sotto lock, ID logico, ledger comune ~/wearusfm_local/jobs/ledger.tsv); il codice gira da un checkout di campagna congelato creato da un job CPU su lrd_all_serial con scripts/slurm/campaign_checkout.sbatch a partire da un riferimento deploy/<nome> pubblicato sul remote "leonardo"; gli output vanno fuori dal repo, sotto $WORK/wearusfm_runs/; i nodi di calcolo non hanno rete; niente editing remoto.
Ricostruisci dai file (docs/pose_temponet_execution_A1_20261121.md, docs/pose_temponet_execution_20261120.md, docs/pose_temponet_budget_A1_handoff_20261121.md, scripts/pose_temponet/{execution,runner,provenance,resources,__main__}.py, scripts/slurm/*.sbatch, scripts/slurm/lib_code_dir.sh, scripts/submit_job.py, e i JSON/log in results/infra/pose_temponet_execution_20261121_A1/launcher/ e results/infra/pose_temponet_execution_20261120/):
(1) La sequenza ESATTA che l'operatore ha usato per A1: push del riferimento di deploy, job di checkout (comando submit_job.py completo, variabili, percorso del checkout risultante, SHA), poi i comandi di invio dei sei controlli (riga submit_job.py completa con sbatch, --export, risorse: partizione, QoS, GPU, CPU, memoria, walltime, account). Copia i comandi veri dai file, non inventarli.
(2) Che cosa fa il launcher pose_temponet.sbatch riga per riga (moduli, venv, PYTHONPATH, CODE_DIR, verifiche di SHA e pulizia), quali variabili d'ambiente pretende, come viene scelto il comando Python eseguito. Si puo' riusare lo stesso launcher per eseguire un modulo Python NUOVO (fuori da scripts/pose_temponet/, per non cambiare il source_sha256 del codice congelato) oppure serve un launcher nuovo? Quali percorsi entrano nel manifest dei sorgenti (provenance.py): un nuovo pacchetto in scripts/<altro nome>/ e un nuovo scripts/slurm/<altro nome>.sbatch lo cambierebbero? E un nuovo file scripts/*.py al primo livello?
(3) Il contesto numerico "gpu8" e i controlli che il runtime congelato fa all'avvio (numerics, thread, variabili): che cosa deve riprodurre un driver esterno per avere predizioni identiche bit per bit a quelle del job 65035631.
(4) Dove stanno su Leonardo gli artefatti che i diagnostici devono leggere (percorsi completi di calibration.pt, selection_states.pt, cache dei latenti del train, e quali stanno su SCRATCH con scadenza) e i loro hash/byte dalle ricevute.
(5) Costi: dalle durate misurate (ricevute e sacct nei JSON) stima tempo e GPU-ore di: inferenza su 20 stati TCN + 4 stati L0/MLP con punteggi su fitting ed esclusi contro y e y*; un controllo di parita' di un'epoca; due fit da 150 epoche a LR 0,003 con valutazioni aggiuntive. Indica QoS adatta (boost_qos_dbg: 30 minuti, massimo 2 job) e walltime prudente. Manutenzione di tutto Leonardo dal 23/11 08:00 al 25/11 08:00: un job deve poter finire prima.
(6) Identita' e ricevute: che cosa di A1 si puo' riusare in sola lettura (calibrazioni, stati, cache, checkout?) e che cosa richiede identita' nuova.
(7) Formato del ledger e della CLI di submit_job.py (argomenti obbligatori, come si esprime un ID logico, dipendenze afterok, che cosa succede a risposta ambigua).
Nel campo defects metti solo eventuali problemi operativi concreti che hai trovato nei documenti o negli script (anche vuoto).`,
  },
  {
    key: 'driver_map', schema: MAP_SCHEMA,
    task: `IL TUO COMPITO: mappa del codice per scrivere un driver diagnostico ESTERNO al pacchetto congelato. Cartella note: ${D}/review/driver_map/ ; scrivi il documento principale in ${D}/review/driver_map/DRIVER_MAP.md
Il driver dovra' fare due cose importando i moduli congelati senza modificarli:
D0 (sola inferenza): leggere calibration.pt dei due fold e selection_states.pt del controllo temporale (TCN_post, L0, MLP); ricostruire feature, y, y*, maschere, supporto, soglie e partizioni; ricaricare ogni stato salvato (per la TCN: 5 learning rate x stato migliore/finale x 2 fold = 20 stati) in una testa nuova; calcolare MSE ed E_dyn (con numeratori e denominatori) contro y e contro y*, su fitting ed esclusi, per fold e pooled; verificare la parita' con i numeri archiviati (selection_temporal_*.json: punto scelto per fold e pooled, punti delle curve pooled); estrarre le loss di training per epoca.
D1 (due fit): stessa testa TCN_post, stesso seme 0, stessa inizializzazione, stesso ordine dei batch e stesso RNG del dropout, stessa ricetta (Recipe()), un solo LR 0,003, 150 epoche fisse, MA con bersaglio di TRAINING y* (pulito) e bersaglio di SELEZIONE/valutazione y (rumoroso) come nell'originale; prima, un cancello di parita': lo stesso driver con bersaglio di training y deve riprodurre i primi punti archiviati della curva del LR 0,003 (esposizioni 0, 0,25, 0,5, 0,75, 1 con update [0,0] [8,8] [15,15] [22,23] [29,30]).
Produci una mappa precisa, con file:riga, firme e frammenti essenziali:
(1) Il percorso completo del job dei controlli per world=temporal, head=TCN_post: da __main__.py/runner.py/workflow.py fino a train.py: quali funzioni, con quali argomenti; come si leggono gli input verificati; come si costruiscono dati/bundle (chiavi del dizionario data: features, target, mask, std, meta, key_time_valid, pose_var...), partizioni fitting/esclusi interni ed esterni, soglia dinamica; dove entra il bersaglio nella loss e dove nella valutazione (sono lo stesso campo? come separarli dall'esterno?).
(2) Le classi di train.py (Trajectory o equivalenti, la coppia sincronizzata dei due fold, l'evaluator): costruttori, metodi (train_to_step, evaluate, state/load_state), RNG (quali generatori, CPU o CUDA, come sono seminati per fold), configure_numerics; come eseguire UN solo LR a orizzonte fisso usando i pezzi esistenti; come salvare lo stato a esposizioni fissate (31, 150, punto scelto).
(3) Lo schema ESATTO di cio' che il codice scrive in selection_states.pt e calibration.pt (chiavi annidate, tipi, che cosa c'e' in best_states / stato corrente / epoch_losses / contatori / RNG) e in selection.json; e come ricaricare uno stato in una testa per la sola inferenza.
(4) Le funzioni di punteggio da riusare per y e y* (metrics.selection_statistics, controls._metric_panel, evaluate_outer_references) e come ottenere dal calibration.pt i componenti L, T, eps, la normalizzazione, a e sigma; se y e y* si ricostruiscono bit per bit.
(5) Insidie: differenze fra torch 2.2 (cluster) e torch 2.14 (Mac) note nel progetto (torch.load(mmap=True) vuole str e non Path; niente uuid nelle proprieta' CUDA; weights_only), dispositivo/dtype, determinismo, dimensione dei blocchi di valutazione (che cambia i bit in FP32), uso di fork_rng, ordine degli import che cambia il contesto numerico.
(6) Quali test CPU esistenti (tests/cpu/test_pose_temponet_{train,controls,workflow,e2e_chain,scientific_integration}.py) costruiscono fixture piccole riusabili per provare il driver in locale con la ricetta di test (_TestRecipe), e come.
Verifica le parti critiche eseguendo codice su fixture piccole (per esempio: costruisci un controllo sintetico minuscolo con le funzioni dei test, allena 2-3 epoche con la ricetta di test, salva e ricarica uno stato, confronta le predizioni bit per bit).
Nel campo defects metti solo difetti concreti del codice congelato che incontri (anche vuoto).`,
  },
]

function seriousDefects(result) {
  if (!result || !Array.isArray(result.defects)) return []
  return result.defects.filter(d => d && (d.severity === 'bloccante' || d.severity === 'importante'))
}

const results = await pipeline(
  AGENTS,
  (a) => agent(PRE + '\n' + a.task, { label: 'audit:' + a.key, phase: 'Audit', schema: a.schema }),
  async (res, a) => {
    const serious = seriousDefects(res).slice(0, 3)
    if (res && seriousDefects(res).length > 3) log('audit:' + a.key + ': ' + (seriousDefects(res).length - 3) + ' difetti seri oltre il terzo non passati allo scettico')
    if (!serious.length) return { key: a.key, result: res, verdicts: [] }
    const verdicts = await parallel(serious.map((d) => () => agent(
      PRE + `\nIL TUO COMPITO: sei uno SCETTICO. Un altro revisore (lente "${a.key}") dichiara il difetto qui sotto. Prova a CONFUTARLO: rileggi il codice citato, riesegui o ricostruisci la prova in modo indipendente, cerca la spiegazione piu' semplice per cui il difetto non esiste o non conta. Se non riesci a confutarlo con una prova, dillo e indica la gravita' che confermi. Se la prova dell'altro revisore non e' riproducibile, il difetto resta "sospetto". Cartella note: ${D}/review/verify/ (file ${a.key}_${d.id}.md).\n\nDIFETTO DICHIARATO\n` + JSON.stringify(d, null, 2),
      { label: 'scettico:' + a.key + ':' + d.id, phase: 'Verifica', schema: VERDICT_SCHEMA }
    )))
    return { key: a.key, result: res, verdicts: verdicts.filter(Boolean) }
  }
)

const out = {}
for (let i = 0; i < AGENTS.length; i++) {
  const r = results[i]
  out[AGENTS[i].key] = r ? r : { key: AGENTS[i].key, result: null, verdicts: [] }
}
return out
