export const meta = {
  name: 'rilettura-bozza-ripresa-infonce',
  description: 'Due verificatori indipendenti rileggono la bozza della review sulla ripresa InfoNCE contro le fonti',
  phases: [{ title: 'Rilettura', detail: 'numeri e riferimenti; conclusioni ed eccessi' }],
}

const BASE = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_resume'
const CLONE = BASE + '/clone'
const DRAFT = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/reviews/claude_review_contrastive_targets_resume_science_20261121.md'
const EVID = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/reviews/claude_review_contrastive_targets_resume_science_20261121_evidence'

const COMMON = `Sei un verificatore indipendente. Devi rileggere contro le fonti la BOZZA di una review scientifica scritta da Claude (revisore) per il progetto WearUsFM, linea 3. Non devi migliorare lo stile: devi trovare errori di fatto, numeri sbagliati, riferimenti sbagliati, citazioni non letterali e affermazioni piu' forti delle prove. Vincoli assoluti: NON modificare la bozza ne' alcun file del repository o del clone; niente ssh, niente rete, niente git commit o push. Scrivi solo nella tua cartella di lavoro.

Percorsi.
- Bozza da verificare (sola lettura): ${DRAFT}
- Cartella delle prove della bozza (script di Claude con le loro uscite .out.txt, rapporti e script dei sei analisti, lettura in sola lettura di Leonardo del 21/11 in leonardo/sola_lettura_20261121.out.txt, sintesi_workflow.json): ${EVID}
- Clone del ramo di GPT codex/contrastive-target-windows-20261118 al commit 918245b6 (sola lettura): ${CLONE} . Contiene il codice (src/wearusfm/, scripts/), la lettura di GPT docs/reviews/contrastive_targets_resume_science_20261121.md, il prompt di GPT docs/prompt_claude_contrastive_targets_resume_science_20261121.md, la proposta firmata docs/reviews/contrastive_common_diagnostics_science_20261119.md, docs/decisioni.md e le evidenze sotto results/infra/contrastive_targets_20261118/ (benchmark_execution_20261121/{C0,CN,CR}, reloads_execution_20261121/{CN_A,CN_B,CR_A,CR_B}, readout_execution_20261121): metrics.jsonl per update, trace-<rank>.jsonl, trace-<rank>.ddp.jsonl, config.json, allocation.sacct.tsv.
- Rapporto originale decompresso (sha256 d4300a2e31c85b39f1f36b8f3146886811d70fe9e24c2d0a9d07e683e84cd4b8): ${BASE}/data/preflight_readout.json . I campi principali stanno in arms.<CN|CR>.resume.pairs.<continuous__reload_a|continuous__reload_b|reload_a__reload_b>.
- La tua cartella di lavoro (creala): ${BASE}/verifica/<chiave> . Il tuo rapporto: ${BASE}/verifica/<chiave>.md (aggiornalo man mano).

Strumenti. Python: ~/.venvs/wearusfm-tok/bin/python (torch 2.14 CPU). Per importare il codice del clone: env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${CLONE}/src:${CLONE}/scripts ~/.venvs/wearusfm-tok/bin/python -B <script>. Shell zsh: per elenchi di file usa bash -c; scrivi \${VAR} quando una variabile e' seguita da due punti. Niente run lunghi: solo letture e conti di pochi secondi. I checkpoint .pt non sono sul Mac.

Regole di verifica. Ricalcola tu i numeri dalle fonti primarie (JSON, metrics.jsonl, tracce, codice): non fidarti dei rapporti degli analisti, che sono fonti secondarie, salvo per esperimenti che non puoi rifare in poco tempo (in quel caso controlla che la bozza riporti fedelmente quel rapporto e segnala che la fonte e' secondaria). Per ogni problema indica: la frase o la cella della bozza (citazione breve e sezione), che cosa non torna, il valore o la formulazione corretti con la fonte, e la gravita': "errore" (fatto o numero sbagliato), "imprecisione" (arrotondamento, riferimento di riga spostato, formulazione ambigua), "eccesso" (conclusione piu' forte delle prove, o deduzione presentata come fatto). Segnala anche le affermazioni che NON riesci a verificare con le fonti disponibili. Se un punto e' corretto non elencarlo; alla fine di' quante affermazioni hai controllato in tutto.

`

const TASKS = [
  { key: 'v1-numeri-riferimenti', prompt: `Compito V1 (chiave: v1-numeri-riferimenti) - Numeri, tabelle, riferimenti al codice e citazioni.
Controlla una per una, in tutta la bozza compreso il "Prompt per GPT" finale:
1. ogni numero e ogni cella di tabella: percentuali dei momenti, mediane, percentili, massimi, conteggi (270, 266, 85, 5.965, 4.979, 6.520, 28.559.133, 197.120...), valori della norma del gradiente agli update 545-548, medie geometriche 548-561 e 562-576, rapporti ("da 2,2 a 2,9 volte la mediana", "1,6 volte", "1,05 volte", "3-7 volte", "0,02-0,04 decadi per update"), chiavi della coda per update di inserimento, pesi per blocco dal 544, frazioni 96,6% e 3,1%, coseno 0,99991, job e nodi, secondi e GPU-ore e core-ore (1,11-1,17; 0,23-0,29; 0,67; 8,9-9,3; 1,8-2,4), consumi 4,74 e 47,33, tetti 18 e 169,6, 17 allocazioni;
2. ogni riferimento a file e righe del codice (run.py:1387-1390, run.py:1429, contrastive_benchmark_readout.py:114-127, contrastive_objectives.py:415-439, contrastive_target_resume.py:189-201, righe 1396 e 1413 del torch del cluster nella lettura di Leonardo) e ogni descrizione di che cosa fa quel codice;
3. le citazioni letterali: la frase della regola firmata ("Un eccesso ripetibile continuo/reload, con reload concordi, richiede localizzazione prima di promuovere il profilo"), la domanda di GPT, i nomi dei campi del JSON e delle metriche;
4. la coerenza interna dei numeri ripetuti in sezioni diverse e nel prompt finale.` },
  { key: 'v2-conclusioni-eccessi', prompt: `Compito V2 (chiave: v2-conclusioni-eccessi) - Conclusioni, deduzioni e possibili eccessi.
Per ogni affermazione conclusiva o interpretativa della bozza (sezioni Verdetto, 2.1-2.7, 3, Raccomandazione per Simone, Limiti e il Prompt per GPT finale) stabilisci se le prove citate la sostengono davvero, se e' presentata come fatto una deduzione, e se manca un limite che andrebbe dichiarato. In particolare metti alla prova:
1. "Lo stato salvato al passo 544 e' ripristinato esattamente" e "esclusa dagli artefatti" per il difetto di ripristino: che cosa coprono davvero le impronte di confine (leggi scripts/contrastive_target_resume.py e come compare_three_paths le confronta) e che cosa resta fuori;
2. "Gli artefatti archiviati localizzano gia' la prima differenza nella riduzione DDP dell'update 545": le norme contrast_global_grad_* identiche al 545 bastano a dire che i gradienti locali coincidono? Per quali parametri? L'all_reduce esplicito ha davvero un layout indipendente dai bucket DDP?
3. "Il raggruppamento diverso riguarda solo il 545": la catena (ordine in _pre_forward nel torch del cluster, punto di campionamento della traccia, un forward per update, norme identiche al 546 in CN) e' completa? C'e' un'altra lettura compatibile con i dati?
4. "Il nodo non e' la causa": l'identita' di A e B su nodi diversi basta?
5. le spiegazioni sul bf16 (salto quantizzato, scala 2^-8, "il livello dello scarto non dipende dall'ampiezza del seme") e sul denominatore (96,6% successivo al fork): sono dedotte o verificate? Sono formulate con la cautela giusta?
6. la valutazione della prova GPU di GPT e dell'intervento con riscaldamento DDP: le previsioni e i criteri sono coerenti con i dati e con il codice? Il "circa un quarto del costo" e' corretto? L'affermazione che una prova a un lato non puo' dare identita' per caso regge?
7. la raccomandazione: segue dalle prove? Rispetta i ruoli (scienza a GPT, firme a Simone, verdetto storico FAILED_REQUIRES_REVIEW invariato, regole nuove dichiarate post hoc)? Il riferimento alla regola firmata e' usato correttamente?
8. le sei "precisazioni sulla lettura di GPT": sono fedeli a cio' che GPT ha scritto davvero nella sua lettura e nel suo prompt? Cita le frasi di GPT.
Cerca anche omissioni rilevanti: un dato delle fonti che contraddice o indebolisce una conclusione e che la bozza non riporta.` },
]

const SCHEMA = {
  type: 'object',
  properties: {
    sintesi: { type: 'string' },
    affermazioni_controllate: { type: 'integer' },
    problemi: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          sezione: { type: 'string' },
          citazione: { type: 'string' },
          problema: { type: 'string' },
          correzione: { type: 'string' },
          fonte: { type: 'string' },
          gravita: { type: 'string', enum: ['errore', 'imprecisione', 'eccesso'] },
        },
        required: ['sezione', 'citazione', 'problema', 'correzione', 'gravita'],
      },
    },
    non_verificabili: { type: 'array', items: { type: 'string' } },
    rapporto: { type: 'string' },
  },
  required: ['sintesi', 'problemi', 'rapporto'],
}

phase('Rilettura')
const results = await parallel(TASKS.map(t => () =>
  agent(COMMON + t.prompt, { label: t.key, phase: 'Rilettura', schema: SCHEMA, model: 'opus' })))
return TASKS.map((t, i) => ({ key: t.key, result: results[i] }))
