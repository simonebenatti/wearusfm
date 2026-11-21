export const meta = {
  name: 'tcn-temporal-diagnosis-review',
  description: 'Two independent reviewers of GPT temporal-control diagnosis and the proposed D0/D1 follow-up (statistics and design), local JSON and code only',
  phases: [
    { title: 'Review', detail: 'statistical reading and variance budget; design and feasibility of D0/D1' },
    { title: 'Verify', detail: 'one skeptic per blocking or important finding' },
  ],
}

const RD = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/review_diag'
const SCRATCH = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad'
const NOTES = '/Users/simonebenatti/wearusfm_local/tcn_temporal_diag_review'
const L = 'results/infra/pose_temponet_execution_20261121_A1/launcher'
const D = 'results/infra/pose_temponet_temporal_diagnosis_20261121'

const CONTEXT = `
CONTEXT (read carefully).

Project: a frozen-encoder pose readout comparison ("TCN head") for an sEMG foundation model. Readout heads are trained on cached
latents of 40 training users (two outer folds of 20 users, about 920 and 934 windows of 160 patches x 384 features, 20 joints).
Before any real fit, the protocol requires two POSITIVE CONTROLS built on the training latents: synthetic targets with a known
function. The temporal control is: clean target y* = L + a*T, observed y = y* + sigma*eps, where L is an affine 384->20 map, T is
a frozen random temporal network with the candidate's structure and receptive field (13 patches) but all hidden layers of width 8,
eps is stationary AR(1) noise with rho = exp(-1/4), independent per window and joint. Calibration chose a = 0.4, sigma = 1.3 in
both folds. All scores are E_dyn = centred dynamic SSE / centred dynamic target SS on the observed y of the HELD half
(the other 20 training users; never the 15 external development users).
Gate (protocol section 4.4, docs/pose_temponet_frozen_protocol_20261117.md lines 82-106): with E_cleanLS = instantaneous linear
WLS fitted on the CLEAN y* of the fitting half and scored on y, and E_oracle = y* scored on y, the candidate TCN_post (72,904
parameters: LayerNorm, an affine branch 384->20 and a deep branch Conv1x1(384,64), three dilated k3 convs 64->64, Conv1x1(64,32),
Conv1x1(32,20), dropout 0.1, AdamW weight decay 0.01, batch 32, 5 learning rates, 150 to 300 epochs, epoch and LR selected on the
pooled MSE of the two held halves) must reach (E_cleanLS - E_TCN)/(E_cleanLS - E_oracle) >= 0.50 in fold 0, fold 1 and pooled.

RESULT (21 Nov): recovery 24.17% / 18.88% / 21.71% -> status "recipe_not_admitted", chain stopped before the real fits.
Held numbers: fold0 cleanLS 0.710431, oracle 0.653451, TCN 0.696657; fold1 0.749920, 0.691103, 0.738818; pooled 0.728479,
0.670659, 0.715926. Selected LR 0.003, epoch 31; E_dyn at epoch 150 of that LR is 0.741556.
Other archived facts: neural L0 trained on y scores 0.718154 / 0.756284 on held; closed-form WLS fitted on y: L0 fitting
0.705764 -> held 0.721315 (fold0), L13 (linear with the 13-patch context, 4992 features, rank 4980) fitting 0.667239 -> held
0.759182. Fitted on the CLEAN y* and scored on y on the FITTING half: L0 0.708654, L13 0.693938 (L13 recovers 27.09% of the
fitting margin in fold0 and 19.93% in fold1). TCN at the selected checkpoint on the fitting half: 0.661873 / 0.710583. TCN on the
held half against the CLEAN target: E_dyn 0.127856 / 0.154431 (neural L0: 0.189922 / 0.211195).

GPT (the scientific owner) wrote a diagnosis and a proposed follow-up, to be reviewed. It is checked out, clean and read-only, at
  ${RD}     (commit e7670758, branch codex/pose-temponet-temporal-diagnosis-20261121)
Main document: docs/reviews/pose_temponet_temporal_diagnosis_20261121.md
Evidence and scripts: ${D}/ (optimization/, control/, architecture/, local_validation.json)
Raw archived artifacts of the controls (JSON only, no arrays): ${L}/
  selection_temporal_TCN_post.json, selection_temporal_L0.json, selection_temporal_MLP.json (curves per LR, selected point,
  selected_fitting_diagnostics, selected_outer_statistics, selected_outer_clean_statistics, closed_form_references),
  calibration_0.json, calibration_1.json (outer_reference_statistics.temporal.{constant,clean_ls,oracle},
  outer_reference_statistics.linear_diagnostics.{L0,L13} with clean-fit statistics on the fitting half, calibration.selected),
  positive_controls_gate_20261121.json.
Code (frozen runtime, same tree): scripts/pose_temponet/{controls,train,workflow,runner,metrics,heads,linear}.py.
The reviewer already re-ran GPT's three scripts: outputs are byte-identical to the committed ones. Do not spend time on that.

GPT's diagnosis in short: the TCN learns part of the signal; the gap between fitting and held and the late degradation suggest
estimation difficulty under noise and insufficient generalisation; a constructive forward-only proof shows the candidate can
represent the generator; the calibration keeps about 8% oracle margin but does not certify that half of it is recoverable with
this sample and recipe.
GPT's proposal: D0 = re-score the selected (epoch 31) and final (epoch 150) checkpoints of LR 0.003 against y and y*, on fitting
and held, reproducing the published scores first; D1 = two diagnostic fits (one per fold), same TCN, init, batches, dropout, weight
decay, LR 0.003, 150 fixed epochs, the ONLY change being the training target y* instead of y; epoch still selected on the pooled
MSE against the original y; clean curves descriptive; no new grid, no automatic extension, no external users. A success on y*
does not admit the noisy recipe. If clean training generalises, GPT's priority is ONE capacity/regularisation variant, frozen
once, then the original control repeated; if it struggles already on fitting, investigate optimisation.

HARD RULES:
- No cluster access, no job, no push, no download. Everything is local JSON, code and (optionally) small CPU simulations.
- Do NOT modify anything under ${RD}. For experiments create your own directory ${SCRATCH}/wf3_<yourname>_work (at most 1 GB;
  delete it with its full literal path when done; leave no process running).
- Python: cd ${RD} && PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ~/.venvs/wearusfm-tok/bin/python <script>
  (one thread always; the environment aborts otherwise). The Mac has 18 GB RAM and little disk.
- DURABLE NOTES, MANDATORY: after EACH numbered task append what you verified and found (command and decisive numbers) to your
  notes file, BEFORE the next task. Create it at the very beginning.
- Report only what you verified. Exact numbers, with the JSON path or file:line they come from. A suspicion is severity "note".
- Severity: "blocking" = the proposal should not go ahead as written, or a claim of the diagnosis is wrong in a way that changes
  the conclusion; "important" = a real gap or error to fix in the proposal or diagnosis; "minor" = worth fixing; "note" = observation.
- You are asked to DISAGREE where the evidence allows it: this is a scientific review. Do not confirm by default.
- Aim to finish in about 20 minutes. First tasks matter most.
`

const FINDINGS = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'four to eight sentences: what you checked and your overall conclusion' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['blocking', 'important', 'minor', 'note'] },
          title: { type: 'string' },
          location: { type: 'string' },
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'numbers with their JSON path, commands and outputs, or quoted code/text' },
          suggested_fix: { type: 'string' },
        },
        required: ['id', 'severity', 'title', 'location', 'claim', 'evidence', 'suggested_fix'],
      },
    },
    verified_ok: {
      type: 'array',
      items: { type: 'object', properties: { what: { type: 'string' }, evidence: { type: 'string' } }, required: ['what', 'evidence'] },
    },
    key_numbers: { type: 'string', description: 'the main quantities you computed, one per line, with units and source' },
    tasks_not_completed: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'findings', 'verified_ok', 'key_numbers', 'tasks_not_completed'],
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    corrected_severity: { type: 'string', enum: ['blocking', 'important', 'minor', 'note', 'not_a_defect'] },
    reason: { type: 'string' },
    evidence: { type: 'string' },
  },
  required: ['refuted', 'corrected_severity', 'reason', 'evidence'],
}

const LENSES = [
  {
    key: 'stat',
    prompt: `You are reviewer STAT. Your notes file: ${NOTES}/stat_notes.md
Tasks, in order of importance:

1. DENOMINATORS AND SELECTION. From the protocol text and the code (workflow.qualify_controls, controls, metrics) and the raw
   JSON, state exactly what the gate compares. Recompute the three recoveries from the raw statistics (dynamic_sse and
   dynamic_target_ss), not from rounded values. Then answer with numbers: where does a head trained on the NOISY y sit on this
   scale when it has no temporal ability at all (neural L0, neural MLP, closed-form L0 fitted on y)? In other words, how many
   points of "recovery" does the clean-fit baseline cost a noisy-trained head before any temporal learning? Is this asymmetry
   stated in the protocol, and does GPT's diagnosis report it? Also check the claim that the best checkpoint of the whole search
   recovers only 21.88% (which checkpoints exist, pooled only or per fold).
2. THE VARIANCE BUDGET. Try to REFUTE the following claim of the reviewer, using only the archived numbers and sound statistics:
   "At this sample size and noise level the estimation-variance cost on held users of a 72,904-parameter head trained on noisy y
   by plain empirical risk minimisation is of the same order as the whole cleanLS->oracle margin (about 0.057 E_dyn), so the 50%
   bar cannot be met without strong effective regularisation; a head of the generator's size (hidden width 8) would lose roughly
   15-30% of the margin and could plausibly pass. The observed 24%/19% is therefore compatible with a pure sample-size effect."
   Useful archived quantities: held excess of the closed-form L0 fitted on y over the L0 fitted on y* (385 parameters per joint),
   the same for L13 where the clean-fit held value is NOT archived (say what you assume), the fitting/held gap of the TCN, the
   decomposition of the held noisy SSE into clean SSE plus noise SS (selected_outer_clean_statistics vs selected_outer_statistics
   and the oracle), parameters shared across the 20 joints versus per-joint parameters, AR(1) effective sample size. Give a
   defensible range, list every assumption, and say clearly what CANNOT be concluded from these numbers. If the claim is wrong
   or overstated, say how and by how much.
3. WHAT D1 CAN AND CANNOT SEPARATE. Under each hypothesis (optimisation failure; poor cross-user generalisation of the function
   even without noise; noise-driven estimation variance that a smaller or better regularised head could fix; noise-driven
   variance that NO head trained with this recipe can fix at this sample size, i.e. a 50% bar unattainable by construction), what
   outcome of D0 and D1 do you predict, with rough numbers? Which pairs of hypotheses give the SAME D0/D1 outcome? What is the
   cheapest additional measurement that separates them (for example: a head of the generator's own family, hidden width 8,
   trained on the same noisy y with the same recipe; the same family started from the true weights and read at the end of
   training; a closed-form readout fitted on y from the generator's true hidden features)? Judge each candidate for validity
   and for the risk of turning into "searching until the control passes".
4. OPTIONAL, only if it fits in 12 minutes of compute: a small CPU simulation with synthetic latents (state how they were
   generated), the REAL generator (controls) and the REAL heads and loss, to see whether the ordering "big head under noise
   recovers little, small well-specified head recovers much more, big head on clean target recovers most" appears at a comparable
   ratio of parameters to samples and noise. Label it clearly as a toy and report the regime you could actually run. Skip it
   rather than produce an unconvincing toy.`,
  },
  {
    key: 'design',
    prompt: `You are reviewer DESIGN. Your notes file: ${NOTES}/design_notes.md
Tasks, in order of importance:

1. FEASIBILITY OF D0 FROM THE CODE. Which states of the temporal TCN selection exist in the saved artifact (train.py
   candidate_states with best_states and current_states per learning rate; workflow.py and runner.py for what is written to
   disk and under which names), for which learning rates and folds, with which update counters? Can y, y*, the components,
   masks, crop, dynamic threshold and the fitting/held partition be re-derived exactly from what the WLS calibration stage saved
   (calibration.pt "components", controls.synthetic_target, workflow.prepare_control_folds)? Which existing functions score a
   state against the clean and the noisy target (how were selected_outer_clean_statistics produced)? What NEW code would D0
   need, and is there anything that prevents reproducing the published scores first? Does D0 need the cluster (where do the .pt
   files live) and therefore a job, an ID and a budget line?
2. D1 DESIGN AUDIT. What exactly must change to train on y* while selecting the epoch on the pooled MSE against y (where is the
   training target bound, where the selection target)? Is "same initialisation, same batches and RNG" achievable when only the
   target changes? Is selecting the epoch on noisy held MSE sensible when training on the clean target? Which recovery
   denominator does D1 use, and is a clean-trained TCN now compared like for like with the clean-fit linear baseline? List the
   confounds GPT's outcome table does not handle (learning rate tuned under noise, single seed, dropout and weight decay tuned
   for the noisy case, the held half used both for selection and for the verdict), and say for each row of the table what can
   and cannot be concluded. Which row is the a-priori most likely outcome, and is the experiment still worth running if so?
3. PROTOCOL COMPLIANCE. Section 4.4 of the protocol says a failed required condition stops before the real fits, forbids
   "searching until it wins" and requires a registered decision for any change of the specification. Read docs/decisioni.md and
   docs/claude_GPT_shared in ${RD} for what is registered about the failed control and about this proposal. Do D0 and D1 need a
   registered decision, an authorisation by the user and a budget before any launch? Does anything in the proposal amount to a
   post-hoc change of the gate, of the selection rule or of the control? Is the proposed "one variant, frozen once, then the
   original control repeated on the same synthetic world" protected against circularity (the same noise seeds and folds used to
   choose and to confirm)?
4. THE CONSTRUCTIVE PROOF. Read ${D}/architecture/oracle_embedding.py: does it embed the generator's weights into the UNMODIFIED
   candidate head (heads.make_head) and compare with the output of controls.generate_components on the central crop? Any
   shortcut that weakens the claim "the candidate can represent the clean target at inference" (normalisation handled outside
   the head, LayerNorm bypassed, dropout, different scaling of L and T)? What does it NOT show?
5. Any blocking issue in the proposal as written.`,
  },
]

const results = await pipeline(
  LENSES,
  lens => agent(CONTEXT + '\n\n' + lens.prompt, { label: `review3:${lens.key}`, phase: 'Review', schema: FINDINGS }),
  (review, lens) => {
    if (!review) return { lens: lens.key, review: null, verdicts: [] }
    const serious = review.findings.filter(f => f.severity === 'blocking' || f.severity === 'important')
    const picked = serious.slice(0, 2)
    if (serious.length > picked.length) log(`${lens.key}: ${serious.length - picked.length} serious finding(s) not sent to a verifier (cap 2 per lens)`)
    return parallel(picked.map(f => () =>
      agent(CONTEXT + `\n\nYou are an independent SKEPTIC. Your notes file: ${NOTES}/skeptic_${lens.key}_${f.id}_notes.md
Another reviewer (lens "${lens.key}") reported the finding below. Try to REFUTE it from the JSON, the code and the documents:
recompute the numbers yourself, read what it cites. If you cannot reproduce it, it is refuted. Decide its true severity.
Aim to finish in about 10 minutes.

FINDING ${f.id} (${f.severity}): ${f.title}
Location: ${f.location}
Claim: ${f.claim}
Evidence given: ${f.evidence}
Suggested fix: ${f.suggested_fix}`, { label: `verify3:${lens.key}:${f.id}`, phase: 'Verify', schema: VERDICT })
        .then(v => ({ id: f.id, verdict: v }))
    )).then(verdicts => ({ lens: lens.key, review, verdicts: verdicts.filter(Boolean) }))
  }
)

return { results: results.filter(Boolean) }
