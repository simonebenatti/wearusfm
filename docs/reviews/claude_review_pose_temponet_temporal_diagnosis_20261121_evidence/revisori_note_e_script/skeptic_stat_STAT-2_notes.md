# Skeptic notes — finding STAT-2 (lens "stat")

Started: 2026-11-21 19:21 CET. Read-only on review_diag (commit e7670758). Work dir (if needed):
/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_skeptic_stat2_work

Finding under test: "D0+D1 cannot separate 'a smaller or regularised head would pass' from 'no random-init head passes
with this recipe at this sample size'" (important). Cited: diagnosis doc line 80 (row 1), line 89 (section 6), line 60.

## Task log

### Task 1 - read what the finding cites (done 19:24)
- Diagnosis doc `docs/reviews/pose_temponet_temporal_diagnosis_20261121.md` (review_diag, HEAD e7670758, `git status --short` empty):
  - line 80 (row 1) literal: "La funzione e' apprendibile con questi utenti se si rimuove il rumore del training. Priorita' a controllo della
    complessita'/regolarizzazione nella ricetta su y. Non e' prova di ottimo ne' ammissione." -> finding quotes it correctly ("priority", not a claim of success).
  - line 89 (sec. 6) literal: "Se D1 separa chiaramente il caso pulito dal rumoroso, preferisco una TCN piu' parsimoniosa o una regolarizzazione diversa,
    scelta e congelata una sola volta prima di ripetere il controllo originale. Non farei una ricerca aperta di varianti fino al passaggio." -> correct.
    Same line: "La dimensione del generatore non va copiata per costruire una testa specializzata a superare il test" (relevant to the suggested arm (a)).
  - line 60 literal: "non garantisce che meta' del margine dell'oracolo sia stimabile con circa 920/934 finestre di 20 utenti per fit ... non e' dimostrato
    che il campione finito ne sia la causa." -> correct: GPT itself names the gap; D0 (line 66) and D1 (lines 72-76) both use the SAME 72,904-parameter head,
    D1 changes only the training target (y* instead of y). No arm trains anything on noisy y other than the already-failed candidate.
  - decision table lines 78-83: four rows = (clean generalises + recovery>=50) / (clean fit >> held) / (clean fit poor or still descending) / (discordant).
    No row and no reading whose outcome is "the 50% bar is not attainable under noisy training at this n". Row 2 mentions "adeguatezza del controllo" only
    for the case where CLEAN training does not generalise.
  - No path declared after a failed frozen-once variant (grep of the doc: lines 89-93 only say new recipe, receipts, thresholds unchanged).
  - line 105: GPT explicitly asks the reviewer to "verificare che D0 e i due D1 separino davvero i casi utili" and to "valutare la priorita' a
    parsimonia/regolarizzazione rispetto a warm start o altre epoche" -> the finding answers a question GPT asked (in scope).
- Protocol `docs/pose_temponet_frozen_protocol_20261117.md` sec. 4.4: line 88 "[l'oracolo] Non e' un bound campionario garantito sul noisy held";
  line 100 "Questa banda e il margine pulito possono essere piu' severi della v2: WLS su y* toglie il rumore nel fit"; line 104 "nessuna ... ricerca finche'
  si vince. Eventuale modifica della specifica richiede decisione registrata."; line 106 "Soglie e griglie sono scelte ingegneristiche fissate prima delle misure."
  -> the protocol never establishes that 50% is attainable by a learner trained on noisy y at n ~ 920 windows; the only declared exit is a registered decision.
- Toy logs (stat_work_files/log_*.txt), every figure quoted in the finding found literally:
  wide_noisy SUMMARY "selected(min held MSE) ep 25 rec 0.311", ep 25 "fit ... rec 0.919"; wide_clean ep 28 rec 0.802 (min held MSE at the last epoch);
  s8_noisy ep 94 rec 0.272 (params 12400); s8_clean ep 78 rec 0.442; s8_truth ep 94 rec 0.666 (range ep 20-94: 0.645-0.689);
  wide_truthfree ep 0/3/10/20/23 = 1.000/0.807/0.718/0.635/0.550.
  NB: these are log files of another reviewer; reproducibility checked in Task 3 below.

### Task 2 - real-run numbers quoted by the finding, recomputed from the archived JSON (done 19:27)
Command: inline python over results/infra/pose_temponet_execution_20261121_A1/launcher/{calibration_0,calibration_1,selection_temporal_TCN_post,selection_temporal_L0,selection_temporal_MLP,positive_controls_gate_20261121}.json
(E = dynamic_sse / dynamic_target_ss; rec = (E_cleanLS - E_head)/(E_cleanLS - E_oracle)).
- held: fold0 cleanLS 0.710431 oracle 0.653451 TCN 0.696657 -> 24.1731%; fold1 0.749920 / 0.691103 / 0.738818 -> 18.8755%; pooled 21.7101%
  (= gate JSON /gate/temporal_head_gate/recovery 0.24173052888786836 / 0.1887545039561508 / 0.21710109906955868, status recipe_not_admitted). CONFIRMED ("24/19% held").
- fitting (oracle reconstructed from linear_diagnostics.L13.temporal_margin_recovery_fitting 0.270896 / 0.1993): fold0 cleanLS_L0 0.708654, L13 clean 0.693938,
  oracle_fit 0.654332, TCN fit 0.661873 -> apparent recovery 86.12%; fold1 0.749626 / 0.738389 / 0.693246 / 0.710583 -> 69.25%. CONFIRMED ("86/69% fitting").
- "affine tax" on the gate scale (heads/readers with NO temporal ability, fitted on noisy y, held): closed-form WLS L0 -19.10% / -16.38%; neural L0 -13.55% / -10.82%
  (pooled -12.28%); MLP -12.24% / -10.65% (pooled -11.50% = gate mlp_recovery). Toy closed-form L0 on y: -8.88% (toy_refs.json selected.held.L0noisy_recovery).
  -> in the real folds any head with an affine branch learned on y starts 11-19 points below zero; the real affine cost is about twice the toy's.
- held clean SSE / margin SSE: TCN 77.42% / 80.61%, neural L0 115.0% / 110.2% (selected_outer_clean_statistics vs clean_ls - oracle SSE).
- Nothing in the archive measures a head other than TCN_post (72,904 par.), L0 and MLP on the temporal world: no archived number bears on whether a
  smaller/regularised temporal head reaches 50% on noisy y, nor on the floor of a well-specified family. So the archive does not already settle H3 vs H4.

### Task 3 - reproduction of the toy (first pass done 19:29; endpoint reruns in progress)
Work dir: scratchpad/wf3_skeptic_stat2_work (copy of stat_work_files/toy.py with only the WORK path changed: `diff` shows line 11 only).
- `toy.py gen` (14 s): toy_refs.json IDENTICAL to the reviewer's (all keys except "seconds"): a 0.4, held E_cleanLS 0.8221200470847861, E_oracle 0.7655237078094925,
  gap 6.884%, closed-form L0 on y recovery -8.879%, fit gap 7.460%, T unread 0.9302. Uses the REAL controls.generate_components and REAL heads.make_head("TCN_post").
- First epochs of the six runs re-run (seeds fixed, one thread): every printed value identical to the reviewer's logs:
  s8_noisy / s8_clean / s8_truth epochs 1-10 (e.g. s8_truth ep 10 "held E 0.78190 mse 1.77616 rec 0.711 ... fit E 0.76777 rec 0.827 clean_rec 0.712");
  wide_noisy / wide_clean / wide_truthfree epochs 1-2 (e.g. wide_clean ep 2 "held E 0.84597 mse 2.20088 rec -0.421 ... fit E 0.83604 rec -0.292").
  -> the logs are genuine deterministic outputs of toy.py. Endpoint reruns (wide 28/28/23 epochs, s8_truth 94, 400 s cap each) launched 19:29.
- Fidelity of the toy to the recipe (read in code): heads.py:46-57 TCN_post = Conv1x1(384,64), three k3 64->64 dil 1/2/3, Conv1x1(64,32), Conv1x1(32,20), dropout 0.1
  after transforms 3 and 4; toy `Small` = same structure with all hidden widths 8 (12,400 parameters = 768 LN + 7,700 affine + 3,932 deep), same init rule.
  train.py:233-234 AdamW(optimizer_groups(wd), lr, betas, eps), constant LR (no scheduler in train.py), clip (train.py:291): the toy loop matches.
  Differences that remain: synthetic latents, one fold, one seed, dropout RNG not the runtime's stream.
- gradcheck (reviewer's script, my path): GPT's zero embedding of the truth into TCN_post -> gradient exactly 0.0 on all extra rows AND extra columns of every deep conv
  (conv0..4 extra rows 0.0, conv1..5 extra cols 0.0), true rows non-zero; embedding RMSE vs y* 4.5e-07. So a truth-initialised arm can be run with the PRODUCTION head class
  and GPT's committed embedding: it is the width-8 family in disguise (zero weights with zero gradient stay zero under AdamW and weight decay). No new head class needed for arm (b).
- Extra robustness run of mine (not in the finding): width-8 head at LR 0.01 (the top of the recipe's grid). At epoch 28: noisy y 5.9% (LR 0.003: 13.9%), clean y* 21.5% (LR 0.003: 31.8%).
  -> the small head's failure in the toy is not cured by the higher LR of the grid (final values appended below when the runs stop).
- (19:33) LR 0.01 small-head runs finished (400 s cap): S8 on noisy y: selected ep 58 15.5%, best 17.4% (ep 63), last ep 67 16.2% (LR 0.003 at ep 67: 26.0%);
  S8 on clean y*: 47.2% at ep 66, still rising (LR 0.003 at ep 66: 42.3%). -> "the small head fails on noisy y" is not an artefact of LR 0.003 at the top of the grid.
- (19:33) endpoint reruns still bit-identical where reached: wide_noisy ep 8 rec 0.028, wide_clean ep 8 rec 0.327, wide_truthfree ep 8 rec 0.751, s8_truth ep 30 rec 0.658 / fit 0.876.

### Task 4 - test of the suggested arm (b): is the truth-initialised "floor" noise-driven? (NEW, my run)
- `toy.py run s8_truth_clean S8 clean truth 60 300`: width-8 head initialised at the TRUE weights, trained on the CLEAN y* (same recipe, LR 0.003):
  held recovery ep 0 100% -> ep 1 73.9% -> ep 2 80.4% -> ep 3 82.6% -> ep 5 80.2% -> ep 10 82.2% -> ep 14 82.4% -> ep 20 84.7%;
  on the FITTING half ep 10 rec 0.808 (clean_rec 0.804), ep 20 0.847 (0.843).
  => 15-20 points are lost by the recipe itself with NO noise in the target and NO generalisation gap (fitting = held): not a noise/sample-size effect.
  Compare s8_truth on noisy y at the same epochs: ep 10 71.1%, ep 14 68.3%, ep 20 67.3%: the noise-driven part is about 11-17 points, the rest is recipe-induced.
  => the reading proposed for arm (b) ("below 50% at 150 means the bar is not attainable under this recipe, so the control specification goes to a
  registered decision") is CONFOUNDED: about half of the truth-initialised shortfall is produced by the recipe on the narrow family, not by noise.
  A companion arm (same head from the truth on y*) or a hedged reading is needed. (cause - dropout on 8-wide layers vs Adam - checked next.)
- (19:35) cause isolated with my copy toy_nd.py (only change: env TOY_DROPOUT overrides the Dropout p; `diff toy.py toy_nd.py` = that block + wd env):
  * S8 from the TRUTH, CLEAN y*, dropout 0: held 98.2 / 99.4 / 99.2 / 98.9 / 99.0% at ep 1-5, 98.5% ep 10, 99.5% ep 12 (one dip 92.2% at ep 8); fitting 99.4% at ep 10.
    -> with dropout off the truth is held; the 15-20 points lost in s8_truth_clean come from DROPOUT 0.1 on the two 8-wide layers (not from Adam or weight decay).
  * S8 from the TRUTH, NOISY y, dropout 0: 91.4% ep 1, 88.6% ep 5, 85.0% ep 10, 83.3% ep 12, still declining (final value appended below).
  So in the toy the "truth floor" 66.6% quoted by the finding = noise-driven drift + a dropout-on-narrow-layers bias of comparable size.
  Consequence for the suggested fix: arm (b) with the recipe's dropout cannot be read as "bar unattainable / control specification" when it falls below 50%:
  it must be run with dropout off, or paired with the same arm on y*, and the reading must use the noise-only part.
  Consequence for the toy evidence: "the small head fails" (27.2% on y, 44.2% on y*) is partly a dropout-0.1-on-width-8 effect, i.e. exactly a
  regularisation mis-specification of a narrow head; checked next with S8 from random init, dropout 0 (launched 19:35, 330 s cap).
- (19:37) Endpoint reruns finished (400 s cap each). Line-by-line comparison of the "ep N held ..." lines (run name and timing stripped) with the reviewer's logs:
  wide_noisy first 17 epochs IDENTICAL (ep 17 rec 0.270); wide_clean first 17 IDENTICAL (>= 50% from ep 12: 0.521; ep 16 0.628; ep 17 0.587);
  wide_truthfree first 17 IDENTICAL (ep 17 0.670); s8_truth first 58 IDENTICAL (ep 58 0.660); s8_noisy and s8_clean first 10 IDENTICAL.
  NOT re-run by me: wide epochs 18-28 and small epochs beyond 58 / 10 (time cap); the quoted end values (31.1%, 80.2%, 55.0%, 66.6%, 27.2%, 44.2%) are read
  from the reviewer's logs, whose deterministic prefix I reproduced exactly.
- (19:37) truth-initialised controls, final values:
  | arm (width-8 head from the TRUE weights)      | target | dropout | held recovery                                   |
  | s8_truth (reviewer's arm (b))                 | y      | 0.1     | 66-67% (ep 20-58; reviewer 66.6% at ep 94)      |
  | s8_truth_clean (mine)                         | y*     | 0.1     | 84.7 / 84.9 / 86.9 / 85.4 / 85.8% at ep 20/25/30/35/40; fitting 84.7-86.2% |
  | s8_truth_noisy_nodrop (mine)                  | y      | 0       | 80.4% ep 20, 78.6% ep 25, 79.5% ep 28           |
  | s8_truth_clean_nodrop (mine)                  | y*     | 0       | 98.5-99.8% (ep 15 99.8%, ep 28 99.6%)           |
  -> additive within 1-2 points: noise about 20 points, dropout on the 8-wide layers about 14 points, both 33-34.
  -> about 40% of the shortfall of arm (b) in the toy is NOT noise-driven. Reading "(b) < 50% => bar unattainable, control specification to a registered decision"
     is not safe as written. In the real folds the affine cost alone is 16-19 points (toy 8.9), so a (b) reading near 50% is plausible and the confound would decide the verdict.
- (19:41) S8 from RANDOM init with dropout 0 (toy_nd.py, 330 s cap, 75 epochs): noisy y 24.5% ep 34, 28.0% ep 60, 29.2% ep 75 (with dropout 0.1: 17.5 / 25.0 / 25.7%);
  clean y* 39.1% ep 34, 46.8% ep 60, 50.4% ep 75 (with dropout: 35.9 / 41.6 / 41.8%).
  -> removing dropout helps the narrow head by 3.5 points on y and 8.6 on y* at epoch 75, but it still fails on noisy y: the reviewer's statement
     "every random-init head tried fails" survives two more arms of mine (S8 at LR 0.01: best 17.4%; S8 without dropout: 29.2%).
     The narrow student is slow even on the clean target (bias), which is the dominant reason, not dropout.

### Task 5 - verdict (19:41)
NOT REFUTED. Severity stays "important".
Core (logic, verified in the text): D0 = re-scoring of TCN_post checkpoints (doc line 66); D1 = same 72,904-parameter TCN_post, only the fitting target changes to y*
(lines 72-76). No arm trains anything else on noisy y; nothing measures a floor under noisy training. GPT concedes the open point (line 60; control/README.md:23
"non certifica l'apprendibilita' a campione finito"), the decision table (lines 78-83) has no row for it, row 1 + sec. 6 + architecture/README.md:59-63
("scegliere un unico emendamento ... capacita'/regolarizzazione sono priorita' piu' motivate") send the single frozen amendment to the head with the gate kept
(optimization/README.md:61 "Non abbassare il gate"), and no path is declared after a failed variant (protocol line 104: only "decisione registrata").
GPT asks exactly this at line 105 -> in scope. Archive: no temporal NEURAL head other than TCN_post was trained on the temporal world (L0 and MLP are instantaneous; the only other temporal reader is the
unregularised closed-form L13 on y, held -85.6% / -69.5%) -> nothing settles it already.
Toy numbers: every quoted figure is in the logs; references identical; deterministic prefix reproduced bit for bit (17 / 17 / 17 / 58 / 10 / 10 epochs).
Three corrections the final review should carry (the finding overstates here):
 1. The toy does not show "no random-init head passes" nor that the bar is unattainable: from the truth the minimal family stays above 50% (66-67% with dropout,
    about 79-80% without). It shows only that a D1-like success (>= 50% from epoch 12) coexists with four random-init arms that stay at or below 31% on y (TCN_post 31.1%; width-8 27.2%, 17.4% at LR 0.01, 29.2% without dropout). One synthetic world, one seed.
 2. Arm (b) as written is confounded: 14 of its 33 points of shortfall are dropout on the 8-wide layers (truth + clean target: 84.7-86.9% with dropout at epochs 20-40, 98.5-99.8% without at epochs 10-28).
    "(b) < 50% => control specification" is unsafe; run (b) with dropout off, or pair it with the same arm on y* and read the noise-only part.
    Arm (b) needs no new head class: GPT's zero embedding into the production TCN_post has exactly zero gradient on all extra channels.
    Arm (a) (width-8 from random init) is the head GPT excludes as a candidate (line 89) and its failure is dominated by slow learning of the clean function:
    low diagnostic value; (b) is the arm that carries the reading.
 3. "Not licensed" is too broad: a D1 success does license the COMPARATIVE priority of capacity/regularisation over more epochs or a warm start (the comparison is literal in
    architecture/README.md:59-61 and in the question at doc line 105; line 80 states the priority without the comparison). What it does not license is the premise that a head-side remedy exists. The rewording should add the control-side
    alternative and the path after a failed variant, not delete the priority.
Scripts and logs of my runs kept in /Users/simonebenatti/wearusfm_local/tcn_temporal_diag_review/skeptic_stat_STAT-2_files/ (small text files).

### Closing (19:42)
- work dir scratchpad/wf3_skeptic_stat2_work deleted with its full literal path (toy_data.npz 297 MB included); no process left; review_diag untouched
  (git status empty, HEAD e7670758de9d0b6e25a9ad880ec3e1f7d573f169). No cluster access, no job, no push, no download.
