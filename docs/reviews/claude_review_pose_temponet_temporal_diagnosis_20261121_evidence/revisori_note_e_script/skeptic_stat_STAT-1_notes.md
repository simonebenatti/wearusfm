# Skeptic notes on finding STAT-1 (lens "stat")

Started 2026-11-21 19:21 CET. Read-only review of
`/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/review_diag`
(commit e7670758). No cluster, no job, no push.

Finding under test (STAT-1, declared "blocking"): the D1 reading rules classify the expected
outcome of clean training (held minimum at the edge, epoch 150) as inconclusive / "optimisation
to investigate"; rows of the table not mutually exclusive; no fixed-epoch reading at epoch 31
declared; D0 and D1 themselves sound.

Tasks:
1. Read the cited text (lines 76, 78-85) literally and check the quotes.
2. Check whether the rows can really co-fire / whether an edge minimum is really classified as claimed.
3. Check the "expected outcome" premise against the archived JSON (curves of the noisy run, clean statistics).
4. Check the toy evidence (stat_work_files/log_wide_clean.txt) and what it supports.
5. Decide severity.

## Task 1 - the cited text, read literally (done about 19:25; clock re-read with `date` at 19:30:02)

File: review_diag/docs/reviews/pose_temponet_temporal_diagnosis_20261121.md (107 lines, commit e7670758).
- line 76, last sentence, verbatim: "Se la lettura resta ambigua o a bordo a 150, esito diagnostico inconclusivo, senza estensioni automatiche." -> quote in the finding is exact (truncated before the comma).
- line 82 (row 3), verbatim: "Anche il fitting pulito resta scarso rispetto al riferimento pulito, o curve ancora in discesa/bordo | Ottimizzazione/implementazione da indagare con target deterministico; non concludere che il campo ricettivo sia insufficiente." -> quote exact.
- line 85, verbatim: "Le righe qualitative non sono nuovi gate numerici. Il solo riferimento quantitativo di sensibilità resta il recupero originale, espresso in modo identico" -> quote exact.
- line 80 (row 1): "Migliore ricostruzione pulita anche sugli esclusi e recupero originale >=50% nei tre pannelli, quando si addestra su y*" ... "Non è prova di ottimo né ammissione."
- line 83 (row 4): "Quadri discordanti o miglioramenti insufficienti a separarli | Diagnosi inconclusiva; nessun rilascio e nessuna scelta automatica della variante."
- line 72: "niente nuova griglia, altri semi o estensione automatica a 300".
- line 74: D1 records "curve pulite, MSE ed E_dyn, fitting e fold esclusi separati"; "Salvare lo stato scelto e il finale"; line 97 counts "quattro letture fitting/esclusi x pulito/rumoroso".
  => the numbers needed for an iso-epoch reading at 31 WOULD exist after D1; what is missing is only the declaration of how to read them.
- No sentence of section 5 conditions "a bordo a 150 -> inconclusivo" on the recovery being below 50%. No sentence declares a reading at exposure 31 for D1 (31 appears only in D0, line 66, and in section 3).

What "bordo" means in the frozen runtime: scripts/pose_temponet/selection.py:132-138 needs_epoch_extension: boundary = 150-5+1 = 146;
True if best pooled MSE OR best pooled E_dyn has exposure >= 146. Protocol docs/pose_temponet_frozen_protocol_20261117.md line 66
("Se minimo MSE oppure minimo E_dyn è nelle epoche146–150, continuare entrambi i fold a300") and line 69 (alarm if still in the last 5 epochs of the maximum horizon).
LR is constant: train.py:233-234 AdamW(lr=lr, betas, eps), no scheduler anywhere in train.py/heads.py (grep sched|cosine|warmup|lr_lambda: no hit).

Toy numbers quoted by the finding, re-read in stat_work_files/log_wide_clean.txt: held rec 0.432 (ep 10), 0.608 (15), 0.705 (20), 0.753 (25), 0.802 (28); SUMMARY "selected(min held MSE) ep 28" = last epoch RUN (time budget), i.e. 28 of 150.
Iso-epoch arithmetic: clean ep 25 0.753 -> bias 24.7; truthfree ep 23 0.550 -> 45.0; 100-24.7-45.0 = 30.3 vs wide_noisy ep 25 0.311. Epoch 20: 100-29.5-36.5 = 34.0 vs 0.295 -> 4.5 off. Arithmetic exact, but it mixes epoch 25 and epoch 23, one seed.
WEAK POINT of the finding's evidence: the toy stops at epoch 28 (19% of the D1 horizon). The edge rule concerns exposures 146-150 with a CONSTANT LR. Whether a clean run still has its minimum there is NOT shown by the toy. -> Task 3 (own run).

## Task 2 - can the rows really co-fire? (done about 19:25, text logic)
Yes on the literal text. Scenario: clean run, recovery on y >= 50% in the three panels at the selected checkpoint, selected exposure >= 146.
 row 1 fires (its own text says "Non è prova di ottimo", so it does not require a non-edge minimum);
 row 3 fires through its second disjunct "o curve ancora in discesa/bordo";
 line 76 says "a bordo a 150 -> esito diagnostico inconclusivo"; row 4 "quadri discordanti -> inconclusiva; nessuna scelta automatica della variante".
 Row 1 and row 3 lead to different consequences (capacity/regularisation variant vs optimisation to investigate), so the conflict is real, not cosmetic.
Counter-reading tried: "a bordo" meant only for the case recovery < 50% and still improving (there "inconclusive, no extension" is the correct reading). The text does not say it; "o" is a plain disjunction.

## Task 3 - the "expected outcome" premise against the REAL archive (done about 19:29)
Command: inline python on results/infra/pose_temponet_execution_20261121_A1/launcher/selection_*.json + calibration_{0,1}.json (pooled cleanLS 0.728479, oracle 0.670659 recomputed).
- selection_temporal_TCN_post.json: recipe initial_epochs 150, last_epochs 5, maximum_epochs 300; selected lr 0.003 exposure 31; minimum_exposures {edyn 27, mse 31}; optimizer_alarms [].
  LR 0.003 pooled recovery by epoch: 10: 3.07%, 20: 19.32%, 27: 21.88%, 31: 21.71%, 50: 13.43%, 100: -11.06%, 150: -22.62%.
  Epoch-to-epoch jitter of the real curve at LR 0.003, last 30 epochs: d(recovery) mean -0.26 points, sd 1.71 points (noisy training).
- REAL precedent of "edge + pass" in the same campaign: selection_linear_L0.json selected lr 1e-4, exposure 299 of 300, minimum_exposures {edyn 298, mse 299},
  optimizer_alarms ['minimum_in_last_five_maximum_epochs'] -- a well-specified head on a target it cannot overfit keeps its minimum at the edge even after the extension to 300,
  and it PASSED its gate (doc line 34: all heads >= 97.46% on the linear control).
- Frozen protocol, docs/pose_temponet_frozen_protocol_20261117.md line 104: "Per ciascuna condizione neurale richiesta: se passa, ammissione con eventuali allarmi annotati;
  se non passa e la testa ha un allarme §4.2, ottimizzazione_non_risolta_entro_il_budget; se non passa senza allarme, ricetta_non_ammessa_dal_controllo_positivo.
  Un allarme di una testa che passa non ferma da solo la catena".
  => in the frozen protocol the edge alarm is SUBORDINATE to the recovery (pass + edge = pass, annotated). GPT's D1 text (line 76 and row 3) reverses that order without saying so.
  This supports the finding and gives a one-sentence fix that needs no new threshold.
- Nothing in the archive says where the minimum of a CLEAN-trained TCN_post falls: no clean-trained curve exists. -> own toy (Task 4).
- Caveat I add: D1 selects on the NOISY held MSE. Even if the clean curve is monotone, the argmin of the noisy MSE in the flat tail is moved by the cross term
  (stat notes: -1.6% / +0.5% of the margin at ep 31; it shrinks like sqrt(clean SSE)). So "minimum in 146-150" is not guaranteed by monotonicity alone; "curve ancora in discesa" (row 3) is.
(Timestamps of Tasks 1-3 corrected: the first versions were guessed, not read from the clock.)

Other places where GPT's reading rules could have been stated, all read: docs/prompt_claude_pose_temponet_temporal_diagnosis_20261121.md (27 lines),
results/infra/pose_temponet_temporal_diagnosis_20261121/control/README.md lines 27-37 ("Nessuna nuova soglia numerica sui punteggi puliti viene introdotta qui";
"il minimo della MSE pulita come descrittivo"), docs/decisioni.md and docs/claude_GPT_shared hunks of e7670758. None declares an iso-epoch reading at 31 for D1
and none says what an edge minimum means when the recovery is already >= 50%.

## Task 4 - own toy to 150 epochs (started 19:27, running)
Script: scratchpad/wf3_skeptic_stat1_work/toy150.py (reviewer STAT's make_latents/edyn_stats, REAL controls.generate_components, REAL heads.make_head("TCN_post"),
REAL metrics.per_window_mse, AdamW lr 0.003 wd 0.01 clip 1 batch 32; data generated in memory, nothing large on disk). TOY, synthetic latents, one fold.
Runs (one thread each, in parallel):
- r64a: clean target, 64-patch windows (same 920 fitting windows, 29 updates per epoch), 150 epochs, shuffle 300000, init seed 0
- r64b: same, shuffle 300001, init seed 1
- r64noisy: noisy target, 64-patch windows, 60 epochs (does the reduced regime reproduce the real signature?)
- full160: clean target, 160-patch windows as reviewer STAT (reproduces STAT's epoch 1: held mse 2.354378, E 0.862251), time-boxed 1500 s
Reduced regime: a = 0.4 selected by the same rule, held gap 8.75%, fit gap 9.38%, E_cleanLS 0.8125, E_oracle 0.7414.

Interim (19:35:10):
- Real archive, recomputed (selection_temporal_TCN_post.json selected_outer_clean_statistics + calibration_{0,1}.json): TCN clean SSE on held = 141,448.3 / 127,958.4
  = 77.42% / 80.61% of the margin (182,695.6 / 158,738.3) -> clean recovery 22.58% / 19.39% vs gate 24.17% / 18.88%; cross term -1.60% / +0.51% of the margin. STAT's numbers confirmed.
- r64noisy FINISHED (60 epochs): argmin held MSE ep 28 (recovery 12.06%), argmin held E_dyn ep 22 (12.95%), epoch 60 recovery -25.44%, fitting recovery on y 134.6% at 60 (fitted noise).
  Same signature as the real run (selected 31, E_dyn minimum 27, 21.7% then -22.6% at 150): the reduced toy is a fair qualitative stand-in.
- r64a / r64b (clean) at epoch 40 / 50 / 60: 78.1 / 83.2 / 86.3% and 79.3 / 84.4 / 86.9%; running argmin of held MSE and E_dyn always within the last 2 epochs so far; fitting clean recovery 95.0 / 95.7% at 60.
- full160 (clean) at epoch 24: 75.9%, argmin = current epoch. (STAT's own run had 71.9% at 24: same seeds, components stored as float32 there -> trajectories diverge by a few points; treat single-epoch values as +-3 points.)

### Task 4 results (runs finished 19:43; full160 stopped by me at epoch 66; logs and script copied to
/Users/simonebenatti/wearusfm_local/tcn_temporal_diag_review/skeptic_stat_STAT-1_files/, 160 kB)
TOY, synthetic latents, one fold, LR 0.003 constant, no deterministic flags. Recovery = held, gate scale.
| run | target | windows | epochs | epoch 31 | epoch 60 | epoch 100 | epoch 150 | argmin held MSE / E_dyn | edge rule (>= 146) |
| r64a | y* | 64 patches | 150 | 74.8% | 86.3% | 90.2% | 92.1% | 147 / 147 (93.3%) | FIRES |
| r64b | y* | 64 patches | 150 | 74.5% | 86.9% | 91.7% | 90.7% | 136 / 136 (92.9%) | does NOT fire |
| full160 | y* | 160 patches | 66 (stopped) | 81.8% | 91.4% | - | - | 59 / 60 at epoch 66 | - |
| r64noisy | y | 64 patches | 60 | 8.7% | -25.4% | - | - | 28 / 22 (12.1% / 13.0%) | - |
- Both clean runs are above 50% from epoch 15-17 onwards (full160 from 12) and stay there.
- The clean held curve is nearly FLAT from about epoch 80-100: tail 100-150 mean 91.5% / 91.2%, least-squares slope +0.057 / +0.012 points per epoch (2.9 / 0.6 points in 50 epochs),
  residual sd 0.93 / 1.31 points, epoch-to-epoch sd 1.2 / 1.3 points, single dips to 85.3 / 85.1%. Fitting clean recovery 95.8 / 95.7% at 150 (flat from 60).
- Scan of the horizon: the edge rule (argmin of MSE or of E_dyn in the last 5 epochs) would fire at 29/51 (r64a) and 22/51 (r64b) of the horizons 100..150
  (120..150: 22/31 and 5/31). => about ONE HORIZON IN TWO, decided by jitter on a plateau, not by an unfinished descent.
What this does to the finding:
- "a held minimum at or near epoch 150 is the expected outcome exactly when clean training works": OVERSTATED. In the toy it happens in 1 seed of 2 at horizon 150 (51/102 over horizons 100-150).
  STAT's own evidence stopped at epoch 28, where the curve is still rising by more than a point per epoch; by epoch 100 it is flat.
- "With a noise-free training target nothing produces an interior minimum unless the clean function overfits users": literally FALSE (r64b: interior minimum at 136 with fitting and held both flat;
  constant LR + dropout + batch noise are enough).
- But the defect of GPT's rule is not removed, it changes nature: with the SAME scientific outcome (clean training recovers about 91%, far above 50%) line 76 would say "inconclusivo" for seed a and nothing for seed b.
  A reading rule whose verdict flips on which of the last 50 flat epochs has the lowest held MSE is a defective rule. Row 3's other trigger ("curve ancora in discesa") is undecidable on such a tail (+0.6 to +2.9 points in 50 epochs against 1 point of jitter).
- Iso-epoch reading in the toy: clean 74.5-74.8% (81.8% full) against noisy 8.7% at epoch 31: here the conclusion is the same at iso-epoch and at the selected checkpoints.
  The iso-epoch reading only matters if the real clean run is slow; it costs nothing because D1 already records the full curves (doc line 74).

## Task 5 - verdict (19:44, clock read with `date`)
NOT REFUTED in its core, corrected in two points, severity lowered from blocking to IMPORTANT.
Stands (verified): quotes exact; line 76 and row 3 do attach "inconclusivo" / "ottimizzazione da indagare" to an edge minimum with no condition on the recovery; row 1 can fire on the same run;
 no reading at exposure 31 is declared for D1; frozen protocol line 104 has the opposite order (pass + alarm = pass with the alarm annotated) and the archive has a real "edge + pass" case
 (selection_linear_L0.json, exposure 299 of 300, alarm minimum_in_last_five_maximum_epochs).
Corrected: (a) the edge is not "the expected outcome", it is about a coin flip in the toy; (b) suggested fix (1) "edge neither alarm nor inconclusive" is too broad:
 with recovery < 50% and the minimum in 146-150 the outcome IS unresolved within 150 epochs (protocol line 104, second branch).
Why not blocking: what D0/D1 run and record is untouched (the finding says so itself); every number needed for the fixed-exposure readings is already recorded by D1 as written (line 74, line 97);
 the rows are declared qualitative and non-gating (line 85), D1 admits nothing (line 76), the 50% gate on the noisy recipe is unchanged; the worst case of the rule as written is a formally inconclusive
 diagnostic or a reading settled after the data, not a wrong admission. The fix is three sentences that mirror protocol line 104 and belongs in the operational package, which is reviewed before any launch.
 It must be in force before D1 runs.
Minimal fix I would ask: 1) order of reading as in protocol line 104: recovery on y >= 50% in the three panels -> row 1, edge only annotated; < 50% with the minimum in 146-150 -> "non risolto entro 150", no automatic extension;
 < 50% without edge -> rows 2 / 3. 2) declare before the run the readings at exposures 31 and 150 and at the selected point, per fold and pooled, fitting and held, against y and y*; the paired difference
 noisy run minus clean run at exposure 31 is the effect of the training noise at equal exposure (same init, batches and dropout stream by construction). 3) drop "curve ancora in discesa" as a trigger or define it.
 A new numeric level for "function learned on fitting" (STAT's point 4) is useful but not required: GPT explicitly declines new thresholds (line 76, control/README.md line 36).
