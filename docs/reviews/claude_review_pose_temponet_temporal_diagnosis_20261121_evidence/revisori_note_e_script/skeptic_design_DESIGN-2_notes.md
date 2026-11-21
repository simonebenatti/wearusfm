# Skeptic notes — finding DESIGN-2 (D1 outcome table not decidable)

Checkout under review (read-only): review_diag at e7670758de9d0b6e25a9ad880ec3e1f7d573f169
Document: docs/reviews/pose_temponet_temporal_diagnosis_20261121.md (107 lines)
Started: 21/11/2026

Goal: try to REFUTE the finding. Recompute numbers, read the quoted text, decide the true severity.

## Step 0 — setup
- Notes file created before any analysis.

## Step 1 — quotes of the finding against the document (read whole file, 107 lines)
- Row 3 text quoted by the finding: VERBATIM at line 82 ("Anche il fitting pulito resta scarso rispetto al riferimento pulito, o curve ancora in discesa/bordo").
- "Le righe qualitative non sono nuovi gate numerici": VERBATIM at line 85.
- "Se la lettura resta ambigua o a bordo a 150, esito diagnostico inconclusivo, senza estensioni automatiche": VERBATIM at line 76.
- Row 1 (line 80): "Migliore ricostruzione pulita anche sugli esclusi e recupero originale >=50% nei tre pannelli" -> priority to complexity/regularisation.
- Row 2 (line 81): "Errore pulito molto inferiore sul fitting che sugli esclusi" -> generalisation across users; simpler reader / adequacy of the control.
- Row 4 (line 83): "Quadri discordanti o miglioramenti insufficienti a separarli" -> "Diagnosi inconclusiva; nessun rilascio e nessuna scelta automatica della variante."
- AGAINST the finding (partial): line 76 DOES name the clean reference: "errore pulito su fitting ed esclusi, confrontato con la predizione cleanLS fittata su y* e valutata ora su y* (nuovo score descrittivo...)". So "the fitting reference is not named (cleanLS on y*, or zero)" is inaccurate: it is named, what is missing is the quantity ("scarso", "molto inferiore").
- AGAINST the finding (partial): row 4 has a stated consequence (inconclusive, no release, no automatic variant). "with no consequence" is inaccurate; what is missing is a next step.
- Line 76 also says "Non introdurre una soglia di 'ottimo pulito' dopo aver visto le curve" (so a threshold frozen BEFORE is compatible with the document) and line 72 declares D1 "analisi esplorativa dichiarata, non una nuova conferma indipendente".

## Step 2 — numbers of the finding recomputed from the JSON (all reproduced)
Command: python inline on results/infra/pose_temponet_execution_20261121_A1/launcher/{selection_temporal_TCN_post,calibration_0,calibration_1}.json
- .selected = {lr 0.003, exposure 31.0, mse 1.8001448213440414, edyn 0.7159262203822878}  OK
- curve lr 0.003, point exposure 150.0: edyn 0.7415560124949256, mse 1.8521482905919395, fold_steps [4350, 4500]  OK
- cleanLS (L0 fitted on y*, scored on y) FITTING: 2096186.735857809/2957984.5843438697 = 0.708654 (fold0); 2157951.3395088115/2878704.615363543 = 0.749626 (fold1)  OK
- L13 clean fit on fitting 0.693938 / 0.738389; temporal_margin_recovery_fitting 0.270895914587907 / 0.19930915281486627  OK
- oracle on fitting reconstructed = E_L0 - (E_L0-E_L13)/rec = 0.654332 / 0.693246  OK (same reconstruction GPT uses at line 46: apparent fitting recovery 86.1181% / 69.2509%)
- held recoveries 24.1731% / 18.8755%, 50% bounds 0.681941 / 0.720512  OK
- Minima per LR (pooled held noisy MSE): 0.0001 -> epoch 181 of 300; 0.0003 -> 64; 0.001 -> 31; 0.003 -> 31; 0.01 -> 93. recipe.last_epochs = 5; optimizer_alarms = []; optimization_unresolved = false.
- Extra check (mine): on held, (E_TCN - E_oracle)*SS_dyn(y) = 138532.5 vs archived clean dynamic_sse 141448.3 (fold0), 128775.7 vs 127958.4 (fold1): the noisy-score recovery is, within ~2%, 1 - cleanSSE_TCN/cleanSSE_cleanLS. So a fitting-side recovery on y is a legitimate clean-error measure ONLY when the model never saw the noise (D1), not in A1.

## Step 3 — what "bordo" means in the frozen runtime, and the overlap claim
- Protocol docs/pose_temponet_frozen_protocol_20261117.md section 4.2 point 2: "Se minimo MSE oppure minimo E_dyn e' nelle epoche146-150, continuare entrambi i fold a300"; point 5: alarm only if the minimum is "nelle ultime 5 epoche dell'orizzonte massimo".
- Code scripts/pose_temponet/selection.py:132-138 `needs_epoch_extension`: boundary = initial_epochs - last_epochs + 1 = 146, on best("mse") OR best("edyn") of the pooled held noisy curve. selection.py:163-166: alarm "minimum_in_last_five_maximum_epochs" only when horizon == maximum_epochs (300).
- So in the frozen recipe a minimum in 146-150 is NOT an optimisation alarm: it triggers the extension to 300. D1 forbids the extension (line 72 "niente ... estensione automatica a 300"), hence a D1 minimum in 146-150 has no status inherited from the protocol; the document gives it TWO different statuses:
  * line 76: "a bordo a 150" -> "esito diagnostico inconclusivo";
  * row 3 (line 82): "curve ancora in discesa/bordo" -> "Ottimizzazione/implementazione da indagare";
  * and if recovery >= 50% in three panels the same run also satisfies row 1 (line 80) -> "Priorita' a controllo della complessita'/regolarizzazione".
  CONFIRMED from the text: three mappings for one observation, no stated precedence. "Opposite" is strong; they are different next steps (section 6 line 91 treats schedule/optimisation and capacity/regularisation as alternative branches).
- "bordo" is definable from the protocol (146-150 on pooled held noisy MSE/E_dyn); "curve ancora in discesa" is not (which curve: fitting/held, clean/noisy, MSE/E_dyn; which slope).
- Mitigations in the document (reasons NOT to call it blocking): D1 is declared exploratory (line 72); rows "non sono nuovi gate numerici" (line 85); success does not admit the noisy recipe (line 76); any variant goes through a new proposal and the unchanged 50% control (line 93); row 4 forbids "scelta automatica della variante".
- But section 6 line 89 makes the choice of the variant depend on "Se D1 separa chiaramente", and section 8 line 105 asks the reviewer precisely to "verificare che D0 e i due D1 separino davvero i casi utili": the reading rule is the object under review.

## Step 4 — empirical test of the prediction "noise-free regression at constant LR still descending at 150 epochs"
The archive has no noise-free run, so this part of the finding was a prediction, not evidence. Toy test on SYNTHETIC latents (not the real ones):
script /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_skeptic_design2_work/sim.py
- 40 synthetic users x 46 windows x 160 x 384 (48 AR(1) factors, per-user loading perturbation "shift", offsets, white noise); fold = alternate users, 920 fitting / 920 held windows, 29 updates per epoch (as in A1 fold 0).
- Frozen code used as is: controls.generate_components (L, width-8 T, AR(1) noise), controls.synthetic_target, heads.make_head("TCN_post", 0), train.Trajectory (AdamW lr 0.003, wd 0.01, batch 32, clip 1, dropout 0.1).
- a picked on the frozen grid with sigma 1.3 so that the held oracle margin is closest to 8%: a = 0.4 (same as the real calibration), held E_cleanLS 0.666092, E_oracle 0.610797, margin 8.30% (real: 0.710431 / 0.653451, 8.02%).
- Three runs of 150 epochs started 19:18 in background: noisy target shift 0.5 (A1 analog), clean target shift 0.5 (D1 analog), clean target shift 1.5.
(results appended below when the runs finish)
- PARTIAL (19:22, 16 epochs of 150; ~10-12 s per epoch, three processes in parallel, one thread each):
  * noisy target, shift 0.5 (A1 analog): held recovery 37.6% at the running minimum (epoch 14): same regime as A1 (24%/19%), a bit easier.
  * clean target, shift 0.5 (D1 analog): held recovery 90.7% at epoch 16, running minimum of held noisy MSE = last epoch so far; held clean MSE 0.0154 still falling (cleanLS clean 0.1343).
  * clean target, shift 1.5: recovery 93.2% at epoch 16, still falling.
  Caveat written before the result: synthetic latents are a rank-48 factor model, far easier than real FM latents; the toy can show the QUALITATIVE shape of a noise-free curve at constant lr 0.003, not the level of the real D1.

## Step 5 — archived evidence on how often the 146-150 boundary fires in A1 (real runs, with noise)
Command: python inline over launcher/selection_{temporal,linear}_{TCN_post,L0,MLP}.json (.curves[].horizon, .selected, .optimizer_alarms, .minimum_exposures)
- A horizon of 300 in a curve means that its minimum at 150 was in epochs 146-150 (selection.py:132-138), so the extension fired:
  * selection_temporal_TCN_post: lr 0.0001 extended to 300 (minimum then at 181); lr 0.003 minimum at 31.
  * selection_temporal_L0: lr 0.0001 and 0.0003 extended to 300; selected lr 0.0003 epoch 183.
  * selection_linear_L0: lr 0.0001, 0.0003, 3e-05 extended; selected lr 0.0001 epoch 299 with alarm "minimum_in_last_five_maximum_epochs".
  * selection_linear_MLP: lr 3e-05 extended. selection_linear_TCN_post and selection_temporal_MLP: none.
  => "a bordo a 150" is an ordinary event in this campaign even with noisy targets (4 of 6 selections have at least one lr extended). The frozen recipe answers it with the extension, which D1 forbids.
- Epoch-to-epoch jitter of the pooled held noisy MSE, lr 0.003 temporal TCN (second half of the curve, from second differences): sd ~0.0021; values 1.80014 (ep 31), 1.81618 (60), 1.83950 (100), 1.85215 (150). The A1 curve at lr 0.003 rises after 31: with noise the minimum is interior.
- PARTIAL (19:32, 60 epochs of 150), from run_*.json via summ.py:
  * noisy target (A1 analog): minimum of held noisy MSE at epoch 14 (recovery 37.6%), then 26.8% at 31, 20.4% at 40, 9.9% at 60; apparent FITTING recovery on y 104% at 20, 131% at 40, 144% at 60 while fitting CLEAN E_dyn rises 0.0743 -> 0.0990: the toy reproduces the A1 pattern (early minimum, late degradation, fitting/held gap) and shows it is noise memorisation in the toy.
  * clean target shift 0.5 (D1 analog): recovery 94.1% at 20, 96.4% at 31, 97.1% at 40, 96.6% at 60; running minimum of held noisy MSE at epoch 57 (1.691741, recovery 97.8%); held clean MSE 0.00950 (20) -> 0.00625 (40) -> 0.00556 (60), minimum 0.004025 at 57: slow descent with jitter, train-mode loss not monotone (0.00854 at 40, 0.01156 at 60).
  * clean target shift 1.5: recovery 97.7% at 60, running minimum at epoch 52.

### Step 4 — FINAL toy results (19:50, three runs of 150 epochs complete; numbers from run_*.json)
Clean target = D1 analog (lr 0.003, 150 epochs, selection on held NOISY MSE):
- shift 0.5: best(mse) epoch 103, best(edyn) epoch 142 -> frozen boundary rule (>=146) NOT fired. Recovery 98.00% at the selected epoch, 97.27% at epoch 150.
  Held clean MSE by block: 0.00631 (ep 31-60), 0.00560 (61-90), 0.00562 (91-120), 0.00561 (121-150); within-block range up to 0.0038-0.0136. Train-mode loss 0.00955, 0.00802, 0.00773, 0.00780. => plateau with jitter from about epoch 60, not a descent.
- shift 1.5: best(mse) = best(edyn) = epoch 120 -> boundary NOT fired. Recovery 98.14%. Held clean MSE by block 0.00480, 0.00454, 0.00446, 0.00434; train-mode loss 0.00859, 0.00801, 0.00781, 0.00745 => very slow residual descent (about -3% per 30 epochs) under jitter three times larger. The minimum of the CLEAN held MSE is at epoch 64, of the noisy one at 120: on a plateau the position of the minimum is arbitrary.
Noisy target = A1 analog: selected epoch 14, recovery 37.59% (A1 real: 24.17% / 18.88%), -15.4% at epoch 150; apparent fitting recovery on y 92.66% at epoch 15 and 165.4% at 150, while the clean-ratio version is 49.95% and 18.05%.
Fix formula check: when the model never saw the noise (clean runs) the fitting-side recovery on y equals 1 - E_clean,fit/E_cleanLS,clean,fit: 0.9460 vs 0.9460 (ep 105), 0.9753 vs 0.9752 (ep 150), 0.9824 vs 0.9823, 0.9744 vs 0.9752. In the noisy run it does not (92.66% vs 49.95%).

READING for DESIGN-2:
- The prediction of the finding ("expected to be still descending at 150 epochs, so the boundary clause WILL fire") is NOT supported: 0 of 2 clean toy runs end at the frozen boundary; the curve reaches a plateau by about epoch 60. It is not refuted for the real latents either (toy latents are easier: 98% clean, 37.6% noisy). Correct wording: the boundary MAY fire, also by chance (minimum position on a jittery plateau: 103, 120, and 142 for E_dyn, four epochs short of 146).
- The toy DOES illustrate the undecidable wording: a 98% pass (row 1) coexists with a train-mode loss and, in one run, a held curve that are still drifting down slowly; whether that is "curve ancora in discesa" (row 3) cannot be decided from the text.
- Caveat: synthetic rank-48 factor latents, one fold, one seed, full support. Qualitative evidence only.

## Step 6 — verdict
NOT refuted in its core; three sub-claims corrected.
Confirmed: (a) boundary at 150 has three mappings in the text (line 76 inconclusive, row 3 optimisation, row 1 if >=50%), no precedence; (b) "molto inferiore", "scarso", "ancora in discesa" have no quantity and no named curve; (c) all cited numbers reproduce; (d) the reading steers the single frozen variant of section 6 and, through row 2, a judgement on the "adeguatezza del controllo".
Corrected: (1) the clean reference IS named at line 76 (cleanLS fitted on y*, scored on y*); (2) row 4 HAS a consequence (inconclusive, no release, no automatic variant): what is missing is the next step; (3) "will fire" is a prediction not supported by the toy; "opposite consequences" should read "different next steps".
Severity: important (not blocking): D0 is unaffected; for D1 a reading rule with precedence, written before the fits in the operational package, is enough and needs no new review round.

## Step 7 — closing checks
- scripts/pose_temponet/train.py:553-554: `pair.advance(recipe.maximum_epochs)` runs only if `needs_epoch_extension` is true, so a curve with horizon 300 in the archive proves that its minimum at 150 was in epochs 146-150.
- Weakness of the suggested fix, rule (2): it labels by the fitting-side recovery alone (>= 0.50 with held < 0.50 is row 2), so held 49% / fitting 51% would be called a generalisation gap. It needs a minimum gap or an inconclusive band. The 0.50 on fitting is a new number: present it as a reading rule frozen before the fits, not as a gate.
- Cleanup 19:52: work directory wf3_skeptic_design2_work deleted with its full literal path; no sim.py process left; review_diag untouched (git status empty, HEAD e7670758de9d0b6e25a9ad880ec3e1f7d573f169).
- Small evidence kept beside these notes: /Users/simonebenatti/wearusfm_local/tcn_temporal_diag_review/skeptic_design_DESIGN-2_sim/ (sim.py, summ.py, run_clean_0.5.json, run_clean_1.5.json, run_noisy_0.5.json; 144 KB).
