# Skeptic notes — finding DESIGN-1 (design lens)

Started 21/11/2026 19:12 CET. Read-only checkout: scratchpad/review_diag @ e7670758.
Finding under test: "D0 and D1 need a registered decision of Simone, a budget line and logical IDs
before any launch; the document never says so" (claimed severity: important).

Sub-claims to test, one by one:
1. Protocol quotes at lines 104, 158, 186, 219 exist and say what is cited.
2. decisioni.md last entries of 21/11: gate failure + "REVIEW_REQUIRED", with the two quotes.
3. "The document never says so": what the diagnosis (sections 5, 7) actually says about authorisation/budget/IDs.
4. execution.py: "retry": "forbidden_without_new_decision"; frozen graph (validate_graph).
5. "D0 is not a local step: it is a cluster job": inputs only on Leonardo (sizes cited), no copy on the Mac.
6. Budget: A1 consumed 3.681111 GPUh of 56.1 authorised, and whether that budget covers new tests.

## 1. Protocol quotes (docs/pose_temponet_frozen_protocol_20261117.md, read in full, 221 lines) — VERIFIED

- line 104 (end of §4.4): "Eventuale modifica della specifica richiede decisione registrata." present verbatim.
- line 158 (§8): "Un esito negativo dei gate torna alla decisione pertinente, non autorizza mutazioni." present verbatim.
- line 186 (§9): "La riserva copre avvii/sforamenti contabilizzati, non retry. [...] Una nuova diagnosi/ripartenza richiede decisione registrata." present verbatim
  (the sentence is the LAST one of the paragraph on budget/packing).
- line 219 (§10 table, row Esecuzione): "Approvazione del 19/11 comprende implementazione, collaudi, benchmark e catena prevista dopo review favorevole e gate; niente secondo
  consenso per gli stessi invii. Superamento tetti, cambio scientifico o retry richiedono nuova decisione" present verbatim.
- Also relevant, not cited by the finding: line 3 "Supera il fermo documentale del 18/11; non autorizza nuovi esperimenti o retry."
  and line 112 "Conservare prove e diagnosticare prima di una nuova decisione."

## 2. decisioni.md entries of 21/11 (lines 6943-6956) — VERIFIED

- line 6946: "Arresto scientifico prescritto, non attesa di autorizzazione." verbatim.
- line 6947: "Per riprendere serve diagnosi e una consegna distinta se cambia la ricetta."
- line 6950: heading "### 2026-11-21 | GPT chat 2 | Diagnosi TCN: seguito proposto, REVIEW_REQUIRED"
- line 6952: Simone's quote verbatim: «perché la TCN non passa il controllo temporale? Analizza cosa non funziona e come possiamo risolverlo, poi preparami il prompt per farlo rivedere a claude».
- line 6956: "Operatore 2-TCN head launcher, nessun invio delegato da questa nota."
- No sentence in the entry says that D0/D1 are authorised, no budget line, no ID. (git diff d3e41d06..e7670758 -- docs/decisioni.md = +9 lines, read in full.)

## 3. What the diagnosis and its companions actually say about authorisation / budget / IDs — VERIFIED (finding is literally right, but overstates the silence)

grep -n -i "simone|decision|autorizz|firma|approv|budget|tetto|ID |deploy|rilasci|pacchetto|consens|registr" on docs/reviews/pose_temponet_temporal_diagnosis_20261121.md:
- The words "Simone", "decisione", "autorizz*", "firma", "approv*", "consens*" occur ZERO times in the main document.
- line 3: "Stato: REVIEW_REQUIRED — analisi conclusa, seguito diagnostico proposto, nessun rilascio".
- line 62: heading "Seguito minimo proposto, da rivedere prima di implementare"; line 68: "Questa lettura e i fit D1 sono solo proposti, non già eseguiti."
- line 97: "non è un preventivo eseguibile ... La nuova implementazione deve misurare questi costi e pubblicare allocazioni e budget prima del rilascio."
- line 99: "Il residuo del budget A1 non trasforma il gate scientifico negativo in un via a prove diverse. La review richiesta qui è sul disegno
  diagnostico, poi serve il pacchetto operativo distinto. Non si creano ora ID o deploy."
- line 105: "Nessuna implementazione, job o modifica della campagna A1 richiesta con questo documento."
Companions in the same commit:
- decisioni.md 6956: "Operatore 2-TCN head launcher, nessun invio delegato da questa nota."
- registry "Problemi trovati": "Nuove ricevute/risorse e compatibilità delle identità devono essere esplicite prima di qualunque futuro rilascio."
- registry "Cosa ho fatto": "Operatore unico resta 2-TCN head launcher dopo review e pacchetto distinto."
- results/.../control/README.md lines 28-31: "prima inferenza dei checkpoint esistenti su y* e y ...; poi, SE AUTORIZZATA E PREPARATA, due soli fit su y*"
  -> the only place where an authorisation is mentioned; grammatically attached to D1 only, D0 is "prima" with no condition; it does not say by whom.
- prompt (docs/prompt_claude_...20261121.md line 27): "Dopo la convergenza prepareremo l'eventuale pacchetto distinto per 2-TCN head launcher".
So: the sequence written by GPT is review convergence -> distinct operational package -> launcher. A decision of Simone registered in decisioni.md
is NOT named as a step anywhere; budget and IDs are explicitly DEFERRED to the package (not forgotten), and the document itself states that the A1
residual is not a go for different tests.

## 4. Frozen runtime — VERIFIED

- scripts/pose_temponet/execution.py:41 `"operator": "2-TCN head launcher", "retry": "forbidden_without_new_decision"` in `_job` (every job).
- plan_controls.json: 24 jobs, Counter(retry) = {'forbidden_without_new_decision': 24}; stages = controls 6, selection 5, refit 5, wls_calibration 2,
  preflight/cpu_benchmark/gpu_benchmark/locked_inference/metrics/collection 1 each. No cell for D0 or D1.
- provenance.py:49-50 STAGES is a closed tuple; __main__.py:56 `--stage ... choices=P.STAGES`.
- runner.py:238-240 run_stage requires `apply_training_pack(execution.job_graph(), ...) == plan["jobs"]` ("allocations differ from measured packing");
  runner.py:220-221 refuses an existing output directory/receipt ("do not duplicate or retry").
- execution.py:94-95 caps 40 GPU jobs / 7 CPU-only jobs; execution.py:106-107 "measured pack differs from the complete frozen matrix".
=> D0 and D1 cannot run from the frozen checkout edebfb6c: they need new code, hence new source identity, checkout, plan and logical IDs.
   (workflow.py:307-324: the frozen code scores the selected state only on HELD, observed and clean; fitting only observed via
   selected_fitting_diagnostics in train.py:570-572. A clean score on fitting and any score of `current_states` do not exist in the runtime.)

## 5. "D0 is not a local step" — VERIFIED (cluster access needed); "job" is an inference

- receipt TCNH-controls-temporal-TCN_post-20261120.json .artifacts.states: bytes 18386382, path /leonardo_work/.../controls/temporal/TCN_post/selection_states.pt
- receipts TCNH-wls-calibration-{0,1}-20261120.json .artifacts.calibration_state: bytes 228630380 and 228396140, calibration.pt on /leonardo_work
- plan_controls.json .inputs.cache_train: bytes 474412856, /leonardo_work/.../consolidation_long_20261105/49046/pose/train/train.npz
- plan_controls.json .inputs.target_reference_train (257675076 bytes) and .inputs.pose_stats: both under /leonardo_scratch/large/userexternal/sbenatti
- find on ~/wearusfm_local, ~/dev/wearusfm (all worktrees) and "~/Documents/ChatGPT/FM per EMG" for names selection_states*.pt / calibration*.pt and for the
  four exact byte sizes: ZERO hits. controls results note line 50: "calibration.pt e selection_states.pt restano su WORK ai percorsi pinnati."
- The diagnosis never calls D0 local: line 66 "Il launcher rilegge con hash verificati gli stati già archiviati"; line 97 lists D0 among the costs still to measure.
- Whether D0 must be an sbatch job or could be a read-only operator run on the login node (as the gate verification was: controls results note line 9,
  "13,76 s, nessuna nuova allocazione") is not decided by any file. Either way it is cluster access with new code, which this analysis did not have.

## 6. Budget — VERIFIED

- launcher/accounting_through_controls_20261121.json: total_gpu_hours 3.681111111111111, total_core_hours 31.974999999999998, allocation_count 16,
  maximum_allocations 29, authorized_caps.gpu_total_hours 56.1 (neural 53.85, cpu_only 115.5, total core 564.3).
- docs/pose_temponet_budget_A1_handoff_20261121.md line 8: "Il moltiplicatore 1,5 riguarda i tetti, non un secondo aumento dei walltime, della dose o del
  numero di modelli."; line 17: "Non aumentare i limiti di memoria, i walltime per job, i massimi di LR/epoche o il numero di celle".
  plan_controls.json .budget_amendment.decision.words = ["va bene se aumenti il budget", "dagli quello che serve moltiplicato per 1.5"].
  => the 56.1 GPUh are ceilings on the planned matrix; two extra fits and a scoring run are new cells. The diagnosis agrees (line 99).

## 7. Extra checks made to try to refute

a) Is there any later authorisation? Local refs only (no fetch): origin/codex/pose-temponet-temporal-diagnosis-20261121 = e7670758 (tip), launcher branch = d3e41d06.
   grep "D0\b|D1\b" docs/decisioni.md -> only line 6955 (the proposal summary). No authorisation, budget line or ID for D0/D1 anywhere.
b) Could D0 come from the local JSON? No. selection_temporal_TCN_post.json: selected_fitting_diagnostics = {"0": {dynamic_sse 1957810.270671099,
   dynamic_target_ss 2957984.5843438697, mse_sum 1553.0214554169368, window_count 920}, "1": {..., window_count 934}} = observed y only, selected checkpoint only;
   clean statistics exist only for HELD (selected_outer_clean_statistics). (1957810.2707/2957984.5843 = 0.661873, as in the context.)
c) What the controls stage opens (runner.py:302-309): P.read_verified_inputs(plan,"train") + the two calibration.pt (runner.py:98-102) -> prepare_control_folds -> run_selection on cuda.
   D0 needs the same inputs plus selection_states.pt. All on Leonardo.
d) Whose decision? None of the four cited protocol lines says "di Simone". Line 219 opposes "nuova decisione" to "Approvazione del 19/11" (Simone's).
   CLAUDE.md of the reviewed checkout, lines 31-32 (Simone, 21/11, "Un solo via"): "Quando servono una modifica scientifica, risorse fuori dal perimetro
   approvato o una decisione richiesta da un criterio di arresto, la chat responsabile segnala la decisione precisa." -> supports "of Simone".
   COUNTER-READING not discussed by the finding: CLAUDE.md lines 98-101 (Simone, 28/10): jobs "con costo massimo inferiore a 5 GPU-ore" for
   "attivita' pertinenti al lavoro affidato" may be prepared and submitted autonomously, "rispettando i budget della campagna"; D1 is 738 s = 0.205 GPUh
   by proportion (150/900 x 4427 s), so D0+D1 are of the order of 1 GPUh (my inference, not a measurement). Lines 108-110 limit it: it "non cambia
   automaticamente obiettivi, architetture o perimetro di un esperimento approvato".
   Same-day practice on line 3 (registry commit e0ecbd4b, 21/11 17:33): a targeted test of 0.47-0.59 GPUh after a stop was put to Simone with cost in GPUh and
   local hours and consumption against the signed budget, and registered with his literal answer. So practice = decision of Simone even below 5 GPUh.
e) "The graph is frozen (validate_graph)": slightly imprecise. validate_graph checks structure, operator, profiles and the 40/7 caps; the freeze of the set of
   cells is runner.py:238-240 (plan["jobs"] must equal apply_training_pack(job_graph(), ...)) and the closed STAGES tuple (provenance.py:49-50). Same conclusion.
f) "D0 ... is a cluster job": the inputs are only on Leonardo (verified), but nothing fixes sbatch vs login-node read. The operator has already run a
   read-only Python evaluation on the login node with no allocation, no ID and no budget line
   (launcher/verify_positive_controls_readonly.command.txt: `ssh leonardo '... python3 -c ...'`; results note line 9: "13,76 s, nessuna nuova allocazione").
   So the point worth carrying is "say explicitly that D0 is a job with ID and budget, not an operator read"; the finding states it as a fact.

## 8. Verdict

NOT REFUTED: every quote, line number and figure cited by DESIGN-1 reproduces.
Overstated in the headline:
 - budget and IDs are not forgotten, they are explicitly deferred (diagnosis lines 97 and 99: "pubblicare allocazioni e budget prima del rilascio",
   "Non si creano ora ID o deploy", "poi serve il pacchetto operativo distinto"), and the diagnosis itself says the A1 residual is not a go;
 - control/README.md line 30 does say "se autorizzata e preparata" (for D1);
 - "of Simone" is an inference (well supported by the 21/11 rule and by same-day practice, but the 28/10 autonomy is a counter-reading).
What survives: no text names the registered decision that protocol line 186 requires for "una nuova diagnosi", and the written sequence
(review convergence -> package -> launcher) can be read as if convergence were enough.
Severity: MINOR (claimed important). Nothing in the reviewed delivery authorises or enables a launch (REVIEW_REQUIRED, "nessun invio delegato",
frozen runtime has no D0/D1 cell, new code/plan/IDs/budget must come in a separate reviewed package); cost of the order of 1 GPUh; no effect on the diagnosis
or on the design of D0/D1. One sentence to carry into the review as a condition for the package, without a new round. It is still a decision that Simone has to take.
No work directory created; no process left running; nothing written under review_diag.
