# r7-governance (rapporto incrementale)

Avvio: 14:03. Clone HEAD 0efa4d45, diff 438c8d65..fbfde797.

## Progressi 1
- (a) Copia della decisione: blob identico `5bdb0d15` a faa4a8d7, fbfde797, 0efa4d45 (sha256 376d1d47...). OK.
- (a) faa4a8d7 NON e' antenato di nessun ref del clone; origin/codex/tokenizer-logcontrast-20261104 nel clone e' a db66b507 (ref locale vecchio). Oggetto presente ma punta corrente del ramo non verificabile dal clone.
- (a) faa4a8d7 si intesta «Decisione proposta a Simone» (riga 3) e chiude «Lo stato attuale è **proposta da discutere prima degli invii**» (riga 52). decisioni.md di faa4a8d7: «Proposta per Simone ... training e invii sospesi».
- (b) CLAUDE.md fbfde797 righe 1-3: «Simone affida a questa chat ...» reintroduce «questa chat», che d9020c20 aveva tolto (registro: «La dicitura «questa chat» in CLAUDE.md poteva attribuire i lanci sia a GPT scientifico sia a 3-launcher»). Spazi mancanti reali (xxd): «Linea3», «del19/11/2026», «per3-launcher», «Base438c8d65», «review94f50942», «decisionefaa4a8d7», «sacctCOMPLETED0:0», «entro7200s».
- (e) evaluation identica come oggetto JSON in plan orig 20261116, 438c8d65, fbfde797 (hash sort_keys e757aeeb...). Fra 438c8d65 e fbfde797 cambiano solo scope, common.numeric_profile, common.benchmark_keep_checkpoints, numerical_amendment, target_recipe_amendment.{known_unresolved_gate,required_new_evidence}. coverage/calibration/activation/resources invariati.
- (d) Aritmetica del prontuario: 17 allocazioni, 18 GPUh, 169,6 coreh, 7,9 h, margine 3,6/32,8, inviluppo 21,6/202,4, storico 6,086667/57,827778, residuo 7,913333/71,505556, incremento 4/32,533333: tutti i conti tornano.
## Progressi 2
- (a) Nessuna firma/approvazione di Simone della proposta faa4a8d7 trovata: grep su decisioni.md, claude_GPT_shared e docs/ di fbfde797 e faa4a8d7 (ref db66b507 per il ramo logcontrast nel clone). Citazioni di Simone presenti al 19/11 (solo fbfde797 decisioni.md:8278 e 8301): «Ti affido l’implementazione dell’emendamento RMS100/fast125 e la preparazione del pacchetto per 3-launcher» e «Gli invii e i controlli su Leonardo spettano a 3-launcher». Nessuna parola su profilo bf16 math/TF32, parita', ripresa 544, cancelli o preventivo 18 GPUh.
- (a) fbfde797 presenta faa4a8d7 come decisione attuata: decisioni.md:8280 «attua la decisione scientifica `faa4a8d7`»; plan.json:368 "science_decision_commit": "faa4a8d7"; prontuario r.5 «decisione scientifica in `faa4a8d7`»; prompt Claude r.8 «la decisione scientifica `faa4a8d7` è attuata».
- RILIEVO PRINCIPALE (governance): preflight_plan.json 438c8d65:190-195 required_before_first_submission conteneva «publish and present resource proposal to Simone»; a fbfde797:315-319 sparito (sostituito da review codice, ledger/quota 3-launcher, audit only). plan.json fbfde797 scientific_decisions.target_windows.execution conserva «Prepare and present new costs before any launch». Emendamento 18/11 sezione «Preventivo da presentare prima degli invii». Nuovo tetto 18 GPUh/169,6 coreh > tetto approvato 17/11 14 GPUh/129,333 (decisioni 17/11 «Tetti invariati: 14 GPU-ore e 129,3333 core-ore»). Codice: solo stage training richiede scientific_decision_pin/budget_decision_pin (contrastive_run.py:160); preflight no.
- (d) delivery_pin verificato: source_digest 40d39f56..., plan b03c0bc6.../14862, preflight e4b39591.../9082, validation 36d1d4d4.../9629, diffstat 46/5524/100: tutto coincide.
- (b) CLAUDE.md fbfde797: blocco H1 sopra «# WearUsFM»; «invii, checkout e controlli Leonardo spettano esclusivamente a 3-launcher» e «prevale sulle assegnazioni operative storiche sottostanti» senza «della linea 3» nella frase; sotto ci sono le assegnazioni a Claude di linea 1 e 2. Parafrasi del mandato senza «dell’emendamento» (decisioni: «implementazione dell’emendamento RMS100/fast125»).
## Progressi 3 (bind-review)
- Prova /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/r7tests/test_r7_binding_provenance.py: 4 passed.
  1) validate_review_binding + make_job accettano un binding fatto a mano con review_commit "f"*40 inesistente e una «review» qualunque nel checkout (contrastive_release.py:106-134: del review_commit si controlla solo il formato, r.109; i byte originali solo in bind_review r.140-143).
  2) bind_review accetta qualunque file in qualunque commit (nessun vincolo su ramo/autore/percorso) — limite dichiarato nel prontuario r.130-131.
  3) due blocchi -> rifiuto «exactly one» (fail-closed): la review di Claude non deve citare l'esempio del prompt.
  4) chiavi in piu' nel blocco (es. scope) -> rifiuto: il blocco non puo' limitare le fasi; make_job (r.227) accetta tutti i modi con lo stesso binding.
- plan generator --check: nessun errore (piani rigenerati identici) nel clone temporaneo.
- verification.json: «bash_n passed» senza comando/log/hash; local_validation.json registra solo la corsa pytest (465 passed, 102,59 s, log b90d18ba...).
## Progressi 4 (risposta R/M/N vs codice)
- M-1 VERIFICATO per mutazione nel clone temporaneo: tolto config_sha256 in validate_audit (contrastive_run.py:103), in read_preflight (contrastive_benchmark_readout.py:418) e la guardia di authenticate_calibration (contrastive_prepare.py:389-390): ciascuna mutazione fa fallire il proprio test in tests/cpu/test_contrastive_targets_guards.py (baseline 5 passed; ogni mutazione 1 failed, 4 passed). match="current execution source inventory" presente nel test storico.
- N-4: test PLAN e testo readout presenti (test_contrastive_targets_guards.py:77-97).
- N-5: local_validation --verify -> {"verified": true, "all_passed": true}. Ma la ricevuta registra SOLO la corsa pytest; «launcher validati» (bash -n) sta in verification.json senza comando/log; «piani rigenerati identici» non e' in nessuna ricevuta (nessun test confronta i byte di preflight_plan.json col generatore). Rieseguiti da me: --check exit 0, bash -n ok sui tre sbatch.
- N-7: historical_artifacts_changed_from_base = [] (git diff --name-only 438c8d65 -- results/infra/contrastive_comparison_20261116).
- N-3: numeric_context_across_runs (readout) richiede 7 run x 4 rank = 28 ricevute; differenze -> FAILED, mancanze -> INCONCLUSIVE, ritorno anticipato prima dei contrasti. Conforme.
- R-4: validate_audit_accounting (release.py:155-192) conforme; nota: expected_job_id mai passato da make_job/dispatcher/runtime/readout (job_id auto-dichiarato nella ricevuta di accounting).
## Progressi 5
- R-2 consumatori NON testati: tolte le chiamate validate_review_binding + controllo review in contrastive_job.py:70-72 e contrastive_run.py:181-184 -> test job/run/release/targets_readout 89 passed sia prima sia dopo (le fixture fanno monkeypatch di validate_review_binding a no-op: test_contrastive_job.py:26, test_contrastive_run.py:73).
- R-4 consumatori NON testati: tolte le chiamate validate_audit_accounting in contrastive_job.py:80-81, contrastive_run.py:193-195, contrastive_release.py:240-242 -> 94 passed prima e dopo (incl. historical_callsites). Testato solo il lettore finale (test_contrastive_historical_callsites asserisce la chiamata).
- Prontuario r.176-177 «il submitter ricava lo SHA da CAMPAIGN_SHA»: FALSO. submit_job.py:190-195 code_sha_from cerca solo EXPECTED_SHA=; per il checkout codice_sha del ledger resta vuoto. (Il checkout stesso verifica HEAD == CAMPAIGN_SHA, campaign_checkout.sbatch:36-48.)
- (e) invariati fra 438c8d65 e fbfde797: src/, contrastive_compare.py, contrastive_audit.py, contrastive_split_benchmark.py, contrastive_rank_exit.py; RATIO_RULE identica (readout r.22); evaluation/coverage/calibration/activation/resources del piano identici.
- Audit report senza SLURM job id (contrastive_audit.py nessun riferimento SLURM): job_id dell'accounting auto-dichiarato; expected_job_id mai passato.

## Ripresa (20/11 03:02)
- Riprendo dai Progressi 5; r7clone pulito a 0efa4d45. Restano: R-1, R-3, N-1, N-2, N-6; soglie in decisioni prima delle misure; coerenza ID/checkout/fasi; punta del ramo logcontrast.
## Progressi 6 (20/11 ~03:30)
- Riconfermato R-2 consumatori non testati: mutazione (via validate_review_binding da contrastive_job.py:70-72 e contrastive_run.py:181-184) -> 89 passed (job/run/release/targets_readout). Clone ripristinato.
- (a) Punta del ramo logcontrast: nel deposito oggetti (alternates del canonico) l'unico discendente di db66b507 e' 3cc74ad6->8e38eb71->faa4a8d7 (11:57:50 del 19/11); nessun commit con genitore faa4a8d7; nessun oggetto dopo le 14:04:46 del 19/11. Punta reale del remoto non verificabile (niente fetch).
- (a) faa4a8d7 voce decisioni.md «Proposta per Simone ... training e invii sospesi» e registro «Preventivo e release restano da approvare»: NON presenti nel decisioni.md di fbfde797 (grep 0). fbfde797 decisioni:8280 «attua la decisione scientifica faa4a8d7».
- (a) plan.json fbfde797:321 "Simone authorizes launches after convergence" (D1-D3 del 16/11) ma :328 target_windows "Prepare and present new costs before any launch"; preflight :258-265 "historical_reserved_cap" 14/129,33 e "historical_unspent_not_new_authorization"; :315-319 required_before_first_submission senza Simone. contrastive_run.py:160-161 pin di decisione/budget solo per training; preparazione (calibrazione/benchmark) solo APPROVED_FOR_PREPARATION + binding review + accounting audit.
- (a) 17/11 decisioni:7846 Simone «direi che se tu sei a posto per questo giro lanci tu, ti do fiducia» con «Tetti invariati: 14 GPU-ore e 129,3333 core-ore» = giro 25/50.
- (d) submit_job.py:190-195 code_sha_from solo EXPECTED_SHA=: prontuario r.176-177 falso (codice_sha vuoto nel ledger per il checkout; lo SHA resta nel comando).
- (d) Contabilita' diagnostici: 4*(154+289+279)/3600=0,802222 GPUh; (4*35+32*722)/3600=6,456667 coreh: corretti.
## Progressi 7 (20/11 ~03:45)
- R-4 consumatori: riconfermato, via le 3 chiamate validate_audit_accounting (job:81, run:195, release:242) -> 94 passed (job/run/release/targets_readout/historical_callsites). Clone ripristinato.
- Copia review Claude in fbfde797 = blob c25d4f6f identico a 94f50942; cartella _evidence (320 file) non copiata: link relativo rotto (NOTA).
- Copia faa4a8d7 (blob 5bdb0d15): i link a common_results_20261119/ (128 file) e a contrastive_common_diagnostics_readout_20261118.md mancano in fbfde797 (presenti solo su faa4a8d7) (NOTA).
- CLAUDE.md fbfde797:177 «protocolli, bracci, criteri, letture e piani li propone GPT per la firma di Simone»; :267-269 regola del budget del passo. faa4a8d7 cambia la prova decisiva («Cambia la prova decisiva», «Cambia il punto di partenza del confronto"): criterio nuovo senza firma.
- make_job (release.py:236) scrive APPROVED_FOR_PREFLIGHT dal solo binding; release APPROVED_FOR_PREPARATION (job.py:134, run.py:168) senza builder nel codice; nessun pin di decisione/budget per calibrazione/benchmark.
- N-1, N-2, N-4 (test 77-97), N-5 (7 log storici pinnati), N-6, N-7: risposte corrispondenti al codice. N-5: ricevuta = sola corsa pytest; bash -n solo dichiarato in verification.json; piani rigenerati senza ricevuta (fatto vero: --check exit 0).
- R-1 ripresa: test usa torch.multiprocessing.spawn (due processi nuovi) a ranks=1: affermazione vera ma limitata.
## Progressi 8 (20/11 ~04:00)
- Generatore: science_plan(plan@438c8d65) riproduce byte per byte plan.json e preflight(…) il preflight_plan.json di fbfde797 (verifica positiva). Ma --check confronta il file con se stesso rigenerato (idempotenza) e test_contrastive_targets_plan.py:38 confronta evaluation con lo stesso file: tautologico (evaluation manomessa preservata dal generatore). Invarianza di evaluation vera solo per diff esterno (NOTA).
- Doc: tutte le modifiche a CLAUDE.md/docs sono solo aggiunte (numstat x 0).
- delivery_pin riverificato (source/plan/preflight/validation, diffstat 46/5524/100).
- Nessun «Simone» con approvazione nelle aggiunte di 3cc74ad6, 8e38eb71, faa4a8d7, fbfde797 (git show -U0 | grep).
- CLAUDE.md fbfde797:183 «nessun job senza il via di Simone»; :150-157 regola 5 GPU-ore e «Non frazionare artificialmente»; :177 firma di Simone sui criteri.
- Prontuario r.159-161 «L'operatore usa il binding ... costruire il job ... Poi committa/pusha metadati» vs prompt Claude r.58 «Dopo la convergenza GPT prepara per 3-launcher la prima release eseguibile»: chi scrive i metadati di release ambiguo (NOTA).
- Bozza finale in preparazione.
## Sintesi finale (20/11 ~04:10)
- BLOCCANTE G-1: proposta faa4a8d7 e tetto 18 GPUh/169,6 coreh senza firma/decisione registrata di Simone; prerequisito «present resource proposal to Simone» tolto dal preflight (438c8d65:191 -> fbfde797:315-319); il solo blocco di Claude rende eseguibile la preparazione (pin decisione/budget solo per training, contrastive_run.py:160-161). Fix: decisione di Simone registrata prima del blocco/primo invio.
- MINORE G-2: CLAUDE.md «questa chat» reintrodotto (contro d9020c20) e precedenza/esclusiva non limitate alla linea 3.
- MINORE G-3: integrazione R-2/R-4 nei consumatori senza test (89 e 94 passed con le chiamate rimosse).
- NOTE: SHA del checkout non dal CAMPAIGN_SHA (submit_job.py:190-195); ricevuta locale solo pytest, --check e test del piano tautologici; job_id SLURM auto-dichiarato; link della copia faa4a8d7 e voce decisioni di faa4a8d7 assenti nel ramo; writer dei metadati di release ambiguo; blocco senza ambito; refusi/parafrasi.
- Prova binding (r7tests/test_r7_binding_provenance.py) rieseguita: 4 passed. Pulizia: r7clone, r7, r7_pt2 cancellati; r7tests conservato come prova.
