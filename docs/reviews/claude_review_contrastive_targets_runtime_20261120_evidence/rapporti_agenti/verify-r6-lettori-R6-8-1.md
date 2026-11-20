# verify-r6-lettori-R6-8-1 (scettico n. 1, rilievo R6-8, NOTA)

Avvio: lettura codice a fbfde797.

## Progressi
- run.py (src invariato nel diff): t0=1613, _accumulate_terms (lookup globale del modulo) 1631, t_step_s=time.time()-t0 a 1731; save()/eval dopo il record (fuori da t_step_s, dentro elapsed_s). elapsed_s = time.time()-t_start (1243 -> 1821): include avvio e tutto il ciclo.
- use_terms = ctx.active or accum>1 or contrastive (1314): con il contrastivo _accumulate_terms e' sempre usato.
- contrastive_run.py main (fbfde797): benchmark CN/CR (full e reload) -> trace_resume_updates; benchmark full C0 -> observe_control. Training: nessun osservatore.
- observe_control patcha trainer._accumulate_terms: a 128 e 256 measure() esegue 6 casi (modello nuovo, DDP nuovo, impronte sha256, backward, copia gradienti su CPU) PRIMA del backward vivo -> dentro t_step_s. finish() dopo R.train -> fuori da elapsed_s.
- Diagnostica gradiente (grad_every=100, grad_steps 256/512/..., denso 513-576): 128 = record "normale", 256 = record "diagnostico" (e i 6 cloni a 256 calcolano anch'essi le 3 norme globali).
- measured_allocation (readout 65-92) invariata nel diff (gia' a 438c8d65); docstring "Conservative ... explicitly including diagnostic overhead", max(elapsed/576, ...).
- Consumo: contrastive_run.py validate_budget (training): somma <= approved_training_gpu_hours; 4*time_limit/3600 <= by_phase[arm] (tetto superiore, non durata imposta).
- Prova CPU (rev_runtime/tmp/sk1_r68/repro_timing.py, trainer reale, 1 processo, 12 update): i sei P._measure_case reali chiamati da un _accumulate_terms patchato come in observe_control all'update 6 -> observer_s 5,130 s; t_step_s[6] 5,839 s contro mediana baseline ~0,64 s (differenza 5,194 s = osservatore); elapsed +4,47 s; total/grad_norm identici bit per bit fra baseline e run osservato (solo il tempo cambia). trace_resume_updates reale (fork 4) gira dentro gli stessi update (tracce 5..12, boundary a 5).
- Costo host per il modello di produzione T100-G (cpu_side_cost.py, 28.559.133 parametri, 109 MiB): costruzione 0,67 s, state_sha256 0,10 s, ...: ~1,0 s per caso, ~6,6 s per punto, ~13 s per i due punti, SENZA backward GPU, DDP, D2H e gather: limite inferiore.
- Effetto con measured_allocation reale (projection_effect.py; ipotesi dichiarate: normale 1,10 s, diagnostico 1,90 s, 50 s fuori dal ciclo, 71 update diagnostici; 128 normale, 256 diagnostico): max() sceglie elapsed/576 (1,2854 s) contro media normale+premio 1,116 s. Osservatore C0 20-40 s -> +2,7-5,4% per update, +1,3-2,3 GPUh su 43 GPUh per C0; tracce CN/CR ~2,6 s -> +0,33 GPUh (arrotondamento a 300 s).
- Documenti: prontuario, risposta, piani, decisioni, decisione scientifica: nessuna menzione dell'overhead degli osservatori nella proiezione. La docstring "explicitly including diagnostic overhead" e' di 438c8d65 (prima degli osservatori) e il suo premio "at the future frequency" riguarda solo il pannello del gradiente.
- Motivi contro la gravita': direzione conservativa; la proiezione e' solo tetto superiore (time_limit <= by_phase, somma <= cap fissato DOPO la lettura, budget_decision_pin); nessun cancello automatico su tempo; padding gia' voluto maggiore (pannello denso a frequenza di benchmark e avvio ~15%); observer_seconds_including_setup e backward_seconds registrati nel rapporto di parita' (le tracce CN/CR no).
- A margine (fuori da R6-8, solo lettura del codice): anche full_batch_gpu_peak_gib di C0 e' il picco cumulativo del processo e include il clone della parita' (peak_scope dichiarato nel rapporto), quindi conservativo come il tempo.

## Esito
Rilievo NON confutato: meccanismo riprodotto, descrizione e riferimenti corretti, gravita' NOTA adeguata (nessun cancello falsato, effetto conservativo di pochi punti percentuali per C0, lacuna di documentazione reale). Copia di sola lettura pulita a 0efa4d45; cartella dati run1 cancellata.
