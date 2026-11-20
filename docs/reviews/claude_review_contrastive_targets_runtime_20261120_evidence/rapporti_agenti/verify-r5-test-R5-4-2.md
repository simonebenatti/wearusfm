# Verifica scettica n. 2 del rilievo R5-4 (r5-test)

Avvio: 2026-11-20. Copia di sola lettura a 0efa4d45 (parity.py e test invariati fra fbfde797 e 0efa4d45: `git diff --stat fbfde797 0efa4d45` tocca solo delivery_pin.json).

## Lettura del codice (fbfde797)
- Numeri di riga del rilievo corretti: parity.py:215-216 (identita' fra rank), 227-228 (schema/profilo), 252-253 (cfg cambiato), 255-256 (cfg di produzione), 257-258 (punto duplicato), 262-263 (batch.index), 294 (live_state_restored), 300-301 (arresto).
- Moduli che importano parity: contrastive_run.py:356 (observe_control, solo nel ramo CUDA a 4 rank) e contrastive_benchmark_readout.py:507 (validate_parity_report). Test che lo toccano: test_contrastive_target_parity.py (diretto) e test_contrastive_targets_readout.py (fixture `point` e read_preflight).

## Riproduzione indipendente (driver mio: tmp/sk2r54_scripts/mut.py, clone tmp/sk2r54_clone a 0efa4d45)
- Base sui file mirati (test_contrastive_target_parity.py + test_contrastive_targets_readout.py): 52 passed.
- Controlli del driver UCCISI: `if hashes and len(set(hashes)) != 1 and not missing:` -> `if False:` => 2 failed (test_gate_requires_both..., readout[gradient]); assess_point:171 live_state_restored -> `if False:` => 1 failed (test_gates_check...[live]).
- Mutazioni del rilievo, una per volta, file mirati: C1c (215), C1d (294 = True), C1t (300), C1f (262), C1l (255), C1m (252), C1n (257), C1q (227) => tutte "52 passed", albero pulito dopo ciascuna.
- Le otto mutazioni INSIEME sull'intero insieme della ricevuta (29 file, PYTHONPATH src:tests/cpu:scripts, --basetemp proprio): **465 passed in 111.03s**, diffstat 8+/8-, albero pulito. RIPRODOTTO.
- Mascheramento escluso: parity.py e' eseguito solo da test_contrastive_target_parity.py e test_contrastive_targets_readout.py (unico che scrive zero_weight_parity.json); observe_control e' importato solo nel ramo CUDA di contrastive_run.main. Nessun altro file di test (grep su tests/cpu) lo nomina.

## Non equivalenza (sonde mie, tmp/sk2r54_scripts/probe.py e probe_mut.py: input che isola ciascuna guardia)
Codice originale -> mutante (clone temporaneo, una mutazione per volta, albero pulito dopo ognuna):
- C1c: rank 1 con student_sha256 cambiato in modo coerente in live_before e nei sei casi -> "FAILED model/teacher/queue/global identities differ across ranks" -> mutante "PASSED"; idem global_ids_sha256 del rank 1.
- C1q: report con profile "other" o schema 2 e due punti validi -> INCONCLUSIVE -> mutante PASSED.
- C1d: _measure_case simulato che altera un parametro vivo -> RuntimeError "parity probe modified the live control state", restored=False -> mutante: nessun arresto, restored=True.
- C1t: stesso input -> mutante: nessun arresto, restored=False, gate FAILED (assess_point:171 fa da rete).
- C1f: batch.index 126 al passo 127 -> ValueError "live batch is not immediately before..." -> mutante prosegue.
- C1l: world_size 2 oppure accum_steps 2 -> ValueError "parity requires production C0..." -> mutante prosegue.
- C1m: secondo punto (255) con lr cambiato -> ValueError "configuration changed" -> mutante registra 2 punti.
- C1n: due misure all'update 128 -> ValueError "duplicate parity control update" -> mutante registra 2 punti.
Le guardie funzionano sul codice originale e sono uccidibili da test semplici: la lacuna e' reale, non sono mutanti equivalenti.

## Motivi cercati per confutare (controlli altrove, percorsi non usati, dichiarazioni)
- Nessun altro test: grep di live_state_restored, "identities differ", "configuration changed", "duplicate parity", "immediately before", "production C0 batch32" in tests/, docs/, risultati: solo la fixture (True), la variante [live] del gate (assess_point:171) e l'asserzione positiva dell'osservatore (test_contrastive_target_parity.py:198). Nessun controllo in contrastive_cuda_check/job/release/sbatch.
- Prontuario e risposta non dichiarano queste guardie fuori test: prontuario:36 "L'osservatore verifica che pesi, teacher, coda, gradienti e RNG ... restino intatti", :40 "byte-identici entro e tra bracci, e fra rank"; risposta R-1 "Test con backward CPU reali e mutazioni delle evidenze" (non promette ogni guardia, non la contraddice).
- Reti di produzione che riducono il peso pratico (non confutano la mancanza del test):
  - C1f: trainer run.py:1617 rifiuta batch.index != step prima di prepare(batch, step) che fissa runtime.step.
  - C1l: validate_recipe (contrastive_run.py:50) impone il config del baseline pinnato f9d03184 (batch 32, accum 1, amp_bf16 True) e contrastive_run.py:329 world_size 4.
  - C1m: un solo cfg per processo passato a ogni _accumulate_terms; irraggiungibile in produzione.
  - C1n: una sola misura per passo (observer.active, pop di captured) e validate_parity_report:229 vuole update [128,256] (doppione -> INCONCLUSIVE).
  - C1q: finish() scrive sempre schema 1 e PROFILE; il readout lega source_sha256 (readout.py:512) e assess_point controlla i flag per caso.
  - C1t: assess_point:171-172 rende FAILED un rank non ripristinato (variante [live] testata: il mio controllo la uccide); il readout lo porta a FAILED_REQUIRES_REVIEW.
  - Senza rete: C1c (stato replicato coerente ma diverso fra rank: i gradienti ridotti dal DDP restano uguali, quindi solo 215-216 lo vede) e C1d (calcolo del flag: forzato True, ne' l'arresto ne' il gate vedono un'alterazione del processo vivo).
- Calcolo di C1d corretto oggi: state_sha256 (pairing.py:13-47) porta su CPU e include dtype/shape, quindi copia CPU e stato CUDA coincidono; runtime.state_dict()["queue"] == runtime.queue.state_dict() (contrastive_objectives.py:179).
- I test non entrano nel source_digest (contrastive_prepare.py:192-201): la correzione resta nei test, come M-1 di 94f5094 (stessa classe: barriere senza test, MINORE), con nuova ricevuta locale ma senza nuovi pin di sorgente.
- Rettifica di riga: il legame del readout sul report di parita' e' contrastive_benchmark_readout.py:511-514 (non 512 soltanto); run.py:1617 (batch.index) e 1629 (prepare(batch, step)), contrastive_run.py:50 e 329 verificati con grep -n.
- Singole sull'intero insieme (stesso driver): C1c -> 465 passed in 153.48s; C1d -> 465 passed in 173.43s; albero pulito.

## Esito
- **refuted = false**, gravita' **MINORE** confermata.
- Il rilievo e' esatto nei fatti e nelle righe: le otto guardie funzionano sul codice di fbfde797, ciascuna e' uccidibile da un input semplice, ma nessun test della ricevuta (465) ne isola una. Byte-identita' dei gradienti coperta (controllo ucciso da 2 test).
- Peso reale: sei guardie hanno una rete (C1f, C1l, C1m, C1n, C1q nel percorso di produzione; C1t nel gate testato). Restano senza rete e senza test C1c (identita' replicata fra rank) e C1d (calcolo del ripristino del processo vivo). Nessun esito sbagliato dimostrato col codice attuale, quindi non BLOCCANTE; e' la stessa classe di M-1 di 94f5094, classificata MINORE, e la correzione sta solo nei test.
- Pulizia: clone tmp/sk2r54_clone e directory di lavoro delle sonde cancellati; restano solo gli script e i risultati in tmp/sk2r54_scripts (pochi KB). Copia di sola lettura intatta: 0efa4d45, `git status` vuoto.
