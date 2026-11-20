# Scettico n. 1 su R5-4 (r5-test): parita', guardie dell'osservatore e del rapporto senza test

Avvio: 2026-11-20. Copia 0efa4d45 (contenuti fbfde797; parity.py identico fra i due commit, nuovo in fbfde797).
Rilievo: MINORE. Guardie di scripts/contrastive_target_parity.py senza test che le isolino.

## Lettura del codice (fbfde797), righe citate
- 215-216 identita' replicate fra rank (assess_point); 227-228 schema/profilo (validate_parity_report);
  252-253 cfg cambiato fra punti; 255-256 cfg di produzione; 257-258 punto duplicato; 262-263 batch.index;
  294 live_state_restored; 300-301 RuntimeError se non ripristinato. Righe coincidono col rilievo.
- Usi in produzione: contrastive_run.py:355-357 (observe_control nel full C0 benchmark);
  contrastive_benchmark_readout.py:503-521 (validate_parity_report sul rapporto del C0).
- Test: tests/cpu/test_contrastive_target_parity.py (unico file che chiama assess_point/measure);
  test_contrastive_targets_readout.py usa point() con schema 1 e profilo corretto.

## Riproduzione (script mio tmp/sk1_r54_scripts/mut.py, clone tmp/sk1_r54_clone a 0efa4d45)
Sostituzioni testuali esatte (1 occorrenza ciascuna, verificata), pytest, ripristino via git (albero pulito dopo ogni prova).
File mirati: test_contrastive_target_parity.py, test_contrastive_targets_readout.py,
test_contrastive_benchmark_readout.py, test_contrastive_run.py.
- BASE: 92 passed in 13.88s.
- Controlli (devono morire): live_state_restored=False (294) -> 1 failed (test_observer_restores...);
  byte-identita' 217 -> 2 failed (test_gate_requires_both..., readout[gradient]). Il banco distingue.
- Mutazioni del rilievo, una per volta: 215 `if False:` (identita' fra rank) 92 passed; 294 `= True` 92 passed;
  300 `if False:` 92 passed; 262 batch.index 92 passed; 255 cfg produzione 92 passed; 252 cfg cambiato 92 passed;
  257 punto duplicato 92 passed; 227 schema/profilo 92 passed.
- Varianti parziali (regressioni plausibili): 227 solo schema (profilo ignorato) 92 passed; 255 senza world_size 92 passed;
  182 global_ids_sha256 tolto dalla tupla replicata 92 passed. TUTTE SOPRAVVIVONO.
- Insieme della ricevuta (29 file di validation_final) con le OTTO guardie del rilievo disattivate INSIEME
  (215+294+300+262+255+252+257+227): rc 0, "465 passed in 122.59s"; albero del clone pulito dopo il ripristino.
  Fatto del rilievo CONFERMATO (per guardie che tolgono solo controlli, il passaggio combinato implica quello singolo).

## Sonde negative mie (tmp/sk1_r54_scripts/test_r54_probes.py, fuori dalla copia; banco dell'osservatore = quello di
## test_observer_restores..., con _measure_case finto che altera lo stato vivo o restituisce case())
- Codice NON mutato: 18 passed in 5.70s -> ogni guardia funziona oggi: 215 (rank 1 coerente entro il rank ma diverso
  da rank 0 in student/teacher/queue/coverage/global_ids -> FAILED con il solo errore "identities differ across ranks");
  227 (schema 2 / profilo "bf16_auto" -> INCONCLUSIVE); 294+300 (parametro, teacher, grad, coverage vivi alterati ->
  live_state_restored False e RuntimeError); 262 (batch.index 126); 255 (batch16, accum2, fp32, world_size 2);
  252 (secondo punto 256 con lr diverso); 257 (128 due volte).
- Ogni mutazione e' uccisa SOLO dalle sonde della sua guardia: 215 -> 5 failed; 294 -> 4; 300 -> 4; 262 -> 1; 255 -> 4;
  252 -> 1; 257 -> 1; 227 -> 2; varianti parziali profile_only -> 1, ws_only -> 1, global_ids_only -> 1. BASE 18 passed.
  La correzione proposta dal rilievo e' fattibile e sufficiente (la mia sonda la realizza).

## Ricerca di motivi per cui non e' un difetto (ridondanze di produzione)
- 262 batch.index: IRRAGGIUNGIBILE in produzione. run.py:1617-1618 solleva se batch.index != step; prepare imposta
  self.step = step (contrastive_objectives.py:208); observed_prepare cattura lo stesso batch nella stessa iterazione e
  _accumulate_terms e' sempre usato (run.py:1314 use_terms = ctx.active or ... or contrastive is not None).
- 255 cfg di produzione: RIDONDANTE. contrastive_run.validate_recipe (:50) esige config_to_dict == ricetta congelata
  (batch 32, accum 1, amp_bf16 True: baseline f9d03184..., prepare.py:118); contrastive_run richiede world_size 4 (:329);
  readout _run_evidence (:205-206) esige cfg == atteso, world_size 4, batch_per_process 8, accum 1; assess_point 161/169/201/186.
- 252 cfg cambiato: IRRAGGIUNGIBILE di fatto: RunConfig e' @dataclass(frozen=True) (run.py:97) e lo stesso oggetto
  passa a ogni passo; il readout lega comunque config_sha256 al config.json canonico del C0 (readout:509-514).
- 257 duplicato: IRRAGGIUNGIBILE (un update per iterazione; failure -> break); in piu' la cattura (259-261) fermerebbe.
- 227 schema/profilo: il rapporto e' scritto da finish() con costanti e legato dal readout a source_sha256; i flag
  effettivi sono controllati caso per caso (186, assert_numeric_flags) e nelle 28 ricevute del readout. Ridondante.
- 300-301 arresto: se 294 calcola False, il gate fallisce comunque (171, testato da test_gates...[live]): nessun falso
  PASSED, solo calcolo sprecato.
- 294 rilevazione: UNICO rivelatore della modifica dello stato vivo (171 e 300 ne dipendono). Il ripristino in se' e'
  testato (RNG/coda/grad in test_observer_restores...; batch/micro stabili su sei backward reali in test_real_cpu...),
  quindi un danno richiede due difetti insieme.
- 215 identita' fra rank: per lo studente e' ridondante con 178 in produzione, perche' il DDP del clone sovrascrive i
  parametri dei rank >0 con quelli del rank 0 prima di `before` (prova mia tmp/sk1_r54_scripts/ddp_init_sync.py, gloo
  2 processi, torch 2.14: rank 1 changed_by_wrap True, equals_rank0_before True). Per teacher, coda, coverage e
  global_ids e' l'UNICO controllo fra rank (i gradienti ridotti sono uguali su tutti i rank per costruzione, 217 non vede).
- Documenti: la risposta (R-1) dice "Test con backward CPU reali e mutazioni delle evidenze"; il prontuario (29-42)
  descrive le verifiche, non dichiara test negativi ne' limiti per queste guardie. Nessuna dichiarazione che esoneri.

## Esito
- refuted = false. Fatto riprodotto con comandi miei (8/8 guardie sopravvivono a 465/465; anche 3 varianti parziali).
- Gravita': MINORE confermata, ma solo per 215 (teacher/coda/coverage/global_ids) e 294: guardie uniche, corrette oggi
  (sonde verdi), senza protezione dalle regressioni; un danno richiede una regressione e una desincronizzazione o
  modifica reale insieme. Per 262, 255, 252, 257, 227 e 300 la mancanza di test e' da NOTA: irraggiungibili o
  ridondanti in produzione. Nessun BLOCCANTE: nessun cammino del codice odierno accetta un'evidenza non valida.

## Non verificato
- Nessuna esecuzione CUDA/NCCL o Leonardo (fuori perimetro): comportamento di 294 con autocast/CUDA e DDP a 4 rank;
  l'init sync del DDP e' provato su torch 2.14 gloo, su 2.2 (Leonardo) e' dedotto (comportamento storico del DDP).
- Mutazioni rieseguite una per una sull'intero insieme di 465: no (solo combinate; per mutazioni che tolgono controlli
  il passaggio combinato implica quello singolo). Ognuna singola sui 4 file mirati (92 test): sopravvive.

## Pulizia
- Clone sk1_r54_clone, copia delle sonde e basetemp cancellati; restano solo script, sonde e risultati in
  rev_runtime/tmp/sk1_r54_scripts (44 KB). Copia di sola lettura pulita a 0efa4d45. Verifica conclusa.
