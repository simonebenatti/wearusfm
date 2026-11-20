# Scettico n. 1 — rilievo R2-N1 (r2-ripresa), fbfde797

Inizio: 2026-11-20

## Lettura del codice
- scripts/contrastive_target_resume.py:320 `boundary_errors`/`support_missing` sono liste uniche per tutte le fasi e i rank; :364-371 vi aggiungono voci con `phase`; :394 idem per il supporto.
- :412 `pair["passed"] = pair["passed"] and not differences and not boundary_errors and not support_missing`: la congiunzione usa le liste globali, non le voci delle due fasi della coppia. Confermato per lettura.
- Consumatori di `pairs[...]["passed"]`: solo :415 (`passed = all(...)`). Il readout (contrastive_benchmark_readout.py:487-494) usa solo `check["status"]` e `check["passed"]`; nessun altro script legge il passed per coppia (git grep). La ricevuta pubblicata contiene pero' l'intero `check` con `pairs` (readout:490).
- Prontuario (docs/contrastive_targets_execution_20261119.md:58-71) e piano (plan.json:403-433): il confine e' un invariante separato; "Su ognuna [coppia] vale la regola storica" elenca solo parametri 1%, buffer/contatori/metadati/stream/RNG/coda esatti, tensori mobili 1%. Nessuna dichiarazione che il passed per coppia includa il confine delle altre fasi. Il commento :416 giustifica solo il supporto mancante come "missing comparison".

## Riproduzione indipendente (trainer CPU reale, 1 rank, fork 2, fine 4, braccio CR)
Script mio: rev_runtime/tmp/sk1_r2n1/test_sk1_r2n1.py (fixture propria, ricalcata su common_paths, piu' un reload_b reale con `exp_avg` azzerato subito dopo `opt.load_state_dict`, patch solo nel processo figlio). Copia di riferimento letta e non modificata (status pulito, HEAD 0efa4d45; contrastive_target_resume.py identico a fbfde797). Esito: 5 passed in 16,61 s; JSON in rev_runtime/tmp/sk1_r2n1/case*.json; basetemp cancellato.
- case0 base: PASSED, tre coppie passed, distanze 0,0.
- case1 Adam exp_avg=0 nel solo reload_b: FAILED_REQUIRES_REVIEW; boundary solo (reload_b, 0, optimizer_moments); continuous__reload_a passed=False, 0 motivi, 0 differenze, distanze 0,0/0,0; le due coppie con reload_b falliscono con 197 motivi (studente 28,9%, teacher 19,9%). Riproduce il rilievo.
- case2 confine teacher alterato nella sola traccia di reload_b (nessuna differenza numerica): tutte e tre le coppie passed=False con 0 motivi e 0 differenze; nessuna coppia porta mai il motivo di confine, che sta solo in boundary_mismatches.
- case3 confine alterato nel solo continuo (caso realistico: stato salvato diverso dallo stato vivo): reload_a__reload_b passed=False con 0 motivi/differenze, benche' A e B coincidano.
- case4 supporto RnC nullo all'update 4 del solo reload_b (traccia e metriche): continuous__reload_a passed=False con 0 motivi/differenze; stato FAILED_REQUIRES_REVIEW per le differenze nelle coppie con reload_b.

## Tentativi di confutazione
- Nessuna dichiarazione (prontuario :58-71, piano :403-433, decisione faa4a8d7 punto 2, risposta R-1) che il passed per coppia includa il confine o il supporto delle altre fasi; il commento :416 giustifica solo il supporto mancante come confronto mancante. Nessun test codifica la semantica congiunta.
- Effetto sui cancelli: nullo. `passed` globale (:415) e `status` (:417-419) restano corretti anche con la correzione proposta (ogni fase sta in due coppie); il readout (:487-494) usa solo status/passed; nessun altro consumatore. Il campo per coppia finisce pero' nella ricevuta pubblicata e autenticata (readout:490, :537-539), quindi chi legge la tabella vede una coppia sana marcata fallita senza motivo.
- Per il caso scientificamente rilevante della decisione ("eccesso continuo/reload con reload concordi") non ci sono errori di confine: il passed per coppia localizza correttamente. La confusione nasce solo quando la prova e' gia' fallita e boundary_mismatches indica fase, rank e campo.

## Esito
Non confutato. Riprodotto in tutti i punti; gravita' NOTA (chiarezza della ricevuta, nessun cancello ne' decisione dipende dal valore per coppia). Osservazione: anche dopo la correzione proposta le coppie colpite resterebbero senza motivo di confine; conviene riportare per coppia anche le voci di confine/supporto delle sue due fasi.
