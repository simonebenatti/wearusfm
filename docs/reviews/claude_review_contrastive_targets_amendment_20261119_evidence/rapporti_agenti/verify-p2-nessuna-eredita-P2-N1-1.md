# Verifica scettica P2-N1 (scettico n. 1)

Rilievo: i test storici sulla ricevuta non esercitano il codice corrente (NOTA).
Copia: 438c8d65. Lavoro con un clone mio in tmp/vp2n1s1 (comandi miei).

## Letture
- tests/cpu/test_contrastive_historical_callsites.py:117-163 (438c8d65): run_frozen su 8905e346 (git archive di src, scripts, contrastive_comparison_20261116); nessuna chiamata al codice corrente. Nel diff e388a1de..0efdc29c il test passa da monkeypatch sul tree corrente (guard_calls == ['calibration_receipt'] verificato sul codice corrente) a run_frozen: copertura del punto di chiamata corrente persa nella consegna.
- frozen_campaign_support.py:16-31: archive() usa `git archive <sha>` (oggetti del commit, non il working tree) e PYTHONPATH/sys.path del solo archivio.
- contrastive_prepare.py:375-391: calibration_report(run_dir) alla r.381 PRIMA della guardia ricevuta r.383-388; confronto completo r.389-390 (include source_sha256).
- contrastive_source_compat.py:161-162: il controllo dell'inventario corrente e' indipendente dal tipo ed e' prima dei controlli per tipo: sul tree corrente la guardia rifiuta qualunque kind.
- Emendamento r.57 e registro (voce 0efdc29c): «I test delle ricevute storiche usano reader congelati» dichiarato.

## Prove riprodotte (clone mio tmp/vp2n1s1/mut a 438c8d65; script tmp/vp2n1s1/{mut.py,run.sh,probe_messages.py,probe_branch.py}; 21 file test_contrastive_*.py + test_rank_contrast.py)
- NONE: 337 passed (102,6 s).
- F (via validate_historical_evidence("calibration_receipt") da prepare.py:384-385, sostituita da pass): 337 passed. Riprodotto da solo (il rapporto originale aveva MF da sola solo sui 164 casi).
- H (via il confronto prepare.py:389-390): 1 failed, 336 passed; fallisce solo test_receipt_is_recomputed_not_trusted_and_all_arms_share_it. I 4 test_modified_historical_receipt_is_rejected restano verdi.
- ZINSTR (strumentazione che non cambia il comportamento, scrive un marcatore quando il ramo r.383 viene eseguito): 337 passed, nessun marcatore. Il ramo della guardia non viene mai eseguito dalla suite contrastiva corrente.
- Controllo positivo (probe_branch.py): run sintetico fixture_run 100/125/v3 con sorgente corrente, ricevuta onesta accettata; ricevuta con source_sha256='0'*64: senza mutazione "historical evidence compatibility: current execution source inventory/pins differ"; con ZINSTR 2 marcatori (ramo raggiungibile); con F "calibration receipt differs from authenticated recalculation". Quindi F e' un mutante equivalente per accetta/rifiuta: cambia solo il messaggio. Un test con match sul messaggio della guardia (la correzione proposta) lo ucciderebbe.
- probe_messages.py sul tree corrente: ricevuta storica intatta e le 4 varianti (weight/selection/reader/source) -> tutte "calibration run config differs from the frozen 512-batch recipe", guard_calls=[] (la guardia non viene mai chiamata). Guardia diretta su ricevuta e run storici -> "current execution source inventory/pins differ".
- 12 run originali (M0 + 11 mutazioni): test_pinned... mai fra i FAILED. E' cosi' per costruzione: git archive legge il commit 8905e346, non il working tree mutato.

## Motivi cercati per confutare (non sufficienti)
- Spostamento al reader congelato dichiarato: emendamento r.57 «I lettori congelati restano il modo di riprodurre le analisi vecchie»; registro 0efdc29c «I test delle ricevute storiche usano reader congelati». Era anche obbligato: a e388a1de la vecchia versione sul tree corrente FALLIVA gia' ("current execution source inventory/pins differ"; 3 failed, 5 passed nei due file storici), perche' il merge di main aveva cambiato l'inventario dei sorgenti. Il commento «Current 100/125 sources must not inherit that exception» si legge come motivazione, non come promessa di verifica.
- La non-eredita' e' verificata altrove sul tree corrente: test_historical_audit_does_not_qualify_new_sources (controllo d'inventario indipendente dal tipo, compat.py:161-162) e test_unmodified_historical_calibration_does_not_qualify_new_targets (match 'calibration run config differs'); la funzione-guardia per la ricevuta e' coperta dalla scena sintetica (test_any_historical_evidence_mutation_is_rejected[receipt, reader_provenance]). Non e' coperto solo il punto di chiamata in authenticate_calibration.
- Questo attenua la parte del rilievo sul commento, ma non confuta il nucleo: nessun test corrente esegue la guardia r.383-388, e i 4 casi parametrizzati ora verificano solo la barriera di ricetta.

## Esito
NON CONFUTATO. Gravita' NOTA confermata: la guardia e' ridondante sul tree corrente (mutante equivalente: senza di essa la ricevuta e' rifiutata dal confronto) e non cambia cosa viene accettato; nessun effetto su ricetta, evidenze, confronti o tetti.
Pulizia: clone tmp/vp2n1s1/mut rimosso, basetemp rimossi; copia git 438c8d65 pulita (0 righe in git status).
