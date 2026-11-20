# verify-r6-lettori-R6-2-1 (scettico n. 1, rilievo R6-2: read_preflight sostituisce le impronte del confronto di ripresa)

Avvio: lettura del codice a fbfde797 (readout read_preflight, target_resume compare_three_paths, authenticate_preflight, prontuario, piani, release).

## Riproduzione (04:10, sonda propria rev_runtime/tmp/v62_probe/test_v62_evidence.py, 2 passati in 26,7 s; copia pulita dopo)
- Codice a fbfde797: target_resume.py:347-361 costruisce evidence[phase] con checkpoint/summary/config/metrics, start.json (reload), origin_sessions, trace-{rank}.jsonl; :423 lo restituisce come evidence_sha256. readout.py:487-489 lo sostituisce con reload_pins (solo _run_evidence dei due reload: config/summary/metrics/checkpoint/origin_sessions/numeric_profile.rank*).
- Sonda unitaria (fixture readout_fixture del repo, compare simulato con tracce): in ricevuta restano solo reload_a/reload_b, nessuna traccia ne' start.json (CN e CR).
- Sonda integrata: compare_three_paths REALE sul fork CPU (fixture common_paths, ranks=1): evidence per fase con trace-0.jsonl e start.json; passata dentro read_preflight: resume.evidence_sha256 = {reload_a, reload_b} senza tracce/start.json.
- Conseguenza dimostrata: cambio coerente di batch_sha256 all'ultimo update nelle tre tracce -> compare resta PASSED, unica chiave diversa evidence_sha256; authenticate_preflight della ricevuta scritta prima del cambio: ACCETTATA. Con le impronte conservate il confronto actual != receipt l'avrebbe rilevato.
- Storico 438c8d65: compare_resume non restituiva evidence_sha256 (readout:381-382 lo aggiungeva): la riga e' un residuo, come detto nel rilievo.

## Tentativi di confutazione (tutti falliti)
- Pin altrove? No: nessun consumatore di resume.evidence_sha256 (git grep a fbfde797: solo le righe scrittrici readout:471/489 e target_resume:423); summary.json non contiene impronte delle tracce; il job readout (contrastive_job.py:89-91) scrive solo la ricevuta --out; il prontuario non prevede un manifest di raccolta che pinni le tracce.
- reload_pins aggiunge qualcosa di unico? No: config/summary/metrics/checkpoint/origin dei reload sono anche nell'evidenza di compare_three_paths; i numeric_profile.rank*.json dei sette run sono gia' in numeric_context_across_runs.profile_receipts_sha256 (readout:357-367, 387). La sostituzione e' una perdita netta di 24 trace-{rank}.jsonl + 4 start.json (2 bracci x 3 percorsi x 4 rank; 2 bracci x 2 reload). I file principali del continuo restano in arms[arm].evidence_sha256.
- Scelta dichiarata? No: prontuario:183 richiede «le tracce» alla lettura (come run, ricevute numeriche e parita', questi ultimi tutti pinnati); piani e risposta R-1 non dicono che le tracce restino non pinnate. Test: la fixture del repo simula compare senza evidence_sha256, nessun test verifica le impronte della ripresa.
- Effetto sul cancello: nessuno. status/passed letti prima della sostituzione (readout:491-494); authenticate_preflight ricalcola e rifiuta ogni modifica che cambia l'esito; usato solo da validate_measured_budget (stadio training, non rilasciato). Start.json: campi chiave gia' verificati contro fork_sha256 (conservato).
- Gravita': difetto di codice reale (residuo della riga storica 438c8d65:382), perdita di provenienza della prova decisiva, senza effetti su esiti, tetti o cancelli -> MINORE confermato; non BLOCCANTE, non solo NOTA.
- A margine (non parte del rilievo): trace_resume_updates scrive anche trace-{rank}.ddp.jsonl (target_resume:204), che compare_three_paths non legge ne' pinna.

## Esito: NON confutato. MINORE. Pulizia: basetemp v62_bt (8 MB) cancellato; resta solo la sonda v62_probe; copia di sola lettura pulita a 0efa4d45.
