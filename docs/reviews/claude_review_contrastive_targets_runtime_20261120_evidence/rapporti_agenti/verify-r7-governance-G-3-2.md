# Verifica scettica n. 2 del rilievo G-3 (r7-governance)

Avvio: lettura di contrastive_release.py, contrastive_job.py:62-81, contrastive_run.py:165-195,
test_contrastive_job.py:15-60, test_contrastive_run.py:61-87, test_contrastive_release.py,
test_contrastive_historical_callsites.py:80-128, test_contrastive_targets_readout.py:66-112.
Prima osservazione: le fixture di job/run sostituiscono validate_review_binding e
validate_audit_accounting con lambda che non registrano le chiamate (test_contrastive_job.py:26-27,
test_contrastive_run.py:73-74). Nessun test confronta il source_sha256 del preflight con source_digest().
Prossimo passo: mutazioni in clone temporaneo, su 4-5 file e poi sull'intero insieme della
validazione locale (29 file test_contrastive* + 3 test delle finestre).

## Riproduzione mirata (cloni temporanei a 0efa4d45, script di mutazione per stringa esatta)
- Mutazione A: contrastive_job.py:70-72 e contrastive_run.py:181-184 sostituite da `pass`.
- Mutazione B: contrastive_job.py:81, contrastive_run.py:195, contrastive_release.py:242 sostituite da `pass`.
- Controllo che i moduli importati siano quelli mutati: inspect.getsource contiene MUT-A/MUT-B (si').
- job+run+release+targets_readout: base 89 passed; mutazione A 89 passed.
- idem + historical_callsites: base 94 passed; mutazione B 94 passed.
Rilievo riprodotto sui file citati. Prossimo passo: insieme completo della validazione locale.

## Insieme completo della validazione locale (29 file: 26 test_contrastive* + 3 test delle finestre)
- base (clone non mutato, ricevuta validation_final verificata con --verify: verified true): 465 passed.
- mutazione A: 465 passed. Mutazione B: 465 passed.
Nessun test dell'insieme della ricevuta rileva la rimozione delle chiamate nei consumatori.

## Estensione: tutte le chiamate consumatrici della review rimosse (A + make_job)
- Mutazione C in aggiunta ad A: contrastive_release.py:221 `declared = validate_review_binding(binding)`
  sostituita da `declared = binding["review_payload"]`. Restano solo bind_review (riga 151) e la CLI check-binding.
- Insieme completo (29 file): 465 passed. Nessun test protegge alcuna chiamata consumatrice di R-2.
- Per R-4 resta protetta solo la chiamata del lettore finale (contrastive_benchmark_readout.py:421,
  registrata con argomenti in test_contrastive_historical_callsites.py:105-128).

## Sonda d'integrazione dello scettico (fuori dal pacchetto, senza monkeypatch dei validatori)
File: rev_runtime/tmp/g3v2_tools/test_g3v2_integration.py. Repo git di prova con review vera
(bind_review), review di "altro pacchetto" (preflight_plan_sha256 diverso, committata),
accounting COMPLETED/7000 s valido, TIMEOUT e COMPLETED/7201 s.
- base: 9 passed (dispatcher, make_job e runtime respingono review di altro pacchetto, review_pin
  rietichettato, TIMEOUT e 7.201 s; accettano binding e accounting validi): premessa «vere nel codice» confermata.
- mutazione A: 3 failed (dispatcher x2, runtime). Mutazione B: 5 failed (make_job x2, dispatcher x2, runtime).
  A+C: 4 failed. La correzione proposta e' fattibile e rileva le regressioni.

## Ricerca di motivi per confutare
- Nessun test dell'insieme confronta source_sha256 del preflight con source_digest(): nessuna rilevazione
  indiretta per digest.
- Prontuario riga 145 e risposta R-2/R-4 affermano il comportamento del codice e i test negativi della
  funzione, non test d'integrazione: nessuna dichiarazione falsa, ma nessuna esclusione esplicita del rischio.
- Mitigazione reale: ogni modifica ai sorgenti cambia source_digest() (tests esclusi, contrastive_prepare.py:192-201),
  quindi una regressione richiede un nuovo blocco di review; i test pero' non aiuterebbero quel revisore.
  Difesa in profondita': make_job, dispatcher, runtime, lettore; per audit/cuda/readout il dispatcher e'
  l'unico controllo della review al momento dell'esecuzione (contrastive_audit/cuda_check/readout non la chiamano).
- Test fuori dal digest (source_paths, contrastive_release.py:58-64): aggiungerli non invalida il blocco di
  review; vanno rigenerati solo ricevuta di validazione locale (pinna i test) e delivery_pin (metadati).

## Esito
Rilievo NON confutato. Riprodotto sui file citati e sull'intero insieme della ricevuta (465 = 465).
Gravita': MINORE (lacuna reale della verifica, nessun effetto sulla catena attuale, codice corretto).
