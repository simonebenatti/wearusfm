# verify-r3-release-R3-M4-1 — scettico n. 1 sul rilievo R3-M4 (copertura dei test delle guardie di release)

Avvio: 2026-11-20 04:00. Clone mio: rev_runtime/tmp/sk1r3m4_clone a 0efa4d45 (pulito). Script miei: rev_runtime/tmp/sk1r3m4_scripts/.

## Letture (04:00-04:10)
- Codice a fbfde797 = 0efa4d45 per i sorgenti (diff fbfde797..0efa4d45 = solo delivery_pin.json).
- test_contrastive_job.py:26-27 (fixture job_env, usata da tutti i test che chiamano build_command) e test_contrastive_run.py:73-74 (release_fixture) sostituiscono validate_review_binding e validate_audit_accounting con no-op: confermato a vista.
- Nessun altro file di tests/cpu chiama validate_review_binding/bind_review/make_job/build_command/validate_release (grep); test_contrastive_targets_readout.py:112 e historical_callsites.py:112 sostituiscono validate_audit_accounting (spia).

## Riproduzione indipendente (04:05-)
- Mutazioni mie (sk1r3m4_scripts/mutants.py, 39 voci riscritte dal codice a fbfde797; dove possibile tolgo solo il raise o la sola condizione, lasciando i pin): tutte applicate una sola volta, sintassi valida.
- Harness mio run_mutants.py: (a) "seven" = i 7 file del rilievo; (b) "full" = esattamente i 29 file della ricevuta validation_final (465 test) per i 25 sopravvissuti dichiarati. Cloni sk1r3m4_clone e sk1r3m4_clone_full a 0efa4d45.
- Base: seven 125 passed in 15,5 s; full 465 passed in 111 s (come la ricevuta: 465 in 102,6 s).
- Probe comportamentali mie (sk1r3m4_scripts/probes/test_sk1_probes.py, 28 test, tutti verdi sul codice non mutato in 6,7 s): ognuna asserisce l'esito della guardia originale, per distinguere mutanti osservabili da equivalenti. Harness run_probes.py su clone sk1r3m4_clone_probe.
- git show <commit>:<path> con "../x", assoluto, "sub/../x": git rifiuta da solo ("outside repository", "exists on disk, but not in") -> MU36 e' candidato equivalente.

## Esiti parziali (04:25)
- seven (7 file, mie mutazioni): MU1 MU2 MU3 MU5 MU6 MU8 MU9 MU19 MU33 SURVIVED; MU4 MU7 MU10-MU18 MU20 KILLED con gli stessi test citati dal rilievo. Coincide finora riga per riga con r3_mutations2.tsv.
- full (29 file, 465 test): MU1, MU2 SURVIVED (465 passed).
- probe: osservabili (la mia probe che asserisce la guardia originale fallisce sul mutante) MU1 MU2 MU3 MU5 MU6 MU8 MU9 MU19 MU21 MU32 MU37; MU33 osservabile solo nel tipo di eccezione (rifiuto con KeyError invece di ValueError: probe "qualunque eccezione" verde); MU35 e MU36 NON osservati (equivalenti: MU35 compensato dal confronto byte di MU39; MU36 compensato da git, che rifiuta da solo percorsi fuori albero).
- MU37 notevole: con job_id "123.batch" il mutante accetta una riga .batch COMPLETED al posto dell'allocazione TIMEOUT (probe test_P_MU37).

## Esiti completi seven e probe (04:20)
- seven (sk1r3m4_seven.tsv, stato finale clean): 14 KILLED = MU4 MU7 MU10 MU11 MU12 MU13 MU14 MU15 MU16 MU17 MU18 MU20 MU31 MU38 (stessi test uccisori del rilievo); 25 SURVIVED = MU1 MU2 MU3 MU5 MU6 MU8 MU9 MU19 MU21 MU22 MU23 MU24 MU25 MU26 MU27 MU28 MU29 MU30 MU32 MU33 MU34 MU35 MU36 MU37 MU39 (125 passed ciascuno). Insiemi identici a r3_mutations2.tsv, con mutazioni scritte da me (piu' strette: tolgo il solo raise/condizione).
- probe (sk1r3m4_probes.tsv, stato finale clean): 21 sopravvissuti OSSERVABILI, cioe' uccisi da una probe semplice: MU1 MU2 MU3 MU5 MU6 MU8 MU9 MU19 MU21 MU22 MU23 MU24 MU25 MU26 MU27 MU28 MU29 MU30 MU32 MU34 MU37. MU33: solo tipo d'eccezione (KeyError, rifiuto comunque). MU35, MU39: NON osservati (si compensano a vicenda: digest sugli originali vs confronto byte). MU36: NON osservato (equivalente: git rifiuta "../", assoluti, "a/../"; anche la variante piu' ampia del revisore `if False:` -> 7/7 probe di rifiuto verdi, perche' ref e SHA corti cadono su _full_sha(review_commit) in validate_review_binding).
- Imprecisione del rilievo: MU36 e' equivalente e va con MU33/MU35/MU39; i sopravvissuti reali sono 21, non 22. La sostanza non cambia.

## Suite completa (29 file, 465 test) — 04:40
- SURVIVED con 465 passed: MU1 MU2 MU3 MU5 MU6 MU8 MU9 (clone A, poi fermato dopo MU9 da watcher e ripulito), MU19 MU21 MU22 MU23 MU24 MU25 (B), MU33 MU37 MU32 MU35 MU36 MU39 (C). In corso MU26 MU27 MU28 (B), MU29 MU30 MU34 (C); carico macchina 40-60, ~5,7 min per giro.
- Argomento statico per i 6 restanti: modificano solo il corpo di build_command (MU26, MU27) e di validate_release (MU28, MU29, MU30, MU34); fuori da test_contrastive_job.py e test_contrastive_run.py nessun file di tests/cpu chiama build_command, validate_release o main() di questi moduli (grep); nessun test confronta l'impronta dei sorgenti con un valore pinnato (le 19 mutazioni gia' provate su contrastive_release.py/contrastive_job.py, stesso inventario, sopravvivono).

## Cercati motivi per confutare (nessuno regge)
- Altri test: nessun file oltre ai 7 esercita le guardie (grep validate_review_binding/validate_audit_accounting/bind_review/make_job/build_command/validate_release); la suite completa non uccide nessun sopravvissuto.
- Dichiarazioni: prontuario (sezioni "Rilascio vincolato alla review", "Verifica locale e limiti") e risposta R-1..N-7 non dichiarano che dispatcher/runtime siano testati con guardie finte; anzi la risposta R-2 dice "Dispatcher e runtime rifiutano review storiche o di un altro pacchetto" e il prompt a Claude (punto 5) cita l'"integrazione fra produttori e lettori". La correzione chiesta da 94f50942 R-2 ("il dispatcher fa questo controllo") non ha regressione: MU22 sopravvive.
- Percorso di produzione: contrastive_preflight_cpu.sbatch:28 e contrastive_preflight_gpu.sbatch:30 eseguono contrastive_job.py; per audit/CUDA/lettura il dispatcher e' l'unico controllo del binding sul cluster (contrastive_run.validate_release copre solo calibrazione/benchmark).
- Equivalenze: solo MU33 (KeyError), MU35/MU39 (compensazione reciproca) e MU36 (git) — 4 su 25; 21 restano osservabili.

## Chiusura (04:55)
- Suite completa: 25 SURVIVED su 25, 0 KILLED (465 passed per ciascuno; anche MU26-MU30 e MU34). Stato finale clean in tutti i cloni.
- Pulizia: cloni sk1r3m4_clone, _probe, _full, _fullB, _fullC e basetemp cancellati; restano sk1r3m4_scripts/ (mutants.py, run_mutants.py, run_probes.py, stop_after.sh, probes/test_sk1_probes.py, probes/test_sk1_extra.py) e le uscite sk1r3m4_{seven,full,fullB,fullC,probes}.tsv. Copia di riferimento rev_targets/clone: HEAD 0efa4d45, pulita.

## ESITO: NON CONFUTATO. Gravita' reale: MINORE (confermata)
- Riprodotto con mutazioni e harness miei: stessi 14 uccisi / 25 sopravvissuti sui 7 file; i 25 sopravvivono anche ai 465 test della ricevuta validation_final.
- Monkeypatch confermati: test_contrastive_job.py:26-27 (fixture job_env, tutti i test di build_command) e test_contrastive_run.py:73-74 (release_fixture). Integrazione reale provata solo per make_job della cella audit (test_contrastive_release.py:144-151) e per la lettura finale (spia, MU31).
- 21 sopravvissuti su 25 sono osservabili e uccisi da probe semplici (28 probe, 6,7 s): il difetto e' reale e correggibile come proposto nel fix.
- Non BLOCCANTE: le guardie oggi funzionano (le probe passano sul codice originale), nessuna evidenza o review non valida accettata, nessun tetto superato. Non NOTA: non e' chiarezza/documentazione, e' assenza di regressioni su guardie critiche di release (status APPROVED, unicita' del blocco, gate accounting e binding in dispatcher/runtime, fix R-2 di 94f50942).
- Imprecisione da correggere nel rilievo: MU36 e' equivalente (git e _full_sha) e va con MU33/MU35/MU39; sopravvissuti osservabili 21, non 22.
