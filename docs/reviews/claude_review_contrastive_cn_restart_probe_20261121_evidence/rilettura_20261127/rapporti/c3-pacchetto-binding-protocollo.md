# C3 - Pacchetto, binding, piano e protocollo (diff 9a2ce7e7..8c72655c)

Esito: nessun bloccante. Script e output in
/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe2/agents/c3-pacchetto-binding-protocollo/
(bind_sim.py, positive_flow.py, negatives.out, positive.out). La copia locale del clone (295 MB) e' stata cancellata; il clone e' pulito.

## 1. Pacchetto (eseguito)
`--package` -> package e9310d67...702d, plan 4f538cf1...853b (attesi). source_digest() = 40d39f56...cde0 = plan.production_source_sha256.
tests/cpu/test_cn_restart_package.py: 12 passed.

## 2. Binding (eseguito su copia locale, ref simulato con update-ref)
Flusso positivo: commit di review su refs/remotes/origin/worktree-claude-training-engine -> --bind-review a 8c72655c -> commit figlio
0fb6b5d5 (genitore 8c72655c) -> worktree pulito -> require_binding() OK (status REVIEW_REQUIRED, training_approved False).
Seconda --bind-review nello stesso checkout: FileExistsError. Binding alterato: "Invalid review provenance". Protocollo alterato dopo il
binding: "Review does not approve this exact CN restart package".

Casi (bind):
- JSON su una riga: ACCETTATO
- JSON su piu' righe (indent=2), riga vuota dopo l'apertura, ordine delle chiavi diverso, spazi dopo i fence: ACCETTATO
- altra sezione ```text dopo il blocco, ```cn-restart-review aperto e non chiuso dopo il blocco, non ASCII, parola in prosa: ACCETTATO
- reviewed_commit 9d34dc69 completo: RIFIUTATO (File changed since Claude review); abbreviato o maiuscolo: RIFIUTATO
- impronte vecchie b0640ca6/9d5811b6: RIFIUTATO; sesto campo: RIFIUTATO
- blocco doppio: RIFIUTATO; CRLF: RIFIUTATO; fence indentati, ~~~, tag cn-restart-review-v1: RIFIUTATO
- commit non antenato del ref, ref assente, commit abbreviato: RIFIUTATO
- commit antenato ma superato da un commit successivo che cambia la review: ACCETTATO (vedi rilievo M-1)

Forma da usare: riga ```cn-restart-review a colonna 0, JSON (consigliato su una riga, come nei test), riga ``` a colonna 0, LF, una sola
volta nel file; nessun altro fence a colonna 0 con quel tag (non citare il vecchio blocco del prompt 9a2ce7e7).
La review attuale fc3eef6b allo stesso percorso non contiene blocchi (solo un ```text alle righe 241-263).
Repository di GPT e canonico: fetch +refs/heads/*:refs/remotes/origin/*, ref presente (439f9a7c).

## 3. plan.json (eseguito)
Chiavi nuove coerenti con SCALAR_ADDEND_PARAMETERS, PROCESS_ONLY_CUDA_RNG, LIMIT, DENOM_FLOOR; legate da test_plan_matches_worker_and_readout
(non legati da test: le stringhe scalar_denominator e reader_only_recovery; il codice alle righe 330/368-371/383 calcola la media dei
moduli locali ordinari sui 4 rank, coerente con la stringa). status REVIEW_REQUIRED, training_approved false, controllati anche da package().

## 4. Protocollo e decisioni (letto, conti rifatti)
Presenti prima della misura: regola cuda_rng.1-3, due scalari 253/257, recupero del solo lettore; comando --validate-only sul login;
lfs project -d + lfs quota -hp; due log .out/.err e phase_timings.jsonl; ledger 1,0 GPU-ora.
Conti: 1,0 GPU-ora; 8,2667 ore locali; inviluppo 1,4 e 11,8667; residui 12,26 e 114,0011; cumulato 5,74 e 55,5989: tutti esatti.
Ledger locale: nessuna voce CONTRAST dopo il 21/11 15:24, nessuna RESTART545 (base coerente, consumo sacct non verificabile oggi).
Superato: la pianificazione rispetto al fermo 23-25/11 (limite 23/11 07:29) nel protocollo e nella voce del 21/11 di decisioni.md;
i riferimenti a 3-launcher/GPT come operatore (CLAUDE.md del 27/11, c47e6557, li dichiara storici).

## 5. Comandi di invio (letto; il render di mac_launcher_command.py e' stato negato dal classificatore e non e' stato eseguito)
mac_launcher_command.py esiste solo nel canonico (daf9b8b2), non nel clone. Non esegue: verifica SHA a 40, comando python3/venv,
submitter = scripts/submit_job.py del worktree o quello condiviso, presenza di --id, byte del submitter locale e condiviso = git show
SHA:scripts/submit_job.py; stampa {workdir, cmd} con la venv e il submitter condiviso.
submit_job.py del clone, del canonico e condiviso: stesso sha256 814da4bc...0d82. Argomenti validi; --agente in {claude, gpt, simone}.
Vincoli del codice vs etichette: vedi oggetto strutturato.
