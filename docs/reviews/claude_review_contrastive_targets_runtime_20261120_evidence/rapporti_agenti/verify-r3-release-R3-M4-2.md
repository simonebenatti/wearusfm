# verify-r3-release-R3-M4-2 — scettico n. 2 su R3-M4 (copertura dei test delle guardie di release)

Avvio: 2026-11-20
Piano: (1) clone temporaneo a 0efa4d45; (2) base dei 7 file; (3) mutazioni con uno script mio, eseguite anche contro TUTTI i test_contrastive_*.py
(non solo i 7 scelti dal revisore) per cercare un test che le uccida altrove; (4) equivalenza delle mutazioni; (5) prontuario/risposta per dichiarazioni.

## Progressi
- Cloni temporanei v4m2_c1..c3 a 0efa4d45 (puliti). Base 7 file: 125 passed in 14,52 s. Base della lista completa della ricevuta
  validation_final (29 file): 465 passed in 108,98 s (= ricevuta 465/102,59 s).
- Monkeypatch confermati a fbfde797: tests/cpu/test_contrastive_job.py:26-27 e test_contrastive_run.py:73-74
  (validate_review_binding e validate_audit_accounting -> no-op); test_contrastive_targets_readout.py:112 la sostituisce con uno stub, mentre
  historical_callsites.py:104-131 usa una SPIA che verifica chiamata e argomenti (per questo MU31 muore). Nessun test chiama build_command/validate_release
  con un binding reale (git grep su tests/).
- Script mio: tmp/v4m2_scripts/mut.py (29 mutazioni: 3 controlli + 26 forme mie delle 25 sopravvissute, MU6 spezzata in due),
  ogni mutazione contro TUTTI i 29 file della ricevuta, con classificazione dei test falliti dentro/fuori i 7 file.
- Conteggio del TSV del revisore (tmp/r3_mutations2.tsv): 14 KILLED, 25 SURVIVED (awk): coincide col rilievo.
- Percorsi di produzione: i launcher scripts/slurm/contrastive_preflight_{cpu,gpu}.sbatch:28/30 eseguono contrastive_job.py
  (dispatcher); verify_git=False non e' usato da alcun chiamante; quindi nessuna guardia sopravvissuta sta su un percorso morto.
- validate_cuda_prerequisite (contrastive_release.py:195) non e' citata da alcun test (git grep su tests/): MU34 e' una lacuna
  anche della funzione, non solo dell'integrazione.
- Equivalenza: MU36 e' equivalente nel comportamento: git rifiuta da se' i percorsi non contenuti ('../x' -> "outside repository",
  '/etc/passwd' e 'a/../b' -> "exists on disk, but not in <commit>"), e git_bytes converte l'errore in ValueError (righe 70-73).
  MU37 e' quasi equivalente con sacct reale: le righe di step hanno TimelimitRaw vuoto (storico 64652559.batch: "|3787||8|"),
  quindi int('') -> ValueError alla riga 184-186; sopravvive solo con la riga .batch artificiale della fixture (7199|8|120).
- Prova diretta di equivalenza (tmp/v4m2_scripts/equiv.py, clone v4m2_c4, uscita tmp/v4m2_out/equiv.out; mutazioni solo in memoria):
  MU36: per '../x', '../clone/README.md', '/etc/passwd', percorso assoluto interno, 'scripts/../scripts/...', './../scripts/...'
  originale e mutante danno entrambi ValueError (cambia solo il messaggio); 'scripts/contrastive_release.py' accettato da entrambi.
  -> MU36 e' un mutante equivalente: la sua sopravvivenza non e' una lacuna dei test.
  MU37: TSV realistico (step con TimelimitRaw vuoto) e job_id '123.batch' -> originale "job ID missing", mutante "not integral":
  respinto da entrambi. Solo con una riga .batch artificiale che porta 120 min (come la fixture) il mutante ACCETTA '123.batch'
  mentre l'allocazione e' TIMEOUT. -> difesa in profondita' contro TSV non realistici/manipolati; lacuna reale ma stretta.
- Riferimento incrociato (non usato come prova): il revisore indipendente r5-test (tmp/r5/results_full2_{x,y,z}.jsonl) ha mutazioni
  analoghe (C3b, C3c, C3e, C3f, C3g, C3h, C3i-C3p, C4d, C4f1-C4f5) che sopravvivono a 465/465 sull'intero insieme della ricevuta.
- Ricerca statica di test che potrebbero uccidere i mutanti fuori dai 7 file: git grep su tutto tests/ di contrastive_release,
  contrastive_job, validate_review_binding, validate_audit_accounting, build_command -> SOLO 5 file, tutti fra i 7 del revisore.
- Mutazione neutra (commento in coda a contrastive_release.py) -> vedi K00: serve a escludere uccisioni banali da digest.
- Non equivalenza dimostrata per MU1/MU2/MU3/MU6: 4 test negativi miei (tmp/v4m2_scripts/test_skeptic_release_negatives.py,
  riusano la fixture `package` del pacchetto; fuori dalla copia) -> 4 passed sul codice originale (4,88 s); ciascun mutante
  A01/A02/A03/A06a e' ucciso dal suo test (1 failed, 3 passed), stato del clone pulito (tmp/v4m2_out/neg_probes.out).
  Quindi la sopravvivenza dipende dall'assenza dei test, non da un'impossibilita' di provarli.
- Protocollo del revisore (7 file) sui mutanti d'integrazione, clone v4m2_c4 (tmp/v4m2_out/seven_core.out): B22, B25, C28, C30
  ciascuno 125 passed; TUTTI E QUATTRO INSIEME (nessuna verifica di binding ne' di accounting in dispatcher e runtime) -> 125 passed
  in 27,93 s; stato pulito.
- Prova piu' forte (tmp/v4m2_out/full_core4.out): OTTO guardie d'integrazione rimosse INSIEME (B22 binding, B23 review, B25 gate,
  B27 release nel dispatcher; C28 binding, C29 review_pin, C30 gate, C34 CUDA nel runtime) -> intero insieme della ricevuta
  465 passed in 226,74 s; stato pulito.

## Documenti (ricerca di una dichiarazione che confuti)
- Prontuario docs/contrastive_targets_execution_20261119.md:145-146: «Il gate è eseguito nella release, nel dispatcher e nel runtime
  della calibrazione»; risposta docs/reviews/contrastive_targets_response_20261119.md:11 (R-2): «Dispatcher e runtime rifiutano review
  storiche o di un altro pacchetto». Nessuna dichiarazione che l'integrazione sia provata altrove o rinviata: anzi e' proprio la
  proprieta' affermata e non coperta dai test. La ricevuta 465/465 elenca solo pytest (contrastive_local_validation.py:38-45).
- git diff 438c8d65 fbfde797 -- tests/cpu/test_contrastive_job.py: i due monkeypatch no-op sono AGGIUNTI in questa consegna, insieme
  all'introduzione delle chiamate nel dispatcher (test_contrastive_release.py e' nuovo, 157 righe).

## Impatto a livello di catena (per la gravita')
- Ridondanti quando si passa dal dispatcher: C28/C30 (coperti da B22/B25 + B27), A21 (coperto da B25 sul cluster), B27/C29/B23
  (coerenza di metadati). Unici: B22 e' l'unico controllo del binding sul cluster per audit, cuda e readout (contrastive_audit.py,
  contrastive_cuda_check.py e il readout non chiamano validate_review_binding: git grep); B25 l'unico gate accounting sul cluster per
  la cella cuda; C34 l'unico controllo del CONTENUTO della ricevuta CUDA prima dei benchmark; A01-A09/A19/A32 unici in
  validate_review_binding/validate_audit_accounting.
- Nessuna di queste lacune fa girare oggi la catena in modo scorretto: le guardie a fbfde797 ci sono e funzionano; manca la rete
  che ne rileverebbe la rimozione o una regressione. Quindi non BLOCCANTE.

## Correzioni al rilievo (non cambiano la sostanza)
- MU36 e' equivalente (vedi sopra): va tolto dalle lacune.
- MU37 e' quasi equivalente con sacct reale; lacuna solo contro TSV non realistici.
- Con MU33/MU35/MU39 gia' indicati dal revisore come quasi equivalenti/ridondanti, le lacune effettive sono circa 20 su 25.
- Per MU21 la mia forma toglie solo la validazione (resta checked_pin(job["audit"])); sopravvive comunque (nessun test chiama
  make_job con una cella non audit).

## Non verificato
- Non ho rieseguito lo script del revisore (r3_mutate2.py): forme di mutazione e driver sono miei.
- Nessuna prova su Leonardo/CUDA (fuori perimetro); i test sono CPU sul Mac.
- Le guardie stesse (correttezza dei respingimenti) non sono il mio oggetto: le ho viste funzionare solo nei casi dei miei test
  negativi (4 casi) e dell'equivalenza (MU36/MU37).

## Righe delle guardie sopravvissute a fbfde797
contrastive_release.py: 68 (MU36), 83 (MU39), 86 (MU35), 92 (MU1), 96 (MU2/MU3), 111 (MU32), 114 (MU8), 121-123 (MU9),
127-128 (MU5), 132-133 (MU6), 165 (MU19), 172 (MU37), 177 (MU33), 240-242 (MU21); contrastive_job.py: 70 (MU22), 71 (MU23),
77-79 (MU24), 80-81 (MU25), 85 (MU26), 141 (MU27); contrastive_run.py: 182 (MU28), 183 (MU29), 195 (MU30), 209 (MU34).

## Esito delle mutazioni mie sull'intero insieme della ricevuta (465 test; tmp/v4m2_out/mut_full.tsv)
```
verdicts: {'SURVIVED': 27, 'KILLED_IN7': 2}
K00_trivial_comment_release              SURVIVED rc=0                  132s  465 passed in 130.13s (0:02:10) | in7= | out7=
K04_no_verify_git(MU4)                   KILLED_IN7 rc=1                130s  1 failed, 464 passed in 128.49s (0:02:08) | in7=test_review_cannot_relab
K13_timeout_ok(MU13)                     KILLED_IN7 rc=1                132s  1 failed, 464 passed in 129.91s (0:02:09) | in7=test_audit_has_no_timeou
A01_first_block_only(MU1)                SURVIVED rc=0                  143s  465 passed in 141.43s (0:02:21) | in7= | out7=
A02_no_status(MU2)                       SURVIVED rc=0                  142s  465 passed in 140.58s (0:02:20) | in7= | out7=
A03_superset_keys(MU3)                   SURVIVED rc=0                  143s  465 passed in 141.55s (0:02:21) | in7= | out7=
A05_plans_not_vs_commit(MU5)             SURVIVED rc=0                  178s  465 passed in 176.01s (0:02:56) | in7= | out7=
A06a_payload_unchecked(MU6)              SURVIVED rc=0                  180s  465 passed in 178.28s (0:02:58) | in7= | out7=
A06b_orig_sha_unchecked(MU6)             SURVIVED rc=0                  179s  465 passed in 177.29s (0:02:57) | in7= | out7=
A08_other_cohort(MU8)                    SURVIVED rc=0                  178s  465 passed in 175.87s (0:02:55) | in7= | out7=
A09_no_preflight_link(MU9)               SURVIVED rc=0                  177s  465 passed in 175.31s (0:02:55) | in7= | out7=
A19_audit_report_source(MU19)            SURVIVED rc=0                  178s  465 passed in 175.73s (0:02:55) | in7= | out7=
A21_make_job_no_gate(MU21)               SURVIVED rc=0                  223s  465 passed in 220.03s (0:03:40) | in7= | out7=
A32_review_outside_checkout(MU32)        SURVIVED rc=0                  224s  465 passed in 220.29s (0:03:40) | in7= | out7=
A33_no_sacct_header(MU33)                SURVIVED rc=0                  222s  465 passed in 218.67s (0:03:38) | in7= | out7=
A35_no_digest_compare(MU35)              SURVIVED rc=0                  225s  465 passed in 220.76s (0:03:40) | in7= | out7=
A36_git_bytes_uncontained(MU36)          SURVIVED rc=0                  225s  465 passed in 221.25s (0:03:41) | in7= | out7=
A37_job_id_any_string(MU37)              SURVIVED rc=0                  226s  465 passed in 222.18s (0:03:42) | in7= | out7=
A39_no_byte_compare(MU39)                SURVIVED rc=0                  211s  465 passed in 208.07s (0:03:28) | in7= | out7=
B22_dispatch_no_binding(MU22)            SURVIVED rc=0                  212s  465 passed in 209.48s (0:03:29) | in7= | out7=
B23_dispatch_review_unlinked(MU23)       SURVIVED rc=0                  212s  465 passed in 209.14s (0:03:29) | in7= | out7=
B24_dispatch_no_cell(MU24)               SURVIVED rc=0                  356s  465 passed in 353.04s (0:05:53) | in7= | out7=
B25_dispatch_no_gate(MU25)               SURVIVED rc=0                  356s  465 passed in 353.68s (0:05:53) | in7= | out7=
B26_dispatch_out_any(MU26)               SURVIVED rc=0                  355s  465 passed in 353.12s (0:05:53) | in7= | out7=
B27_dispatch_release_unlinked(MU27)      SURVIVED rc=0                  362s  465 passed in 357.70s (0:05:57) | in7= | out7=
C28_runtime_no_binding(MU28)             SURVIVED rc=0                  361s  465 passed in 355.94s (0:05:55) | in7= | out7=
C29_runtime_review_pin(MU29)             SURVIVED rc=0                  362s  465 passed in 356.92s (0:05:56) | in7= | out7=
C30_runtime_no_gate(MU30)                SURVIVED rc=0                  437s  465 passed in 433.53s (0:07:13) | in7= | out7=
C34_runtime_no_cuda(MU34)                SURVIVED rc=0                  437s  465 passed in 433.79s (0:07:13) | in7= | out7=
surviving reviewer ids (full 465-test set): 25 ['MU1', 'MU2', 'MU3', 'MU5', 'MU6', 'MU8', 'MU9', 'MU19', 'MU21', 'MU22', 'MU23', 'MU24', 'MU25', 'MU26
```
- Controlli: K00 (commento neutro, cambia il source digest) 465 passed -> nessuna uccisione banale da digest; K04 (=MU4) e K13 (=MU13)
  uccisi dai test attesi nei 7 file -> lo strumento rileva le mutazioni quando un test c'e'.
- Le mie 26 forme delle 25 sopravvissute: TUTTE 465 passed; nessun test fuori dai 7 file le uccide. Stessi 25 id del rilievo.
- Cloni c1-c3 puliti a 0efa4d45 dopo la corsa (git status vuoto).

## Esito
NON CONFUTATO. Rilievo riprodotto in modo indipendente e confermato anche sull'intero insieme della ricevuta (465 test), non
solo sui 7 file. Correzioni minori: MU36 equivalente, MU37 quasi equivalente (con MU33/MU35/MU39 gia' indicati: ~20 lacune
effettive su 25). Gravita' reale: MINORE (difetto reale della verifica, nessun effetto BLOCCANTE dimostrato: le guardie a
fbfde797 sono presenti e il checkout e' congelato).
- Pulizia: cloni v4m2_c1..c4 e basetemp cancellati; conservati tmp/v4m2_scripts (mut.py, equiv.py, summarize.py, test_skeptic_release_negatives.py) e tmp/v4m2_out; copia di riferimento a 0efa4d45, pulita.
