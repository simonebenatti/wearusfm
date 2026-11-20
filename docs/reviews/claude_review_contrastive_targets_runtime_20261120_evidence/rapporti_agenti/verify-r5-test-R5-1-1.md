# Verifica scettica n.1 del rilievo R5-1 (r5-test)

Avvio: 2026-11-20 04:20:19. Copia: 0efa4d45 (codice fbfde797).

## Metodo (comandi miei, script in rev_runtime/tmp/sk1_r51_scripts/)
- Cloni temporanei sk1_r51_a (strumentato) e sk1_r51_b (mutazioni) a 0efa4d45; contrastive_release.py identico fra fbfde797 e 0efa4d45 (diff fbfde797..0efa4d45 = solo delivery_pin.json).
- Riferimenti nei test (grep su tests/cpu, non solo ricevuta): solo 5 file citano contrastive_release; test_contrastive_job.py:26 e test_contrastive_run.py:73 sostituiscono validate_review_binding con `lambda value: None`; historical_callsites e targets_readout sostituiscono solo validate_audit_accounting. Nessun test invoca la CLI bind-review/check-binding/make-job in sottoprocesso.
- Prontuario (docs/contrastive_targets_execution_20261119.md:123-131) e risposta (contrastive_targets_response_20261119.md:11, R-2) descrivono il legame ma NON dichiarano test negativi su status/blocco/piani (a differenza di R-4, riga 13, che li elenca). Nessuna prova negativa manuale prevista nella procedura.
- mutants.py: 15 mutazioni del rilievo (S01-S15) + 3 controlli che devono morire (K01 sorgente dichiarata, K02 byte-identita' bind, K03 override); ciascuna da sola, poi `git checkout`.
- instrument.py: sentinelle a costo zero prima di ogni `raise ValueError` e all'ingresso delle funzioni di release; un solo giro dell'intero insieme della ricevuta per vedere quali rami di rifiuto i 465 test raggiungono.
- Sottoinsieme (5 file che citano release), base senza mutazioni: 94 passed in 32.88s.

## Risultati (1): strumentazione sull'intero insieme della ricevuta (29 file)
- Clone strumentato: **465 passed in 349.69s** (carico ~50-60), 1327 sentinelle.
- Solo `tests/cpu/test_contrastive_release.py` (22 test) entra in contrastive_release.py; nessun altro dei 29 file.
- Rami di rifiuto MAI raggiunti da alcun test: L97 (chiavi/schema/**status**, riga 96), L115 (piani di questa coorte, 114), L123 (legame interno del preflight, 121-122), L128 (byte dei piani al reviewed_commit, 127), L133 (copia/payload, 132), L84/L87/L78 (sorgenti/digest al reviewed_commit, 83/86/77), L231 (input di make_job, 230), L48 (pin nel checkout, 47), L69 (forma git_bytes, 68), L102 (formato hash, 101), L110 (review_commit completo, 109), L99, L108, L228, L42.
- Sonde: `status` sempre 'APPROVED' (22/22 passaggi); numero di blocchi solo 0 (1 test storico) o 1 (22): mai >=2; a riga 116 `sci_ok` e `pre_ok` sempre True (l'unico rifiuto a 119 e' il caso `[source]`, src_ok=False).
- Raggiunti: L50 pin mismatch (science/preflight/review modificati, sacct), L93 zero blocchi (review storica), L119 sorgente cambiata, L73 git show fallito (relabel, dentro verify_reviewed_sources), L143 byte-identita' del bind, L225/L235 cella unica e override in make_job.
- Conseguenza: per le barriere a condizione OR non raggiunte, togliere la condizione non cambia nessun esito dei 465 test (equivalenza sul test set).

## Risultati (2): mutazioni singole sui 5 file che citano release (94 test)
- Base 94 passed; S01 status, S02 un blocco, S03 coorte, S04 hash dichiarati dei piani, S05 legame interno preflight, S06 sorgenti al reviewed_commit, S07 byte dei piani al reviewed_commit, S08 copia/payload: tutte **94 passed** (sopravvivono). (in corso S09-S15 e controlli K01-K03)
- Sottoinsieme, seguito: S09 (make_job senza validazione), S10 (input di make_job), S11 (pin nel checkout), S12 (forma git_bytes), S13 (chiavi extra): **94 passed** (sopravvivono). (in corso S14, S15, K01-K03)

## Risultati (3): la lacuna e' reale e chiudibile (test usa e getta fuori dalla copia)
- `sk1_r51_scripts/test_sk1_negative.py` (4 test sulla fixture `package` della ricevuta: status REVIEW_REQUIRED; secondo blocco citato in un recinto ````text; preflight_plan_sha256 dichiarato errato; piani di un'altra cartella con review coerente e commit che li contiene), eseguito in clone sk1_r51_c:
  - codice originale: **4 passed** (le barriere funzionano davvero);
  - S01 -> FAILED solo test_sk1_status_must_be_approved; S02 -> solo test_sk1_two_blocks_are_rejected; S03 -> solo test_sk1_plans_of_another_cohort_are_rejected; S04 -> solo test_sk1_declared_preflight_hash_must_match (ognuno 1 failed, 3 passed); albero pulito dopo ogni prova.
- I test non entrano nel source_digest (source_paths: src/wearusfm, scripts/contrastive_*.py, sbatch, train_jepa.py, lib_code_dir.sh): la correzione non cambia i sorgenti rivisti (va solo rigenerata la ricevuta locale che pinna i file di test).
- Regex: un esempio in recinto a 4 backtick conta come blocco (2 trovati), una menzione inline no (1): l'avvertenza finale del rilievo e' esatta ed e' fail-closed (bind-review rifiuta); il prompt per Claude (docs/prompt_claude_contrastive_targets_runtime_20261119.md) contiene 1 blocco di esempio con segnaposto.
- Sottoinsieme, completato: anche S14 (formato hash) e S15 (review_commit completo) **94 passed**. Controlli del driver UCCISI come atteso: K01 sorgente dichiarata -> test_changed_source...[source]; K02 byte-identita' del bind -> test_bind_requires_review_bytes...; K03 override -> test_job_builder_releases_only... (1 failed, 93 passed ciascuno). Albero di sk1_r51_b pulito.

## Risultati (4): intero insieme della ricevuta con mutazione singola
- S01 (tolto `payload["status"] != "APPROVED"`, release.py:96): **465 passed in 360.76s**, rc 0, albero pulito. Riprodotta la prova del rilievo.
- S02 (`len(blocks) != 1` -> `< 1`, release.py:92): **465 passed in 428.67s**, rc 0, albero pulito.
- In corso: S01..S15 applicate insieme sull'intero insieme (controllo incrociato della strumentazione).
- S01..S15 applicate INSIEME sull'intero insieme: **1 failed, 464 passed in 170.78s** -> fallisce solo test_review_cannot_relabel_the_current_code_as_an_older_commit. Causa verificata su test_contrastive_release.py: S06+S07 da sole -> stesso test FAILED (1 failed, 21 passed); le 14 senza S07 -> 22 passed; le 14 senza S06 -> 22 passed. Le due verifiche Git (125 e 126-128) sono ridondanti fra loro per quel test (commit `0`*40 inesistente: `git show` fallisce in entrambe). Il rilievo dice "una sola" barriera: resta esatto. Nessun contrasto con r5: il suo combinato degli extra (465 passed) non conteneva insieme C3a e C3b (le due verifiche Git).
- Contenuto al reviewed_commit mai provato: disattivati insieme i soli confronti di byte/digest (righe 83, 86, 127) lasciando le chiamate `git show`: **94 passed** sui 5 file (che comprendono i 22 test che raggiungono il modulo). Il test di relabel prova solo l'esistenza del commit, non la differenza dei contenuti (L84 e L128 mai raggiunti).

## Motivi cercati per confutare (nessuno regge)
- Controllo altrove: no. make_job (221), contrastive_job.build_command (contrastive_job.py:70) e contrastive_run.validate_release (contrastive_run.py:182) delegano tutti alla stessa validate_review_binding; nessuno ricontrolla status o numero di blocchi. Il dispatcher ricontrolla solo lo status del job ('APPROVED_FOR_PREFLIGHT', scritto da make_job).
- Percorso non usato in produzione: no, e' il legame di release dei job (prontuario 123-131).
- Dichiarazione esplicita: no. La risposta (R-2, riga 11) e il prontuario non dichiarano test negativi per status/blocco/piani ne' accettano la lacuna; R-4 (riga 13) li dichiara per l'accounting, dove esistono. Precedente M-1 di 94f5094: tre barriere senza test = MINORE.
- Attenuanti reali (non confutano): alcune voci sono difesa in profondita' -> formato hash (101) e' equivalente in accettazione, perche' 117-118 confrontano con hexdigest calcolati (cambia solo il messaggio); hash dichiarati dei piani (117-118) e byte dei piani al reviewed_commit (126-128) si coprono a vicenda; 125 e' coperta da 117 (sorgente dichiarata); 221 e 230 sono rifatte dal dispatcher (contrastive_job.py:70 e :110 con validate_input_bindings, contrastive_prepare.py:107-109). Senza ridondanza restano: status, schema, blocco unico, piani di questa coorte, legame interno del preflight, copia/payload, chiavi, pin della review nel checkout, review_commit.

## Esito
- refuted = false. Gravita' confermata MINORE: codice corretto oggi (i 4 test negativi passano sul codice originale), nessuna catena che accetti review non valide; manca la protezione dalle regressioni, e una regressione su status o blocco passerebbe la ricevuta da 465.

## Non verificato
- Integrazione sul pacchetto vero (bind-review con review REVIEW_REQUIRED o con due blocchi sui piani reali): non rifatta per non creare commit nemmeno in un clone; le barriere sono provate sul codice reale con la fixture git della ricevuta.
- Nessuna esecuzione CUDA/Leonardo (fuori perimetro).

## Pulizia
- Cloni sk1_r51_a/b/c e basetemp cancellati; restano solo script e log in rev_runtime/tmp/sk1_r51_scripts (252 KB). Copia di sola lettura pulita a 0efa4d45. Fine: verifica conclusa.
