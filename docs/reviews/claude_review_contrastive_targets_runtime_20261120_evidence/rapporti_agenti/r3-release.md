# r3-release — PUNTO 3 RELEASE (review di fbfde797, punta 0efa4d45)

Avvio: 2026-11-19 14:03

## Letture iniziali (14:10)
- Copia a 0efa4d45; diff fbfde797..0efa4d45 = solo delivery_pin.json. source_digest() alla punta = 40d39f56...cde0, plan b03c0bc6...021a, preflight e4b39591...6dd92: coincidono con delivery_pin (comando: python -c "import contrastive_release as C, contrastive_prepare as P; ..." nel clone r3c).
- pytest test_contrastive_release.py + test_contrastive_targets_guards.py: 27 passed in 6,6 s.
- test_contrastive_job.py e test_contrastive_run.py monkeypatchano validate_review_binding e validate_audit_accounting a no-op: l'integrazione reale nel dispatcher/runtime NON e' coperta dai test -> da provare a mano.
- Python sul cluster 3.11.6 (summary.json storici) -> hashlib.file_digest disponibile (gia' usato da diagnostics eseguiti).
- Audit storico 64652559: riga allocazione TIMEOUT|0:0|3787|8|60, riga .batch COMPLETED: il gate legge solo JobIDRaw==job_id -> respinto; AllocCPUS=8 su lrd_all_serial confermato dallo storico.

## (a) bind-review — esperimenti (r3_scripts/r3_bind.py, clone r3c; commit della review simulato come commit orfano)
Accettati: blocco multi-riga come nel prompt (reviewed_commit fbfde797 o 0efa4d45, entrambi), JSON su una riga, blocco in testa/fondo (anche senza newline finale), spazi dopo il nome, blocco dentro ````text, CRLF in tutto il file, chiusura ```json, testo UTF-8 con accenti.
Accettato anche (NOTA): chiave "status" duplicata, REJECTED poi APPROVED (json.loads: vince l'ultima).
Respinti: due blocchi (modello+reale, anche se il modello sta in ````text come nel prompt), blocchi identici doppi, "``` nome", info "nome json", maiuscole, ~~~, menzione inline, status APPROVATO/approved/APPROVED_WITH_NOTES, schema v2, chiave in piu', chiave mancante, SHA abbreviato/maiuscolo, reviewed_commit=base 438c8d65 ("execution sources differ ... contrastive_benchmark_readout.py"), commit inesistente, source della base, piano storico, hex maiuscolo, JSON con virgola finale (JSONDecodeError), lista (TypeError, non ValueError: comunque chiuso).
Respinto A7: recinto di CHIUSURA indentato (il regex esige "\n```" a colonna 0); l'apertura invece non e' ancorata.
Copia locale: newline in piu' o CRLF -> "not byte-identical"; review_path assente/con ".." -> respinto; copia fuori dal checkout o symlink verso l'esterno -> respinto (relative_to).
Storici: review 94f50942, release 17/11 (contrastive_preflight_release/cuda_benchmark_release) -> 0 blocchi, respinti; il prompt per Claude contiene 1 blocco-modello (respinto per SHA segnaposto) -> se copiato nella review insieme al blocco reale, la review e' respinta (2 blocchi).

## (b)+(d) release solo audit e dispatcher (r3_scripts/r3_release.py; release sintetica fc6e1758 figlia di 0efa4d45, diff = solo review copy + binding + job + config audit)
- B0 validate_review_binding nel commit di release: OK; B1 build_command(audit,cpu,7200) con binding reale: OK (contrastive_audit.py --config ...).
- Respinti: job senza package_review / None; review_payload manomesso; original_review_sha256 diverso; binding verso piano storico; review del job != binding; id calibrazione con mode audit; arm/repeat sull'audit; 3600 s; kind gpu; out == fresh_root o fuori; source storica; status diverso; mode readout con id audit; job audit storico 17/11 (releases/audit.job.json).
- ACCETTATI (NOTA): review_commit falso ("2"*40) e original_review_path falso nel binding: a runtime non sono verificati (solo _full_sha); la provenienza originale e' controllata solo da bind-review sul Mac; check-binding non la riverifica.
- Mutazioni del commit di release: nuovo scripts/contrastive_extra.py, src/wearusfm/__init__.py, plan.json, preflight_plan.json, cancellazione di un sorgente, launcher sbatch -> tutti RESPINTI.
- ACCETTATI (MINORE): nuovo src/sitecustomize.py (eseguito a ogni avvio Python con PYTHONPATH=$CODE_DIR/src: stampa "R3 INJECTED ..."), nuovo scripts/argparse.py (eseguito al posto della stdlib da contrastive_release.py --help), nuovo .py in src/ fuori da src/wearusfm. Nessun controllo di ascendenza o di diff (grep merge-base|is-ancestor|rev-list|diff --name = 0). "Soli metadati operativi" e "figlio del commit rivisto" non sono imposti dal codice.
- Nessun hash circolare: git grep dei tre hash in scripts/src/tests = 0; i piani non citano review/binding.

## (c) gate accounting (r3_scripts/r3_audit.py)
- Diretto: accettati COMPLETED 0:0 7000/7200/0 s, colonne extra/ordine diverso, sacct -p con | finale, altre allocazioni nel TSV. Respinti: 7201, 7380 COMPLETED, TIMEOUT (anche 0:0 con .batch COMPLETED, e riga storica 64652559), FAILED, 0:15, CANCELLED, OOM, 16 CPU, 60 min, "02:00:00", negativo, "7000.0", solo .batch/.extern, due righe allocazione, header JobID, tab, riga di data prima dell'header, TSV vuoto, "COMPLETED ", audit di sorgenti storici, schema/audit_sha/source/logical_id diversi, job_id int o "", job_id != TSV, pin sacct incompleto (sacct assente -> KeyError), TSV o audit modificati dopo il pin, accounting None.
- Applicato: make_job (K2 TIMEOUT respinto; K4 audit senza gate come previsto), build_command (D2 respinto; D1 supera il gate e si ferma dopo), validate_release runtime calibrazione (V5/V6 respinti; V1 ok), read_preflight (Q2/Q3/Q4 respinti; Q1 supera il gate e si ferma al CUDA mancante).
- expected_job_id non e' mai passato dai chiamanti; il report audit non registra SLURM_JOB_ID: l'ID SLURM e' solo dichiarato dall'operatore e confrontato col TSV prodotto dallo stesso operatore.
- V7: validate_release accetta audit_sacct come percorso non pinnato (il dispatcher pero' esige il pin identico nel job).

## Formato del blocco accettato (da codice contrastive_release.py:26,90-103 e prove A/R)
- Apertura: esattamente ``` + "contrastive-release-review" (minuscolo), poi solo spazi/tab facoltativi e a capo. Niente spazio fra ``` e nome, niente altre parole d'info, niente ~~~.
- Chiusura: la prima riga che inizia con ``` a colonna 0 dopo l'apertura (blocco indentato, in elenco o in citazione ">" -> respinto). Testo dopo ``` sulla riga di chiusura tollerato.
- Corpo: un oggetto JSON stretto con ESATTAMENTE 6 chiavi: schema="contrastive-target-release-review-v1", status="APPROVED", reviewed_commit (40 hex minuscoli; fbfde797b48644744c36547533acf0e8a7793535 o 0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba entrambi accettati), source_sha256=40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0, science_plan_sha256=b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a, preflight_plan_sha256=e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92 (hex minuscoli). Indentazione/righe libere.
- Unicita': una sola occorrenza in tutto il file, anche contando esempi/modelli (il modello del prompt in ````text conta come blocco).
- Posizione: ovunque (testa, fondo, anche senza newline finale). File UTF-8; la copia nel checkout deve essere byte-identica al blob del commit della review (CRLF o newline in piu' -> respinto).
- Locale: validate_review_binding legge la copia con read_text() (codifica di locale): con LC_ALL=en_US.US-ASCII una review con accenti da' UnicodeDecodeError; con C/POSIX Python 3.11 usa UTF-8 (provato sul Mac).

## RIPRESA 2026-11-20 03:02 (nuova istanza dopo l'interruzione del 19/11 alle 14:25)
- Stato trovato: mutazioni MU1-MU10 in r3_mutations.tsv (MU11 in corso in r3m, contrastive_release.py modificato nel clone r3m, NON nella copia di riferimento). Copia di riferimento a 0efa4d45, pulita.
- Rileggo il codice da zero e riverifico le affermazioni chiave prima di riportarle.

### Riproduzione 03:15 (cloni nuovi r3x, r3y a 0efa4d45; uscite in tmp/r3x_bind.out, r3y_release.out, r3y_audit.out)
- Hash: source 40d39f56...cde0, plan b03c0bc6...021a, preflight e4b39591...6dd92 = delivery_pin; preflight.source_sha256 e science_plan.sha256 coerenti. Inventario 125 file.
- pytest release+guards+job+run: 71 passed in 11,18 s. test_contrastive_job/run monkeypatchano validate_review_binding/validate_audit_accounting (righe 26-27 e 73-74): integrazione reale non coperta.
- r3_bind.py, r3_release.py, r3_audit.py rieseguiti: TUTTI gli esiti del 19/11 riprodotti identici (A1-A12, R1-R23, L1-L6, H1-H4; B0-B1, N1-N19b, M1-M9; G1-G7, R1-R35, K1-K4, D1-D3, V1-V7, Q1-Q4).
- Hash circolari: nessuno. git grep dei tre hash a 0efa4d45: solo preflight (source+plan), delivery_pin, ricevute di validazione; plan.json cita solo "preparation_review_commit": "94f50942" (storico). Inventario senza docs/ o results/.
- Superficie fuori inventario: nessun modulo di scripts/ non contrastive importato dalla catena; src/ contiene solo wearusfm; nessun file non-.py in src/wearusfm; HISTORICAL_INITIAL_REFERENCE pinnato da costante; nessun .gitattributes. Restano quindi solo i ganci di avvio/ombra (sitecustomize, moduli omonimi in scripts/ o src/).

### (a) casi aggiuntivi 03:30 (r3_scripts/r3_bind2.py, uscita tmp/r3x_bind2.out)
- ACCETTATI: E1 apertura a meta' riga ("Testo ```contrastive-release-review\n" -> conta come blocco: regex non ancorata a inizio riga), E2 recinto a 4 backtick, E4 BOM, E5 apertura indentata con chiusura a colonna 0, E6 menzione ```contrastive-release-review``` inline (non conta: serve a capo dopo il nome), E8 chiavi in altro ordine, E9 righe vuote, E12/E12b chiavi duplicate (vince l'ultima: "status":"REJECTED" poi "APPROVED" passa), E13 escape APPROVED, E16 tab dopo il nome.
- RESPINTI: E3/E3b citazione ">", E7 blocco reale + esempio del formato in ```text (2 blocchi), E10 file latin-1 (UnicodeDecodeError), E14 "}```", E15 blocco vuoto prima del reale, E17 null, E18 chiave in piu' NaN.
- CLI: bind-review rc 0 (7,4 s); check-binding rc 0; --out esistente -> FileExistsError rc 1; SHA abbreviato rc 1. Review con accenti: LC_ALL=C rc 0 (UTF-8 mode), LC_ALL=en_US.US-ASCII rc 1 UnicodeDecodeError (read_text nel validate; bind usa decode UTF-8).
- Tempo validate_review_binding: 4,36 s sul Mac (127 git show), 0,02 s senza git.

### (b) checkout simulato 03:35
- Worktree staccato e bloccato dal commit di release sintetico bbc1622b (figlio di 0efa4d45), file e cartelle a-w come campaign_checkout.sbatch: 0 scrivibili, status pulito. check-binding nel worktree (PYTHONPATH=src, cwd=checkout) rc 0 in 5,6 s; build_command(audit, cpu, 7200) dal job JSON committato: OK in 3,7 s. git show funziona nel worktree in sola lettura (oggetti condivisi col repo canonico).

### (b)/(d) altri esiti 03:50
- check-binding CLI su binding con review_commit "2"*40 (inesistente) e original_review_path inesistente: rc 0, accettato. La provenienza e' verificata solo dentro bind-review; nessun comando la riverifica dopo (MINORE: un binding scritto a mano o modificato dopo bind-review passa ovunque, purche' la copia nel checkout contenga un blocco APPROVED con gli hash correnti).
- Commit di release con stesso albero ma ORFANO (nessun genitore) o con genitore 438c8d65: check-binding rc 0 in entrambi (merge-base --is-ancestor fbfde797 HEAD = no). "Figlio del commit rivisto" non imposto; innocuo per inventario/piani (byte confrontati), ma con un rilascio orfano l'oggetto fbfde797 resta raggiungibile solo da altri ref (gc).
- Formato sacct storico (accounting_and_integrity.txt del 17/11) e' senza intestazione e con colonne diverse: con lo stesso comando il gate rifiuterebbe ("unambiguous allocation fields"). Serve -P con intestazione JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw (prontuario lo dice, senza il comando). AllocCPUS=8 e TimelimitRaw in minuti confermati dalla riga storica 64652559 (TIMEOUT|...|3787|60|8).
- Report dell'audit (contrastive_audit.py, invariato) senza SLURM_JOB_ID; expected_job_id mai passato dai chiamanti (grep: 4 chiamate di produzione senza). L'ID SLURM dell'accounting e' solo dichiarato e confrontato col TSV.
- Nessun nome di campagna/deploy storico collide con contrastive-targets-audit-20261119 (grep su docs/results/scripts).

### Formato del blocco (riverificato 04:00, codice contrastive_release.py:26 e 90-103 a fbfde797)
- Regex: ```` ```contrastive-release-review\s*\n(.*?)\n``` ```` con DOTALL, findall su tutto il file; deve trovare ESATTAMENTE 1 occorrenza.
- Apertura: sottostringa "```contrastive-release-review" (minuscolo, nessuno spazio fra ``` e nome), poi solo spazi/tab e a capo. NON ancorata: conta anche a meta' riga, indentata o dentro un recinto ````text; 4 backtick accettati. Da evitare qualunque altra riga che contenga quella sottostringa seguita da a capo (esempi, modello del prompt).
- Chiusura: il primo "\n```" dopo l'apertura, quindi ``` a colonna 0 su riga propria (indentata, in elenco o citazione -> respinto o JSON invalido).
- Corpo: oggetto JSON con esattamente 6 chiavi (ordine e spazi liberi; duplicati risolti con l'ultimo valore): schema "contrastive-target-release-review-v1", status "APPROVED", reviewed_commit 40 hex minuscoli, source_sha256/science_plan_sha256/preflight_plan_sha256 64 hex minuscoli.
- Valori accettati oggi: reviewed_commit fbfde797b48644744c36547533acf0e8a7793535 (content_commit; anche 0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba passa), source 40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0, plan b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a, preflight e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92.
- File UTF-8 (bind decodifica strict UTF-8); posizione libera; la copia nel checkout deve essere byte-identica al blob del commit della review (copiare con git show, non con un editor).
- Review realistica (testo con accenti, recinti ```bash/```text prima e ```python dopo, blocco in fondo): bind-review CLI rc 0, 1 blocco trovato.

### Bozza rilievi (04:10)
- Nessun BLOCCANTE trovato nel punto 3.
- MINORE-1 superficie fuori inventario / figlio non imposto (M7-M9 accettati, orfano accettato).
- MINORE-2 provenienza della review non riverificabile dopo bind-review (check-binding/dispatcher/runtime accettano review_commit inesistente).
- MINORE-3 ID SLURM dell'accounting solo dichiarato (report audit senza SLURM_JOB_ID, expected_job_id mai passato).
- MINORE-4 copertura dei test: mutazioni sopravvissute (vedi r3_mutations2.tsv), integrazione dispatcher/runtime con guardie monkeypatchate a no-op.
- NOTA-1 tolleranze del parser (apertura non ancorata, 4 backtick, chiavi duplicate: vince l'ultima; un esempio APPROVED conta come approvazione).
- NOTA-2 read_text() con codifica di locale nel validate (ASCII -> UnicodeDecodeError), bind usa UTF-8.
- NOTA-3 procedura delle ricevute per make-job non-audit: storico calibration.job.json pinnava l'audit con percorso assoluto Leonardo; make_job "locally available" fallisce sul Mac con quel pin (FileNotFoundError); servono pin relativi nel commit di release (o make-job sul cluster); pin() di default e' assoluto. Formato sacct storico senza intestazione respinto dal gate: indicare il comando.
- NOTA-4 V7: validate_release accetta audit_sacct non pinnato (stringa); il dispatcher lo esige pinnato.

### Mutazioni completate 04:25 (r3_scripts/r3_mutate2.py, clone r3mm a 0efa4d45, 7 file di test, base 125 passed; esiti in tmp/r3_mutations2.tsv; stato finale clean)
- 39 mutazioni: 14 KILLED, 25 SURVIVED, 0 non applicate.
- KILLED: MU4 (nessun verify_git), MU7 (copia non byte-identica), MU10 (hash dichiarati vs correnti, solo dal caso [source]), MU11-MU18 (durata 7380, <7200, TIMEOUT, CPU, TimelimitRaw, ExitCode, allocazioni duplicate, ID logico), MU20 (audit_sha256), MU31 (gate nella lettura: spia in test_contrastive_historical_callsites), MU38 (override della cella).
- SURVIVED: MU1 piu' blocchi, MU2 status, MU3 chiavi esatte, MU5 piani vs commit rivisto, MU6 payload/sha del binding, MU8 altra coorte, MU9 legame interno del preflight, MU19 source del report audit, MU21 gate in make_job per celle non audit, MU22/MU28 binding nel dispatcher/runtime, MU23 review del job, MU24 cella nel dispatcher, MU25/MU30 gate accounting nel dispatcher/runtime, MU26 out fuori da fresh_root, MU27 release con stesso binding/accounting, MU29 review_pin, MU32 review fuori dal checkout, MU33 intestazione sacct (quasi equivalente: KeyError), MU34 prerequisito CUDA nel runtime, MU35/MU39 (ridondanti fra loro: digest e byte), MU36 contenimento del percorso in git_bytes, MU37 job_id non numerico.
- Lettura: le guardie funzionano (prove dirette sopra), ma i test non ne rilevano la rimozione; dispatcher e runtime sono testati con le due guardie monkeypatchate a no-op.

### Chiusura 04:30
- Cloni r3c, r3m, r3mm, r3x, r3y, worktree r3wt (sbloccato e rimosso), basetemp r3x_pt1, r3_work e log vuoti cancellati. Conservati r3_scripts/ (r3_bind.py, r3_bind2.py, r3_release.py, r3_audit.py, r3_mutate.py, r3_mutate2.py) e le uscite r3x_bind.out, r3x_bind2.out, r3y_release.out, r3y_audit.out, r3_mutations.tsv, r3_mutations2.tsv.
- Copia di riferimento rev_targets/clone: HEAD 0efa4d45, status pulito.
- Esito del punto 3: nessun BLOCCANTE; 4 MINORI (fuori inventario/figlio non imposto; provenienza review non riverificata; ID SLURM solo dichiarato; copertura dei test), 4 NOTE.
