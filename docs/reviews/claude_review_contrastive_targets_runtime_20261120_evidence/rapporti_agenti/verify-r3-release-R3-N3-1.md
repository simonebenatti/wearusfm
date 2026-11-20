# verify-r3-release-R3-N3-1 — scettico n. 1 sul rilievo R3-N3 (NOTA)

Avvio: 2026-11-20. Copia di riferimento rev_targets/clone a 0efa4d45 (sola lettura).

## Rilievo da confutare
R3-N3: procedura delle ricevute per make-job delle fasi successive e comando sacct non esplicitati
(pin assoluti di Leonardo -> FileNotFoundError sul Mac; pin assoluti del Mac -> fallimento sul cluster;
sacct storico senza intestazione respinto dal gate; nessun generatore per accounting JSON / release delle fasi successive).

## Progressi
- Letture (codice a fbfde797): contrastive_release.py pin() r.34-37 assoluto per default; checked_pin r.40-51 strict;
  make_job r.240-242 legge audit+accounting+TSV per le celle non audit; contrastive_job.checked_file r.22-32 legge i byte;
  build_command r.80-81 rilegge audit e accounting sul cluster prima di qualunque altra cosa costosa.
- Storico: releases/calibration.job.json audit.path=/leonardo_work/.../audit.json, 20.275.571 byte; accounting_and_integrity.txt
  riga 1 senza intestazione (14 campi: comprende i sei richiesti, piu' altri).
- Documenti: prontuario r.140-146 fissa i campi del TSV `JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw` ma non
  il comando; r.178-179 rinvia i comandi definitivi alla consegna post-review; r.187 "I sei checkout congelano solo le nuove
  evidenze"; prompt r.58-60 "checkout con soli metadati". Nessuna indicazione su pin relativi o su dove gira make-job per le
  fasi successive (grep su risposta, decisioni, registro, piani: nulla).
- Generatori: AUDIT_SCHEMA e APPROVED_FOR_PREPARATION prodotti solo nei test (git grep a fbfde797).
- Test: test_contrastive_release.py:105 usa "sacct": C.pin(tsv) (assoluto); make_job provato solo sulla cella audit (r.144-151).
- submit_job.py r.12: ID 'sottomesso' mai risottomesso ("Usare un ID nuovo"); preflight failure_policy: nessun retry.

## Riproduzione (clone temporaneo vn3_1_clone a 0efa4d45 + commit di review sintetico 9c5da9f3; script vn3_1_scripts/exp.py)
- A: make_job(binding, CONTRAST-CALIB..., audit = pin storico /leonardo_work/.../audit.json) -> FileNotFoundError '/leonardo_work'
  (resolve strict in checked_pin). Controllo: make_job della cella audit OK (nessuna lettura locale).
- B: ricevute fuori dal checkout con C.pin() di default (percorso assoluto del Mac): make_job OK; build_command(job,'gpu',1800)
  con ricevute presenti supera il gate (si ferma a KeyError 'config', contrastive_job.py:103); con ricevute assenti (come su
  Leonardo) FileNotFoundError nel checked_file(job['audit']) -> il job fallirebbe all'avvio.
- C: ricevute copiate nel commit con pin relativi: make_job OK sul "Mac"; in un secondo clone (ROOT diverso, come il checkout
  del cluster) build_command supera il gate (KeyError 'config' a r.103). Il codice attuale consente dunque la procedura giusta
  senza modifiche ai sorgenti (che la review congela).
- F: pin interno dell'accounting fatto come nel test (test_contrastive_release.py:105, "sacct": C.pin(tsv), assoluto) con
  audit/accounting relativi: make_job OK, build_command in altro ROOT FileNotFoundError sul TSV.
- D (sacct): righe 1-3 storiche senza intestazione -> "audit sacct requires unambiguous allocation fields" (r.178);
  stesse 14 colonne storiche CON intestazione -> formato accettato (TIMEOUT respinto dal gate di stato; COMPLETED/120 accettato);
  -n -P con i sei campi -> respinto. La causa e' solo l'assenza dell'intestazione (-n), non le "altre colonne": imprecisione lieve.
- E: C.pin default assoluto, C.pin(relative=True) relativo, contrastive_prepare._pin relativo dentro ROOT (due helper con default diversi).
- Generatori: nessuno per accounting JSON o release APPROVED_FOR_PREPARATION (git grep, solo test). Nessun controllo del job JSON
  sul cluster prima dell'invio (launcher exec diretto di contrastive_job.py; campaign_checkout.sbatch crea solo il worktree).
- Baseline: pytest tests/cpu/test_contrastive_release.py nella copia di riferimento: 22 passed in 10,29 s; copia pulita.

## Motivi cercati per confutare (insufficienti)
- Prontuario r.143 fissa i campi del TSV (intestazione implicita), non il comando: mitiga solo la parte sacct, che e' comunque
  a prova di errore (rifiuto in make_job sul Mac, prima di ogni invio; sacct si puo' rieseguire).
- Prontuario r.178-179 rinvia i comandi definitivi alla consegna post-review; r.140 e r.187 dicono che le evidenze fra fasi si
  pubblicano/congelano nei checkout, ma non come pinnarle ne' dove gira make-job. Docstring di make_job: "must be locally available".
- Prontuario storico 16/11 r.47 "Pin relativi rendono il ricalcolo identico sul Mac e sul cluster" (ricevute di calibrazione),
  ma il documento e' marcato storico e superato dal 19/11, e per l'audit lo storico usava il percorso assoluto di Leonardo.
- _read non legge .gz: un pin relativo richiede il report audit non compresso (~20 MB) nel commit, a differenza dell'archivio
  storico (audit.json.gz 1,1 MB): un motivo in piu' per esplicitare la procedura.

## Esito
NON confutato. Gravita' reale: NOTA (documentazione della procedura post-audit). Non BLOCCANTE: esito dipendente dall'operatore,
fail-closed (dispatcher rifiuta, nessuna evidenza invalida accettata), release solo audit non toccata, costo GPU di un avvio
fallito trascurabile. Unico caso che arriva al cluster: pin assoluto del Mac (default di C.pin e schema del test), con perdita
dell'ID a tentativo unico (submit_job.py:12, failure_policy). La correzione documentale (pin relativi nel commit di release e
comando `sacct -j <ID> -P -o JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw` con intestazione) basta col codice
attuale; il rafforzamento repository=True cambierebbe i sorgenti e richiederebbe una nuova review.
- Pulizia: cloni vn3_1_clone/cluster/cluster2, ricevute e basetemp cancellati; restano vn3_1_scripts/ (setup_review.py, exp.py). Copia di riferimento a 0efa4d45, pulita.
