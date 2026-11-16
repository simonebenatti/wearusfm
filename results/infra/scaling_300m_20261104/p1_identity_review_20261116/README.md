# Review P1 identity, 16/11/2026

Consegna esaminata: c2ae7ee02b3df5bdc3310b90d6608e211753dd07.
Aggiornamento documentale successivo letto: 01f853e (esito suite Claude).
Nessuna SSH, raccolta remota, inferenza, modifica del codice Claude o job.

Review principale e protocollo da congelare in `docs/reviews/`:

- `gpt_review_scaling_p1_identity_20261116.md`;
- `scaling_p1_identity_diagnostic_protocol_20261116.md`.

Tre audit indipendenti, **46 controlli complessivi**:

- `p1-identity-code-20261116.py/json/md`: 9 controlli su AST e sequenze
  stat/hash simulate; controesempio del fallback che accetta la sostituzione
  dell'inode a byte uguali. Non è una riproduzione locale di Lustre.
- `p1-mixed-review-20261116.py/json/md`: 8 controlli su fixture del builder
  e schema del lettore, provenienza distinta per P1; nessuna cache reale
  caricata. Le modifiche al runner archiviato riguardano soltanto argomenti
  `--repo/--out`, per rendere la riproduzione indipendente dal percorso tmp.
- `p1-evidence-budget-20261116.py/json/md`: 29 controlli sui sacct/log/stat
  pubblicati, costi esatti con frazioni e hash delle fonti.
- `p1-identity-npz-determinism-20261116.json`: prova locale piccola di
  uguaglianza dei byte NPZ tra BytesIO seekable e percorso. La stessa prova
  è richiesta nel runtime effettivo prima delle fixture grandi.

Le note dei revisori conservano i riferimenti `/private/tmp` d'origine;
le copie definitive e i loro hash sono qui. I JSON dei tre audit sono stati
rigenerati dal revisore principale e risultano byte-identici.

Riproduzione, con una copia locale pulita della consegna e directory d'output
temporanea; i primi due comandi bastano con la libreria standard:

```bash
python3 p1-identity-code-20261116.py --repo /percorso/c2ae7ee --out /tmp/code.json
python3 p1-evidence-budget-20261116.py --repo /percorso/c2ae7ee --out /tmp/budget.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ~/.venvs/wearusfm-tok/bin/python -B p1-mixed-review-20261116.py \
  --repo /percorso/c2ae7ee --out /tmp/mixed.json
```

Il runner mixed inserisce `src/` e `scripts/` della copia selezionata in testa
al path. Il repository scientifico non viene modificato. Nessun payload
wearable incluso: solo prove sintetiche, statistiche e pin dei materiali
pubblicati. La suite 3.555/3.555 è un risultato riportato da Claude, non
rieseguito da questa review.

Fonti primarie esterne consultate il 16/11/2026:

- [Manuale Lustre, §§21 e39.4.2.2](https://doc.lustre.org/lustre_manual.pdf):
  LSoM e statahead/AGL; non prova del difetto specifico di Leonardo.
- [stat(2)](https://man7.org/linux/man-pages/man2/stat.2.html) e
  [statx(2)](https://man7.org/linux/man-pages/man2/statx.2.html): campi e
  sincronizzazione degli attributi, non snapshot transazionale del contenuto.
- [inode(7)](https://man7.org/linux/man-pages/man7/inode.7.html): significato
  distinto di atime/mtime/ctime.
- [sync(2)](https://man7.org/linux/man-pages/man2/sync.2.html) e
  [fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html): ambito,
  sincronizzazione e gestione degli errori.

Nessuna versione client o configurazione Lustre di Leonardo è inferita
dalla versione corrente dei manuali; vanno registrate nel job diagnostico.
