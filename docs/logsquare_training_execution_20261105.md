# T100-LQ: esecuzione dopo la correzione dell'inizializzazione

Release dedicata `2293b94ffb5f2bad2639d30a297c14eea034d233`: due soli file,
approvazione e pin. Diagnostico63591184 positivo e ricalcolo indipendente; decisione
in `docs/reviews/logsquare_initialization_results_20261105.md`.

Piano attivo: `results/infra/logsquare_training_20261105/training_plan.json`.
Release attiva nello stesso percorso, con quattro controller, guard, evidenze e review
pinnati. I vecchi file del 04/11 restano immutati e non sono più il piano da eseguire.

| Fase | ID logico | Risorse | Limite | Tetto GPUh / coreh |
| --- | --- | --- | --- | --- |
| Checkout | LOGSQ-training-checkout-20261105 | 2 CPU, 4G | 10 min | 0 / 0,333333 |
| 5.000 | LOGSQ-T100-LQ-5000-20261105 | 4 GPU, 32 CPU, 480G | 01:30:00 | 6 / 48 |
| 19.618 | LOGSQ-T100-LQ-19618-20261105 | 4 GPU, 32 CPU, 480G | 04:00:00 | 16 / 128 |

Stima da rho_update già approvato: 16,7052 GPUh; tetto22GPUh/176,333333coreh
checkout incluso. Nessun aumento di dose, loss, seme o epsilon.

Checkout nuovo `logsquare-training-20261105` al commit pubblicato che contiene questo
documento e i pin A4; SHA effettivo e ID nel launch_manifest della cartella risultati.
Nessun checkout di campagna preesistente modificato. Nuovo run in
`$SCRATCH/wearusfm_runs/runs/logsquare_20261105/T100-LQ`, ricevute in
`$WORK/wearusfm_runs/results/logsquare_training_20261105/T100-LQ/{5000,19618}`.

Invii con submitter/ledger comuni, override espliciti dei due tempi. Prima fase dopo
checkout COMPLETED e verificato; continuazione con afterok del nuovo job a5.000.
Se il checkout già completato non è più nella tabella attiva, il gate è la verifica
indipendente di sacct e dei file, senza dipendenza ormai soddisfatta. Nessun retry cieco.

R1/R2, D1–D3 e N1–N4 restano quelli approvati. Una continuazione parziale per time_limit
non viene riavviata da questo controller. Nessuna modifica ai buffer per superare i gate:
pre-controllo del controller, verifica dello studente effettivo in ogni rank e verifica
del gain nei checkpoint. Entrambe le fasi usano stesso codice e piano.

A4: aggiornato prima `extraction_plan.training_plan` al nuovo file del05/11,
poi `inference_plan.extraction_plan`. I 110 sorgenti valutativi restano invariati;
checkpoint futuri, ricevuta e budget delle estrazioni restano non eseguibili finché
mancano. Benchmark rho_infer a5.000 prima delle estrazioni; la continuazione non lo
aspetta. Pin e copie WORK dei checkpoint da verificare prima della scadenza SCRATCH.

Autorizzazione operativa già acquisita, anche per questa ripartenza dopo diagnosi e
correzione revisionata. Nessun nuovo consenso, nessun monitor in background.
