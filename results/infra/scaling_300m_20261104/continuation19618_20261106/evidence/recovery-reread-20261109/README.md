# Rilettura in sola lettura prima del deploy del recupero v2 (09/11/2026)

Letta dal Mac con `continuation19618_20261109_v2/remote_read.sh` su login05, fra le 11:43:36 e le 11:44:06 CET, dopo il
rinnovo del certificato ssh. Nessuna scrittura sul cluster fuori da `/tmp` del login.

- `remote_listing.txt`, `remote_sha256.txt`: uscite grezze divise a `@@@@SPLIT@@@@`. Tolte solo le tre righe d'avviso di ssh e la
  riga del saldo di un altro account (`CNHPC_1526560`), fuori perimetro.
- `recheck.json`: uscita di `make_training_plan.py remote_listing.txt remote_sha256.txt --check`, `passed: true`. Il piano
  ricostruito da queste letture e' identico a `continuation19618_20261109_v2/training_plan.json`: stessi pin del run a 5.000, delle
  ricevute, del controllo 30M, delle cache e degli ingressi. Cartelle del run, delle ricevute e del pre-controllo v2, checkout e
  riferimento di deploy del recupero: assenti.
- `attest_spent.py`, `attestation.json`: il `sacct -P` originale dei sei job gia' eseguiti (checkout v1 63695634, training v1
  63696864, diagnosi 63860499, 63861166, 63861825, 63864244) contro le righe `spent` dell'inviluppo firmato
  (`budget_envelope.json`, SHA-256 `224743df...`): durate, CPU, GPU e stato coincidono per tutti e sei (`all_match: true`).
  Picco di memoria della diagnosi 63861166, che ha calcolato gli stessi insiemi del pre-controllo su 4 CPU: 2.993.724 KiB.
