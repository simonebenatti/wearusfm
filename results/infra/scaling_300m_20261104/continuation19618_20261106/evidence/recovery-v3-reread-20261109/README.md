# Rilettura in sola lettura prima del deploy del recupero v3 (09/11/2026, 12:21)

`continuation19618_20261109_v3/remote_read.sh` su login, dal Mac. Tolte dalle copie: le tre righe d'avviso di ssh, le
associazioni e il saldo di altri account, e i job MONALISA in coda (19 righe `mona_*`), fuori perimetro.
`recheck.json`: `make_training_plan.py remote_listing.txt remote_sha256.txt --check` del v3 con esito `passed: true`. Il piano
ricostruito e' identico a `continuation19618_20261109_v3/training_plan.json`. Cartelle del run, delle ricevute e del
pre-controllo v3, checkout e riferimento di deploy del v3: assenti. Nessun job di questa linea in coda.
