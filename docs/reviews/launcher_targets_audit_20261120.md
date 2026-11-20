# Linea 3 — checkout e audit RMS100/fast125, 20 novembre 2026

Operatore: 3-launcher. Istruzioni 1b6fdeaee22cca4a0a0a3c31728db71dee47a817;
esecuzione 9abde2951cd9b2a25aa699faceb93132de6732f8. Nessuna modifica tecnica.
Piano rigenerato e confrontato byte per byte con la consegna; submitter comune
814da4bc… verificato. Ref di deploy pubblicato e SHA remoto verificato.

## Sequenza e controlli

Prima del checkout: saldo 655.982 ore locali, WORK libero circa 41,28 GiB,
corsia seriale UP, manutenzione confermata 23/11 08:00 CET–25/11 08:00 CET.
Riserva della coorte 12 GiB; mantenuta ulteriore riserva prudenziale di 18 GiB
per altre scritture (gate operativo 30 GiB liberi prima di ciascun invio).
Snapshot di tutti i job del progetto, reservation e quota conservati nelle
evidenze. Il job scaling sbenatti 64837401 è PENDING (32 CPU, 19:45:00);
i job degli altri utenti del progetto sono inclusi nel computo residuo.
Non si sommano nuovamente le risorse storiche già addebitate al saldo.

Un primo controllo login ha incontrato il ref non ancora visibile perché il
push era ancora in corso: nessun job era stato inviato. Atteso il completamento
del push e verificato lo SHA, il controllo congelato è stato rieseguito con
esito positivo. Nessuna modifica al piano e nessun retry di ID SLURM.

Binding APPROVED, audit arm=null, config C0/calibration_only=true,
RMS100/fast125, quattro pin input e argv contrastive_audit.py --steps 1024
verificati sul login. Stdout integrale in checkout_login_check.txt.

Checkout **64936161**, COMPLETED 0:0, 47 s, 4 CPU, limite 4 minuti:
consumo 0 GPU-ore / 0,052222 core-ore. L'audit è subordinato al controllo
congelato sul checkout effettivo e a una nuova fotografia operativa.

La consegna si limita a questo audit. Nessuna calibrazione o fase GPU viene
inviata automaticamente.

## Audit sottomesso

Seconda fotografia operativa 12:04:40 CET: saldo 655.982 ore locali, residuo
degli altri job del progetto 5.225,126667 ore (proxy billing=CPU), inviluppo di
questa fase 17,466667 ore. Saldo netto prudenziale 650.739,406666 ore senza
ricontare storico o impegni già nel saldo. WORK libero 41,13 GiB, oltre il gate
operativo di 30 GiB; corsia seriale UP, reservation e calendario ricontrollati.
Prima fotografia: residuo altri job 5.229,1 ore e saldo netto 650.735,433333.

Controllo congelato audit-login positivo: ledger checkout, sacct COMPLETED 0:0,
HEAD, pulizia, permessi, binding, build_command e validate_recipe verificati.
Audit **64937087**, ID CONTRAST-AUDIT-targets-v3-20261118, campagna
CONTRASTIVE-targets-v3-20261118, 8 CPU / 30 GiB / 120 min. Un solo invio.
Stdout integrale ed evidenze in audit_execution_20261120/.
Rapporto atteso: `$WORK/wearusfm_runs/results/contrastive_targets_20261118/audit.json`.
Esito, hash, accounting autenticato e consuntivo finale restano da raccogliere.

## Raccolta conclusiva

Audit **64937087 COMPLETED 0:0**, 4.022 s (1 h 07 min 02 s), 8 CPU, limite
120 minuti. Rapporto positivo: **1.024 passi / 1.024 record**, copertura globale
e pannello superati. Frazione query InfoNCE ammissibili: 99,8006% sui 768 update
attivi della copertura; 99,8413% sui 256 update del pannello. RnC attivo nel
100% degli update in entrambi. CapgMyo resta esente secondo la regola già firmata;
nessuna nuova esenzione. Si tratta di disponibilità di candidati e supporto,
come dichiarato nel rapporto: non è una misura di qualità del modello.

Config canonica, quattro pin degli input, digest sorgenti e numero di record
confrontati con il job congelato: tutti coincidenti. Rapporto remoto originale
20.275.571 byte; SHA256
`eb7cbdf49ffdd00148647e82c5f51768edbd17b5ef515d960a1f6edec35754d3`.
Archiviato senza modifiche in audit_execution_20261120/audit.json.gz;
SHA256 gzip `41472b53ff924b26f73ffc859e0e560c553f873915055fd8d1ba0029926bb9de`.

Ricevuta sacct: audit.sacct.tsv, 92 byte, SHA256
`8d2c71fbcbe883cf906b8349d5b104882307a26e859cc95629bf295f1c413cfe`.
Contiene intestazione e unica riga dell'allocazione 64937087. Creato
audit_accounting.json, schema contrastive-audit-accounting-v1, SHA256
`594d72932f2ca15440ee4bf1255a0abe035467933703efcd23309a6e14a32726`.
Comando check-audit del codice congelato eseguito con esito zero e ricevuta
accounting_validation.json. Sorgenti locali scripts/src byte-identici alla
versione eseguita. Nessuna eccezione di timeout o scostamento dai limiti.

| Risorsa | Richiesto | Effettivo | Residuo rispetto al richiesto | Inviluppo con margine |
|---|---:|---:|---:|---:|
| GPU-ore | 0 | 0 | 0 | 0 |
| Checkout core-ore | 0,266667 | 0,052222 | 0,214445 | 0,666667 |
| Audit core-ore | 16 | 8,937778 | 7,062222 | 16,8 |
| Totale core-ore | 16,266667 | **8,99** | 7,276667 | 17,466667 |

Billing osservato uguale alle CPU allocate, quindi addebito incrementale
locale 8,99 ore. Il saldo alla sottomissione meno questa fase effettiva è
655.973,01 ore, prima dei nuovi consumi delle altre linee; non è un saldo
corrente misurato. Cumulativo linea 3 storico più nuova fase: 6,086667 GPU-ore
e 66,817778 core-ore. Report, log e ricevute raccolti: consegna conclusa.
Nessuna calibrazione GPU inviata; la release successiva resta alla chat scientifica.
