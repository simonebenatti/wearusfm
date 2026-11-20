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
