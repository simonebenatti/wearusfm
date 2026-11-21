# Riprese RMS100/fast125 — controllo preliminare interrotto

**Seguito operativo bcb6bd1d:** riconciliazione, prova breve e unico replay
completo riusciti. Checkout 65023425 sottomesso; le note sull'arresto qui
sotto restano lo storico del primo tentativo, non lo stato corrente.

Consegna 95acb366, codice 12e215a978b9d30ea6d670f58af1a0f97387d477.
21/11/2026, 3-launcher. Piano rigenerato byte-identico e pin verificati;
ref deploy pubblicato e SHA remoto verificato. Nessuna modifica tecnica.

Snapshot prima del controllo checkout 12:11:16 CET: saldo disponibile
653.928 ore locali, WORK libero circa 36,84 GiB, oltre la riserva di 12 GiB
per la coorte più 18 GiB per altre scritture. Nessun job debug dell'utente;
reservation/manutenzione confermata 23/11 08:00–25/11 08:00 CET. Snapshot
completo in reloads_execution_20261121/checkout_operational_snapshot.txt.

Il check congelato **before_checkout è terminato con codice SSH 255**, dopo
accounting dei prerequisiti e verifica positiva di tutti i file storici tramite
sha256sum e dopo il caricamento dei moduli. Il log non contiene un errore Python
remoto di binding o ricetta; contiene il CalledProcessError del generatore
locale sull'uscita SSH 255. Non è dimostrata la causa del distacco. Non è
dimostrato un difetto scientifico o del pacchetto; non si assume però passato
il resto del controllo, che non ha prodotto l'attestazione finale.

Il gate è stato rispettato: **nessun checkout o replica sottomessi**.
Verifica successiva in sola lettura alle 12:12:03 CET: nuova connessione
riuscita su login02, ref deploy corretto, checkout e radice reloads assenti.
I cinque ID sono assenti sia dal ledger sia dalle sue directory di prenotazione.
Nessun ID SLURM assegnato. Consumo allocazioni di questa consegna: 0 GPU-ore /
0 core-ore. Non vengono inclusi in questo consuntivo i controlli di login.

Log integrale in reloads_execution_20261121/checkout_login_check.txt.
Stato **STOPPED_BEFORE_SUBMISSION**. Non ripetuti automaticamente check o invii;
la fase attende la lettura della diagnosi e la decisione operativa successiva
prevista dal prontuario. Codice, fork, soglie e piani rimangono invariati.

## Replay autorizzato del controllo, senza retry SLURM

Emendamento bcb6bd1d, stesso SHA di esecuzione e piano. Riconciliati tutti
i cinque ID ancora liberi, destinazioni assenti e nessun processo del precedente
controllo rilevato nella nuova sessione. Prova breve su checkout dei continui
login02, import NumPy 1.26.4/Torch 2.2.0a0 e binding APPROVED: **14,935740 s,
exit 0**. Unico replay integrale before_checkout su connessione nuova senza
multiplexing, diagnostica SSH separata: **101,777361 s, exit 0**. Parsing delle
ricevute conferma binding APPROVED, audit autenticato e quattro validazioni
recipe_valid/release_valid, ID e fork coerenti per CN_A/B e CR_A/B.

Originale checkout_login_check.txt intatto; stdout/stderr/meta/SSH del replay
sono file distinti. Non cambia la ricetta, nessuna tolleranza e nessun criterio
di controllo. Causa precisa del precedente 255 ancora indeterminata.
Checkout **65023425** inviato una sola volta dopo il controllo positivo.
