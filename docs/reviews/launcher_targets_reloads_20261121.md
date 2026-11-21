# Riprese RMS100/fast125 — esecuzione e raccolta concluse

**Stato corrente: quattro riprese COMPLETED 0:0**, endpoint 576 e raccolta
verificata. Checkout 65023425 concluso. Seguito operativo bcb6bd1d e
autorizzazione unica di Simone, procedura 92d1fc0a recepita a df5c8c43.
Le note sugli arresti e sull'attesa qui sotto sono lo storico delle fasi precedenti.

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

## Attesa del checkout e controllo anticipato

Alle 12:22:40 CET, checkout 65023425 ancora **PENDING QOSMaxCpuPerUserLimit**,
0 CPU allocate, 0 s. È attivo TCNH-wls-calibration-0-20261120 (65023397,
8 CPU, residuo 3:52:35); TCNH-wls-calibration-1 (65023399, 8 CPU) è anch'esso
in coda per lo stesso limite. Nessuna modifica a priorità, risorse o job di altre linee.

3-launcher ha anticipato per errore before_CN_A mentre il checkout era ancora
PENDING: il guard congelato lo ha rifiutato sulla riga sacct del predecessore,
prima della validazione della replica e di qualsiasi sbatch GPU. Log e snapshot
preservati come CN_A_early_pending_guard.txt e CN_A_early_pending_snapshot.txt.
È un errore di sequenziamento operativo, non un errore di ricetta o di fork.
Il controllo sarà ripetuto soltanto quando il prerequisito diventa COMPLETED 0:0.
L'orchestrazione locale ora verifica questa condizione prima ancora di chiamare
il controllo della replica; nessun cambiamento al codice congelato.

Stato corrente **WAITING_FOR_CHECKOUT**, non FAILED_REQUIRES_REVIEW scientifico.
Tutte le quattro GPU restano non inviate; nessun ID GPU prenotato e nessun costo
di allocazione della tranche finora. Gli invii restano quelli già autorizzati.

## Esecuzione e raccolta del 21/11, concluse alle 15:06:21 CET

| Cella | Job | Esito | Durata SLURM | Aggiornamenti |
|---|---:|---|---:|---|
| Checkout | 65023425 | COMPLETED 0:0 | 39 s | codice 12e215a9 |
| CN_A | 65033016 | COMPLETED 0:0 | 265 s | 545–576 |
| CN_B | 65033756 | COMPLETED 0:0 | 213 s | 545–576 |
| CR_A | 65033804 | COMPLETED 0:0 | 205 s | 545–576 |
| CR_B | 65034107 | COMPLETED 0:0 | 210 s | 545–576 |

Ogni ID inviato una sola volta dal submitter persistente e dal ledger comune;
nessun retry SLURM. Picco misurato dalla cronologia: due riprese simultanee.
CR_B inviata dopo CN_B COMPLETED. Confermati prima degli invii saldo, quota,
reservation, finestra di manutenzione, pin, identità del submitter, checkout
pulito e protetto, audit, ricette e fork. Quota WORK letta anche da Lustre
per il progetto 20147512: 988,4–989,2 GiB su 1 TiB durante gli ultimi invii,
oltre la riserva complessiva di 30 GiB. Nessun job o checkout di altre linee modificato.

Il controllo CN_B via connessione multiplexata si è chiuso con SSH 255,
senza ricevuta finale e senza invio. Checkout corretto e pulito alla verifica
separata; la connessione nuova senza multiplexing ha poi completato tutto
il controllo congelato con exit 0. Anche una lettura di stato successiva
via multiplexing ha restituito 255; la lettura su connessione nuova è riuscita.
Questo localizza il problema operativo sul trasporto/sessione, senza
identificarne la causa precisa e senza attribuirlo alla ricetta. Soglie e
comandi scientifici invariati. Gli invii sono stati eseguiti direttamente
con interprete e submitter persistenti del piano, senza wrapper temporaneo.

Raccolti e verificati SHA-256 e byte di **88 file** delle quattro celle,
più quattro log SLURM originali e ricevute. Per ciascuna: copia iniziale del
fork identica, fase A/B corretta, 32 metriche e 32 record in ciascuna delle
otto tracce, quattro profili numerici, conclusione max_steps a 576. I file
origin_sessions sono identici alle copie storiche locali già raccolte e
sono verificati anche contro gli originali remoti. I checkpoint rimangono
su WORK, non sono copiati nel repository; percorso, byte e SHA sono nel
[rapporto operativo](../../results/infra/contrastive_targets_20261118/reloads_execution_20261121/collection_readout.json).
Entrambi i fork sorgente sono ancora identici alle impronte firmate,
463.142.486 byte ciascuno.

Le due riprese CN producono lo stesso checkpoint finale byte per byte:
`bb39f33998df27c2a0b7b6ca7554c77afccf08bb7405831a17a28c4ef6594301`.
Le due CR analogamente:
`89e88059d4e4b9659c0676cab9afe3dcd2c21d6d9878cdd963e0b7cf77784bb2`.
Tutti e quattro i file misurano 463.190.144 byte. Batch, input e RNG
coincidono nelle sei coppie e nei quattro rank; le tracce non differiscono
escludendo il campo phase. Le metriche aggregate mostrano già a 545 piccoli
scarti numerici: primo campo A/B band_shape, 1,49e-8 per CN e 1,19e-7 per CR;
continuo/ripresa grad_norm, circa 9,54e-12 per CN e 1,57e-12 per CR.
Le impronte finali student/teacher nei summary **differiscono dai continui**.
Non si deduce da questi hash l'ampiezza della differenza tra i tensori.
Nessun checkpoint è stato caricato in questa raccolta: il verdetto della
ripresa resta alla successiva lettura CPU congelata.

Consuntivo della tranche, checkout incluso: **0,992222 GPU-ore e 7,981111
core-ore**, contro richiesta 8 / 64,266667 e inviluppo 9,6 / 77,466667.
Con le fasi precedenti: circa 4,740000 GPU-ore / 47,048889 core-ore.
Fonte sacct e cronologia completa archiviate. Nessun altro job inviato.

Verifica riproducibile della raccolta: `python3
results/infra/contrastive_targets_20261118/reloads_execution_20261121/verify_collection.py`.
La verifica controlla byte, hash, passi, fasi, fork, accounting e concorrenza;
le differenze delle tracce sono descrittive e non introducono nuove soglie.
