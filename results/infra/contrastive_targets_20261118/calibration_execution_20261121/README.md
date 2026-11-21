# Calibrazione RMS100/fast125 raccolta il 21 novembre 2026

Checkout 65014846, calibrazione 65015116: COMPLETED 0:0. Codice eseguito
e05c05826daa9a6feda47f1f75742ea329c84f5f, campagna CONTRASTIVE-targets-v3-20261118.
Rapporto `PASSED_FOR_REVIEW`, training_approved=false. Nessuna fase successiva.

Rapporto originale calibration_report.json SHA256:
20cb900522b62b46f17d1b65cbbe1ef579a4cdd5e935a7ab31b7a309ed516532.
I 14 file di rapporto/run non-checkpoint sono verificati contro remote_sha256.txt.
Il checkpoint resta sul cluster, SHA e byte in checkpoint_remote_receipt.txt.

calibration/ contiene summary, config, 512 metriche, quattro ricevute numeriche,
provenienza e letture dei pannelli. allocation.sacct.tsv collega i veri ID alle
risorse richieste e allocate. Ricevute del submitter e snapshot prima degli
invii sono conservati nella stessa cartella.

Lettore eseguito sul checkout congelato. Il primo comando testuale del prontuario
ha sostituito PYTHONPATH e perso i percorsi CINECA (import NumPy fallito prima
della lettura). Log in calibration_report_reader.failed_env.log. Ripetuto il solo
lettore con `PYTHONPATH="src:scripts${PYTHONPATH:+:$PYTHONPATH}"`, come nei login
check già approvati, stesso interprete e stessi sorgenti. Esito e versioni in
calibration_report_reader.log. Nessuna modifica a script, ricetta o soglie,
nessun nuovo job o retry SLURM.
