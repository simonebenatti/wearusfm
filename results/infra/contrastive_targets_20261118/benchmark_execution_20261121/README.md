# Tre continui RMS100/fast125 — raccolta operativa

Esecuzione congelata 746112dd711903e5c92ade2efd90834264b2676a.
Checkout 65017186, C0 65017240, CN 65017310, CR 65020151: tutti COMPLETED 0:0.

60 file originali raccolti, SHA e byte verificati contro remote_sha256.txt e
remote_bytes.txt. I file .pt restano su WORK; impronte e dimensioni di tutti i
checkpoint finali e a 544 in checkpoints_remote.txt. allocation.sacct.tsv
conserva gli ID e il billing; state.json collega ID logici, SLURM e codice.

C0/ contiene aggregato e quattro rapporti individuali di parità, verificati
fra loro; validate_parity_report ricalcola PASSED ai punti 128/256. CN/ e CR/
contengono per ciascun rank le 32 tracce 545–576, incluse le osservazioni DDP.
Tutti hanno summary, config, 576 metriche, mappa origini e quattro profili.

collection_verification.json e verify_collection.py conservano i controlli di
raccolta: ricette normalizzate, dose, sorgenti, dati, profili, parità e tracce.
Le mappe delle origini coincidono; batch e input SHA CN/CR coincidono nelle
tracce 545–576. Questo non dimostra identità di tutti i tensori a ogni passo
del training. Mediane delle norme contrastive pesate sul tratto 513–576 sono
descrittive; non sostituiscono la lettura CPU completa successiva.

Nessuna ripresa, lettura CPU completa o training lungo lanciato. La consegna
termina con questa raccolta per la discussione di Simone e della chat scientifica.
