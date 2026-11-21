# Consegna di calibrazione · condizioni di invio

La pubblicazione del resoconto dell’audit e dei due script di prova Gloo documenta la preparazione. Per consegnare un job a 3-launcher occorrono le ricevute della prova P-5 su quattro processi e il documento operativo della fase, con lo SHA di esecuzione e il piano degli argomenti rigenerato a quello SHA.

L’audit già concluso alimenta direttamente la calibrazione. Il job deve leggere l’originale in WORK, autenticare l’allocazione 64937087 e la review, controllare gli SHA dei quattro input e scrivere in una destinazione nuova. Il controllo sul Mac dei file e della ricetta conserva i risultati in `../releases/calibration_20261120/local_check.json`. La verifica finale sul login compete a 3-launcher.
