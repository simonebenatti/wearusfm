# Controprova D3-4 (raccolta)

Esito: i fatti reggono, ma la gravità corretta è minore e non importante.

Verifiche fatte:
- Sequenza, righe 169-191: non c'è nessun comando di copia (grep di rsync/scp: nessun risultato). La cartella di uscita e i log sono però nominati nella tabella alle righe 54 e 56.
- Le gambe stanno in root/ordinary e root/warm (contrastive_cn_restart.py, righe 131-134): confermato.
- Il log del checkout è wearusfm_runs/logs/campaign_checkout_%j.out (campaign_checkout.sbatch, righe 9-10). Il passo 7 non lo cita, perché parla solo dei «due log di SLURM .out e .err» del job GPU.
- Il passo 7 non cita le uscite del passo 5 (~/wearusfm_local/cn_restart545_20261127, righe 120-121), la quota del passo 3 né gli state.json del submitter.
- L'archivio reloads_execution_20261121 contiene checkout.state.json, CN_A.state.json, checkout_login_check.txt e *_login_check.txt: confermato.
- «Tutti i file che non sono .pt» è la regola; l'elenco che segue è introdotto da «come nell'archivio storico». C'è un'ambiguità lieve, ma la regola è chiara.

Gravità: la raccolta avviene dopo il job e non brucia alcun ID. I file mancanti restano recuperabili: i log sono su WORK, che è persistente, e gli state.json e le uscite del passo 5 sono sul Mac. Il passo 7 chiede già di controllare dimensioni e git status prima del commit. Il rischio di copiare i .pt sul disco stretto del Mac esiste ma è recuperabile. Gravità: minore.
