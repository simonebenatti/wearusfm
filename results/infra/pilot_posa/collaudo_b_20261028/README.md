# Risultati B, job62523672

**SUPERATO**,5.000 passi. Codice congelato24139ab901c052357a46a65bc20466f2eb1e8358.
Analisi: [review GPT](../../../../docs/reviews/collaudo_b_results_20261028.md).

Le copie del run sono byte per byte dal percorso remoto indicato in `collection_manifest.json`; gli SHA256 sono calcolati
localmente dopo il trasferimento. Nessun checkpoint, cache di feature o segnale wearable grezzo incluso nel repository.
`accounting.json`/`sacct_utc.txt` documentano tempi e costi; `ls_l.txt` documenta i file finali e il checkpoint non vuoto.
Le istantanee `b_startup.json`/`logstartup.txt` conservano lo stato storico pending, non lo stato finale.

Riproduzione dalla radice del repository (numpy nell'ambiente locale):

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python scripts/arms_collaudo.py results/infra/pilot_posa/collaudo_b_20261028 --arm R3-lungo --steps 5000 --expected-sha 24139ab901c052357a46a65bc20466f2eb1e8358 --listing results/infra/pilot_posa/collaudo_b_20261028/ls_l.txt --out results/infra/pilot_posa/collaudo_b_20261028/verdetto.json
MPLCONFIGDIR=/private/tmp/wearusfm-b-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python docs/reviews/analyze_collaudo_b_20261028.py --plot
```

`verdetto.json` e' il risultato applicabile, prodotto dal lettore congelato. `analisi.json` e la figura sono descrittivi post hoc;
non aggiungono criteri. Due ricalcoli indipendenti hanno confermato ratio0,8828655883 e IC95[0,8394701934;0,9322308875].

`verdetto_listing_file_assoluto.json` conserva una prima invocazione con input non conforme alla CLI: era stato passato
`remote_checkpoint_ls_l.txt`, cioe' `ls -l /path/checkpoint.pt`, invece dell'elenco della directory. Il parser usa il nome completo
come chiave e non trova `checkpoint.pt`. La successiva raccolta grezza `ls_l.txt`, conforme alla documentazione, risolve il solo
controllo del checkpoint senza modificare codice, risultati o soglie. Il file era sempre presente (454.215.776 byte).

Nessun nuovo job richiesto da questa analisi; nessuna autorizzazione alla continuazione automatica di R3.
