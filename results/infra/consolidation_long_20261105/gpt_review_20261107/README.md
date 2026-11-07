# Review scientifica e audit locale del lungo — 07/11/2026

Consegna immutabile: `c109bfef5a3f80cc4b2334e3554c2171db938467` sul ramo
`claude/consolidation-long-exec-20261105`. Questi artefatti sono una review,
non nuove ricevute runtime o pin di fase. Nessun job o accesso operativo al
cluster; nessuna nuova predizione, fit, scelta di input o protocollo.

- `chain_audit.json`: 4.934 controlli di provenienza, contabilità, codice,
  copertura e ledger; limita esplicitamente le attestazioni sui payload remoti.
- `classification_verification.json`: ricalcolo indipendente dai conteggi
  appaiati incorporati nella summary, punti e IC coincidenti (0,0 pp).
- `pose_verification.json`: ricalcolo indipendente dalle somme per utente;
  1.683 valori, errore massimo 4,39e-14.
- `scientific_readout.json`: estratto aggregato per tabelle e figura, nessun
  identificatore personale esportato.
- `prespecified_contrasts.md`: tutti i contrasti registrati, senza selezione
  dei soli risultati favorevoli.
- `long_results.png` / `.pdf`: figure di stime puntuali. Gli IC appaiati sono
  nell'appendice, non dedotti dalla sovrapposizione di intervalli marginali.
- `preservation_inventory.json`: inventario dai pin (circa 19,61 GiB), non
  una verifica remota di esistenza né una ricevuta di copia.
- `critic.txt`: lettura critica della sintesi e dell'appendice.

Aggiornamento successivo del 07/11, dopo la decisione separata di Simone
«Tutto, 19,6 GiB»: `preservation_verification.json` verifica i manifest della
copia eseguita da Claude (consegna `424f0ec`, job 63781257). Copertura 352/352
file dell'inventario, più due artefatti della sintesi; 354 file e 174 directory
totali, 21.062.138.097 byte. Nessuna nuova lettura dei payload remoti. Lo script
`docs/reviews/check_long_preservation_20261107.py --repo . --out /private/tmp/long-preservation-audit`
riproduce il controllo usando Git e il ledger locale. Le dipendenze fuori
LONG_ROOT restano separate. Questa aggiunta non cambia i 107 job scientifici
o gli hash originali della lettura critica; il totale con la copia è 108.

La sintesi completa pubblicata da Claude contiene tutti i risultati qui letti.
Le predizioni grezze e i checkpoint non sono stati riletti; la riproduzione
numerica parte dalle statistiche archiviate. Nessun NPZ/PT è aggiunto a Git.

## Riprodurre

Da una copia del repository con il commit di consegna disponibile:

```bash
git worktree add --detach /private/tmp/wearusfm-long-review-source c109bfef5a3f80cc4b2334e3554c2171db938467
```

Usare Python 3.11 con NumPy; l'audit del ponte A1 importa anche i moduli del
progetto e richiede l'ambiente locale WearUsFM. Il worktree di input resta
pulito; gli output devono stare altrove. Esempi dal ramo che contiene la review:

```bash
~/.venvs/wearusfm-tok/bin/python docs/reviews/check_long_chain_20261107.py --snapshot-root /private/tmp/wearusfm-long-review-source --out /private/tmp/long-review-chain --ledger ~/wearusfm_local/jobs
~/.venvs/wearusfm-tok/bin/python docs/reviews/check_long_classification_20261107.py --snapshot-root /private/tmp/wearusfm-long-review-source --out /private/tmp/long-review-classification.json
~/.venvs/wearusfm-tok/bin/python docs/reviews/check_long_pose_20261107.py --snapshot-root /private/tmp/wearusfm-long-review-source --out /private/tmp/long-review-pose
~/.venvs/wearusfm-tok/bin/python docs/reviews/render_consolidation_long_20261107.py --repo . --out /private/tmp/long-review-figure
```

Esecuzioni locali con `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
I tre controlli sono stati rieseguiti dopo aver copiato gli script nel ramo di
review: i tre JSON di verifica coincidono byte per byte con le prime esecuzioni
degli agenti indipendenti. Non è stata ripetuta la suite runtime: nessun sorgente
scientifico o di campagna è cambiato.

La corrispondenza col ledger corrente vale per la copia locale verificata il
07/11; per una riproduzione futura conservare i 107 stati originali. I percorsi
remoti nei pin indicano l'origine dei file, non una dipendenza di rete degli script.
