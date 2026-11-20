# Audit RMS100/fast125 — evidenze operative

Checkout 64936161 e audit 64937087: COMPLETED 0:0, esecuzione 9abde295.
`audit.json.gz` contiene il rapporto remoto originale, 20.275.571 byte una volta
decompresso; SHA256 originale:
`eb7cbdf49ffdd00148647e82c5f51768edbd17b5ef515d960a1f6edec35754d3`.
Nessun campo del rapporto è stato cambiato. Per rieseguire la validazione,
decomprimere con `gzip -dk audit.json.gz` dalla cartella delle evidenze, poi
dalla radice del repository eseguire il comando congelato:

```sh
PYTHONPATH=src python scripts/contrastive_release.py check-audit --audit results/infra/contrastive_targets_20261118/audit_execution_20261120/audit.json --accounting results/infra/contrastive_targets_20261118/audit_execution_20261120/audit_accounting.json
```

Il validatore è byte-identico al commit di esecuzione; uscita zero conservata
in accounting_validation.json. `audit.sacct.tsv` è la ricevuta richiesta, con
intestazione e una sola riga dell'allocazione principale. `allocation.sacct.tsv`
include anche il checkout e i TRES richiesti/allocati. Le ricevute state.json
collegano ID logico, ID SLURM e SHA del codice.

Rapporto positivo: 1024 record, pannello e copertura globale superati;
pin di config, quattro input e sorgenti verificati. Nessuna fase successiva
inviata. La consegna operativa è chiusa con questa raccolta.
