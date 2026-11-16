# P1-r2: rilascio singolo con criterio v2 — 16/11/2026

Simone ha approvato esplicitamente la modifica del criterio e l'invio diretto da GPT
di un solo P1, con guardia nuova e log completi. La citazione integrale e il perimetro
sono in `criterion_amendment.json`. Nessun ulteriore consenso e nessun retry.

La diagnostica 64610181 resta **NOT_REPRODUCED / READERS_CLEAR_IN_TEST**: non si
afferma che Lustre sia la causa dimostrata. L'esito negativo non blocca più questo
solo tentativo, legato al report completo `c2e32e9c…`. Le vecchie decisioni di stop
rimangono nel registro; l'emendamento le supera per questo P1 soltanto.

## Controlli preservati

- Helper `scaling_output_identity.py` identico a quello provato, SHA `717362c2…`.
  Due letture integrali attorno alla validazione del contenuto; impronte identiche,
  identità del file preservata, una sola eventuale lettura aggiuntiva per la sola
  stabilizzazione degli attributi temporali. Audit di ogni osservazione ed errore.
- Invariati i 103 sorgenti scientifici, checkpoint, recipe, teacher, dati, split,
  verifiche immutabili degli ingressi e validatore del contenuto delle cache.
- Posa/P2 già completate sono conservate; i due P1 falliti non vengono adottati.
  Il nuovo P1 produce tutte e tre le cache nella cartella nuova `p1-r2`.
- Manifest, report diagnostico e rapporto CPU hanno pin SHA-256 e dimensione;
  sono copiati nel nuovo checkout immutabile. Ricevuta e sidecar riportano il
  criterio, il vero esito diagnostico e il diverso commit del produttore P1.
- 163 test CPU superati; review indipendente e preflight in questa cartella.

## Un solo checkout e un solo P1

| Operazione | ID logico | Risorse richieste | Massimo con 180 s di margine |
|---|---|---|---|
| Checkout | SCALING-P1-recovery-checkout-20261116 | 2 CPU, 4 GiB, 5 min | 0,266667 core-ore |
| P1-r2 | SCALING-300M-eval19618-p1-r2-20261116 | 1 GPU, 8 CPU, 100 GiB, 2 h | 2,05 GPU-ore; 16,4 core-ore |

Il checkout nuovo `scaling_p1_recovery_20261116_v1` viene creato dal launcher
versionato su seriale. Il nuovo ref deploy punta allo SHA completo pubblicato;
nessun checkout o ref attivo viene aggiornato. P1 dipende da `afterok` sul checkout.
Entrambi gli script passano su stdin tramite `scripts/submit_job.py`, ledger comune,
con `/dev/stdin` come argomento script di sbatch. Stop al primo invio non sottomesso.
Il piano con lo SHA effettivo e le due righe vengono pubblicati dopo il commit
del pacchetto, prima del deploy e dei job. Nessun runner o monitor in background.

`budget.json` contiene il consuntivo e il massimo cumulativo: 7,035834 GPU-ore su
7,30; 56,286667 core-ore allegate su 58,40; P1 cumulativo 3,070834 su 3,10 GPU-ore;
CPU-only 350,207223 su 351 core-ore. Nessuna diagnostica nuova.

## Raccolta successiva

Accettare soltanto COMPLETED 0:0, ricevuta completa, tre cache e tre sidecar con
hash verificati, eventi della nuova guardia completi, input e codice invariati.
Conservare stdout SLURM, `p1.log`, `output_identity_events.jsonl`, ricevuta e
accounting con il testo sacct originale. Un nuovo errore ferma la catena; nessun
altro invio è autorizzato da questo piano. I lettori futuri mantengono i controlli
originali e la provenienza mista dichiarata, senza riscrivere pose/P2.
