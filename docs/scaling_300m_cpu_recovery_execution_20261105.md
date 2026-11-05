# Recupero CPU scaling 5.000 — proposta dopo review Claude 7

**Stato: preparato per review, nessun nuovo job inviato.** Review Claude
`1c21047c77f956a1819fe09e8520d3a36dd7a4ae` acquisita integralmente con le due voci
originali del registro. Il recupero dei soli EPN612/UCI è approvato nell'impostazione;
il dispatcher nuovo richiede la review prima dell'uso. Il tetto CPU qui proposto
richiede la decisione numerica di Simone prima del primo fit, come B-1.

Non attribuiamo retroattivamente a Simone il numero310. Le autorizzazioni ai
lanci e alle letture restano valide: non richiediamo di nuovo il via ai singoli
job del piano una volta chiuso l'emendamento. Non si invia alcun fit in attesa
della decisione e non si avvia il seguito19.618.

## B-1: inviluppo concreto

La proposta è **310 core-ore CPU-only cumulative** per completare la valutazione
a5.000, incluse le22,027778 già consumate. Zero nuove GPU-ore. Il tetto dei fit
è240core-ore di walltime prenotato; ciascuno usa8CPU,30GiB,al massimo3ore,
con massimo15shard. Non sono240core-ore aggiuntive per ciascun modello.

| Voce | Core-ore richieste o già consumate | Margine scheduler | Totale contabile |
|---|---:|---:|---:|
| Consumato CPU-only, incluso il TIMEOUT |22,027778|già compreso|22,027778|
| Recupero EPN612 120min + UCI10min,8CPU |17,333333|0,8|18,133333|
| Fit sonde, massimo15shard |240|6|246|
| Ridge posa,8CPU/60min |8|0,4|8,4|
| Sei aggregati,2CPU/10min |2|0,6|2,6|
| Confronto,8CPU/30min |4|0,4|4,4|
| Due checkout metadata residui,2CPU/5min |0,333333|0,2|0,533333|
| Totale delle voci | | |**302,094444**|
| Riserva entro310 | | |**7,905556**|

`resource_policy.json` in `results/infra/scaling_300m_20261104/cpu_recovery_20261105/`
contiene il conto e lo stato delle approvazioni. I180s per job sono margine
prudenziale legato a `OverTimeLimit=3min`, non tempo richiesto né garanzia assoluta
di terminazione: il precedente padre ha registrato184s oltre il limite.
Il consuntivo conserva sempre il tempo padre effettivo. La riserva copre
scostamenti contabili/terminazione, non altri tentativi o l'aumento del cap240.
Se un tetto non basta si ferma la fase, senza adattarlo automaticamente.

La formula resta quella di `scaling_probe.validate_resources`, compreso il
moltiplicatore1,5, i120s per shard e il massimo dei sei C per il refit. Dopo i
15report completi si pubblicano shard reali, copertura esatta3×5×4 e arrotondamento
in alto al minuto. Si ammette il piano solo se ogni shard≤10.800s, totale≤240coreh
e numero≤15. I conti di Claude suggeriscono12–14shard negli scenari, non provano
la fattibilità su EPN ancora incompleto. Nessuna riduzione di griglia/fold/tolleranze.

## B-2: dispatcher e identità

Nuovo file `scripts/slurm/scaling_readers_recovery.sbatch`, derivato dall'originale
con diff minimo. Originale, controller e numerica restano intatti. Il dispatcher
ha un vincolo esplicito su `EXPECTED_SHA=aef665ba7ea66fc80291846487907bb3ef9107df`
e importa dal vecchio `CODE_DIR` congelato; non si usa lo SHA della nuova consegna
come identità del codice numerico nei report.

Differenze deliberate: nome/log distinti; default3h; benchmark per singolo dataset
EPN612/UCI soltanto300M naturale; tetti7200/600s rispettivi; fit10800s/240coreh,
numero massimo15 invariato. Moduli, venv,8thread, bootstrap, guardie di scheduler,
pin, output nuovi, identità, runtime e stop al primo errore restano identici.
Il test `test_scaling_readers_recovery_launcher.py` verifica un elenco chiuso di
queste differenze contro i byte del launcher frozen, oltre a percorsi positivi
e rifiuti alle soglie;42test passati. Nessuna modifica al calcolo dei fit.

## Riuso, ordine e invii preparati

`accepted_benchmarks.json` lega i13report a hash, percorso remoto, ricevuta del
submitter, accounting, raw sacct, log e riga di completamento. I tre NinaPro del
job63593508 conservano parentTIMEOUT e il costo; Claude ne accetta il riuso.
Il vecchio audit parzialeEPN resta intatto. I due report nuovi devono avere anche
`numeric_sha256=1ac8481eb8b53fbdd423404a927cca02e908197f16ed6494268b56e5ddc24b1c`.

1. Review del dispatcher, pubblicazione e verifica SHA della consegna.
2. Controllo di ledger, job esistenti, pin e runtime congelato; invio del solo
   `SCALING-probe-recover-epn612-20261105` tramite submitter comune.
3. Raccolta/accettazione EPN, incluso statoCOMPLETED0:0 e sei C conclusi;
   altrimenti stop. Poi invio `SCALING-probe-recover-uci_emg-20261105`.
4. Raccolta/accettazione UCI, completamento manifest15report e budget dei fit.
   Se C raggiunge max_iter si applica la regola congelata e si segnala; non si
   cambia max_iter né si rende implicitamente riuscita una misura parziale.
5. Decisione registrata310 e review soddisfatte; pubblicazione pin/argv misurati,
   checkout budget immutabile e verifica. Solo allora ridge/fit, aggregati e
   confronto nell'ordine previsto. Nessuna dipendenzaSLURM sostituisce i gate.

Comandi esatti dei due recuperi in `submission_plan.json`: `--script-stdin` del
dispatcher pubblicato/pinnato, vecchi piano e runtime remoti. Non serve un
checkout per i due benchmark; budget e manifest del confronto richiedono i due
checkout metadata già contabilizzati. Ogni output è in una nuova directory sotto
`cpu/probe_benchmark_recovery_20261105/`, fuori dagli output del tentativo fallito.

La QoS seriale consente10job sottomessi complessivi per utente e8CPU attive:
si legge la coda dell'utente prima di ogni invio e si considerano anche le altre
linee. Si inviano solo gli slot disponibili, verificando gli esiti prima della
tranche seguente. Nessun cambio di thread, QoS o nodo per aggirare questi limiti.
Il recupero è sequenziale; non si accodano fit prima delle misure.

## B-3 e facoltativi

Il freezeS8 è emendato separatamente in
`docs/scaling_300m_continuation_budget_freeze_20261105_v2.md`: QoSnormal compatibile
con un nodo in luogo della non ammissibilebprod; inviluppo futuro CPU350 con
benchmark reali endpoint19618. JSONv1 conservato come storico, v2 nuovo riferimento.
Nessun criterio scientifico, dose o formulaGPU cambia; nessuna continuazione è
autorizzata. Questo emendamento precede qualsiasi fit a5.000.

F-1 accolto nel manifest del riuso; F-2 accolto con margini scheduler; F-3 annotato
senza cambiare runtime; F-4 conserva la regola sulla convergenza; F-5 elimina
`cpu_reserved_core_hours_so_far` obsoleto dallo stato corrente; F-6 distingue
review interne dalla reviewClaude7. Questa consegna richiede una nuova review
Claude del dispatcher concreto, non attribuisce a Review7 copertura del nuovo codice.
