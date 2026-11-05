# Recupero CPU scaling 5.000 — budget e dispatcher approvati

**Stato: budget approvato da Simone e dispatcher approvato dalla Review Claude8 (4e6542f); correzione `/dev/stdin` applicata; EPN612 sottomesso63620780.** Review Claude
`1c21047c77f956a1819fe09e8520d3a36dd7a4ae` acquisita integralmente con le due voci
originali del registro. Il recupero dei soli EPN612/UCI è approvato nell'impostazione;
Review8 approva il dispatcher, subordinatamente alla correzione degli argv qui applicata. Simone ha approvato il
tetto CPU pubblicato in4c670bd: la condizione decisionale B-1 è soddisfatta.

L'approvazione riguarda310 core-ore cumulative, incluse quelle già consumate.
Le autorizzazioni ai lanci e alle letture restano valide: nessun nuovo consenso
ai singoli job di questo piano. Restano i gate tecnici
prima degli invii; il seguito19.618 non è autorizzato.

## B-1: inviluppo concreto

Il tetto approvato è **310 core-ore CPU-only cumulative** per completare la valutazione
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
dispatcher pubblicato/pinnato e **`/dev/stdin` prima dello stadio** dopo le opzioni
di sbatch; vecchi piano e runtime remoti. Questa forma è obbligatoria anche per
fit/ridge/aggregati/confronto su stdin. La ricevuta deve riportare lo stesso
`script_stdin_sha256` del pin pubblicato. Non serve un
checkout per i due benchmark; budget e manifest del confronto richiedono i due
checkout metadata già contabilizzati. Ogni output è in una nuova directory sotto
`cpu/probe_benchmark_recovery_20261105/`, fuori dagli output del tentativo fallito.

La QoS seriale consente10job sottomessi per utente **nella QoS lrd_all_serial**
e8CPU attive. Si legge `squeue -u sbenatti -p lrd_all_serial` e si considerano
tutte le linee nella stessa QoS, senza contare i job GPU in altre QoS. Si inviano
solo gli slot disponibili, verificando gli esiti prima della
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
review interne dalla reviewClaude7. La Review8 copre il dispatcher concreto; la sua unica correzione obbligatoria
agli argv è applicata. Review7 non viene estesa retroattivamente al nuovo codice.
I tetti S8 v2 sono una proposta condizionale, non budget autorizzato da Simone.

## Invio effettivo dopo Review8

Piano5275b0d pubblicato e SHA remoto verificato prima del lancio.
EPN612 **63620780**, sottomesso20:22:50; snapshot20:23:27 **PENDING(Resources)**,
8CPU/30GiB/2h, zeroGPU. Ricevuta comune e hashdispatcher verificati, log ancora
assente perché il job non è iniziato. UCI non ancora sottomesso: resta il gate
sull'esitoEPN. Nessuna duplicazione, fit o automazione.

### 05/11/2026 — GPT scaling: EPN recuperato e accettato, UCI pronto

Job63620780 COMPLETED0:0 alle21:13:23:2820s,8CPU=6,266667coreh,
MaxRSS4,065578GiB. SeiC convergenti, nessun warning/limite; C1008933iter.
Report4b52dc33, audit, log, sacct-P originale e accounting hash-bound raccolti in
cpu_recovery_20261105/epn612_completed. Piano/code/ricetta/runtime1ac848 invariati;
primi5fit identici al parziale eccetto tempi, supporto e ruoli uguali alle altre viste.
Ricevuta dispatcher c2bd verificata; audit offline riproducibile e controllo
indipendente del critico concordi. Matrice14/15, CPU cumulativa28,294444/310.
Simone conferma di verificare EPN e avviare UCI senza nuovo consenso. Dopo questo
commit/push verificato si invia il solo UCI con argv5275b0d,8CPU/30GiB/10min.

### 05/11/2026 — GPT scaling: il recupero EPN chiude il report mancante, non la valutazione

Nessun nuovo gate fallito. Il TIMEOUT originale resta registrato; il suo parziale
non sostituisce il nuovo report completo. Questi benchmark misurano risorse e
convergenza sul train: nessun punteggio downstream disponibile e nessun verdetto
30M/300M. Restano UCI e il piano completo dei fit entro240coreh/15shard/3h.
Le cache remote non sono rilette da questo audit offline; valgono i controlli
precedenti di estrazione/validazione. Nessun training o estensione19.618.

### 05/11/2026 — GPT scaling: UCI recupero sottomesso63625951

Dopo accettazione EPN pubblicata6e58861 e SHAorigin verificato, invio UCI con
argv esatto del piano5275b0d: submitter/ledger comune, ricevuta63625951,
21:32:52,8CPU/30GiB/10min/zeroGPU. Runtime remoto aef pulito, piano d2cc,
dispatcher c2bd, directory nuova e coda seriale vuota verificati prima dell'invio.
Receipt copiato byte per byte e controllato. Alle21:33:40 PENDING(Resources).
Nessun nuovo consenso: esecuzione richiesta esplicitamente da Simone dopo EPN.
I fit non sono ancora inviati: manca il report UCI e il preventivo completo misurato.

### 05/11/2026 — GPT scaling: distinguere benchmark da qualità

EPN/UCI sono benchmark train-only: controllano convergenza e risorse su fold0,
non producono score sugli utenti esclusi. I FM30M/300M e le estrazioni sono già
completati; restano sonde complete, ridge posa e comparatore. Nessun nuovo errore
al rilascio UCI. Stima temporale comunicata: ordine di un giorno di calcolo CPU
per i fit, più coda, da precisare con tutti15report e packing effettivo. Nessuna
promessa di completamento entro domani e nessun monitor in background attivato.
