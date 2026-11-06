# Review operativa indipendente del rilascio CPU — 05/11/2026

**VERDETTO: APPROVATO. Nessun blocco rilevato** nei due piani con pin reali e nei sette comandi iniziali. Sono stati controllati:

- `docs/scaling_300m_cpu_execution_20261105.md`;
- `results/infra/scaling_300m_20261104/cpu_readers_20261105/pose_plan.json`, SHA `a500cc3c56bde3938a2aa384bf6a942fd6eaf02be0ebd5935ac251f732cb2198`;
- `.../probe_plan.json`, SHA `d2cc7da1cf08473067d96b52bff532463453ff5ce633814743b1a53dda5bbb63`;
- `.../submission_plan.json`, SHA `f6d0c0e7585023b763b314271936988ca75344d6e6995663e1b844d726a3403a`;
- launcher `scripts/slurm/scaling_readers.sbatch`, inviluppo e review Claude comprensiva dell'aggiunta `8b763610`.

## Pin e identità dei dati

**Quattordici cache**: dieci classificazione nuove, due posa300M nuove, due posa30M riusate. Tutti i pin dei piani si riconciliano con output delle ricevute complete, provenienze e letture SHA remote raccolte; checkpoint e codice di estrazione sono quelli effettivi. Nessun segnaposto o train posa parziale.

Per le sonde sono stati verificati gli originali delle quattro ricevute gzip, con le dieci provenienze già auditate. Cache/receipt/provenance/input/split/ricetta sono quelli dichiarati nel piano. Le cinque popolazioni coincidono esattamente con la sezione `datasets` storica; ricalcolati i ruoli dai soggetti e controllati **entrambi gli algoritmi di hash**: JSON canonico del piano e fingerprint dei fold nelle provenienze. P1 esclude i soggetti ufficiali di test secondo gli split congelati. Nessun cambio di fold, C, tolleranza, seed, numero iterazioni o popolazione.

Per la posa il controllo30M è identico al piano storico5000. I nuovi pin300M coincidono con la receipt completa `a57c9420…`; train/val sono appaiati allo storico per hash dei bersagli, utenti, finestre, registrazioni e riferimenti. I due fold da20 utenti sono invariati. Eseguito e passato il vero **`scaling_pose_readout.validate_plan`**, che è strutturale; non è una lettura delle cache.

Contratto, manifest S5 e split sono assoluti nel checkout numerico congelato. Identici nei due piani, ove comuni; verificati SHA/bytes dei file locali contro i pin e contro `git show aef665b:<path>`. Ricette e mappe dei sorgenti dei due lettori coincidono con le funzioni del codice congelato. Verificata identità dei sorgenti coinvolti e del launcher rispetto al commit runtime. Nessuna modifica scientifica, nuova vista o dose.

## Comandi, runtime e risorse

Ricostruiti indipendentemente **tutti i sette argv completi** e confrontati con `submission_plan.json`: submitter comune, agente/campagna corretti, ID e output unici, `--gpu-ore 0`, piani assoluti e loro SHA reali, `CODE_DIR` numerico `scaling_300m_evaluation_20261105_v1`, `EXPECTED_SHA=aef665ba7ea66fc80291846487907bb3ef9107df`. Nessun bypass del launcher, fit anticipato o flag di ricetta.

| Stadio | Job | CPU | Tempo ciascuno | Core-ore massime |
|---|---:|---:|---:|---:|
| Validazione posa |1|2|30min|1|
| Benchmark posa |3|8|10min|4|
| Benchmark sonde30M naturale/300M matched |2|8|45min|12|
| Benchmark sonde300M naturale |1|8|90min|12|

I sette stadi numerici sommano **29 core-ore**. Tutti30G, nessuna GPU, login08/lrd_all_serial come imposto dal launcher; numero di thread, moduli, venv e bootstrap comune padre/figli rimangono quelli congelati. Il benchmark sonde elabora cinque dataset sequenziali per vista, quindi servono quindici report effettivi; nessuna selezione held-out nel benchmark.

Checkout documentale aggiuntivo: **2CPU/4G/300s =1/6 core-ora**, nella riserva. Prima tranche **29,166667 core-ore**. Somma della campagna verificata:

`179 stadi numerici + 1/3 checkout originario + 3 × 1/6 checkout documentali + 1/6 residuo = 180 core-ore`.

La riduzione a cinque minuti riguarda soltanto i nuovi checkout; nessun walltime di valutazione è stato ridotto. Tre è un massimo cumulativo, non tre checkout già necessari o inviati. Un TIMEOUT del checkout consuma il suo tentativo e ferma la fase; nessun retry automatico.

## Ordine e regole di arresto

Ordine documentato corretto: pubblicazione/verifica del commit dei piani → checkout completo verificato → validate posa completo accettato → sei benchmark. È una scelta operativa prudente far attendere anche i benchmark delle sonde, pur indipendenti dalla posa. Prima di ogni invio si ricontrollano ledger e fallimenti noti. Nessun fit è predisposto in coda prima dei benchmark.

Il commit esatto e argv del checkout documentale vengono registrati separatamente **dopo il commit dei piani, prima dell'invio**: `metadata_checkout.publication_sha` è una nota di processo, non un SHA finto né un argv eseguibile. Non considero completato quel passaggio da questa review. Root ha comunicato separatamente l'assenza remota degli output e del nuovo checkout e la pulizia del runtime; questo revisore non ha eseguito SSH.

O-6/O-8 sono esplicite: primo errore, rifiuto runtime/threadpools o TIMEOUT → stop agli invii successivi, raccolta di evidenze e costo, comunicazione a Simone; nessun retry o cambio di nodo/thread/librerie/ricetta per passare. Il limite180 è mantenuto operativamente con consuntivo cumulativo, non falsamente attribuito a un unico guard automatico.

O-7 è recepita: ogni accounting JSON sarà affiancato dal `sacct -P` originale; per le sonde il job è legato ai cinque report tramite submitter/argv/pin. Dopo misure reali restano necessari formula posa con quattro accounting, budget sonde con quindici benchmark, copertura esatta delle60 combinazioni e ≤136core-ore/≤15shard. Nessun budget di fit inventato nei piani attuali.

## Limiti

Solo audit locale di file/pin/schema/comandi. **Non eseguito `scaling_probe.validate_plan` integrale con i percorsi remoti**, non caricate le cache NPZ né i checkpoint, nessun benchmark/fit. Le guardie complete di integrità, finitezza/supporto e runtime saranno eseguite nei job, e un loro fallimento resta bloccante. Le comparazioni scientifiche e il gate misurato dei fit restano da produrre. Nessun SSH, job, commit o modifica al repository; unico file scritto da questa attività: questo memo.
