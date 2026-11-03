# Review del piano CPU posa tokenizer a 5.000 passi — 03/11/2026

Esito favorevole al nuovo checkout CPU, alla validazione e al benchmark sul solo train. Non sono emersi blocchi concreti nel launcher o nel piano effettivo. La ridge dei tre bracci è già prevista e autorizzata, ma la sua sottomissione resta subordinata alla lettura dei due report preparatori e al criterio temporale congelato sotto. Non va inviata anticipatamente con la sola dipendenza `afterok`.

Review locale, senza modifica del codice, test duplicati, fit, comandi remoti o job. Le cache e le ricevute integrali restano sul cluster: qui sono stati verificati i pin contro le evidenze raccolte dall'operatore. La fase CPU `validate` rileggerà e verificherà gli artefatti integrali.

## Artefatti verificati

| Artefatto | SHA256 |
|---|---|
| `scripts/slurm/tokenizer_pose_readout.sbatch` | `f9fd6a4ac9f7365c1a59b9e0845c118476c1e6841d73efa5be3aba9f3752e5ec` |
| `scripts/tokenizer_pose_readout.py` | `18a15436e48045447661d3aaab6a8770819527b4ff97a2837d9be3930bad7fb3` |
| `results/infra/tokenizer_evaluation_20261103/pose_readout_plan_5000.json` | `9efbf8279b1a6fdbcd551bae2326e36b74f551c7469f5cbd46842c49906b39ba` |
| `results/infra/tokenizer_evaluation_20261103/pose_readout_resource_policy_5000.json` | `ddc9c2e9b8ff6f352194c8034cbd32e23f8f0363ea73dd3f77d96934c3fd5742` |
| `results/infra/tokenizer_evaluation_20261103/pose_readout_verification_5000.json` | `30a4eb21355f09ec2319848d80bedd40cf72e63a50078a892801a8bf551e8309` |

Il launcher è quello del commit `ff06920cffd3e72fd9e912a744a9d8b4525b8748`. Il lettore e i tre moduli numerici importati non differiscono dal codice già approvato al commit `afb644ea10ecc2a716160b4d8a4904e805c366fb`; tutti e quattro gli hash nel piano coincidono con i file locali.

## Provenienza e supporto comune

Verificati i sei pin delle cache train/val contro gli output attestati dalle tre ricevute e contro `completed_5000/sha256sum_cache_and_provenance.txt`. Gli hash delle sei provenienze originali locali coincidono con i pin e con la raccolta remota. I pin delle tre ricevute indicano i file integrali remoti e coincidono con `sha256sum_full_receipts.txt`: non sono gli hash dei riepiloghi locali.

Braccio, teacher, dose 5.000, checkpoint, codice di estrazione `f2061b8bc91a6355213793a9719121135ad48fa2`, modello effettivo, statistiche posa e superamento della verifica dopo il caricamento sono coerenti fra provenienze e ricevute. I checkpoint sono quelli già congelati nel piano di estrazione; le ricevute riportano completezza e checkpoint invariato.

Le tre firme di metadati, target/mask/std/pose_var e maschera centrale coincidono esattamente, separatamente per train e val. Conteggi: **1.907 finestre, 40 utenti, 200 registrazioni train; 18.548 finestre, 15 utenti, 1.950 registrazioni val**. Riferimenti storici e due fold di utenti alternati sono esattamente quelli già fissati. Il nuovo piano non cambia selezione, ordine, soglia dinamica o supporto del confronto.

La ricetta resta teacher mean-K64 FP16, LayerNorm stateless FP32, ridge FP64 istantanea con crop6 coerente su feature, target e mask. La selezione di alpha usa soltanto i due fold train, con intercetta non penalizzata e fallimento esplicito se il minimo resta al bordo della griglia dopo le estensioni previste. Le dinamiche dipendono dal `pose_var` originale e dalla soglia già congelata. Nessun risultato val è usato per decidere risorse o ricetta.

## Sequenza e launcher

`validate` verifica tutte e sei le cache e i due riferimenti, senza calcolare metriche scientifiche. `benchmark`, sul braccio esplicito **T25-L**, apre soltanto la cache e il riferimento train: misura statistiche sufficienti e tre sistemi di autovalori con alpha fisso, senza selezione o inferenza sulla val. I due job preparatori possono essere eseguiti in parallelo.

Il launcher richiede checkout di campagna congelato e piano contenuto nel checkout, output assoluto canonico nuovo fuori dal codice, un nodo, CPU coerenti con la fase e memoria positiva non superiore a 30 GB. Imposta i thread a 2 per la validazione e a 8 per benchmark/ridge; nasconde CUDA. Per validazione e benchmark occorrono gli override SLURM di CPU e durata indicati dal piano, perché il default del file è quello della ridge.

La ridge esegue le tre celle in sequenza e produce già il confronto appaiato. Non è necessario né prenotato un job `compare` separato. Il controllo dell'esito preparatorio e del tempo prima della ridge appartiene alla catena operativa; il launcher non legge autonomamente i report precedenti.

## Criterio temporale fissato prima del benchmark

Si definiscono:

- `V`: `ElapsedRaw` del job padre SLURM di validazione completo, in secondi, inclusi controlli e I/O.
- `S`: `statistics_seconds` del benchmark a 8 thread.
- `E`: `three_eigensystems_seconds` del medesimo benchmark.
- `R = max(10, n_val / n_train) = 10`, dai conteggi già fissati.

La previsione conservativa è:

`T_need = ceil_60s(1,5 × [2V + 3 × ((1 + R)S + 13E)] + 120 s)`.

`2V` riserva tempo per validazione e riletture; il fattore tre copre i tre bracci. `R × S` è un proxy volutamente prudente del lavoro sulle finestre val, senza eseguire fit o leggere score val nel benchmark. **`13E` è un margine empirico per le valutazioni della griglia alpha, non tredici decomposizioni:** la CV prepara i suoi due solver una sola volta e il refit prepara il terzo. Il 50% aggiuntivo e i 120 secondi coprono variabilità e finalizzazione. La formula non è un limite matematico garantito del runtime.

La ridge può essere sottomessa soltanto dopo che validazione e benchmark risultano entrambi `COMPLETED 0:0`, con report completi, identità e supporti coerenti, ambiente del benchmark a 8 thread e **`T_need <= 1800 s`**. Un superamento richiede diagnosi dei tempi: non si ampliano silenziosamente tetto o ricetta e non si scelgono bracci sulla val.

## Risorse

| Fase | CPU | Durata massima | Memoria | Core-ore massime |
|---|---:|---:|---:|---:|
| Validazione | 2 | 30 minuti | 30 GB | 1 |
| Benchmark train | 8 | 15 minuti | 30 GB | 2 |
| Ridge con confronto | 8 | 30 minuti | 30 GB | 4 |
| Nuovo checkout | 2 | 10 minuti | 4 GB | 0,333333 |

Le **7 core-ore** delle tre analisi sono già nella matrice autorizzata. Il nuovo checkout aggiunge **0,333333 core-ore CPU-only**, portando il tetto complessivo della matrice con i tre checkout a **114 core-ore CPU-only**. Questa fase usa **0 GPU-ore**. La memoria è coerente con l'elaborazione sequenziale di una cache val FP16 D384 alla volta, predizioni FP64 e normalizzazione a batch64; non è introdotta una matrice delle similarità fra finestre né un allargamento temporale delle feature.

L'esito di questa review autorizza tecnicamente la preparazione già approvata e conferma il percorso condizionale verso la ridge. Non è una valutazione scientifica dei tre modelli: i nuovi score devono ancora essere prodotti e interpretati nel perimetro di un solo seme e della validazione di sviluppo già utilizzata.
