# Review delle estrazioni tokenizer a 5.000 passi — 03/11/2026

Esito favorevole: nessun blocco concreto rilevato nel launcher e nel piano dei tre job `extract-pose` già autorizzati. La verifica è stata eseguita soltanto sui file locali, senza job, comandi remoti, modifica dei piani o duplicazione dei test. L'assenza attuale di STOP e gli hash dei checkpoint remoti sono attestati dalla raccolta dell'operatore; il controller li verifica nuovamente nell'allocazione, prima e dopo l'estrazione.

## Identità della consegna

| Artefatto | SHA256 verificato |
|---|---|
| `scripts/slurm/tokenizer_evaluation.sbatch` | `450b5de086dea37b3b0b476267c3c5caae82dc41c348e0fcd77367ee0ce035dc` |
| `scripts/tokenizer_evaluation.py` | `00d5f42583bc3203df5041558486647dd05f66f59df8adf3ae7728cf3e6c127b` |
| `results/infra/tokenizer_evaluation_20261103/extraction_plan_5000.json` | `c0e5d501cd42c81d60aaa390098644ec5e3bf351f3e06031233647934dd2e985` |
| `results/infra/tokenizer_evaluation_20261103/resource_policy_5000.json` | `175fe3de16a5051bf68347f90a87b3de3f55aec2c4762e441477032ab3540461` |
| `results/infra/tokenizer_evaluation_20261103/verification_5000.json` | `5cb391ee9c70e30b0ee9be24eafc88be51577c9d4640faea92bddf8410c0a7e0` |

I pin dei tre checkpoint, dei rispettivi snapshot di configurazione e delle ricevute coincidono esattamente con `results/infra/tokenizer_execution_20261103/5000/collection.json`. Ho ricalcolato gli hash e verificato le dimensioni dei nove file locali raccolti: ricevuta, configurazione e riepilogo per ciascun braccio. Tutte le configurazioni complete sono uguali alle rispettive ricette congelate a 5.000 passi; le ricevute riportano protocollo corretto, `complete: true`, dose 5.000 e codice di training `afb644ea10ecc2a716160b4d8a4904e805c366fb`.

| Braccio | Job di training nella ricevuta | SHA256 del checkpoint nominato |
|---|---:|---|
| T25-L | 63385800 | `bfde1d69a85338875a45b4802979d235045e40e1bea7cfc0596c98ac021f4deb` |
| T100-L | 63386190 | `0ec89584fafb8319708f3a8a035ce8124d91cebc8d05ec4e40db4def5ee0b10e` |
| T100-G | 63386228 | `49dd518b5eea632a16bc7ff88ab0fb013f30d61458164c9521867572c83daa80` |

Il codice sotto `src`, il controller, le sue dipendenze di preflight e posa e `train_jepa.py` non differiscono dal commit di training. Il nuovo launcher non introduce cambiamenti del modello, degli ingressi o della matematica dell'estrazione.

## Contratto dell'estrazione

`load_context` ricostruisce la configurazione dalla CLI congelata e verifica la configurazione completa, la firma dati, le statistiche posa e la dose. `validate_checkpoint` carica rigorosamente il **teacher**, controllando anche l'attivazione effettiva GELU/identity e il gain salvato: la sola parità delle chiavi dei pesi non è usata come prova dell'attivazione. Non viene calibrato il gain sui dati di valutazione.

Il piano punta a `checkpoint_step0005000.pt` e agli snapshot `config.json` nelle directory delle ricevute della fase. Non dipende dal `checkpoint.pt` o dal `config.json` mobili della directory di training; la continuazione del medesimo braccio può quindi procedere conservando questi artefatti. Il controller ricontrolla hash e stato dei file prima di pubblicare la ricevuta completa.

Manifest, scale, durate delle finestre e statistiche coincidono con i pin del preflight. SHA della comparazione e dello split v1 locale verificati. I riferimenti posa coincidono con `pose_temporal_20261102/plan.json`: **1.907 finestre, 40 utenti e 200 registrazioni train; 18.548 finestre, 15 utenti e 1.950 registrazioni val**. Le cache B19618 student sono usate esclusivamente per identità della selezione e dei target, non per feature o predizioni. Il nuovo encoder è sempre teacher.

Il controller conserva selezione, seme, finestre e preprocessing dell'estrattore originale; verifica metadati, target, mask, std e pose_var contro i riferimenti. Tutti i valori delle nuove feature e predizioni sul tempo reale di input devono essere finiti; il padding resta distinto. Gli output si scrivono in nuove directory esterne al codice. La ricevuta `complete` viene scritta soltanto dopo entrambi gli split e tutte le guardie finali.

I cinque riferimenti di classificazione coincidono con le cache originali B del piano PSD3, con label native. Sono già presenti nel piano per completezza; **non viene eseguita classificazione a 5.000 passi**. La sezione checkpoint 19.618 ancora vuota è compatibile con questa fase.

## Launcher e risorse

Il launcher verifica braccio, dose e combinazione `extract-pose:none:5000`, checkout di campagna congelato, piano contenuto nel checkout e uscita canonica nuova fuori dal codice. Richiede un nodo, otto CPU, un dispositivo CUDA visibile e walltime esatto di 2.700 s. Imposta il codice del checkout in testa al percorso Python e invoca il controller senza override scientifici.

Sono previsti **tre job da 1 GPU, 8 CPU, 100 GB e 45 minuti**, pari a **2,25 GPU-ore** e 18 core-ore allocate con GPU. Le 2,25 GPU-ore sono già incluse nelle 8,25 GPU-ore della matrice di valutazione: non vanno sommate nuovamente al preventivo generale. Il nuovo checkout da 2 CPU per 10 minuti aggiunge **0,333333 core-ore CPU-only**, portando il tetto CPU-only della matrice, inclusi i due checkout di training ed estrazione 5k, a **113,666667 core-ore**.

I 45 minuti sono un tetto operativo prudente già previsto, non una garanzia del tempo effettivo. Nessun retry cieco è autorizzato dal piano. Dopo tre ricevute valide si può congelare il piano CPU con tutti i pin, misurare il benchmark sul solo train e produrre il confronto ridge comune; questa review non anticipa risultati scientifici né seleziona i bracci da continuare.
