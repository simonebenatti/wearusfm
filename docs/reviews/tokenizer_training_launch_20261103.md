# Review indipendente del lancio tokenizer — 03/11/2026

Esito favorevole per il launcher, il piano delle allocazioni e la copertura scientifica delle valutazioni. Nessun impedimento rilevato al lancio dei tre training autorizzati. Questa verifica è locale: lettura del codice, confronto degli hash e riconto aritmetico indipendente. Non sono stati duplicati i test, né eseguiti comandi remoti o job. La revisione finale del wrapper `scripts/tokenizer_train.py`, in aggiornamento, è affidata separatamente all'altro revisore.

## Oggetto verificato

| File | SHA256 alla review |
|---|---|
| `scripts/slurm/tokenizer_train.sbatch` | `d9e7680777d3083a362bf4b2537c13e9c1d6ff716a4a7ce50c3e5576cb6a2381` |
| `results/infra/tokenizer_execution_20261103/training_plan.json` | `04a389158a0390c2ae0dffc2c77b43a1edec5e7ee80150943c1e06a684d50cb8` |
| `results/infra/tokenizer_execution_20261103/full_resource_policy.json` | `ee8afdc72898af2aab32d6a67712622ea5a1b1e412456791c4b141c4e4b088ac` |

Il piano e il rapporto di preflight corrispondono ai rispettivi pin. Il rapporto indica 64 batch, batch globale 32, 2.048 finestre esposte per braccio, completezza e parità di stream, query, target e supporto posa. I 100 pin scientifici del piano coincidono sia con il checkout locale sia con il commit di preflight `ef8016a520f12a71eb9890202e836f56622d000b`: coprono tutti i file Python sotto `src`, `scripts/train_jepa.py` e `scripts/tokenizer_preflight.py`. Nessun cambiamento della matematica di training è introdotto dal nuovo launcher.

## Tempi e risorse

Le medie degli aggiornamenti, esclusi i primi otto, sono esattamente quelle del rapporto: 0,6952644913 s per T25-L, 0,6997349929 s per T100-L e 0,7058875770 s per T100-G. I rapporti rispetto a T25-L sono 1, 1,0064299294 e 1,0152792007.

Ho ricalcolato per ciascuna fase:

`ceil_900s((3814 / 5000) × max(1, rapporto_preflight) × passi_aggiuntivi × 1,20 + 600 s)`.

Il risultato è **5.400 s per i primi 5.000 passi** e **14.400 s per i successivi 14.618**, in tutti e tre i bracci. I valori `expected_seconds_proxy` e `minimum_from_policy_seconds` sono coerenti. I 19.618 passi includono i primi 5.000: non vengono contati due volte. Il riferimento storico include overhead del training; il preflight fornisce un rapporto tecnico, non una misura del collo di bottiglia del loader a regime. I tetti restano quindi allocazioni prudenti, non garanzie di durata.

| Voce | GPU-ore massime | Core-ore CPU-only massime | Core-ore allocate con GPU |
|---|---:|---:|---:|
| Sei fasi di training, 4 GPU e 32 CPU | 66 | 0 | 528 |
| Dodici estrazioni, 1 GPU e 8 CPU | 8,25 | 0 | 66 |
| Valutazioni CPU | 0 | 113 | 0 |
| Nuovo checkout, 2 CPU per 10 minuti | 0 | 0,333333 | 0 |
| Totale futuro | **74,25** | **113,333333** | **594** |

La matrice delle estrazioni contiene sei coppie train/val posa da 45 minuti, tre P1 da 30 minuti e tre P2 da 45 minuti. La parte CPU contiene 96 core-ore di shard, 4 di aggregati, 1 di confronto, 8 di ridge, 2 di benchmark e 2 di validazione. I core dei job GPU non sono sottratti una seconda volta dal budget CPU-only.

Con il preflight registrato di 199 s su 4 GPU, il totale consuntivo GPU finora più tetti futuri è **74,471111 GPU-ore**. Il conto con l'intera riserva originaria di 2 GPU-ore sarebbe 76,25. Rispetto al precedente residuo di riferimento di 63,2925 GPU-ore, il primo conto eccede di **11,178611 GPU-ore**: lo sforamento è esplicito ed è già autorizzato da Simone. Non è stato verificato un nuovo saldo CINECA. Il checkout già concluso aggiunge 0,018889 core-ore CPU-only al consuntivo, separatamente dal futuro.

## Launcher e dipendenze

Il launcher richiede un braccio registrato e una dose registrata, un nodo e 32 CPU, quattro dispositivi CUDA e il walltime esatto della fase. La continuazione a 19.618 deve essere sottomessa con **`--time=04:00:00`**; il default del file è 90 minuti e, senza override, la guardia rifiuta correttamente quella fase.

`lib_code_dir.sh` verifica SHA, pulizia e sola lettura del checkout, imposta la directory di lavoro e pone il suo `src` in testa al percorso Python. Il piano deve risiedere nel checkout di campagna congelato; ricevute e risultati devono essere nuovi e fuori dal codice. Prima del wrapper sono ricontrollati rapporto di preflight e hash delle sorgenti scientifiche. Il launcher non sceglie bracci o checkpoint in base alla val e non sottomette altri job.

Il disegno prevede tre catene indipendenti: training fresco a 5.000, poi continuazione dal proprio stato verificato a 19.618. L'autorizzazione comprende tutti e tre i bracci e non dipende dai risultati intermedi. Le ricevute e gli snapshot di configurazione delle fasi devono restare distinti dal `config.json` e dal `checkpoint.pt` mobili della directory di training; i valutatori approvati consumano checkpoint nominati e pin verificati.

## Copertura scientifica e artefatti successivi

I tre valutatori approvati coprono la matrice: `tokenizer_evaluation.py` estrae il teacher di posa a entrambe le dosi e P1/P2 solo a 19.618; `tokenizer_pose_readout.py` produce due confronti ridge a tre bracci con crop6 e selezione sul solo train; `tokenizer_probe.py` copre i cinque dataset all'endpoint, 420 fit logistici e il confronto comune. Gli IC restano condizionati a un seme di pretraining e alla validazione di sviluppo già utilizzata.

Non occorrono i nuovi checkpoint o le nuove cache prima di sottomettere il training. I loro hash vengono aggiunti ai piani delle estrazioni e delle analisi dopo la produzione e verifica dei rispettivi artefatti, prima di eseguire quelle fasi. Il piano ridge richiede tutti i sei pin train/val della dose, anche per il benchmark; il piano delle sonde richiede tutti i quindici pin dell'endpoint, anche per un singolo shard. Questo determina quando congelare i piani CPU, senza aggiungere un prerequisito al training o richiedere placeholder.

Nessun nuovo obiettivo, cambio delle soglie, selezione sulla val o campagna ulteriore è introdotto da questi file. Un eventuale fallimento tecnico va diagnosticato con i tetti registrati; non è previsto un retry cieco.

## Chiusura della review del wrapper

Riletti i controlli finali di `tokenizer_train.py` SHA256
`d948fe30ad35543bdf7aa05fb174fd550cc398f0f2b63efc546592a1a3e2dda2`.
Il critico conferma compatibilità col formato reale torch2.2 del trainer: RNG di ciascun rank
validato con Generator temporanei, CUDA obbligatorio, schema AdamW a un gruppo con momenti
finiti, firma dati esatta e contatori d’allarme validi. Non è richiesta uguaglianza dei RNG
fra i due salvataggi distinti; stati addestrati e provenienze restano verificati.
Ricevuta precedente con protocollo esatto e `complete is True`. Nessun rilievo aperto.

33 test del wrapper passati in2,40s, incluse esecuzione simulata fresca5k→propria19618,
immutabilità del checkpoint5k e rifiuto di STOP, fase incompleta, errore subprocess, sorgenti
modificate, ricevute false o provenienze errate. Test SHA256
`1d1fd942d685a18775f2464adaac5e802a370471f6b7860d992b91bedae5bcb6`.
Il totale dei quattro moduli nuovi è107test passati (33+35+19+20); nessun nuovo testGPU.
