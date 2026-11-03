# Valutazioni tokenizer a 5.000 passi — esecuzione 03/11/2026

I tre training iniziali sono COMPLETED 0:0. Ricevute complete, configurazioni integrali,
checkpoint nominati e SHA sono verificati; nessun STOP. Le continuazioni già sottomesse
restano autonome: nessuna scelta dei bracci sulla base dello screening.

| Braccio | Job training | Secondi allocati | GPU-ore effettive | Checkpoint SHA256 |
|---|---:|---:|---:|---|
| T25-L |63385800|3834|4,260000|bfde1d69a85338875a45b4802979d235045e40e1bea7cfc0596c98ac021f4deb|
| T100-L |63386190|3926|4,362222|0ec89584fafb8319708f3a8a035ce8124d91cebc8d05ec4e40db4def5ee0b10e|
| T100-G |63386228|3916|4,351111|49dd518b5eea632a16bc7ff88ab0fb013f30d61458164c9521867572c83daa80|

Totale training5k **12,973333 GPU-ore**, 103,786667 core-ore allocate insieme alle GPU.
La raccolta locale dei nove file receipt/config/summary è identica byte per byte agli
hash remoti. Evidenze in `results/infra/tokenizer_execution_20261103/5000/`.
Inventario SLURM del 03/11 alle11:49UTC: continuazioni63386315/63386337/63386403
PENDING/Priority; dipendenze iniziali soddisfatte.

## Piano congelato prima del lancio

`results/infra/tokenizer_evaluation_20261103/extraction_plan_5000.json`,
SHA256 `c0e5d501cd42c81d60aaa390098644ec5e3bf351f3e06031233647934dd2e985`.
Tre checkpoint_step0005000.pt e snapshot config/receipt immutabili, indipendenti dai
file checkpoint.pt/config.json aggiornati dalle continuazioni. Sei input e riferimenti
storici fissati; posa teacher con train40utenti/200registrazioni/1907finestre e
val15utenti/1950registrazioni/18548finestre. Le cache storiche attestano selezione,
target e maschere; non si richiede uguaglianza delle nuove feature.

Nessuna modifica ai valutatori scientifici di afb644ea10ecc2a716160b4d8a4904e805c366fb.
Aggiunto soltanto `scripts/slurm/tokenizer_evaluation.sbatch`, launcher sottile con
controllo del checkout, allocazione, dose, pin e uscita nuova. Review del critico
favorevole in [rapporto di lancio](reviews/tokenizer_evaluation_5000_launch_20261103.md).
Sintassi bash verificata; pin e configurazioni verificati indipendentemente.
I35test del controller erano già passati; nessun cambiamento che ne richieda la ripetizione.

## Risorse e procedura

Nuovo checkout CPU `tokenizer-eval5k-20261103`, creato da commit pubblicato via
`campaign_checkout.sbatch`:2CPU/4G/10min, **0,333333 core-ore CPU-only aggiuntive**.
Questa aggiunta operativa è coperta dall'autorizzazione corrente; totale di riferimento
CPU-only della matrice con i due checkout training/eval5k:113,666667coreh.
Gli eventuali checkout successivi verranno contabilizzati separatamente.

| ID logico | Risorse | Tetto |
|---|---|---:|
| TOKENIZER-T25-L-pose5000-20261103 |1GPU/8CPU/100G/45min|0,75GPUh|
| TOKENIZER-T100-L-pose5000-20261103 |1GPU/8CPU/100G/45min|0,75GPUh|
| TOKENIZER-T100-G-pose5000-20261103 |1GPU/8CPU/100G/45min|0,75GPUh|

**2,25 GPU-ore già comprese negli8,25 della matrice**, non un'aggiunta;18coreh allocate
con leGPU. Risorse, output e ID in `resource_policy_5000.json` nella stessa cartella
del piano. Output su SCRATCH, fuori dai checkout; nessuna sovrascrittura. Il ledger
comune e l'inventario remoto verificano assenza di duplicati prima dei lanci.

Dopo commit/push e verifica SHA, deploy su riferimento nuovo, checkout CPU, verifica
COMPLETED0:0/SHA/pulizia/sola lettura, quindi tre invii via submitter comune.
I launcher ricevono il piano relativo, braccio,5000,extract-pose,none,output assoluto;
CODE_DIR ed EXPECTED_SHA sono quelli del nuovo checkout. Non usare dipendenze SLURM
su vecchi job già conclusi: evidenze complete sopra sostituiscono il legame scaduto.

## Seguito autorizzato

Dopo tutte e tre le ricevute di estrazione complete: confronto supporti/target,
piano CPU con sei cache reali, benchmark train-only e ridge a crop6 entro i tetti
pubblicati. P1/P2 rimangono riservati a19.618. Lo screening5k è descrittivo: nessun
verdetto finale o selezione del vincitore, nessuna nuova loss o dose. Nessun retry cieco.

## Sottomissioni eseguite

Checkout **63399280 COMPLETED0:0**,18s,0,010000coreh CPU-only, SHA
**f2061b8bc91a6355213793a9719121135ad48fa2** verificato, pulito e in sola lettura.
Pubblicato prima dei lanci su origin e nuovo deploy/tokenizer-eval5k-20261103.

| Braccio | Estrazione posa5k | Primo stato verificato |
|---|---:|---|
| T25-L |63399428|RUNNING|
| T100-L |63399479|RUNNING|
| T100-G |63399484|RUNNING|

Tre invii riusciti tramite submitter/ledger comuni, allocazioni1GPU/8CPU/100G.
Ricevute originali in results/infra/tokenizer_evaluation_20261103/submission/,
indice submissions_5000.json. Continuazioni63386315/63386337/63386403 ancoraPENDING.
Nessun risultato scientifico delle estrazioni o ridge ancora disponibile.
