# Review delle estrazioni tokenizer a 19.618 passi — 04/11/2026

**Esito favorevole al lancio delle nove estrazioni previste.** Nessun blocco concreto
rilevato nel piano, nei pin raccolti, nelle risorse o nel riuso del controller e del
launcher. La review e' indipendente e locale: non ha eseguito SSH, sottomissioni,
modifiche del codice scientifico o nuove letture dei checkpoint sul cluster.

## Identita' verificate

| Artefatto | SHA256 |
|---|---|
| `results/infra/tokenizer_evaluation_20261103/extraction_plan_19618.json` | `a01799f892d473769a8770beefafd96d0a46896968bee98d2c5de3376eb9e660` |
| `results/infra/tokenizer_evaluation_20261103/resource_policy_19618.json` | `862329258bd2d10279ad91ae38cecb2961575bee5e8a90e18cfafc048f205d8c` |
| `results/infra/tokenizer_evaluation_20261103/verification_19618.json` | `6a7dc0aa637ba2ff5040d1a10cf48bfe6cdc6a68565c18f19b25fa0fb5b9b3b9` |
| `scripts/tokenizer_evaluation.py` | `00d5f42583bc3203df5041558486647dd05f66f59df8adf3ae7728cf3e6c127b` |
| `scripts/slurm/tokenizer_evaluation.sbatch` | `450b5de086dea37b3b0b476267c3c5caae82dc41c348e0fcd77367ee0ce035dc` |

Ricalcolati i nove hash locali di ricevuta, configurazione e riepilogo e confrontati
con l'inventario remoto raccolto in `results/infra/tokenizer_execution_20261103/19618/`.
I file coincidono byte per byte, inclusi i riepiloghi privi di newline finale.
La ricevuta di ciascun braccio identifica esattamente il job atteso, il protocollo
di training, `complete: true`, la dose 19.618 e il codice
`afb644ea10ecc2a716160b4d8a4904e805c366fb`; il riepilogo conferma `stopped=max_steps`
e lo stesso SHA. La raccolta `sacct_completed.txt` attesta COMPLETED 0:0 per tutte
le continuazioni.

| Braccio | Job | SHA256 checkpoint nominato | Byte |
|---|---:|---|---:|
| T25-L | 63386315 | `bf87d87c4ba68412652b5f049c2e22ffb85782cf4b2a9291440610994a9506db` | 454215840 |
| T100-L | 63386337 | `c6a55d1b1fdcd226e6f31ad6ecb4bb118e0ed6e6567da66dc2a0ddeaaa11d7ee` | 454381728 |
| T100-G | 63386403 | `a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056` | 454381728 |

I pin usano `checkpoint_step0019618.pt` e gli snapshot config/receipt della fase
19.618, non i file mobili del run. Hash e dimensioni dei checkpoint coincidono con
gli inventari remoti raccolti; i tre checkpoint nominati a 5.000 hanno ancora gli
hash precedenti. Le configurazioni integrali sono uguali alle ricette congelate,
con il solo `max_steps` impostato a 19.618. Gli hash del piano training nelle
ricevute coincidono con il file locale congelato. Il training complessivo delle
due fasi ha consumato **50,433333 GPU-ore**, ricalcolate dai sei job raccolti.

L'assenza di STOP, dei nuovi output, del nuovo checkout e di duplicati in coda o
nel ledger e' attestata dalla raccolta dell'operatore, non da una seconda sessione
remota del revisore. Il controller ricontrollera' hash, configurazione, teacher,
gain, attivazione, dose e STOP nell'allocazione, prima di estrarre e prima della
ricevuta finale. Il confronto con lo SHA training esatto e lo stato SLURM sono
stati verificati in questa consegna: il solo controller accetta uno SHA training
formalmente lungo 40 caratteri e non consulta SLURM.

## Matrice e contratto scientifico

Tutte le sezioni del piano 5k restano identiche, compresi sei input, riferimenti,
percorsi e checkpoint 5k; e' riempita soltanto la mappa dei tre checkpoint 19.618.
Ricalcolati gli hash della comparazione e dello split v1 locali. Il codice in
`src/` e i controller/dipendenze di estrazione verificati non differiscono dal
commit di training. Controller e launcher conservano inoltre gli hash gia'
revisionati per la tranche 5k.

Le nove celle sono esattamente i tre bracci per posa/P1/P2, senza duplicati di ID
logico o output. La posa estrae teacher train e val nella stessa allocazione;
rimangono 1.907 finestre/40 utenti train e 18.548 finestre/15 utenti val, con
parita' esatta dei metadati, target e maschere contro i riferimenti. P1/P2 usano
il teacher, le righe e label native originali e `x_pool` FP32 da 768 feature.
Il crop6 resta specifico del rapporto posa, senza modifica del pooling P1/P2.

## Risorse e condizioni operative

| Estratto | Job | Risorse per job | Tetto GPU-ore |
|---|---:|---|---:|
| Posa train+val | 3 | 1 GPU / 8 CPU / 100G / 45 min | 2,25 |
| P1 | 3 | 1 GPU / 8 CPU / 100G / 30 min | 1,50 |
| P2 | 3 | 1 GPU / 8 CPU / 100G / 45 min | 2,25 |

Somma verificata: **6 GPU-ore e 48 core-ore allocate con GPU**, gia' comprese nelle
8,25 GPU-ore di estrazione dell'intera matrice. Il nuovo checkout da 2 CPU/4G/10 min
aggiunge separatamente **0,333333 core-ore CPU-only**.

Il submit deve impostare esplicitamente **30 minuti per P1**: il launcher ha
default 45 minuti e rifiuta il walltime diverso da quello previsto per lo stage.
Le nove directory di uscita sono assolute, distinte e fuori dal codice. Piano e
policy devono essere pubblicati, poi eseguiti da un nuovo checkout congelato,
pulito e in sola lettura; il checkout 5k rimane invariato. Usare submitter e ledger
comuni. Le estrazioni includono gia' la validazione del checkpoint, quindi non
richiedono un ulteriore job GPU `validate`.

Dopo ricevute di estrazione complete e parita' verificata si congelano i piani CPU
con i pin reali e si legge il benchmark prima dei fit. I walltime sono tetti
operativi, non garanzie di durata; un fallimento richiede diagnosi, senza retry
cieco o modifica della ricetta. Nessun monitor in background viene riattivato.
