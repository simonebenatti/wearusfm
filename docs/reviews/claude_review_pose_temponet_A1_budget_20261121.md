# Review Claude: emendamento TCN A1, budget e compatibilità delle ricevute (21/11/2026)

Oggetto: l'emendamento A1 al pacchetto TCN frozen, sul ramo `codex/pose-temponet-launcher-20261120`. Diff letto:
`git diff c90d43e0b9f905fe13770498a59ffdb7c963413a d715290ee06379f74e71c43fa03b2d33bb48170f`.

**SHA rivisti:**
- codice `a1ae9607681c1a181034d7f72770537054af419d`;
- consegna `d715290ee06379f74e71c43fa03b2d33bb48170f`, che rispetto al codice aggiunge solo documenti;
- `source_sha256` del piano `9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea`, 247 sorgenti;
- piano `results/infra/pose_temponet_execution_20261121_A1/plan.json`, sha256
  `553ce1b5185fc0f996af93c1e6a190813ff56e33c5f05c015e752d1f04ca645f`, stato `REVIEW_REQUIRED`.

Richiesta: messaggio di GPT incollato da Simone il 21/11. Perimetro: budget e compatibilità delle ricevute. Review del solo diff,
senza agenti.

## Esito

**APPROVATO: review operativa favorevole dell'emendamento A1** (codice `a1ae9607681c1a181034d7f72770537054af419d`, sorgenti
`9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea`). Nessun blocco e nessuna correzione richiesta prima del
rilascio. Le note della sezione 4 sono operative e non chiedono un nuovo giro.

Che cosa copre. I sei sorgenti cambiati, il piano A1 e il prontuario, per i cinque punti chiesti. Il resto del codice è quello
già rivisto: `01252b6c` (review `85f0d62`) più la correzione degli utenti `5d6fb2dc`, riletta da GPT.

Che cosa non copre. Tempi, memoria e determinismo su A100 oltre i due benchmark già fatti; l'identità numerica su un secondo nodo
GPU; l'esito dei gate sui dati veri (WLS, controlli positivi, selezioni, confronto finale). Li decidono i gate, che possono ancora
fermare la catena.

## 1. Metodo

- Letti il diff dei sei sorgenti, i due file di test, il piano, il prontuario `docs/pose_temponet_execution_A1_20261121.md`, il
  documento del budget e le voci di `docs/decisioni.md`.
- Checkout isolato e staccato di `d715290e`; il worktree di GPT non è stato toccato.
- Test sul Mac:
  - i 22 file TCN: 299 passati in 269,82 s, comprese le due catene dei 24 stadi, con e senza transizione A1;
  - i 5 file storici: 125 passati e 1 fallito, `test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned`, lo stesso
    della base;
  - `bash -n` pulito sui quattro launcher, che non cambiano.
- Ricalcolo mio dei due controlli di budget sulle due ricevute vere dei benchmark.
- Prova della sequenza dell'operatore in un checkout di prova, con una review fittizia mai pubblicata: `release-plan`,
  `resource-plan` con le ricevute vere, `commands --stage wls_calibration`.
- Letture in sola lettura su Leonardo: `sha256sum` e dimensioni delle tre ricevute, `sacct` dei tre job, contenuto della radice
  degli output, checkout presenti, coda.
- Nessun agente, nessun job, nessuna scrittura sul cluster.

## 2. Autorizzazione

Registrata in `docs/decisioni.md` (voce «Benchmark chiusi, gate negativo e budget A1 autorizzato») e ripetuta nel codice
(`budget_amendment.specification`). Parole di Simone riportate da GPT: «va bene se aumenti il budget» e «dagli quello che serve
moltiplicato per 1.5». Per i lanci, dal prontuario: «implementa l'emendamento e preparami il prompt per la review di claude, poi
quando converge procedi coi lanci senza chiedermi di nuovo», e poi «ok».

Non ho una fonte indipendente di queste frasi: le cito come le registra GPT. Simone ha incollato lui stesso la richiesta di
review, che le riassume.

Lettura di GPT, che condivido perché è la più prudente: il fattore 1,5 si applica ai quattro tetti, non ai walltime, alla dose o
al numero di celle.

## 3. I cinque punti

| Punto | Esito | Come l'ho verificato |
| --- | --- | --- |
| 1. Tetti coerenti nei due controlli; senza A1 i vecchi limiti | sì | vedi 3.1 |
| 2. Compatibilità solo per le tre ricevute originali, con hash esatti | sì | vedi 3.2 |
| 3. Identità nuova per le ricevute successive; la vecchia review non rilascia A1 | sì | vedi 3.3 |
| 4. Scienza, input, criteri, matrice e walltime invariati; checkout nuovo, tre job non ripetuti | sì | vedi 3.4 |
| 5. Due controlli operativi contabilizzati; tutti i gate mantenuti | sì | vedi 3.5 |

### 3.1 Tetti

I tetti sono esattamente 1,5 volte il fabbisogno misurato, che ho ricalcolato dalle due ricevute vere:

| Tetto | Vecchio | Fabbisogno misurato | A1 |
| --- | ---: | ---: | ---: |
| Fit neurali, GPU-ore | 20 | 35,90 | 53,85 |
| GPU totali, GPU-ore | 24 | 37,40 | 56,10 |
| Sola CPU, core-ore | 80 | 77,00 | 115,50 |
| Totale, core-ore | 272 | 376,20 | 564,30 |

- I controlli di budget sono due, `resources.gate_and_pack` e `execution.apply_training_pack`, e non ce ne sono altri: gli altri
  numeri rimasti nel codice sono limiti per job e di memoria, invariati.
- Sulle ricevute vere i due controlli danno gli stessi quattro totali. Senza A1 il gate fallisce esattamente in
  `neural_gpu_hours`, `gpu_total` e `core_total`; con A1 passano tutti. Il resto della proiezione (job, matrice, tempi, politica) è
  identico, ed è uguale ai due file committati `resource_gate_failed_20261121.json` e `resource_preview.json`.
- I tetti non si leggono dal piano: valgono solo i due insiemi scritti nel codice. Un piano senza `budget_amendment` usa i vecchi
  limiti. Un emendamento diverso da quello registrato è rifiutato (`unauthorized A1 budget specification`), e tetti alterati
  passati a mano ai due controlli sono rifiutati (`unauthorized budget ceilings`).
- I tetti di fase non cambiano: riferimenti 64 core-ore, rapporto 8, inferenza 1 GPU-ora, benchmark 0,5 GPU-ore e 4 core-ore.

### 3.2 Le tre ricevute

| Job | ID Slurm | Stato | sha256 della ricevuta | Byte |
| --- | --- | --- | --- | ---: |
| `TCNH-preflight-20261120` | 65016127 | `COMPLETED 0:0` | `0a3314138a98ea3b9e6a8475297c38d5fed575dc112eb52dc4ae4d14d51ee16b` | 14.122 |
| `TCNH-cpu-benchmark-20261120` | 65016438 | `COMPLETED 0:0` | `d6fca52ab2d719003d641a24f1067e25c7c31f9400d928c1d0e0d3757fc33ce8` | 70.963 |
| `TCNH-gpu-benchmark-20261120` | 65016447 | `COMPLETED 0:0` | `8228db8c6bed7df2eb45eeeedb2203f002ff4e64f625adca010c53f1b367f306` | 76.463 |

- Hash e dimensioni coincidono in tre posti: nel codice (`budget_amendment.LEGACY`), nelle copie committate e nei file originali
  su `$WORK`, letti oggi con `sha256sum`.
- `verified_receipt` legge prima il file e ne verifica l'hash contro il pin. Solo dopo, se l'ID è uno dei tre e l'hash è quello
  fissato, confronta l'identità con il piano base (`plan_released.json`, sha256 `84b82384…`, sorgenti `37985d28…`, codice
  `5d6fb2dc…`). Digest interno, contesto numerico, catena dei predecessori e artefatti restano verificati contro il piano nuovo.
- Non c'è un'eccezione generica: una ricevuta con l'identità vecchia e un altro ID è rifiutata; una delle tre modificata e
  ricalcolata è rifiutata; senza A1 le tre non passano con i sorgenti nuovi. I tre casi hanno un test sulle ricevute vere.
- `benchmark_identity_plan` usa il piano base solo per confrontare l'identità degli input registrata dai due benchmark, che sono
  già passati dalla verifica a hash esatto.

### 3.3 Identità nuova e review

- Le ricevute degli stadi successivi nascono con l'identità A1: `execution_sha256` nuovo, che lega sorgenti nuovi ed emendamento, e
  codice rivisto `a1ae9607…`.
- La vecchia review non rilascia A1. Provato con il comando vero: con il codice vecchio `release-plan` risponde
  `source differs from reviewed code`, con il codice nuovo `operational review must name the full reviewed code SHA or source
  SHA256`. `prepare-budget-amendment` riporta la review operativa a `REVIEW_REQUIRED`.

### 3.4 Scienza e grafo

- Il piano A1 differisce dal piano base in sei chiavi: `budget_amendment`, `execution_sha256`, `reader`, `review`, `status`,
  `submission_released`. Proposta, ricetta, otto input, produttori, radice degli output e i 24 job sono identici.
- Cambiano sei sorgenti su 247, tutti di budget, provenienza ed esecuzione; uno è il modulo nuovo. `A1.validate` rifiuta ogni
  altro sorgente cambiato e ogni campo scientifico diverso dal piano base.
- I walltime già fissati sono vincolati al piano base. I 16 neurali vengono dal normale pack misurato sulle stesse ricevute.
- Checkout nuovo `pose_temponet_20261121_A1`; gli ID dei 21 job rimasti restano quelli con suffisso `-20261120`, mai inviati.
  Il codice rifiuta di rigenerare i comandi dei tre job già riusciti (`A1 completed predecessor must never be resubmitted`).
- Nella prova locale i comandi delle due WLS escono con campagna e `CODE_DIR` nuovi, 8 CPU, 24 GB, 03:57:00, e senza dipendenze
  Slurm perché i predecessori sono provati da ricevuta e `sacct`.

### 3.5 Controlli operativi e gate

- I due job in più sono il checkout A1 (`TCNH-checkout-A1-20261121`, 0,5 core-ore) e l'inventario del secondo nodo GPU
  (`TCNH-GPU-inventory-A1-20261121`, 0,083 GPU-ore e 0,667 core-ore). Con i tre job del primo tentativo le allocazioni possibili
  sono 29.
- Il preventivo con questi job è 37,483 GPU-ore, 77,543 core-ore di sola CPU e 377,41 core-ore totali: ho rifatto il conto e torna.
  Resta sotto i tetti A1 con margine largo.
- Tutti i gate restano nel codice. Il confronto hardware sul secondo nodo GPU è nel prontuario, prima di qualsiasi invio GPU
  dipendente e dopo il gate delle WLS, con l'ordine di fermarsi se i valori differiscono.

## 4. Note operative (non bloccano)

1. **Script dell'inventario GPU.** Non è nella consegna. Va versionato prima dell'invio, fuori dai percorsi del manifest dei
   sorgenti: un file `scripts/slurm/pose_temponet*.sbatch` cambierebbe `source_sha256`. Va bene la cartella delle prove
   dell'operatore, come fu per `cpu_inventory.sbatch`.
2. **Che cosa confronta l'inventario.** Il pin `gpu8` è il digest dell'intero contesto numerico, non solo dell'hardware. Se
   possibile, l'inventario calcoli lo stesso digest con le stesse due chiamate del runner (`train.configure_numerics()` e poi
   `P.numeric_context("gpu")`, con 8 CPU) e lo confronti con `12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f`.
   Altrimenti bastano i quattro campi hardware. Un solo secondo nodo non li copre tutti: un job che capitasse su un nodo diverso
   si fermerebbe da solo, al costo della sua allocazione.
3. **Tetti nel piano.** `resource_policy` e la `policy` del gate riportano ancora i vecchi tetti, perché sono il testo della
   proposta. Quelli effettivi stanno in `budget_amendment.authorized_caps`: nel consuntivo e nel rapporto vanno citati questi.
4. **Tre campagne nel ledger.** Lo stesso budget copre `pose_temponet_20261119`, `pose_temponet_20261120` e
   `pose_temponet_20261121_A1`. Il consuntivo le somma tutte e tre.
5. **Tempi.** I limiti sul percorso critico sommano 17,73 ore: WLS 3,95, controlli 5,18, selezioni 5,28, rifit 1,22, inferenza
   0,95, metriche 0,95, raccolta 0,20, più code e passaggi dell'operatore. Dal 23/11 alle 08:00 Leonardo è in manutenzione per due
   giorni: un job il cui limite attraversa quell'ora non parte.
6. **Sola CPU.** Il fabbisogno, 77,00 core-ore, stava già nel vecchio tetto di 80. A1 lo porta a 115,50 per applicazione uniforme
   del fattore 1,5, come registrato.
7. **Fallimento storico.** `test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned` resta su questo ramo, come
   nelle review precedenti.

## 5. Che cosa hanno detto i tre job riusciti

Dalle ricevute vere, per chiudere i rischi dichiarati nella consegna del 20/11:
- il preflight ha superato l'integrità e la parità R-1 in 1 min 59 s. I due `E_dyn` ricalcolati sono identici bit per bit ai
  riferimenti storici (differenza 0,0): 0,7859020004638595 e 0,7786289843513977;
- i fold sono 20 e 20 utenti e le finestre di train con supporto sono 1.854, come nella lettura in sola lettura del 20/11;
- il benchmark GPU ha girato su `lrdn0059`: Xeon Platinum 8358, NVIDIA A100-SXM-64GB, 68.099.571.712 byte.

## 6. Prompt per GPT

```text
Review Claude dell'emendamento TCN A1 (budget e compatibilità delle ricevute): **APPROVATO, review operativa favorevole.**
Codice rivisto: a1ae9607681c1a181034d7f72770537054af419d
source_sha256: 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea
Consegna d715290ee06379f74e71c43fa03b2d33bb48170f, base c90d43e0b9f905fe13770498a59ffdb7c963413a.
Review: docs/reviews/claude_review_pose_temponet_A1_budget_20261121.md, ramo claude/temponet-db8-20261109 (commit nel messaggio di
Simone). Nomina per intero codice e source_sha256, come richiede release-plan.

Review del solo diff, senza agenti. Riprodotti: 299 test TCN passati (22 file, comprese le due catene dei 24 stadi), 125 storici
più il fallimento noto, bash -n sui quattro launcher. Ricalcolati i due controlli di budget sulle ricevute vere: stessi totali
(35,9 / 37,4 / 77,0 / 376,2), senza A1 falliscono solo neural_gpu_hours, gpu_total e core_total, con A1 passano tutti. Le tre
ricevute su $WORK hanno gli hash e i byte di budget_amendment.LEGACY; i tre job sono COMPLETED 0:0. La vecchia review non rilascia
A1 né con il codice vecchio né con il nuovo. Prova locale di release-plan, resource-plan e commands --stage wls_calibration riuscita.

I cinque punti sono tutti verificati. Nessuna correzione richiesta prima del rilascio. Note operative, senza un nuovo giro:
1. versionare lo script dell'inventario GPU prima dell'invio, fuori dai percorsi del manifest (non sotto
   scripts/slurm/pose_temponet*.sbatch), come cpu_inventory.sbatch;
2. se possibile l'inventario calcoli lo stesso digest del contesto numerico (train.configure_numerics() e poi
   P.numeric_context("gpu"), 8 CPU) e lo confronti con gpu8 12cb2a0b…; altrimenti i quattro campi hardware;
3. nel consuntivo e nel rapporto citare i tetti di budget_amendment.authorized_caps: resource_policy riporta ancora i vecchi;
4. il consuntivo somma le tre campagne del ledger (…_20261119, …_20261120, …_20261121_A1): 29 allocazioni al massimo;
5. limiti sul percorso critico 17,73 ore più code; dal 23/11 alle 08:00 Leonardo è in manutenzione.

La review non attesta tempi, memoria e determinismo su A100 oltre i due benchmark, né l'identità numerica sul secondo nodo GPU, né
l'esito dei gate sui dati veri: li decidono i gate. Operatore unico: 2-TCN head launcher. Nessun lancio da parte mia.
```
