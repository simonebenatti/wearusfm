# Review Claude: due diagnostici contrastivi da stato comune (`365cb1e`)

18/11/2026, sera. Revisore: Claude. Linea 3: codice, protocollo e lettura spettano a GPT scientifico, i lanci a 3-launcher.
Claude non lancia job.

## Richiesta

Prompt di GPT passato da Simone: «Rivedi la consegna 365cb1e29bc410013d4ee0494b9c2e92835bc7f1 su
origin/codex/tokenizer-logcontrast-20261104, diff da 09eeaf93.» Si chiede inoltre: «Verifica disegno, implementazione,
invarianti, lettura dei risultati, pin e risorse. Pubblica verdetti separati ed eventuali rilievi con prove riproducibili».
Infine: «Nessun lancio: aspettiamo convergenza prima del deploy».

Perimetro: i 14 file di `09eeaf93..365cb1e`. Ho letto anche i due commit successivi del ramo. `35ea94b` aggiunge il prompt di
3-launcher e `submission_plan.json`. `d9020c2` chiarisce i ruoli in `CLAUDE.md` e aggiunge due voci di registro. Nessuno dei due
tocca codice, launcher o risorse. Punta del ramo riletta alle 22:25: ancora `d9020c2`.

## Verdetti

| Blocco | Verdetto | Bloccanti |
|---|---|---|
| Disegno | **APPROVATO** | nessuno |
| Implementazione | **APPROVATA** | nessuno |
| Prerequisiti del collaudo | **APPROVATI** per lo SHA `365cb1e29bc410013d4ee0494b9c2e92835bc7f1` con il piano di `35ea94b` | nessuno |

Non chiedo modifiche al codice: lo SHA congelato resta quello del piano. Le note N-1...N-4 riguardano la lettura e il testo del
protocollo e si correggono senza un nuovo giro di review. Conviene scriverle prima dell'invio, con un commit di soli documenti:
lo SHA di esecuzione resta `365cb1e`. Se arrivano dopo le misure, vanno marcate come scritte dopo i dati, sempre prima della
lettura.
I suggerimenti S-1 e S-2 valgono per il prossimo giro. L'approvazione non autorizza training: entrambi i diagnostici scrivono
sempre `training_approved: false`. Un'etichetta `MEASURED_FOR_REVIEW` o `CONTROL_INPUTS_VERIFIED` non supera il gate storico.

## 1. Disegno

**Replay CUDA a peso zero.**
- All'update 255 (passo 254, `fill_steps` 256) il fattore dei pesi e' 0.
- I tre bracci eseguono lo stesso grafo: anche C0 calcola InfoNCE e Rank-N-Contrast, con peso 0
  (`contrastive_objectives.py`, `attach` e `weights`).
- Le tre ricette della coorte differiscono solo per `contrastive.arm` (prova 2 di `checks.out.txt`).
- A stato, batch e RNG ripristinati, una differenza fra bracci puo' quindi venire solo dal calcolo o da uno stato non
  ripristinato. Il disegno separa bene le due cose. Le repliche misurano la variabilita' del calcolo. I quattro profili
  separano precisione, backend e determinismo. Le impronte per rank attestano che gli ingressi sono gli stessi.
- I profili toccano davvero il calcolo: il modello usa `scaled_dot_product_attention` in backbone, proxy e decoder, e `index_add`
  nelle loss.
- L'ordine C0, CN, CR, CR, CN, C0 bilancia un eventuale effetto della posizione.
- Il profilo `bf16_auto` lascia i valori predefiniti del trainer: stesso autocast (`run.py`, `amp`), nessuna impostazione TF32 nel
  codice di training.

**Ripresa da stato comune.**
- Le due riprese partono dagli stessi byte a 580, verificati con SHA-256 prima di ogni fase.
- La prosecuzione in memoria e' anch'essa un processo ripreso a 576. Il confronto isola quindi il riavvio a 580, non il
  confronto con un run mai interrotto. Il protocollo lo dice, insieme all'effetto della ricostruzione dei bucket DDP.
- La ricetta lo permette (prova 3):
  - lo schedule WSD fa iniziare il decay a 39.237, quindi 580 e' un punto di salvataggio valido;
  - 588 non supera `total_steps`;
  - fra 577 e 588 si salva solo a 580 e alla fine, si valuta solo a 588, e nessun rapporto di copertura cade fra 581 e 588
    (blocco 1024);
  - fra 581 e 588 i pesi contrastivi sono pieni (fattore 1).

**Lettura.** La tabella e' fissata prima delle misure. Distingue invarianti esatti, differenze numeriche e ripetibilita' fra le
due riprese, e lascia aperte le attribuzioni quando le tre coppie hanno ampiezze simili. Limiti dichiarati in modo adeguato:
- fixture emg2pose a batch 8, un solo backward;
- coda della calibrazione traslata di 258 passi;
- denominatore della ripresa pari allo spostamento 580→588, non la soglia storica dell'1%.

## 2. Implementazione

Invarianti controllati nel codice:
- **Stato prima di ogni backward:**
  - studente e teacher sono verificati contro l'impronta iniziale della coorte;
  - coda e copertura sono nuove e caricate;
  - batch e microbatch hanno la stessa impronta;
  - l'RNG e' ripristinato per rank dopo la costruzione del modello (`contrastive_common_state_cuda.py:186-206`).
- **Consumo RNG:** confrontato dopo il backward, per rank e fra profili.
- **Riduzione DDP:** l'accordo bit per bit fra rank e' controllato per ogni caso.
- **Gradienti:** copiati prima del clipping (`:214-228`). Norme e coseni per blocco in FP64, con normalizzazione simmetrica.
  I parametri con gradiente assente da una sola parte sono elencati e contati come zeri.
- **Dropout muscolare:** l'unico consumo RNG per passo e' il dropout su CUDA (`channel_identity.py:58-61`, generatore globale).
  Il DataLoader ha un generatore proprio (`run.py:1518-1523`), quindi creare l'iteratore alla ripresa non sposta l'RNG globale.
- **Ripresa:**
  - la copia diagnostica cambia solo `keep_checkpoints`: l'impronta dell'intero contenuto e' ricalcolata dopo aver rimesso la
    lista originale (`contrastive_common_checkpoint.py:61-79`);
  - le tracce leggono batch, ingressi sul dispositivo e RNG prima del forward, e lo stato al confine prima dell'update 581.
    Studente, teacher, momenti, gruppi con il learning rate del passo e stato contrastivo sono confrontati con il checkpoint
    a 580;
  - le impronte non dipendono dal dispositivo (`pairing.py`, `.cpu()`), quindi il confronto fra GPU e checkpoint su CPU vale;
  - nello stato contrastivo confrontato in modo esatto ci sono solo conteggi e identita', nessun valore numerico.
- **Distanze:** `_tensor_distance` rispetto allo spostamento del primo membro della coppia da 580, con minimo 1e-12. Include i
  buffer, che nel modello sono costanti: contribuiscono zero al numeratore e al denominatore.

Prove riproducibili (cartella `claude_review_contrastive_common_diagnostics_20261118_evidence/`):
- `tests7.out.txt`: i due file pytest della consegna, eseguiti da `git archive 365cb1e`, **7 passati in 18,07 s**.
- Digest dei sorgenti storici ricalcolato: `46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320`, uguale a quello
  del driver e delle due release. `bash -n` del launcher passa.
- `smoke_gloo_result_claude.json`: lo smoke Gloo a quattro rank, rieseguito da me (17 s). Le coppie coincidono byte per byte con
  `smoke_gloo_result.json` di GPT:
  - nessun disaccordo di confine, ingressi o RNG;
  - `reload_a` e `reload_b` identici;
  - continuo contro ripresa: studente 7,523e-7, teacher 3,447e-9; prima differenza di metrica all'update 5.

Comando dei test, dalla radice dell'albero estratto:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src:$PWD/tests/cpu:$PWD/scripts ~/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=<cartella nuova> tests/cpu/test_contrastive_common_state_cuda.py tests/cpu/test_contrastive_common_checkpoint.py`

## 3. Prerequisiti del collaudo

- **Pin** (`checks.py`, prova 1):
  - il piano di `35ea94b` congela `365cb1e`;
  - le sei impronte di launcher, driver, risorse e generatore coincidono con i file del commit;
  - submitter `814da4bc66dbf18d...` uguale in `365cb1e`, in `main` e nel file del worktree `wearusfm-gpt-science`, pulito.
- **Risorse** (prova 4):
  - incrementale 6 GPU-ore e 48,2667 core-ore, come dichiarato;
  - cumulativo 11,2844 su 14 GPU-ore e 99,6378 su 129,33 core-ore.
- **Launcher:**
  - uscite fuori dal repository;
  - controlli di `CODE_DIR`, SHA, pulizia, sola lettura, 32 CPU, 240 GiB e 30 minuti;
  - cartella d'uscita nuova;
  - stesso ambiente dei launcher contrastivi gia' usati.
- **Leonardo, sola lettura alle 22:20** (`leonardo_prereq.out.txt`):
  - quota del progetto 977,5 GB su 1 TB: 46,5 GiB liberi, contro i 14 richiesti per le due riprese. Qui `shutil.disk_usage` vede
    lo stesso valore, quindi il controllo dei 7 GiB nel driver e' significativo;
  - i tre checkpoint d'ingresso sono presenti con le dimensioni attese (234.339.066 byte e due da 463.189.952);
  - cartella d'uscita, checkout di campagna e ref `deploy/contrastive-common-20261118` sono assenti;
  - nessun job dell'utente su `boost_qos_dbg`.

  Le impronte dei file da 442 MiB non le ho ricalcolate sul nodo di login: le verificano i job prima di lavorare.
- **Durata:** i benchmark da 576 passi sono durati 11-13 minuti (`accounting.txt`). Una ripresa da 4-12 passi in tre fasi sta
  largamente nei 30 minuti.

## Note (non bloccano)

- **N-1, lettura CUDA.** La coppia `bf16_auto`/`bf16_math` cambia due cose: la scelta del backend SDPA e TF32 di cuDNN.
  `controls()` spegne `cudnn.allow_tf32` solo fuori da `bf16_auto` (`:73-75`); il valore predefinito di PyTorch e' acceso.
  La differenza va letta come «backend SDPA e impostazioni TF32». I flag effettivi sono nel rapporto (`flags` per rank).
- **N-2, lettura CUDA.**
  - Come riferimento del rumore conviene usare le tre coppie di repliche (C0, CN, CR), non solo C0/C0. Cosi' un effetto della
    posizione nell'ordine non viene scambiato per un effetto del backend.
  - Manca una riga per un caso. Se le repliche differiscono anche nel profilo deterministico, la sorgente sta fuori dagli
    algoritmi controllati (NCCL o un operatore senza flag) e non si attribuisce al backend.
  - Se il profilo deterministico si ferma su un operatore senza implementazione deterministica, gli altri tre profili restano
    validi. Il codice lo prevede.
- **N-3, dichiarazione della fixture.** `build_batches` toglie le etichette di posa alle due finestre del rank 0
  (`contrastive_cuda_check.py:73-74`). Il rapporto lo registra (`removed_pose_rank: 0`), il protocollo no: basta una frase.
- **N-4, lettura della ripresa.** In 8 passi lo spostamento del teacher e' circa (1-m) volte quello dello studente. Le distanze
  relative del teacher si confrontano fra le tre coppie, non con quelle dello studente.

## Suggerimenti per il prossimo giro

- **S-1.** Nelle tracce, `rng_before_sha256` unisce i generatori CPU e CUDA in un'unica impronta (`:116`). Un disaccordo verrebbe
  rilevato, ma non attribuito al generatore giusto. Due impronte separate costano nulla.
- **S-2.** Nessun test copre la precedenza degli stati di `orchestrate()` e `run()`. Manca anche il ramo d'errore di
  `diagnostic_seed` con un campo alterato fuori dalla lista. Test brevi, senza effetto sul digest.

## Per 3-launcher

- **P-1.** I percorsi `--script-stdin` del piano puntano al worktree `wearusfm-logcontrast`, che 3-launcher condivide con GPT.
  Le sei impronte vanno riverificate subito prima di ogni `sbatch`, come gia' previsto al punto 1 del prompt di 3-launcher.
- **P-2.** Su `boost_qos_dbg` girano al piu' due celle per utente alla volta: la terza aspetta, senza duplicati.

## Limiti di questa review

- Sul Mac non ci sono GPU, NCCL, bf16 vero ne' torch 2.2: CUDA, NCCL e il profilo deterministico li vedranno solo le celle.
- Nessuna analisi di mutazione dei test e nessun agente revisore. E' una verifica rapida, come chiesto da Simone il 18/11
  («non possiamo fare troppi giri di review tra voi 2»).
- Ho letto come contesto i documenti della lettura 64775571 e della parita' del gradiente, senza rivederli.

## Prompt per GPT

```text
Review Claude della consegna 365cb1e (diagnostici da stato comune): disegno APPROVATO, implementazione APPROVATA, prerequisiti del collaudo APPROVATI per lo SHA 365cb1e29bc410013d4ee0494b9c2e92835bc7f1 con il piano di 35ea94b. Nessun bloccante, nessuna modifica al codice: lo SHA congelato resta quello.
Ramo origin/worktree-claude-training-engine, file docs/reviews/claude_review_contrastive_common_diagnostics_20261118.md, prove nella cartella _evidence/ accanto (7 test passati, digest 46938f31... ricalcolato, 6 pin su 6, smoke Gloo rieseguito identico al tuo, lettura di Leonardo: 46,5 GiB liberi, input presenti, uscite e ref assenti).
Note da recepire nel testo del protocollo e della lettura, senza nuovo giro. Meglio prima dell'invio, con un commit di soli documenti (lo SHA di esecuzione resta 365cb1e); se arrivano dopo le misure, marcarle come scritte dopo i dati, sempre prima della lettura:
N-1 bf16_auto/bf16_math cambia backend SDPA e TF32 di cuDNN, non solo il backend;
N-2 rumore di riferimento dalle tre coppie di repliche e riga per «repliche diverse anche nel profilo deterministico»;
N-3 dichiarare che il rank 0 della fixture non ha posa;
N-4 distanze del teacher confrontate fra coppie, non con lo studente.
Per il prossimo giro: S-1 impronte RNG separate CPU/CUDA nelle tracce, S-2 test della precedenza degli stati e del campo alterato in diagnostic_seed.
Per 3-launcher: riverificare i sei pin subito prima di ogni sbatch e non piu' di due celle dbg in esecuzione.
```
