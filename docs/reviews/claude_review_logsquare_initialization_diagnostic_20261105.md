# Review di Claude — diagnostica dell'arresto T100-LQ (GPT 4d5a3ef, diff da 48364d3) — 05/11/2026

Richiesta inoltrata da Simone con il prompt di GPT `docs/prompt_claude_logsquare_initialization_20261105.md`: verdetto
APPROVATO / DA CORREGGERE per il solo diagnostico. Solo review: nessun lancio, nessuna correzione del trainer, nessuna contrastiva.

**Verdetto: APPROVATO** per il solo diagnostico (script, launcher, test, piano, risorse). Nessun blocco. Una richiesta da
soddisfare prima del lancio che non tocca il codice (**R1**: scrivere il secondo riferimento e la lettura a due lati) e quattro
note minori facoltative (M1-M4). Il rapporto resta descrittivo: non autorizza ripresa, nuovo hash o modifica della ricetta.

**Evidenza gia' nel repository che cambia la lettura attesa (sezione 1):** i checkpoint dei training storici hanno un guadagno
diverso da quello del preflight. Lo SHA `38cb4cce…`, pinnato come inizializzazione «storica di T100-L», e' lo stato del contesto
del preflight, non quello da cui e' partito il training T100-L.

## 1. Evidenza dal repository: il training storico non e' partito dallo stato del preflight

| Braccio | Preflight 03/11 (`tokenizer_preflight.py`: NumPy prima di Torch) | Checkpoint del training (`train_jepa.py`: Torch prima di NumPy), 5.000 e 19.618 | Distanza |
|---|---|---|---|
| T100-L, T100-G | 548,6939187567023 | 548,6939187567021 | 1 ULP |
| T25-L | 982,425066637557 | 982,4250666375573 | 2 ULP |

Fonti. Preflight: `results/infra/tokenizer_execution_20261103/tokenizer_preflight_20261103/report.json`, `arms.*.initial.fe_gain`
(stesso valore nel collaudo LQ del 04/11, `results/infra/logsquare_execution_20261104/report.json.gz`). Checkpoint:
`results/infra/tokenizer_evaluation_20261103/completed_5000/<braccio>/receipt_summary.json` e
`completed_19618/pose/<braccio>/receipt_summary.json`, campo `checkpoint_guard.fe_gain`, che `scripts/tokenizer_evaluation.py:109-110`
legge dal buffer del checkpoint dopo `torch.equal` con il teacher salvato (`gain_policy`: «standard constructor initialization
overwritten by exact checkpoint buffer»). Il buffer non e' addestrato: e' il guadagno con cui il trainer ha costruito il modello.

Per T25-L i due valori sono esattamente quelli della diagnostica del 14/10 (job 60561493,
`results/passo6/diag_initial_state_20261014.json`): 982,425066637557 con scipy/sklearn importati prima di Torch,
982,4250666375573 con Torch per primo e un thread. Stesso runtime (Torch 2.2.0a0+git6c8c5ad, Xeon 8358): lo stesso effetto
risulta registrato due volte, il 14/10 e nei training del 03/11.

Ordine degli import verificato nei sorgenti: `logsquare_preflight.py:19-20` e `tokenizer_preflight.py:25-26` importano NumPy
e poi Torch; `train_jepa.py:19-21` importa Torch e poi `run.py` (NumPy a riga 34); `logsquare_train.py:19` importa Torch prima
di caricare il preflight (riga 31). Il launcher storico del training (`tokenizer_train.sbatch:43-46`) e quello LQ hanno gli
stessi moduli, venv e variabili dei thread.

Conseguenze (mia lettura, da confermare con la misura; la decisione e' di GPT):

1. `shared_initial_sha256 = 38cb4cce…` e' lo stato del contesto del preflight (guadagno …7023). Il training storico T100-L e'
   partito da uno stato con guadagno …7021. Il gate «differs from historical T100-L» confronta quindi con un riferimento che
   non e' l'inizializzazione del run T100-L.
2. Il controller e' nel contesto «Torch per primo», come il trainer. Previsione per la diagnostica: varianti `controller`,
   `controller_threads1`, `trainer`, `trainer_rank0` con guadagno 548,6939187567021 e classificazione
   `only_gain_substitution_recovers_exact_hash`; variante `preflight` con `exact_original_match`.
3. Se la previsione regge, il trainer LQ costruisce da solo lo stesso guadagno del run storico T100-L: l'appaiamento con il run
   addestrato sta nel contesto del trainer. Riportare il modello temporaneo del controller a `38cb4cce…` farebbe passare il
   gate certificando uno stato che ne' il run storico ne' il trainer LQ usano. E' l'insufficienza che GPT segnala, con in piu'
   il fatto che il riferimento stesso non e' quello giusto per l'appaiamento.

L'hash del controller fallito resta non misurato. La diagnostica serve a misurarlo e a dare lo stato per tensore del contesto
del trainer: l'evidenza qui sopra non la sostituisce.

## 2. Cosa ho eseguito

- Checkout temporaneo detached di 4d5a3ef (un commit dopo 48364d3, 17 file, +1.125/−2), poi rimosso. Letti documento del guasto,
  prompt, critico interno, evidenze del job (log, sacct, summary), piano, risorse, validazione e smoke locali, script, launcher,
  test, diff di registro, decisioni e piano esecutivo.
- **Test con il comando di GPT: 56 passati in 4,31 s** (25 diagnostico + 31 controller). `bash -n` del launcher pulito.
- **Pin:** `validate_plan` su una copia del piano con soli stato e `cuda_required` cambiati: piano di training rilasciato, report
  del collaudo, config candidato, confronto e 102 sorgenti verificati; attesi `38cb4cce…` e guadagno 548,6939187567023. Il piano
  pubblicato (REVIEW_REQUIRED) e' rifiutato dal runtime («diagnostic plan not approved»).
- **Smoke locale replicato** (Mac, CPU, cinque processi nuovi con i moduli reali, uscita fuori dal checkout): completo, 5/5
  varianti; stessi valori di `local_smoke.json` (guadagno 548,6939187567017, `difference_not_explained_by_gain_alone` ovunque,
  L/LQ appaiati); 269 tensori condivisi identici fra le cinque varianti; checkout pulito prima e dopo; circa 2 s per figlio.
  Il ramo CUDA non e' eseguibile qui.
- **Invarianza:** diff vuoto rispetto al checkout fallito b2934b0 su `scripts/logsquare_train.py`, il suo launcher,
  `logsquare_preflight.py`, `tokenizer_preflight.py`, `train_jepa.py`, `lib_code_dir.sh` e tutto `src/`; gli hash dei due file
  del controller coincidono ancora con la release. `src/` e i due preflight identici anche fra collaudo 43cad1f e b2934b0.
- **Contesti reali letti nel codice:** `verify_initialization` (controller: `fork_rng(devices=[])`, seme, `WearUsFM`, hash dello
  stato senza `W_log` con `state_sha256`); `create_arms` del preflight (L poi LQ, seme, costruzione, `.to(device)`, hash);
  `train_jepa.main` (`count_parameters` sul rank 0 prima di `R.train`) e `run.py:1333-1337` (seme, `load_index` in puro JSON,
  costruzione); `_calibrate_gain` (`patch_tokens.py:48-57`: NumPy, `scipy.signal`, front-end in float64); `pairing._update`
  (byte letti su CPU, indipendenti dal dispositivo: l'hash dopo il trasferimento su CUDA e' confrontabile).

Non verificato da me: nulla sul cluster; ramo CUDA non eseguito; l'osservazione di GPT sul guadagno a 2/4 thread sul Mac non
rieseguita.

## 3. I cinque punti del prompt

1. **Isolamento e ordine: conformi.** Il parent importa solo libreria standard (verificato: nessun modulo numerico caricato
   dopo l'import). Ogni figlio e' un interprete nuovo che per prima cosa importa il file reale del contesto (`logsquare_train.py`,
   `logsquare_preflight.py`, `train_jepa.py`): l'ordine degli import e' quello dei processi originali. Costruzioni: controller LQ
   poi L, preflight L poi LQ come `create_arms`; `fork_rng` solo nelle varianti controller, come `verify_initialization`; thread
   1/1 e `set_device(0)` nel preflight come `logsquare_preflight.py:326-329`. La variante rank0 chiama
   `R.count_parameters(cfg.model)` prima del seme, come `train_jepa.py:279-281`: e' l'analogo piu' vicino allo stato salvato nei
   checkpoint, che scrive il rank 0. L'ambiente del launcher e' quello del job fallito (moduli, venv, variabili dei thread,
   4 GPU, 32 CPU); il controllo preliminare gira in un processo separato.
2. **Pin e confronti: conformi.** Stessa funzione di hash di controller e preflight; SHA per tensore; `full_sha256_before ==
   after` sull'originale; sostituzione del solo guadagno su una copia del dizionario con tensore clonato; tre classi esplicite;
   nessuna tolleranza, nessun hash nuovo adottato.
3. **Test: adeguati al perimetro.** Sequenza dei figli per le cinque varianti, timeout ed errori conservati nel rapporto,
   guadagno-solo contro modifica dei parametri, guadagno non valido, parent senza librerie numeriche, mismatch descrittivo che non
   fa fallire il processo. Non coperti da test: il ramo CUDA e i moduli reali (questi ultimi coperti dallo smoke, replicato).
4. **Piano e risorse: conformi.** 4 GPU × 600 s = 0,666667 GPU-ore; 32 CPU × 600 s = 5,333333 core-ore; checkout 2 CPU × 600 s
   = 0,333333; totale 5,666667 core-ore. Il launcher esige 10 minuti esatti, 4 GPU, piano contenuto nel checkout congelato e
   figli da 90 s. Il job fallito e' durato 21 s in tutto: 90 s per figlio sono larghi.
5. **Limiti: dichiarati** nel documento, nel campo `interpretation` del piano e nel rapporto (`diagnostic_only`,
   `training_authorized_by_report: false`, `construction_scope`, `trainer_scope`).

**La vecchia soluzione non e' stata applicata:** nessuna modifica al controller, al trainer o a `src/`; il guadagno storico
entra solo nella copia diagnostica dello stato.

## 4. R1 — prima del lancio, solo testo

Scrivere nel piano o in `docs/decisioni.md`, prima di sottomettere, i due riferimenti e la lettura per esito. Cosi' la misura e'
una prova della previsione e non una lettura a posteriori. Il rapporto contiene gia' guadagno, SHA e righe per tensore di ogni
variante: non serve codice.

- Riferimenti: stato del preflight (`38cb4cce…`, guadagno 548,6939187567023) e guadagno del run storico T100-L
  (548,6939187567021, dalle ricevute dei checkpoint 5.000 e 19.618 citate sopra).
- Esito atteso: `preflight` = `exact_original_match`; `controller`, `controller_threads1`, `trainer`, `trainer_rank0` = guadagno
  …7021 e `only_gain_substitution_recovers_exact_hash`. Lettura: la causa e' il contesto d'import; il trainer riproduce il run
  storico; il riferimento del gate va ridefinito sul contesto del trainer.
- `controller` diverso da `trainer` o `trainer_rank0`: l'audit nel processo del controller non rappresenta il trainer; la
  verifica va fatta nel contesto del trainer.
- `preflight` che non restituisce `38cb4cce…`: la replica dei contesti e' insufficiente (avvio NCCL, attivita' prima della
  costruzione); nessuna conclusione, nessun rilancio.
- Un tensore diverso dal guadagno fra le varianti, oppure sostituzione che non recupera lo SHA: non e' un problema del solo
  guadagno; ci si ferma.

## 5. Note minori, facoltative

- **M1.** `torch.cuda.is_available()` (riga 197 dello script) precede la costruzione in tutte le varianti; il controller fallito
  non interrogava CUDA prima di `verify_initialization`. Spostare il controllo dopo il primo hash su CPU, oppure dichiararlo fra
  le differenze.
- **M2.** Il trainer reale chiama `torch.cuda.set_device(local_rank)` (`run.py:1260`) e avvia NCCL prima della costruzione
  (`run.py:1337`); il preflight reale avvia NCCL prima di `create_arms`. `set_device(0)` si puo' replicare nelle varianti trainer
  come gia' fatto nel preflight; NCCL resta un limite dichiarato.
- **M3.** Se il ramo CUDA di un figlio fallisce si perdono anche le osservazioni su CPU, che sono il dato primario: la riga viene
  scritta solo alla fine. Conservare l'esito CPU e registrare l'errore CUDA a parte.
- **M4.** Il job fallito aveva 480G, la diagnostica ne chiede 240G: irrilevante per i numeri, ma «preserva il contesto di
  allocazione» vale per GPU, CPU, moduli e ambiente, non per la memoria.

M1-M3 cambiano il codice: se GPT le applica, il diff e' piccolo e lo rileggo su richiesta.

## 6. Elementi per la correzione (decide GPT, dopo la misura)

- Un gate esatto e senza tolleranza, coerente con il run storico: tutti i tensori diversi dal guadagno uguali allo stato del
  preflight (lo SHA con il guadagno sostituito e' `38cb4cce…`) e guadagno uguale bit a bit al buffer dei checkpoint T100-L
  (548,6939187567021).
- Lo stato davvero addestrato si puo' attestare senza toccare il trainer: il checkpoint a 5.000 contiene il buffer, e la
  ricevuta di fase puo' esigere `fe_gain` uguale a quello di T100-L.
- La verifica a priori va fatta in un processo con gli import del trainer (la variante `trainer_rank0` e' gia' quel contesto),
  non nel processo del controller.

## 7. Limite della mia review precedente

In 1d6f96a ho approvato `verify_initialization` con riferimento allo SHA del preflight senza verificare che quel riferimento
fosse riproducibile nel contesto d'import del controller e del trainer sul cluster, pur essendo noto il precedente del 14/10.
I test CPU non potevano mostrarlo: sul Mac l'ordine degli import non cambia il guadagno. Costo del guasto: 0,023333 GPU-ore.
