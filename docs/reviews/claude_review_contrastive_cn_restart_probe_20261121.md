# Review Claude: prova mirata sulla ripresa InfoNCE (`9d34dc69`), release chiusa per due correzioni

21/11/2026. Revisore: Claude. Linea 3: scienza e codice a GPT, lanci a 3-launcher. Claude non ha lanciato nulla.
Il verdetto storico resta `FAILED_REQUIRES_REVIEW`.

## Richiesta

Prompt di GPT passato da Simone (`docs/prompt_claude_cn_restart_probe_20261121.md` a `9a2ce7e7`): rivedere il solo diff
`918245b6..9d34dc690cf60fcbfe405279dc17546655fa3f5b` del diagnostico a due riprese InfoNCE. Simone, in chat con Claude: «Se approvi il
diagnostico, pubblica il nuovo blocco di review previsto nel prompt e restituisci lo SHA completo. Il lancio spetta poi a
3-launcher.»

Metodo:
- **Lettura mia** dei quattro script, del lanciatore, del piano, del protocollo e della voce di `decisioni.md`.
- **Impronte ricalcolate.** `--package` restituisce `package_sha256 b0640ca6…` e `plan_sha256 9d5811b6…`, come nel prompt.
  Il digest dei sorgenti di produzione è invariato: `40d39f56…`.
- **Letture in sola lettura su Leonardo**, descritte in fondo: sorgente e API del torch del cluster, e la funzione di
  confronto di GPT eseguita sui checkpoint storici veri.
- **Sette revisori indipendenti**, uno per parte, più due scettici per ogni rilievo serio: 41 analisi in tutto.
- **Prova della catena intera su CPU** con il trainer vero e 4 processi Gloo.
- **Rilettura della bozza** contro le fonti da parte di un verificatore indipendente: circa 95 affermazioni, nessun
  errore di fatto, 14 precisazioni applicate qui.

## Esito

**Diagnostico non approvato in questo stato. Il blocco di review non è emesso e la release resta chiusa.**

- **Il disegno è approvato.** Due riprese dallo stesso fork, osservazione per rank e per tensore al 545, criteri e
  falsificatori registrati prima della misura: è l'attuazione corretta del limite del 20/11.
- **L'implementazione delle due gambe regge.** Le celle eseguono la vera ripresa CN_A, il riscaldamento è a effetto nullo
  e l'osservatore non altera la riduzione. È verificato con il codice vero, su CPU.
- **Due difetti bloccanti stanno nel solo confronto finale.** Con il codice attuale il report uscirebbe `INCONCLUSIVE`
  in ogni caso, dopo aver consumato le GPU. Ognuno si corregge in poche righe.
- **Nove correzioni non bloccanti** conviene farle nello stesso commit, perché due di esse toccano regole che vanno
  fissate prima della misura.
- **Prossimo passo.** Corretti i punti, rileggo solo il diff ed emetto il blocco sul commit nuovo.

## Difetti bloccanti

### B-1. La lettura dei vettori fallisce sul torch del cluster

- **Dove.** `scripts/diagnostics/contrastive_cn_restart_observe.py:144`:
  `torch.load(tensor_path, map_location="cpu", mmap=True, weights_only=True)`, con `tensor_path` di tipo `pathlib.Path`.
- **Che cosa succede.** Il torch `2.2.0a0+git6c8c5ad` di Leonardo, con `mmap=True`, accetta solo una stringa:
  `serialization.py:1012-1014` solleva `ValueError("f must be a string filename in order to use mmap argument")`.
  `compare.py` cattura l'eccezione e scrive il report con stato `INCONCLUSIVE`.
- **Prova.** Il sorgente letto sul cluster. Più revisori hanno riprodotto il fallimento sul Mac, simulando quel
  controllo su osservazioni prodotte dal codice vero con Gloo. È lo stesso errore del job 64706211 del 17/11.
- **Perché i test non lo vedono.** Il torch 2.14 del Mac accetta i `Path`. Sette test di rifiuto di `load_observation`
  usano `pytest.raises(ValueError)` senza `match`: in cinque di essi l'errore del torch 2.2 passerebbe per il rifiuto
  atteso.
- **Correzione.** `torch.load(str(tensor_path), …)`. Due revisori hanno verificato che basta.

### B-2. Il confronto a 576 della gamba riscaldata non può mai passare

- **Dove.** `scripts/diagnostics/contrastive_cn_restart_compare.py:241` e `:248`:
  `require(not endpoints["warm"]["metadata_or_schema_different"], …)`.
- **Che cosa succede.** Il checkpoint salva `torch.cuda.get_rng_state_all()` del rank 0, cioè lo stato dei generatori di
  tutte e quattro le GPU viste da quel processo (`run.py:1482`). Tre di queste GPU il rank 0 non le usa.
  - Un training partito da zero risemina tutti i generatori con un seme per rank (`run.py:1463`).
  - Una ripresa li semina con `cfg.seed` (`run.py:1383`) e poi ripristina solo quello della propria GPU
    (`run.py:1446-1450`).

  Quindi `cuda_rng.1`, `cuda_rng.2` e `cuda_rng.3` di qualunque ripresa differiscono da quelli del continuo, per
  costruzione e senza alcun effetto sulla traiettoria. La gamba riscaldata è una ripresa: la riga 248 solleva
  `ValueError` e il report esce `INCONCLUSIVE`, prima ancora di leggere i gradienti del 545.
- **Prova sui dati veri.** Ho eseguito su Leonardo, in sola lettura, la funzione `checkpoint_differences` del commit
  candidato sui checkpoint storici:

| Coppia | Tensori in virgola mobile diversi | `metadata_or_schema_different` | `exact_payload` |
|---|---|---|---|
| Continuo e ripresa CN_A | 1.990 (270 studente, 270 teacher, 540 momenti, 910 della coda) | `cuda_rng.1`, `cuda_rng.2`, `cuda_rng.3` | falso |
| Riprese CN_A e CN_B | 0 | nessuno | vero |

- **Conseguenza anche senza la riga 248.** `exact_payload` resterebbe falso e un recupero completo del continuo sarebbe
  classificato `PARTIAL_SUPPORT`: `SUPPORTED` è irraggiungibile.
- **Correzione proposta.** Nel solo confronto fra gamba riscaldata e continuo, trattare questi tre percorsi come metadati
  di processo noti:
  - escluderli dall'identità scientifica e riportarli a parte;
  - richiedere che coincidano con quelli della gamba ordinaria, che deve riprodurre CN_A;
  - continuare a confrontare `cuda_rng.0`, `torch_rng` e `rank_rng`.

  La regola va scritta in `decisioni.md` e nel protocollo prima della misura, con la prova storica qui sopra. Va scritto
  anche che ogni altra differenza non in virgola mobile a 576 continua a rendere la prova inconclusiva: oggi questa
  regola è nel codice ma non nei criteri registrati. È coerente con il criterio già registrato: «Le differenze di sola
  serializzazione vengono separate».
- **Perché i test non lo vedono.** `compare()` non è mai eseguito per intero dalla suite, e nessun payload di test ha
  questa forma.

## Da correggere nello stesso commit

Nessuno di questi punti basterebbe a chiudere la release. I primi due toccano però regole che dopo la misura sarebbero
post hoc.

1. **Soglia di 1e-6 e parametri scalari.** Il limite è relativo alla sola norma del gradiente ridotto. Il modello ha due
   parametri di un solo elemento, `anchor_heads.rms.bias` (ID 253) e `anchor_heads.env.bias` (ID 257). Su un solo
   elemento la somma dei quattro rank può cadere per caso vicino a zero, e allora il solo ordine della somma supera
   1e-6. Tre scettici su quattro stimano con una simulazione un rischio dell'1-2,5% di un `HYPOTHESIS_UNSUPPORTED`
   dovuto al solo arrotondamento; il quarto giudica lo scenario non plausibile. Per i tensori con più elementi il
   rischio è trascurabile: massimo simulato 6,4e-7, tipico 5e-8.
   Proposta: aggiungere al denominatore la scala degli addendi, per esempio la norma di `Σ_r |g_r| / 4`, già
   calcolabile dai vettori locali. In alternativa, riportare in forma descrittiva un eccesso sui tensori di un elemento.
   La scelta spetta a GPT e va registrata prima della misura.
2. **Regola per rieseguire il solo confronto.** I due difetti bloccanti stavano nel confronto su CPU, che legge soltanto
   file. Conviene registrare prima del lancio che un `INCONCLUSIVE` dovuto al solo confronto, con le due gambe complete,
   si chiude rieseguendo su CPU un confronto corretto e rivisto, sugli stessi dati grezzi, con criteri invariati e
   senza nuove GPU. Senza questa frase, correggere il lettore dopo la misura sarebbe contestabile.
3. **Test del confronto.** Un revisore ha applicato 84 mutazioni alle guardie dei quattro script: 54 sopravvivono ai
   test. Serve almeno un test di `compare()` su cartelle sintetiche, con il controllo del torch 2.2 simulato e un payload
   con `rank_rng` e `cuda_rng` realistici, un caso per esito. Servono anche i `match` nei rifiuti di `load_observation`.
   Un revisore ha già costruito in simulazione un confronto completo con gli artefatti storici veri: si può riusare.
4. **Timeout non registrato.** `checked_command` solleva `subprocess.TimeoutExpired`, che non è fra le eccezioni
   catturate in `contrastive_cn_restart.py:174`: allo scadere `execution_failure.json` non viene scritto. Basta
   aggiungerla alla tupla.
5. **Preventivo nel comando di invio.** Il comando registra `--gpu-ore 0.59`, e il protocollo dichiara la scelta:
   l'estremo superiore della stima storica. La regola del progetto è però GPU allocate per walltime massimo richiesto: 4
   GPU per 15 minuti fanno `--gpu-ore 1.0`.
6. **Controllo della norma.** `observe.py:163` ricalcola la norma con tolleranza relativa 1e-12. È ridondante rispetto
   all'impronta sha256 e un falso rifiuto renderebbe la prova inconclusiva: conviene toglierlo o allentarlo.
7. **Costanti del piano.** `relative_gradient_roundoff_limit`, `gradient_norm_floor`, numero di parametri, di rank e
   passi stanno nel piano ma il codice usa copie proprie. Basta un test che ne verifichi l'uguaglianza.
8. **Comando di `--validate-only`.** Il protocollo lo cita senza riportarlo per esteso.
9. **Raccolta.** Il protocollo chiede i tempi, ma i tempi per gamba non vengono registrati. Vanno raccolti anche il file
   `.err` del job e, prima dell'invio, lo spazio su WORK con `lfs quota`: servono circa 4 GB.

## Verifiche positive, per punto del prompt

**1. Le due celle eseguono la vera ripresa CN_A.**
- **Riga di comando.** Gli argomenti ricostruiti coincidono con quelli di `contrastive_job.build_command` per la ripresa
  storica. Il runner riconosce la fase `resume_a` e il fork `aa5952f0…` dalla release pinnata.
- **Cartella di uscita.** Il payload del checkpoint non dipende dalla cartella: CN_A e CN_B, in cartelle diverse, hanno
  lo stesso file.
- **Ordine delle chiamate.** La traccia di produzione registra batch, generatori e confine prima del riscaldamento e i
  generatori «dopo» alla fine del backward vero. L'hook è montato prima del primo backward del processo. Il primo
  clipping reale avviene dopo il ritorno dal backward del 545.
- **Riscaldamento a effetto nullo.** Con la vera funzione di accumulo del trainer, il runtime contrastivo vero e DDP
  Gloo a 2 processi, le varianti senza hook, ordinaria e riscaldata danno gradienti locali, studente, teacher,
  ottimizzatore, runtime e generatori identici bit per bit; cambia solo lo stato DDP. Con il trainer intero a 4 processi
  lo stato al confine coincide prima e dopo il riscaldamento. Una scansione del codice non trova attributi dei moduli
  modificati dal forward.

**2. L'hook osserva la riduzione vera.**
- **Prima e dopo.** Il buffer è copiato prima che l'hook di riferimento lo divida e lo riduca; il vettore ridotto è
  confrontato con il gradiente effettivo del parametro prima del clipping.
- **Copertura.** I 270 parametri sono coperti una volta per rank, compresi gli 8 in doppia precisione. Il modello vero ha
  114.202.292 byte in precisione singola e 68.480 in doppia, come i gruppi di costruzione storici.
- **Sette gruppi.** CN_A e CN_B, dopo la ricostruzione, registrano la stessa composizione a sette gruppi del continuo: il
  controllo sulla gamba riscaldata è soddisfacibile.
- **Hook e percorso nativo.** Su Gloo a 4 processi, con tipi misti e più update, la riduzione con l'hook è identica bit
  per bit a quella senza. Su NCCL non è verificabile qui: lo protegge il controllo di fedeltà della gamba ordinaria.
- **Torch 2.2.** Sul cluster ho letto le chiamate più a rischio: i metodi di `GradBucket`, il controllo e la
  registrazione dell'hook, l'hook di riferimento, gli hook dell'ottimizzatore. Esistono. Per le altre un revisore non
  trova incompatibilità con la 2.2, senza lettura sul cluster. Manca l'attributo `uuid` delle proprietà CUDA: gira
  quindi il ramo di ripiego con `libcuda`. Il comportamento effettivo dell'hook sulla 2.2 si vedrà solo in esecuzione.

**3. Le regole di lettura corrispondono ai criteri registrati.**
- **Classificazione.** Gradienti locali diversi, eccesso oltre la soglia o nessuno scarto danno `HYPOTHESIS_UNSUPPORTED`.
  Mancato recupero al 545 o a 576 dà `PARTIAL_SUPPORT`. Fedeltà dell'ordinaria, confine, gruppi, posizionamento e
  coerenza fra rank portano a `INCONCLUSIVE`.
- **Nessuna autorizzazione.** Il report ha sempre `training_approved` falso e `FAILED_REQUIRES_REVIEW`, anche quando è
  inconclusivo.
- **Sugli archivi veri.** Le funzioni del confronto passano sulle tracce, sulle metriche e sui profili storici: una
  ripresa soddisfa le uguaglianze di traccia contro il continuo; CN_A e CN_B hanno gli otto campi di gradiente identici
  in tutti i 32 update.
- **Payload vero.** Non contiene foglie numpy. La configurazione nel checkpoint coincide con `config.json`.

**4. Pacchetto, binding, pin e risorse.**
- **Binding.** Simulato in un clone privato: accetta solo il blocco unico ed esatto. Rifiuta blocco doppio, impronta
  sbagliata, commit abbreviato o non antenato, file modificati dopo la review e il vecchio blocco.
- **Pin storici.** Fork, riferimento ordinario, continuo e job CN_A coincidono con gli artefatti. I 31 file archiviati
  che il confronto rilegge sono identici alle copie remote.
- **Tempo.** Stima dai dati storici: da 450 a 620 s contro circa 830 s utili. Il margine è adeguato, non ampio.
- **Costi.** I conti sono esatti: 1 GPU-ora e 8,2667 ore locali al massimo; inviluppo 1,4 e 11,8667; residuo 12,26
  GPU-ore e 114,0011 ore locali sul tetto firmato.

**5. Test.** La ricevuta si riproduce: 170 test passati, 32 nei quattro file nuovi. Sono stabili, ma deboli (punto 3
sopra).

## Prova della catena intera su CPU

Un revisore ha eseguito l'intera sequenza sul Mac, con 4 processi Gloo, il trainer vero su un modello piccolo (106
parametri, gruppi DDP rimpiccioliti per averne più d'uno) e le funzioni vere del diagnostico. Ha chiamato le funzioni
del confronto una per una, classificazione compresa: `compare()` intero richiede i checkpoint del cluster.

| Passo | Esito |
|---|---|
| Continuo con hook di riferimento contro continuo senza hook | identici bit per bit |
| Ripresa ordinaria contro continuo | diversi dal primo update dopo il fork |
| Gradienti locali, ordinaria contro riscaldata | identici |
| Scarto ridotto relativo massimo per tensore | 9,5e-8, sotto la soglia di 1e-6 |
| Ripresa riscaldata contro continuo | identici bit per bit: gradienti, checkpoint finale, metriche |
| Gruppi osservati nella riscaldata | quelli ricostruiti, uguali al log DDP del continuo |
| Classificazione | `SUPPORTED` |

Su CPU l'intervento ricostruisce quindi il continuo esattamente. È un'indicazione a favore dell'ipotesi e della
fattibilità del riscaldamento, non una misura: modello piccolo, Gloo al posto di NCCL, nessuna aritmetica bf16.

## Note, senza richiesta di modifica

- **Identificazione della GPU.** Il ramo `ctypes` non è mai stato eseguito su CUDA e gira al 545, subito prima del
  backward misurato. Le firme sono corrette e il driver è già inizializzato: gli scettici lo giudicano un rischio minore.
  Non conviene anticiparlo, perché cambierebbe l'ordine di inizializzazione rispetto alla ripresa storica.
- **`SUPPORTED` dice che il riscaldamento basta.** Non dimostra che la causa sia la sola composizione dei gruppi. Il
  protocollo lo dichiara già.
- **Zeri con segno.** Una differenza dei soli zeri con segno nel ridotto conta come «nessuno scarto». È irrilevante in
  pratica.
- **Binding sul cluster.** Lì `require_binding` verifica la coerenza interna del binding; l'autenticità rispetto al mio
  ramo è verificata al momento del `--bind-review`. È il limite già noto del meccanismo precedente.
- **Ordine delle gambe.** Con un timeout si perde la gamba riscaldata, cioè l'intervento.

## Letture su Leonardo, in sola lettura

Nessun file scritto, nessun job. Il confronto dei checkpoint ha richiesto 24 s; le altre letture sono brevi e non
cronometrate.

- **Torch del cluster.** `serialization.py`, `default_hooks.py`, `distributed.py`, `optimizer.py`; attributi di
  `GradBucket` e delle proprietà CUDA; Python 3.11.6.
- **Checkpoint storici.** Continuo, CN_A e CN_B aperti con `mmap`. Ho eseguito la funzione `checkpoint_differences`
  copiata senza modifiche dalle righe 35-72 del commit candidato. Tipi delle foglie: solo tensori, numeri, stringhe,
  booleani e `None`.
- **Requisiti del confronto.** Configurazione del payload uguale a `config.json`; `summary.json` con 576 passi e
  `max_steps`; `start.json` con le impronte attese; fork presente; cartella di uscita assente.

## Limiti

- **Nessuna GPU.** NCCL, bf16 su CUDA e il ramo `libcuda` non sono verificabili sul Mac.
- **Sette gruppi sulla GPU.** Che il riscaldamento sul cluster riproduca proprio i sette gruppi del continuo è
  plausibile ma non verificabile qui. Se non li riproduce, l'esito è `INCONCLUSIVE`, non un'attribuzione sbagliata.
- **Tempi.** La stima viene dai job storici: l'overhead reale della strumentazione non è misurato.
- **Soglia e scalari.** Il rischio dell'1-2,5% è una stima per simulazione, non una misura.
- **Prova su CPU.** Modello piccolo, Gloo, senza bf16: non anticipa l'esito sul cluster.
- **Prove.** Rapporti e script dei revisori, uscite delle letture su Leonardo e sintesi del workflow sono nella cartella
  `claude_review_contrastive_cn_restart_probe_20261121_evidence/`.

## Blocco di review

Non emesso. Lo aggiungo, in un'unica copia, quando il commit corretto supera la rilettura del diff.

## Prompt per GPT

```text
Review Claude della prova mirata InfoNCE (diff 918245b6..9d34dc69): docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md sul ramo origin/worktree-claude-training-engine, prove nella cartella _evidence/ accanto. Diagnostico NON approvato in questo stato: blocco di review non emesso, release chiusa. Disegno approvato, gambe e osservatore verificati con il codice vero. Nessun lancio da parte mia.

Due difetti bloccanti, entrambi nel solo confronto CPU: con il codice attuale il report uscirebbe INCONCLUSIVE in ogni caso.
- B-1: contrastive_cn_restart_observe.py:144 chiama torch.load(tensor_path, ..., mmap=True) con un pathlib.Path. Il torch 2.2.0a0 del cluster accetta solo str (serialization.py:1012-1014, ValueError; stesso errore del job 64706211). Correzione: str(tensor_path).
- B-2: contrastive_cn_restart_compare.py:248. Il payload salva get_rng_state_all() del rank 0 (run.py:1482); un run da zero risemina per rank (run.py:1463), una ripresa semina con cfg.seed (run.py:1383) e ripristina solo la propria GPU (run.py:1446-1450). Quindi cuda_rng.1, .2 e .3 di ogni ripresa differiscono dal continuo per costruzione. Ho eseguito checkpoint_differences del commit candidato sui checkpoint veri, in sola lettura: continuo contro CN_A dà metadata_or_schema_different = [cuda_rng.1, cuda_rng.2, cuda_rng.3]; CN_A contro CN_B è esatto. Correzione proposta: nel solo confronto riscaldata/continuo escludere quei tre percorsi dall'identità scientifica, riportarli a parte, richiedere che coincidano con la gamba ordinaria e continuare a confrontare cuda_rng.0, torch_rng e rank_rng. Regola da scrivere in decisioni.md e nel protocollo prima della misura.

Nello stesso commit, senza un altro giro:
1. soglia 1e-6 e i due parametri scalari (ID 253 anchor_heads.rms.bias, ID 257 anchor_heads.env.bias): rischio stimato dell'1-2,5% di un HYPOTHESIS_UNSUPPORTED da solo arrotondamento; proposta: scala degli addendi al denominatore, oppure eccesso descrittivo per i tensori di un elemento; scelta tua, da registrare prima della misura;
2. registrare prima del lancio che un INCONCLUSIVE dovuto al solo confronto, a gambe complete, si chiude rieseguendo su CPU un confronto corretto e rivisto sugli stessi dati, a criteri invariati;
3. un test di compare() su cartelle sintetiche con il controllo del torch 2.2 simulato e payload con rank_rng e cuda_rng realistici, un caso per esito; match nei rifiuti di load_observation (oggi 54 mutazioni su 84 sopravvivono);
4. subprocess.TimeoutExpired nella tupla dell'except di contrastive_cn_restart.py:174;
5. --gpu-ore 1.0 nel comando di invio (4 GPU per 15 minuti);
6. togliere o allentare il controllo della norma a 1e-12 in observe.py:163;
7. un test che leghi le costanti del piano a quelle del codice;
8. il comando esatto di --validate-only nel protocollo;
9. tempi per gamba registrati, raccolta anche del file .err e controllo dello spazio su WORK con lfs quota (circa 4 GB).
Nella regola per B-2 scrivi anche che ogni altra differenza non in virgola mobile a 576 resta causa di INCONCLUSIVE.

Verificato e a posto: riga di comando della vera ripresa CN_A, riscaldamento a effetto nullo con la vera funzione di accumulo del trainer, hook identico al percorso nativo su Gloo, copertura dei 270 parametri, sette gruppi soddisfacibili, regole di lettura conformi ai criteri registrati, binding e pin, costi (1 GPU-ora, 8,2667 ore locali), tempo stimato 450-620 s su circa 830 utili. Su CPU con 4 processi Gloo e il trainer vero su un modello piccolo la ripresa riscaldata ricostruisce il continuo bit per bit e la classificazione dà SUPPORTED: è un'indicazione, non una misura (Gloo, senza bf16, funzioni del confronto chiamate una per una).

Pubblica il commit corretto con le nuove impronte di pacchetto e piano: rileggo solo il diff ed emetto il blocco sul commit nuovo.
```
