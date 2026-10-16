# Contratto delle loss distribuite (DDP e accumulo)

*Revisione 2 (09/10/2026, sulla review di GPT):* la ripresa continua la stessa sequenza di batch invece di aprire uno stream nuovo (punto 4).

**Perche'.** Decisione del 09/10/2026 (`docs/decisioni.md`, «Lavoro in parallelo Claude / GPT»): prima di scrivere DDP si concorda come le
loss espongono numeratori e denominatori, cosi' che 1 GPU e N GPU (con o senza accumulo) diano **lo stesso aggiornamento** sullo stesso batch
globale. Proprietari: le riduzioni in `training/jepa.py` e `model/anchors.py` sono di GPT; il ciclo in `training/run.py` e' di Claude. Questo
foglio fissa l'interfaccia fra i due; il codice si scrive dopo l'accordo.

## 1. Il problema, sul codice di oggi

Ogni componente c (JEPA, `log_rms`, `log_env`, `band_shape`, `ms_fast`, `ms_slow`, `rvq`, `keep`) e' una media con un proprio denominatore:

| Componente | `loss_per_window=True` (finestra 1) | `loss_per_window=False` (sanity) |
|---|---|---|
| `jepa` | media sulle finestre con >= 1 query nascosta della media per finestra (`masked_mean` con `sample_index`) | `F.mse_loss(h, tgt)`: media su query x dimensione |
| ancore | come sopra, con la validita' propria di ciascuna ancora (zona di bordo, bande disponibili) | `masked_mean` senza `sample_index`: somma degli errori validi / numero di elementi validi |
| `rvq` | come le ancore, sulle finestre RVQ | idem |
| `keep` | media sui campioni con almeno una famiglia valida (`SpectralKeepHead.loss`) | uguale (non dipende dal flag) |

La perdita del batch globale e' `L = sum_c w_c * L_c` con `L_c = N_c / D_c` (numeratore e denominatore **su tutto il batch globale**),
`w_jepa = jepa_weight` (default 1; 0 nel braccio ancore-only, dal 16/10/2026), `w_ancora = anchor_weight` (ogni ancora e `rvq`; 0 nel
JEPA puro), `w_keep = keep_weight`. Con DDP ogni GPU calcola `N_c^r / D_c^r` sul suo pezzo
e DDP **media i gradienti**: si ottiene `(1/R) sum_r N_c^r / D_c^r`, che e' diverso da `sum_r N_c^r / sum_r D_c^r` appena i `D_c^r` differiscono
(e differiscono: zona di bordo, finestre corte senza scala lenta valida, bande oltre il limite del canale). Cambierebbe di nascosto il
bilanciamento JEPA / ancore.

## 2. Interfaccia proposta

**In `training/jepa.py` (GPT):** una funzione che restituisce i termini invece delle medie, per esempio

    jepa_loss_terms(student, teacher, inp, visible, cfg, ...) -> LossTerms
    LossTerms.num:     dict[str, Tensor]  # componente -> numeratore locale, CON gradiente
    LossTerms.den:     dict[str, Tensor]  # componente -> denominatore locale, senza gradiente (float64 o int64)
    LossTerms.weight:  dict[str, float]   # componente -> peso nella perdita totale (jepa_weight, anchor_weight, keep_weight)
    LossTerms.log_num / log_den           # diagnostiche solo per il log (jepa_<maschera>, jepa_topo_<topologia>, rvq_acc...), senza gradiente

con, per ogni componente:
- **per finestra:** `num` = somma, sulle finestre locali valide, della media per finestra; `den` = numero di finestre locali valide;
- **per token** (`loss_per_window=False`): `num` = somma degli errori validi (per `jepa`: somma dei quadrati su query x dimensione);
  `den` = numero di elementi validi. Cosi' `num/den` riproduce esattamente il comportamento attuale;
- **keep:** `num` = somma delle perdite per campione sui campioni validi; `den` = numero di campioni validi;
- nessun elemento valido: `num` = 0 col gradiente (come oggi `x.sum() * 0`), `den` = 0.

`jepa_losses` resta come involucro: `losses[c] = num_c / max(den_c, 1)` e `total = sum_c w_c * losses[c]`, **identico a oggi** su un
processo solo (i test attuali restano validi senza modifiche). Il `ValueError` «nessun token nascosto» resta.

**In `training/run.py` (Claude):** per ogni aggiornamento, con R processi e k micro-batch per processo:

    D_c = all_reduce_sum( sum_{micro-batch} den_c )            # denominatore globale dell'aggiornamento, prima di ogni backward
    per ogni micro-batch:  loss = sum_c w_c * num_c * R / max(D_c, 1);  backward   (DDP divide per R: gradiente = quello globale)
    una volta: clip, optimizer.step, EMA; WSD indicizzato dagli aggiornamenti

I valori registrati in `metrics.jsonl` diventano medie globali: `all_reduce` di `num` (staccato) e `den`, poi il rapporto. Le diagnostiche
per tipo di maschera e topologia si riducono allo stesso modo.

## 3. Il punto da decidere: il denominatore con l'accumulo

Senza accumulo (k = 1; finestra 1 al 30M e al 100M: 4 GPU x 8 finestre) basta un `all_reduce` dei `den` **dentro il forward, prima del
backward**: esatto, costo trascurabile. Con l'accumulo (k > 1, probabilmente il 300M) il backward del primo micro-batch richiede gia' i
denominatori dei micro-batch successivi, che non sono ancora passati dal forward. Opzioni:

- **(a) conteggi dai dati, prima del forward** *(proposta)*: una funzione `loss_counts(inp, visible, anchor_targets, cfg) -> dict[str, int]`
  (GPT, accanto alle riduzioni) calcola i `den` dai soli dati e maschere, senza modello; il trainer li somma su micro-batch e processi prima
  del primo backward. A ogni passo il trainer **verifica** che il `den` restituito dal forward coincida con quello precalcolato, e si ferma
  se no: la duplicazione della logica e' sorvegliata a ogni passo. Serve che la validita' dei token nascosti sia ricostruibile dai dati
  (`patch_valid` dipende da lunghezza, fs e patch: da verificare insieme).
- (b) due passaggi: forward di tutti i micro-batch per i conteggi, poi forward+backward. Esatto ma costa un forward in piu' per micro-batch.
- (c) accumulo con media per micro-batch: **non esatto**, escluso dalla decisione del 09/10.

L'accumulo si implementa solo se la misura della memoria del 300M lo rende necessario; l'interfaccia (a) va comunque prevista ora.

## 4. Altri punti del contratto

- **Parametri inutilizzati:** la testa `keep` nel braccio A (peso 0), la testa RVQ spenta, eventuali teste senza target in un passo. Prima
  scelta: `find_unused_parameters=True`; si misura il costo e si valuta `static_graph` solo se conviene.
- **Teacher:** fuori dal modulo avvolto da DDP (nessun gradiente, nessuna sincronizzazione dei buffer); EMA identica su ogni processo
  perche' i pesi dello studente lo sono dopo `optimizer.step`. Una verifica dell'uguaglianza dei pesi fra processi ogni `eval_every`.
- **Diagnostica dei gradienti per componente** (`keep_grad_every`, `torch.autograd.grad`): non compatibile col normale funzionamento di DDP.
  Proposta: sotto DDP si calcola sul processo 0, sul modulo non avvolto, in un forward separato sul suo micro-batch e senza sincronizzazione;
  e' una misura descrittiva locale, e si registra come tale.
- **Ripresa: la stessa sequenza, non uno stream nuovo** (correzione del 09/10 sulla review di GPT; la versione precedente di questo punto
  proponeva uno stream nuovo alla ripresa, contro l'accordo). Requisito: a numero di processi e di worker invariati, dopo una ripresa i batch
  (sessioni, ritagli, montaggi virtuali, maschere) sono **identici** a quelli di un run continuo, per un'interruzione a qualunque passo.
  Oggi non e' cosi' neppure su una GPU: alla ripresa lo stream riparte da un seme nuovo, e ogni worker porta con se' uno stato (generatore,
  pool di blocchi gia' letti, sessioni scartate). Disegno proposto (dataloader, Claude):
  - il batch di indice b del processo r lo produce il worker w = b mod W; lo stato del worker dipende solo da (seme, r, w, segmento g = b div S),
    con S passi per segmento (multiplo di W, es. 1.200): a ogni inizio di segmento il worker riparte con `default_rng([seme, r, w, g])` e il
    pool vuoto;
  - alla ripresa al passo s, ogni worker ricostruisce il suo stato **rigiocando a secco** i batch del segmento corrente fino a s: le stesse
    estrazioni casuali (sessioni, tessere, inizi, montaggi, maschere) senza leggere i segnali ne' calcolare filtro e target. Costo: al piu'
    S/W batch per worker, da misurare (stima: secondi);
  - invariante, sorvegliato da un test: nessuna estrazione casuale dipende dai valori dei segnali letti (oggi vero: le scale hanno un seme
    proprio, i target non hanno casualita');
  - nel checkpoint si registrano W, il numero di processi e S; una ripresa con valori diversi si rifiuta (nessuna ripresa elastica);
  - il dropout del muscolo ha un generatore per processo, salvato nel checkpoint con gli altri stati casuali di ogni processo;
  - test d'accettazione: gli hash di tutti i batch (`training.pairing.batch_sha256`) di un run continuo coincidono con quelli di un run
    interrotto e ripreso, con interruzioni dentro un segmento e sul confine, a 1 e a 2 processi; stati del modello uguali entro le tolleranze
    del punto 5;
  - effetti: il pool si svuota a ogni confine di segmento (perdita di letture trascurabile con S/W ~ 200 batch per worker); a parita' di seme
    i batch dei run nuovi differiscono da quelli del codice attuale (stessa distribuzione): i confronti accoppiati si fanno sempre fra run
    dello stesso codice.
  - **Implementato** (09/10/2026, `pretraining_loader.batch_stream`, `RunConfig.stream_segment` = 1.200): lo stato e' per (segmento,
    classe c = b mod W) invece che per worker, perche' alla ripresa da `start` il worker w produce la classe (start + w) mod W, non w; i
    blocchi leggono i dati al primo uso, cosi' il pool si ricostruisce a secco senza letture; il trainer verifica a ogni passo che l'indice
    del batch sia il passo; il DataLoader ha un generatore proprio (creare l'iteratore estrae un seme, e dal generatore globale sposterebbe
    il dropout nei run ripresi). Prove: `tests/cpu/test_stream_resume.py`, con sessioni scartate, montaggi virtuali, scale in cache o no,
    1 e 3 worker, interruzioni dentro e sul bordo dei segmenti, e il training ripreso uguale bit per bit a quello continuo (batch e pesi).
- **Arresto coordinato:** loss o gradienti non finiti, allarme, limite di tempo e STOP decisi in modo che tutti i processi si fermino allo
  stesso aggiornamento (un `all_reduce` del flag); checkpoint scritto solo dal processo 0, dopo una barriera.

## 5. Test d'accettazione (tolleranze dichiarate prima)

Si fanno **sulla CPU del Mac** con DDP `gloo` (2 e 4 processi via `torch.multiprocessing`), in float64: costano 0 GPU-ora. Poi **una**
conferma su Leonardo con NCCL e bf16, su `boost_qos_dbg` (preventivo a parte).

1. Stesso batch globale (32 finestre): 1 processo senza accumulo contro 4 processi x 8, contro 2 processi x 8 x 2 accumuli. Confronto di
   loss totale e di ogni componente, gradienti dopo la riduzione, pesi dopo `optimizer.step`, teacher dopo l'EMA.
   Tolleranza float64: differenza relativa <= 1e-10. Su GPU in bf16: da dichiarare dopo una misura della variabilita' fra due run
   identici su 1 GPU (la tolleranza non puo' essere piu' stretta del rumore del bf16 stesso).
2. Maschere eterogenee fra processi; **un processo senza nessun target valido** per un'ancora; un processo con finestre corte senza scala
   lenta valida.
3. `loss_per_window=False`: stesso confronto (il comportamento del sanity deve restare riproducibile).
4. Parametri inutilizzati (braccio A della keep), arresto coordinato su NaN iniettato in un solo processo, ripresa da checkpoint con N
   processi: stessa sequenza di batch e stessi stati del run continuo (punto 4, «Ripresa»).
5. Il percorso a un processo senza DDP resta bit per bit quello di oggi (i test attuali passano invariati).

**Da concordare con GPT:** forma e nomi di `LossTerms`; se l'opzione (a) e' praticabile (validita' ricostruibile dai dati); chi scrive i
test della parte delle riduzioni (proposta: GPT i test di `jepa_loss_terms`/`loss_counts`, Claude quelli DDP del ciclo).

## 6. Consegna GPT del 12/10/2026: interfaccia implementata

La sezione precedente conserva la proposta originale. La struttura e l'opzione (a) sono accettate; ora sono implementate le API seguenti
in `training/jepa.py`, senza cambiare la firma di `jepa_losses` e senza modificare il trainer di Claude:

    jepa_loss_terms(student, teacher, inp, visible, cfg, anchor_targets=None,
                    rvq_on=None, rvq_codes=None, generator=None, kinds=None) -> LossTerms

    loss_counts(inp, visible, cfg, anchor_targets=None, rvq_on=None, rvq_codes=None,
                *, model_cfg, keep_target_active=None) -> dict[str, Tensor]

- `model_cfg` e' l'FMConfig del modello effettivo (es. student.cfg), obbligatorio: durata patch, dim e teste opzionali non sono in JEPAConfig.
- `keep_target_active` e' il buffer calibrato di student.spectral_keep, obbligatorio solo con keep_weight > 0. I target osservati da soli
  non bastano. `loss_counts` non fa encode/decode e non consuma RNG. Il callback RVQ deve essere un lookup puro dei codici congelati.
- `num`, `den`, `weight`, `log_num`, `log_den` come sopra; tutti i denominatori sono scalari int64 detached, anche in bf16. Le diagnostiche
  di maschera/topologia hanno chiavi complete quando `kinds` e' passato, incluse selezioni locali assenti con conteggio0.
- Aggiunta `log_sum`: contatori detached da SOMMARE (rvq_targets, target dell'altra selezione RVQ, keep_valid_samples), non medie di medie
  ne' medie fra rank. Accuracies sono in log_num/log_den con numero di codici validi; loss RVQ rispettano loss_per_window.
- `terms.backward_loss(global_den, world_size=R)` esegue solo `sum(weight * num * R / max(global_den,1))`; niente collective, divisione
  aggiuntiva per micro-batch, optimizer o EMA. Con world_size=1 e den locale restituisce il totale locale ricostruito.
- LossTerms e' registrato come pytree: i numeratori sono visibili sia alla ricerca dei parametri inutilizzati sia al sink degli output
  DDP, non un oggetto opaco. Roundtrip locale provato; resta da estendere il test gloo del trainer agli output nuovi.
- Il percorso `jepa_losses` conserva le medie storiche e l'errore su una maschera senza query. `jepa_loss_terms` ammette anche zero query
  locali con num differenziabile0 e den0: spetta al trainer controllare i conteggi GLOBALI e fare stop coordinato se JEPA non ha target.
- Le componenti sono quelle attivate dagli stessi cfg/head/target su tutti i rank. Il trainer deve validare chiavi e conteggi globali e
  confrontare quelli precalcolati con quelli REALI del forward a ogni passo; nessun collective su chiavi locali variabili.
- Keep conserva media equiponderata delle famiglie dentro ogni campione, poi somma delle loss dei campioni validi/count campioni: nessuna
  sostituzione con una media globale per famiglia. Le statistiche effettive sono prodotte dallo stesso forward della testa.

**Confini:** GPT ha implementato riduzioni e precounts; Claude collega forward DDP, all_reduce dei denominatori su rank/micro-batch,
no_sync per accumulo dove opportuno, diagnostiche globali e arresto coordinato. La guardia che rifiuta il training DDP vero resta accesa
finche' questa integrazione e i test distribuiti non sono completati. Test global-batch contro micro-batch sul modello CPU float64:
rtol e atol 1e-10 per loss/gradienti/pesi/EMA; keep/readout e diagnostiche che fanno cast float32 hanno tolleranze float32, non 1e-10.
NCCL/bf16, prestazioni e memoria restano da collaudare con protocollo e autorizzazione separati.

## 7. Integrazione nel trainer (Claude, 12/10/2026)

`training/run.py`, sulla consegna GPT `5030a19`. La guardia che rifiutava il DDP vero e' tolta, e con lei il ramo provvisorio della media
locale (`ddp_local_mean_for_tests`, `--ddp-local-mean-collaudo`): i test distribuiti sotto passano.

- **Percorsi.** Un processo senza accumulo: `jepa_losses`, il percorso di sempre (verificato bit per bit contro `run.py` precedente: pesi,
  teacher, RNG, metriche e ordine dei campi). DDP e/o `RunConfig.accum_steps` > 1 (`--accum-steps`): `_accumulate_terms`.
- **Micro-batch.** `pretraining_loader.split_batch`: il batch del processo in `accum_steps` parti di campioni contigui, canali reimpacchettati
  (vicini spostati, patch fino al massimo della parte). Lo stream non dipende dall'accumulo: le estrazioni sono quelle del batch intero.
- **Prima di ogni backward:** `loss_counts` (con `model_cfg` = student.cfg e, con la keep, il buffer `target_active` della testa) per tutti i
  micro-batch; stesse chiavi su micro-batch (altrimenti errore) e processi (hash delle chiavi, MAX e MIN in un all_reduce); conteggi
  sommati e ridotti (int64, all_reduce SUM): D_c. Se D_jepa = 0 nel batch globale: arresto coordinato.
- **Per micro-batch:** forward dentro il modulo DDP (`no_sync` su tutti tranne l'ultimo), `backward_loss(D, world_size=R)`; prima del
  backward un solo all_reduce MAX di: conteggi del forward diversi dai precalcolati, perdita non finita, chiavi delle diagnostiche diverse da
  quelle del primo micro-batch o fra processi. Qualunque flag: tutti i processi si fermano allo stesso aggiornamento, senza backward, senza
  sovrascrivere l'ultimo checkpoint, con STOP.
- **Un aggiornamento:** un clipping, un `optimizer.step`, una EMA, WSD indicizzato dagli aggiornamenti (invariato).
- **Metriche:** un all_reduce SUM (float64) dei numeratori staccati, `log_num`, `log_den`, `log_sum` sommati sui micro-batch; nel record:
  componente = num / max(D, 1), `total` = somma pesata, diagnostica = log_num / log_den solo se log_den > 0 (come `jepa_losses`, che omette
  le selezioni vuote), contatori `log_sum` sommati. Nessuna media di medie.
- **Test** (CPU, `gloo`, float64): `tests/cpu/test_ddp_equivalence.py` — lo stesso batch globale a 1 processo col percorso di sempre, a 1
  processo con 2 e 4 micro-batch, a 2 processi con 1 e 2 micro-batch, a 4 processi con una finestra ciascuno: loss per componente e
  totale, gradienti dopo la riduzione, pesi dopo clipping e AdamW, teacher dopo l'EMA entro 1e-10; processi uguali bit per bit; per token e
  per finestra, ancora singola e multi-scala con montaggi virtuali; un processo intero senza target RMS. Le diagnostiche per tipo di maschera
  e topologia (calcolate in float32 in `jepa.py`) entro 1e-5 relativo. Il test fallisce se si usano i denominatori locali (verificato).
  Inoltre arresti coordinati per conteggi sbagliati e JEPA vuota, split, accumulo su un processo attraverso `train()` con ripresa bit per
  bit; in `tests/cpu/test_training_ddp.py` il ciclo completo a 2 processi con e senza accumulo, ripresa, NaN e conteggi sbagliati su un solo
  processo.
- **Non provato:** NCCL e bf16 su GPU (protocollo a parte, tolleranza da fissare dopo il rumore del bf16 fra due run identici); RVQ e keep
  attraverso il trainer DDP (i termini e i conteggi li ha provati GPT; il trainer passa `rvq_codes` e `target_active`, ma nessun test
  distribuito li esercita: la keep richiede un file di calibrazione, l'ancora RVQ e' spenta nella finestra 1); `pairing_audit` e
  `keep_grad_every` restano rifiutati con DDP e con l'accumulo.
