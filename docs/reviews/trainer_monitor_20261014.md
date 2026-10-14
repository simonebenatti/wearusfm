# Review punti 3-4: trainer e monitor di stabilita'

Review GPT del 14/10/2026. Base integrata: main5030a19; codice esaminato: trainer16f5a5d, IPC49509a2 e follow-up frozen/rami,
monitor5f62444, invariati fino a Claude366ab11. Solo lettura del worktree di Claude: le correzioni richieste restano sul suo branch.
Non e' un'approvazione dell'intero branch, un merge o una prova NCCL/bf16.

## Rilievi riprodotti

### P2 — La ripresa accetta un seed diverso e cambia la ricetta dello stream

`src/wearusfm/training/run.py:745-769`, stream costruito a `:887-892`.
Le guardie verificano schedule, worker/segmento/world_size e identita' parziale Step1, ma non il seed della ricetta dei dati.
Riproduzione sul trainer reale: small_config senza keep, un update seed0; stessa cartella e checkpoint, max_steps2 ma seed1:
la ripresa viene accettata e arriva al passo2. BatchStream riceve il nuovo cfg.seed: non e' la continuazione della sequenza originale.
Una CLI di ripresa con seed dimenticato/default diverso puo' quindi alterare silenziosamente una campagna.

Richiesta a Claude: validare la ricetta di ripresa dal checkpoint prima di creare stream/modelli o scrivere output; almeno seed,
batch globale/accumulo e campi loader/dataset che modificano i dati. Consentire esplicitamente solo variazioni operative previste,
come estensione max_steps e limite di tempo; branch_from resta il percorso dichiarato per cambiare ricetta/schedule.
Test richiesti: seed/batch/maschere diversi rifiutati, estensione max_steps ammessa, replay con configurazione invariata ancora esatto.
La riproduzione conferma seed; non pretende di aver testato separatamente tutte le altre variazioni.

Questo non invalida i test o i run fatti con la stessa configurazione. Nessun difetto nuovo dimostrato della formula di riduzione DDP.
Il problema e' preesistente in5030a19, non una regressione introdotta da16f5a5d. La guardia serve prima di congelare le catene lunghe
della finestra1; non si dichiara che riprese con configurazione invariata siano difettose.

### P2 — Una chiave del forward mancante sfugge alla guardia coordinata

`src/wearusfm/training/run.py:499-503`.
backward_loss(global_den) viene chiamata prima del controllo di chiavi/conteggi del forward. Con fault injection che rimuove un
componente dal forward ma non dal precount, solleva ValueError("inconsistent component keys") prima di raggiungere i flag comuni.
In DDP un peer puo' essere nella collective; torchrun puo' terminarlo, ma non e' lo stop coordinato dichiarato dal contratto.

Richiesta: validare struttura/chiavi/conteggi e sincronizzare il verdetto prima di costruire la loss/backward. Test a2processi con
componente mancante su un solo rank e checkpoint preservato. L'eccezione e' riprodotta sul helper reale; nessun hang permanente
dimostrato o errore osservato nei run ordinari, i cui componenti rispettano il contratto.

### P2 — L'identita' del pannello fisso non protegge i valori normalizzati

`src/wearusfm/training/run.py:600-609` e `:875-878`.
fixed_eval_set.json contiene solo seed e identificativi sessione/trial/start/stop. Nel trainer reale, con cfg invariato, si puo' riprendere
passando scale di sessione diverse: le finestre hanno gli stessi identificativi e il file viene accettato, ma i segnali nel pannello
cambiano. Una variazione di loss non e' piu' attribuibile al solo modello.

Richiesta: proteggere il contenuto del pannello (segnali normalizzati, fs, QC, maschere, target) o una provenienza equivalente completa,
incluse scale/manifest/dati/preprocessing, e rifiutare drift alla ripresa. La dimostrazione usa scale esterne, non un cambio di loader
che una guardia di calibrazione Step1 potrebbe gia' rifiutare. Il caso plain JEPA e' coperto, senza keep.

### P2 — Le diagnostiche opzionali trattano l'assenza di uno strato come loss zero

`src/wearusfm/training/run.py:634-641`; chiavi condizionali reali in `training/jepa.py:293-300`.
Il codice somma le chiavi presenti e divide tutte per il numero totale di batch. Se jepa_topo_grid_2d vale2 solo nel primo di due batch,
riporta1, anche se il secondo non contiene quel tipo di query. Tipo/topologia rari risultano artificialmente piu' facili.

Richiesta: usare numeratori/conteggi validi per componente/strato (il contratto LossTerms li fornisce), separare contatori dalle medie
e non interpretare un insieme assente come osservazione zero. Aggiungere test con strato presente in un solo batch.

## Verifica e limiti

- Baseline corretta:38testCPU pertinentiPASS, separati dalla review del trainer.
- Monitor fornito:5testPASS,1deselected (caso con worker) nel controllo root; suite914PASS riportata da Claude non rieseguita integralmente.
- Repro reali seed e scale: PASS, rispettivamente3,66s e3,98s; PASS qui significa che il comportamento scorretto e' riprodotto.
- Review indipendente trainer: nessun errore trovato di scala globale/clipping/EMA/freeze; provaCPUgloo testa opzionale usata solo
  nel no_sync del rank0 e assente nell'ultimo micro di entrambi: gradienti esatti su entrambi per2update (test PASS in3,31s).
  Selezione regressioni dell'agente:4PASS prima del blocco nel test con2worker; interrotta, non suite completa.
- Repro opzionali e codice persistente: `docs/reviews/repro_training_monitor_20261014.py`. Nessun dato reale o codice di Claude modificato.
- Errori/valori non finiti del pannello solo su rank0 e ripresa DDP con diagnostiche richiedono test distribuiti dedicati; non si
  afferma un hang permanente senza una riproduzione (torchrun puo' arrestare gli altri processi).

Comando per le quattro riproduzioni (nel worktree Claude allo snapshot indicato):

```sh
PYTHONPATH=src:tests/cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-evaluation/docs/reviews/repro_training_monitor_20261014.py'
```

## Bozza v2: requisiti prima della firma, non soglie adottate

Letta `docs/regola_stabilita_bozza.md` in366ab11. Pannello fisso, allarme persistente e distinzione fra training/generalizzazione sono
la direzione corretta; il collaudo2C storico resta non passato. Non si adottano qui delta0,10/k3 o altri numeri.

- Definire la chiave dell'obiettivo: il percorso frozen oggi scrive keep_branches, non total; distinguere loss pesata e non pesata.
- Il minimo storico mantiene un riferimento selezionato: spiegare/collaudare la scelta senza alzare o abbassare soglie sui run passati;
  se si calibra, separare la calibrazione dal collaudo che decide e congelare regola/cadenza prima di quest'ultimo.
- Nel JEPA con teacher EMA anche il target evolve: una salita sullo stesso ingresso riguarda il sistema student/teacher,
  non dimostra da sola regressione dello studente contro un target immutabile. Nei rami frozen i target spettrali sono invece fissi.
- Prima di usare il pannello per un verdetto: correggere i due P2, prova di ripetibilita'/non-interferenza GPU/DDP e gestione coordinata
  dei non finiti/errori, distinguendo avviso nei run lunghi e arresto nel collaudo. Sonde P1/P2 restano gli endpoint downstream.
