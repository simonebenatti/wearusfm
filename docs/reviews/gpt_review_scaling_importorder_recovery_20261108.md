# Scaling 300M: review della correzione e piano di recupero

08/11/2026. Diagnosi Claude `913b4ae4bdfd8de709641ef39383a9fa7e0020f0`,
correzione `0c5b974af2483faf767a2e171183ae1af0dcdb19`, registro Claude
`c981c97a1f80e28d4264b66ab5035a35fffab1b0`.

**APPROVATO il fix tecnico e l'impostazione del recupero a 19.618. Il commit
0c5b974 non è ancora un pacchetto operativo rilasciabile: conserva il piano
del tentativo fallito.** Occorrono piano e pin nuovi e il collegamento dello
stadio B al tentativo corretto. Non occorre un'altra campagna diagnostica GPU.
GPT non esegue deploy, invii o operazioni sul cluster; implementazione ed
esecuzione restano a Claude. L'estensione a 49.046 è una proposta distinta.

## 1. Esito tecnico e responsabilità della review

Il job 63696864 si è arrestato nella preparazione dell'insieme fisso, prima
del DataLoader di training e del primo update 5.001. Non è una divergenza
del 300M e non modifica la lettura scientifica accettata a 5.000. Il recupero
deve partire dal checkpoint completo originale a 5.000, riverificato, non
dalla directory del tentativo fallito.

La precedente review verificava sorgenti, copia degli stati e continuità
della ripresa, ma non la dipendenza numerica dall'ordine di caricamento delle
librerie nel runtime di Leonardo. Questa copertura mancava. Il cancello
esatto ha fermato correttamente il job prima degli aggiornamenti.

Riscontri indipendenti sui materiali pubblicati:

- **42 controlli di audit superati.** Il solo cambiamento semantico nel
  wrapper è `import torch` prima dell'import del trainer. `src/` e il CLI
  originale non cambiano.
- Torch-first riproduce **8/8 batch fissi e 64/64 pannelli** originali;
  NumPy-first riproduce il contenuto del tentativo fallito. Le finestre sono
  identiche. Il CLI originale ricostruisce byte per byte anche i due JSON
  d'identità, verificati contro i pin del run accettato:
  `af2f7cb1…` (fixed set), `7b21e1c0…` (pose panel).
- **40 test del launcher superati** in 45,62 s. Riprodotti i **7 errori di
  pin** in 3,32 s: tutti derivano dal piano v1 che richiede il vecchio SHA
  del wrapper. Non sono sette nuovi difetti numerici; non vanno ignorati o
  trasformati in test saltati nel pacchetto nuovo.
- Wrapper precedente: `0628ad6eb83e1f9c0be873925798f8cc9a320c1bcac98a001763f6345c7db863`.
  Corretto: `ed4b2ca473ea2057d7dc3e9b102a2e49eadf42d0c89ee30ef41e7c761e926edb`.

I numeri riassunti nel messaggio di Claude richiedono una rettifica non
bloccante. Nei dump pubblicati il massimo relativo dei segnali è
3,03×10⁻⁶, quello assoluto 4,77×10⁻⁷; un array ha nove elementi diversi.
Il massimo assoluto delle ancore è 3,94×10⁻⁷. Restano differenze piccole,
ma non sempre «1–2 campioni» o entro 1,6×10⁻⁷ relativo. Questi massimi
riguardano i batch scaricati, non l'intero training. Le 412 impronte dei file
raccolte ora costituiscono una baseline utile; da sole non provano
retrospettivamente l'identità SHA dell'intero corpus di 75.760 file posa.
Le riproduzioni delle identità storiche sono la prova forte per il caso.

## 2. Risposte a, b e c

**a — Correzione sufficiente e minima: sì, per il guasto dimostrato.**
Ripristina l'ingresso numerico della CLI originale senza abbassare il
cancello. Il test statico tutela l'ordine; la prova reale raccolta e i
controlli d'identità nel worker coprono il caso concreto. Conservare
l'uguaglianza esatta dei batch. Non sostituirla con una tolleranza per
ottenere un PASS.

Per un giro futuro ha senso separare identità semantica (sorgenti, finestre,
ricetta) e impronta numerica, oppure conservare gli input della diagnostica.
Congelare soltanto quei batch, però, non garantirebbe che il resto del flusso
di training usi lo stesso backend. Non serve affrontare questo ridisegno
per recuperare il run attuale; non si modificano checkout storici.

**b — Ripresa esatta nel contratto scientifico: sì, con il limite già dichiarato.**
Si preservano checkpoint student/teacher, AdamW e contatori, RNG per rank,
posizione dei dati, configurazione e schedule originale. Il fix ripristina
il percorso numerico del run accettato, invece di introdurre una variante.
Non dimostra ex novo l'identità bit per bit di tutta la futura traiettoria
CUDA/DDP rispetto a un training mai interrotto. Il precedente limite della
prova CPU resta valido. Non assumiamo che piccole perturbazioni siano
sempre irrilevanti per SGD: le rimuoviamo ripristinando il percorso originale.

**c — Stadio B: non cambiare gli import per uniformarli al training.**
L'estrazione storica del controllo 30M a 19.618 usa già gli entry point
NumPy-first. I nove file pertinenti confrontati con il suo commit effettivo
`453ce6c496745f2d73aa3007406721586e6dd767` sono byte-identici. Il supervisore
`scaling_evaluation19618.py` avvia `pose_eval.py` e `channel_readout.py` in
processi Python nuovi: non ereditano gli import del worker di training.

I cancelli dello stadio B verificano target e supporti posa, finestre grezze,
etichette, utenti e fold; non impongono identità con gli hash dei segnali
filtrati e delle ancore del fixed set di training. Le feature 300M non devono
essere numericamente uguali alle feature 30M. I nuovi artefatti riceveranno
i propri pin. Non emerge quindi un nuovo blocco di valutazione causato dal
solo fix del trainer; i cancelli reali devono comunque passare.

Lasciare invariati estrattori, lettori, proiezioni e bootstrap del dispatcher
fra benchmark, fit, aggregazione e confronto. Cambiare solo la valutazione
300M introdurrebbe invece una differenza evitabile rispetto al controllo.
Anche il preflight confrontava le due taglie nello stesso contesto; il suo
esito tecnico non va reinterpretato come prova bitwise della CLI originale.

**Limite dei fingerprint runtime:** i rapporti reali dei due ordini producono
lo stesso inventario ordinato di threadpool, pur con risultati diversi.
`numeric_sha256` e versioni uguali non certificano il binding BLAS effettivo.
La lista non ordinata della posa può rilevare indirettamente un ordine
diverso. Non è corretto nemmeno dire che ogni operazione dei lettori sia
OpenBLAS solo perché NumPy viene prima: esistono import transitivi e operazioni
Torch. Un eventuale canary numerico nel processo effettivo è un miglioramento
futuro, non una condizione aggiuntiva di questo recupero.

## 3. Consegna concreta a Claude: un pacchetto di recupero

Preparare sul ramo di esecuzione un unico emendamento operativo circoscritto:

1. **Piano di recupero immutabile v2**, con lo SHA del wrapper corretto,
   nuovi ID logici di checkout/training, nuovo ref di deploy, directory del
   run e delle ricevute nuove. Preservare piano v1, checkout e run fallito.
   Nessun recupero da un checkpoint parziale. Riverificare i pin originali
   del run a 5.000: il tentativo fallito non ha raggiunto il controllo finale
   dopo il training. Conservare la copia verificata dei quattro file-seme.
2. **Riprendere 5.000→19.618 senza cambiare ricetta**: batch globale 32,
   quattro GPU, schedule WSD originale fino a 49.046, stesso LR/clip/seme,
   stessa dose e guardie su memoria/contatori/identità. Conservare il limite
   SLURM 14:45 e il timeout operativo prescritto. Lo stop al primo errore
   resta; l'emendamento non abilita retry automatici.
3. **Collegare la raccolta e lo stadio B al nuovo tentativo.** Nel codice
   attuale `training_job()` in `scaling_plans19618.py` usa ancora l'ID logico
   `SCALING-300M-train19618-20261106`; `stage_table('training')`,
   `build_extraction_plan()` e `build_reader_plans()` leggono il vecchio
   `training_plan.json`. Anche checkout e percorso delle prove vanno legati
   coerentemente al nuovo piano. Non basta aggiornare lo SHA del wrapper:
   altrimenti si continuerebbe a raccogliere il job fallito. Fare la modifica
   operativa minima, con un'unica sorgente dell'identità del recupero.
   In particolare `checkout_job('training')` fissa ID e directory v1;
   `evidence/checkout-training` esiste già e `collect()` rifiuta directory
   esistenti. Dare anche alle prove del nuovo checkout una destinazione
   distinta, conservando quelle vecchie. Non allargare l'allowlist di
   `release_check` per permettere modifiche runtime dopo la review.
4. **Conservare lo stadio B scientifico già approvato.** Nessuna nuova
   derivazione numerica, nessuna riestrazione del controllo, nessun cambio
   di S8/lettori/bootstrap. Aggiornare soltanto i collegamenti operativi che
   la nuova consegna rende necessari. Il commit del pacchetto rivisto sarà
   la nuova base di `release_check`; i checkout successivi possono ancora
   aggiungere solo documenti ed evidenze come già previsto.
5. **Chiudere i test del pacchetto**, compresi i sette che oggi si fermano
   sul pin v1, le prove di ripresa e l'incastro piano nuovo → ricevuta →
   estrazione. Coprire anche il rifiuto delle prove/ID del tentativo fallito.
   Non modificare pin storici né ridurre le asserzioni per rendere verde la
   suite. La diagnosi cluster già raccolta evita nuove prove GPU della causa.
   Rendere verificabile anche il legame fra il piano v2 selezionato e i suoi
   hash nella ricevuta (`plan_sha256`, `execution_source_sha256`), oltre ai
   percorsi. Pubblicare righe d'invio nuove: generatore e righe v1 contengono
   ancora i vecchi riferimenti. Il launcher accetta già piano e output come
   argomenti e non richiede un cambiamento per questa sola ragione.
6. **Pubblicare consuntivo e inviluppo emendato**, separando il tentativo
   fallito, i quattro job diagnostici, checkout di recupero, nuovo training
   e valutazione. Allegare `sacct -P` originale e accounting: nei materiali
   diagnostici qui rivisti i costi sono riproducibili dalle durate dichiarate,
   ma manca l'attestazione scheduler indipendente dei cinque job.

La verifica di questo emendamento coerente conclude il rilascio del recupero;
non introduce review fra checkout, training e singoli invii di valutazione.
I documenti futuri continuano a essere prodotti dai costruttori approvati e
a superare i cancelli meccanici. Non è richiesto un nuovo giro di diagnosi
o una revisione generale dell'infrastruttura.

### Conto del recupero a 19.618

Dalle durate pubblicate: fallimento 0,227778 GPUh / 1,822222 coreh;
diagnosi 0,153333 GPUh / 1,844444 coreh. Totale **0,381111 GPUh e 3,666667
coreh**, delle quali 0,617778 coreh in job senza GPU. Non sono persi i 5.000
passi validi.

Un tentativo sostitutivo completo conserva il tetto di **59 GPUh**, più
0,2 di tolleranza scheduler; estrazioni 7+0,15. Sommando il costo già speso
per guasto/diagnosi si ottengono **66,731111 GPUh** per recupero e lettura,
non 66,35. Non si cancella il tentativo fallito dal conto e non si interpreta
il vecchio «nessun retry» come autorizzazione automatica al nuovo tentativo.

CPU: il massimo già pubblicato dei lettori è 345,166667/350. Se si attribuiscono
qui i 0,617778 coreh diagnostici senza GPU e un checkout aggiuntivo da 2 CPU,
5 min + 3 min di margine (0,266667), il conto è **346,051111/350**.
È un conto di fattibilità, non un piano eseguito: il sottotetto checkout
0,8 richiede un emendamento esplicito se si aggiunge quel job. Claude deve
pubblicare il totale effettivo con i checkout realmente richiesti; i core
dei job GPU restano separati. Il vecchio freeze e i suoi pin restano intatti.

## 4. Parere scientifico su 49.046: favorevole, conservando 19.618

Non userei il guasto come ragione per allungare il training. Esiste però
una ragione scientifica indipendente: la valutazione lunga del 30M mostra
ulteriore apprendimento della posa, e il 300M a 19.618 è ancora sul plateau
del medesimo schedule. Il decay inizia a 39.237.

La review della linea 2 pubblicata in `57c4d7e`,
`docs/reviews/consolidation_long_results_20261107.md`, riporta sul contrasto
49.046−19.618:

- ridge post-backbone: E_dyn 0,839→0,786, riduzione **6,32% [5,87; 6,83]**;
- testa posa allenata: 0,833→0,779, riduzione **6,52% [5,92; 7,15]**;
- EPN nel pool storico: **+1,93 pp [1,42; 2,44]**. I due indici primari
  aggregati P1/P2 non dimostrano invece un beneficio congiunto del lungo.

Questi risultati motivano la prova, non ne predicono l'esito sul 300M.
Semine uniche e popolazioni di sviluppo restano limiti; il progresso sul
tratto finale combina nuovi update e decay, senza isolarne le cause.

Proposta: **5.000→19.618, poi 19.618→39.237→49.046**, mantenendo la
valutazione già prevista a 19.618. A 49.046 si confronteranno le due taglie
alla stessa dose e alla fine dello stesso schedule; non si sceglierà il
checkpoint che appare migliore dopo la lettura. Il controllo 30M lungo
esiste: per riusarne le cache servono la verifica di viste, supporti e
ricetta del piano nuovo, non basta l'esistenza del checkpoint.

39.237 è qui un confine tecnico sensato per la ripresa, non una richiesta
automatica di una terza campagna di sonde. L'eventuale estensione può essere
decisa prima dei risultati a 19.618; se approvata, il training successivo
può procedere mentre si eseguono i lettori CPU del checkpoint intermedio.
Non occorre trasformare il risultato a 19.618 in un nuovo gate scientifico
se non è questa la decisione concordata.

Preventivo di training dalla formula congelata, 2,933010711 s/update,
20% di margine, 1.200 s di overhead per segmento, arrotondamento a 900 s:

| Segmento | GPUh attese con overhead | Tetto GPUh | Walltime richiesto |
|---|---:|---:|---:|
| 5.000→19.618, recupero | 48,97 | 59 | 14 h 45 |
| 19.618→39.237, proposta | 65,27 | 79 | 19 h 45 |
| 39.237→49.046, proposta | 33,30 | 40 | 10 h |
| Totale futuro da 5.000 | **147,54** | **178** | **44 h 30 complessive** |

Solo l'estensione oltre 19.618 costa circa **98,57 GPUh attese** e un tetto
**119 GPUh**, più 0,4 di tolleranza sui due job. Il totale dei tre segmenti
ha 0,6 GPUh di tolleranza e 1.424 coreh nominali allocate ai job GPU
(1.428,8 con la stessa tolleranza). I tempi di coda
non sono inclusi. La formula per un unico job 19k→49k darebbe 117 GPUh,
29 h 15: oltre il limite di 24 h. Il numero «116» non è il tetto risultante
dalla formula per i due segmenti.

**Questo non è il budget completo di una campagna a 49.046**: mancano nuove
estrazioni, lettori e checkout. Non sono compresi i costi già sostenuti.
La dose 49.046 e il relativo pacchetto restano alla decisione di Simone;
nessuna autorizzazione viene dedotta dalla correzione del guasto.

## 5. Evidenze e limiti

Artefatti della review:
`results/infra/scaling_300m_20261104/importorder_review_20261108/`:
audit riproducibile e JSON, output dei test, mappa dei cancelli dello stadio B,
prove sui runtime e calcolo dei costi. README con comandi di riproduzione.

La review legge solo Git, codice e materiali locali raccolti da Claude.
Non sono stati riletti gli array diagnostici remoti, il checkpoint o sacct
live; non sono stati effettuati nuovi forward, training o lanci. Le prove
confermano il caso riprodotto e il piano di recupero, non una garanzia su
ogni possibile runtime o futura traiettoria numerica.
