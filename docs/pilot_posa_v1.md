# Seconda ondata del pilot degli obiettivi: R3, ancore + testa di posa su emg2pose, senza JEPA — `pilot_posa_v1`

**Stato: FIRMATA da Simone il 26/10/2026** («Firmo la regola di lettura del §5 di docs/pilot_posa_v1.md, le deroghe a v10 §6.1 e §6.5/§3.7, le soglie congelate e la scelta sulla mano non sollecitata; seconda ondata CON il controllo R3c. Via al codice e ai test; nessun job senza il mio via.») **: regola §5, deroghe (§2), soglie del §2 (iii)-(iv) e del §6 (punto 7), mano non sollecitata tenuta, CON R3c; codice e test in corso, nessun job senza il via. Storia della proposta:** passata tre volte dal `critico` lo stesso giorno: prima review DA RIVEDERE (8 bloccanti, 9 minori),
seconda DA RIVEDERE (3 bloccanti, 6 minori), terza DA RIVEDERE con un solo bloccante (la condizione (b) del §3, riscritta attorno a tre controlli
d'uguaglianza) e 5 minori, tutti ripresi in questa versione; il `critico` rivede protocollo e codice insieme prima di qualunque job (§6, punto 3).
In attesa della firma di Simone su: regola di lettura (§5), deroghe dichiarate a v10 §6.1 e §6.5/§3.7 (§2), soglie congelate del §2 (iii)-(iv) e
del §6 (punto 7), regola di validita' dei campioni di posa (§6, punto 5, dopo il collaudo dell'ingest). Nessun job, nemmeno di ingest, senza il
via di Simone** (Simone, 26/10/2026: «Sì, prepara la proposta della seconda ondata con R3 (ancore + testa di posa su emg2pose, senza JEPA) e il
preventivo; nessun job senza il mio via»). Origine: esito della prima ondata (`results/pilot_obiettivi_20261024/README.md`: nessun braccio
passa; R1 sotto R0, su P1 con l'intervallo intero sotto lo zero; R2 e R5 non si distinguono da R0 nel senso della regola, con R5 su P1 a
-0,53 [-1,07; -0,02], intervallo appena sotto lo zero) e sintesi del 24/10
(`docs/sintesi_pretraining_20261024.md`: ancore-only = A, il readout non e' la causa). R3 era gia' la seconda ondata prevista da
`docs/pilot_obiettivi_v1.md` §2 («R3 ancore + testa di posa su emg2pose senza JEPA»); R4 (R2 + posa) resta fuori da questo preventivo.

## 1. Domanda

Un bersaglio **esterno, alla scala del movimento** (gli angoli articolari della mano, 20 gradi di liberta', ricostruiti dal motion capture in
emg2pose), predetto a ogni istante dai latenti del backbone accanto alle ancore fisiche e **senza il termine JEPA**, produce un encoder che una
sonda lineare legge meglio di A (R0 del pilot) a pari esposizione? Misura: S della validazione interna (P1: NinaPro DB2/DB3/DB6; P2: EPN-612,
UCI-EMG) col readout medio `pool`, lo stesso di tutta la storia del progetto. Nessuno dei dataset della validazione interna e' emg2pose
(`docs/pilot_obiettivi_v1.md` §4; le cache di P1 e P2 vengono da `results/passo*` di NinaPro, EPN-612 e UCI-EMG): il bersaglio nuovo non puo'
entrare nella misura per via diretta.

**Perche' questa e' la leva con piu' evidenza** (fatti gia' nel repo, letti con le regole congelate): (i) ancore-only = A in tutte e quattro le
letture del readout per canale (24-25/10); (ii) JEPA-only collassa nei due semi (21/10); (iii) il rango effettivo non cresce con la taglia da 30M a
300M (F4); (iv) prima ondata: cambiare il JEPA in tre modi (bersaglio a, unita' per regione, finestra a 150 ms) non da' nulla, e il bersaglio a sta
sotto R0 nei due compiti («regola non superata», non «peggiora»: su P2 il delta medio supera il punto, -1,27, ma l'intervallo tocca lo zero).
*Interpretazione (mia):* tutta
l'informazione lineare che la sonda legge viene dai bersagli esterni (le ancore: log RMS, forma spettrale, inviluppo, bande multi-scala), che sono
statistiche locali del segnale; il JEPA e' peso morto. Un bersaglio esterno che codifica lo **stato del movimento** e' l'unica cosa non ancora
provata nella direzione che i dati indicano. L'encoder di A batte le 4 feature di Hudgins di 4 punti su P1 e 14 su P2: la domanda e' se un
bersaglio piu' ricco allarga quel margine.

## 2. Bracci, riferimento e deroghe da firmare

| Run | Cosa cambia rispetto ad A (JEPA target b + ancore 0,2) | Stato |
|---|---|---|
| **R0** riferimento del criterio | A rifatto a 2 epoche, seme 0, LR 1e-4: **e' il tronco R0 del pilot, gia' fatto** (`PILOT-R0-A-30M-s0`, codice 705ec04, `checkpoint_postdecay.pt` con SHA in `results/pilot_obiettivi_20261024/provenienza_criterio.json`); rapporti `pool` P1/P2 a 2 epoche gia' letti (S 18,84 / 81,22) | fatto |
| **R3** ancore + posa | `--jepa-weight 0` (come ancore-only di W1ARM: `anchor_weight` 0,2, ancora multi-scala, maschere e teacher EMA invariati; l'ancora RVQ e' spenta anche in R0 e in ancore-only, `rvq_codes` nullo nei config) **piu' una testa di posa** sui latenti del backbone: a ogni patch temporale di una finestra di emg2pose, `LayerNorm(media sui K latenti di z[s, p]) -> Linear(d, 20)` predice i 20 angoli standardizzati; errore quadratico medio sugli elementi validi (patch, giunto), ridotto **per finestra** come gli altri termini; **peso 1,0** (§2, scelta iii); nel contratto delle loss (`loss_counts`: numeratore e conteggio `pose`, batch intero = micro-batch) | da scrivere (§6) |
| **R3c** controllo, **consigliato e separabile** | ancore-only allo stesso disegno (`--jepa-weight 0`, nient'altro): il tronco a 2 epoche e il ramo a 1 epoca di «A senza JEPA», mai misurati a 2 epoche (a 4 epoche ancore-only contro A valeva +0,51 [IC sopra zero] su P1 e -0,40 su P2, `decisioni.md`, 23/10) | da lanciare (opzione) |

**Che cosa misura ciascun confronto.** Il criterio (§5) e' **R3 contro R0**: «R3 batte il default». Da solo non attribuisce l'eventuale guadagno
alla posa, perche' togliere il JEPA sposta gia' P1 di mezza soglia: R3c serve a questo, col confronto descrittivo **R3 contro R3c** («la posa
aggiunge alle ancore») e **R3c contro R0** («togliere il JEPA a 2 epoche»), stessa chiamata e stessi endpoint. Senza R3c l'attribuzione resta
aperta e la proposta lo dichiara; con R3c il costo della prima ondata raddoppia (§7). La configurazione candidata al default, se R3 passa e la
conferma a due semi regge, e' **R3 cosi' com'e'** (ancore 0,2 + posa 1,0, senza JEPA); A + posa (JEPA acceso) non e' provato e non e' candidato.

**Dove si attacca la testa e perche'.** Il readout `pool` del criterio e' la media dell'encoder locale sui canali piu' la media del backbone sui
latenti e sul tempo (`spectral_step1.pool_p1p2`: `enc.z.mean(dim=2)` mediato sugli istanti validi, larghezza 2d). La media sui K latenti di
`z[s, p]` e' la parte del backbone di quel readout, per istante: la testa di posa chiede che proprio quel vettore porti, linearmente, lo stato
della mano. La parte locale (un canale, una patch) non vede la mano e resta alle ancore.

**Bersaglio di posa.** `joint_angles` (20 valori per campione, stesse righe dell'EMG nel dataset `timeseries` degli HDF5: allineamento per indice
di campione, nessuna interpolazione nostra), della mano registrata. **Che cosa sono** (fact-checker del 26/10 sul paper arXiv 2412.02725 e sul
repository `facebookresearch/emg2pose`, `docs/fatti_da_verificare.md` n. 28, stato `raccolto`): angoli ricostruiti da **motion capture ottico**
(26 telecamere a 60 Hz) con cinematica inversa offline, **interpolati linearmente a 2 kHz** e filtrati passa-basso a 15 Hz (l'ordine fra filtro e
interpolazione non e' dichiarato); 20 gradi di liberta'
(pollice CMC flesso-estensione e abdu-adduzione, MCP e IP; per indice, medio, anulare e mignolo MCP abdu-adduzione e flesso-estensione, PIP, DIP),
**in radianti secondo i commenti del codice** (le tabelle del paper sono in gradi: il collaudo dell'ingest misura il massimo assoluto per giunto);
**i fallimenti della cinematica inversa (12,7% dei frame nel paper) sono righe coi 20 angoli tutti a zero**, senza campo o maschera nel file
(`get_ik_failures_mask` del repository li ricava a runtime). Ridotti a **200 Hz** (media su 10 campioni a 2 kHz: la posa nativa e' a 60 Hz e
filtrata a 15 Hz, 200 Hz non perde nulla) e salvati in float16 fuori dall'albero processato dell'EMG (che il manifest v1.1 congela), in
`$SCRATCH/data/processed/emg2pose_pose/u<utente>/<stem>/pose_200hz_f16.npy`, con una maschera di validita' per blocco. **Regola di validita',
da congelare dopo il collaudo dell'ingest (§6, punto 5) e prima dell'ingest intero.** Il nucleo e' certo: una riga coi 20 angoli tutti a zero
(tolleranza di `numpy.isclose`) non e' valida. **Il margine attorno ai blocchi di zeri non si fissa ora:** se gli zeri sono stati messi a 60 Hz e
poi interpolati, ai bordi dei blocchi ci sono rampe non nulle, e un frame fallito isolato diventa una «V» verso lo zero senza nessuna riga
esattamente nulla (un frame dura 33,3 campioni: coincide con un campione a 2 kHz un frame su tre); se il passa-basso a 15 Hz ha agito dopo, la
contaminazione dura piu' di un frame. Il collaudo dell'ingest lo misura con una statistica che non dipende dall'ipotesi (§6, punto 2), e il
margine (in campioni dal bordo di ogni blocco di zeri, e la regola per le «V») si congela in `decisioni.md` da quella misura. Un blocco a 200 Hz e'
valido solo se tutti i suoi 10 campioni lo sono. **Standardizzazione per giunto** con media e deviazione standard calcolate dal job di ingest sui
soli blocchi validi delle registrazioni dello split `pretraining` (le 8.117 registrazioni fuori dal pretraining, 1.950 di validazione e 6.167 di
test fra cui le 3.539 del test per fasi nuove, restano fuori), congelate in `pose_stats.json` con sha256 nel config del run; per patch da 25 ms il
target e' la media dei blocchi validi a 200 Hz **il cui centro cade nella patch** (esattamente 5, salvo ai bordi della finestra; la griglia a 200 Hz
parte dall'inizio della registrazione, quella del front-end dal primo campione della finestra, che e' a multipli di 200 ms da un'ancora della
prova: sfasamento fino a mezzo blocco, **2,5 ms, dichiarato** e trascurabile rispetto alla patch e a una posa filtrata a 15 Hz); un elemento
(patch, giunto) e' valido se almeno un blocco valido vi cade; una finestra senza posa (tutti gli altri dataset) non contribuisce al termine. La
posa entra solo nelle finestre di emg2pose: **24,6% del tempo campionato** (quota dell'unita' emg2pose nel manifest v1.1 alla frazione 1, pesata
nel tempo: `share` 0,24618; 287,8 h di finestre e 158 utenti nello split di pretraining secondo il manifest, 288,9 h di registrazioni dal CSV dei
metadati, 35% delle ore del corpus), cioe' **~23% delle finestre** (le finestre di
emg2pose durano 4 s contro una media di 3,77); l'altro 77% delle finestre addestra le sole ancore. Il loader pesca da un pool di 16 blocchi da 8
finestre: un micro-batch da 8 senza emg2pose capita circa una volta su cinque o sei (stima del `critico` col modello del pool; nel pannello
fisso di R0 11 sotto-batch da 8 su 32 non ne hanno), e **un passo intero senza posa sui 4 processi nello 0,1-1% dei passi**, decine o centinaia nel
tronco: il codice prevede i due casi (§6, punto 2) e quei passi si escludono dai descrittivi del termine (si logga il conteggio).

**Mano non sollecitata.** Nello split `pretraining` di emg2pose **4.973 registrazioni su 17.136 (66,3 h su 288,9, il 23%) hanno il bracciale
sulla mano che in quella fase non era sollecitata a muoversi** (`moving_hand` diverso da `side` e da `both`; `~/wearusfm_local/reports/passo4/
emg2pose_metadata.csv`); 540 di queste (7,3 h) hanno l'altro lato nel test ufficiale per fasi nuove (`decisioni.md`, punto 11 del 09/10, che le
chiama «lato fermo»). Il paper non le descrive come ferme: «the hand that is not prompted to move during a particular stage», libera di muoversi
(fact-checker, fatto 28). La loro posa e' misurata come le altre e non e' costante per costruzione. **Scelta proposta, da firmare con la regola
§5:** restano nel termine e nelle statistiche; l'ingest riporta la varianza della posa per registrazione, cosi' la scelta e' rivedibile prima del
training senza rifare l'ingest. Alternativa a costo zero: escluderle dal termine (`--pose-unprompted off`).

**Deroghe dichiarate, da firmare insieme alla regola.**
1. **v10 §6.1** («Tutti i target sono osservativi, calcolati dal raw»): la posa e' misurata da un altro sistema, non calcolata dall'EMG. E' un
   bersaglio esterno, non piu' auto-supervisionato in senso stretto, anche se e' una grandezza fisica continua e non un'etichetta di gesto. La
   prima frase del §6.1 (mai la waveform grezza) e le ancore non cambiano.
2. **v10 §6.5 e §3.7** (teste supervisionate «attivate per-batch secondo il dataset», «bilanciamento per campionamento, non per pesi di loss»,
   «le teste di regressione portano il lead time vincolato di §3.7», cioe' l'attivita' elettrica che precede la forza di 30-100 ms): la testa di
   posa e' attivata per finestra secondo il dataset, com'e' previsto, e il campionamento resta quello del manifest (D9); ma **(a)** il termine ha
   un peso di loss, come le ancore (§6.2: «peso complessivo 0,1-0,3»), perche' qui si bilancia fra termini sulla stessa finestra, non fra
   dataset; **(b)** il target e' **allo stesso istante della patch (ritardo zero), senza lead time**: il backbone e' bidirezionale nel tempo
   (`model/backbone.py`, attenzione su tutti gli istanti validi), quindi il latente in t vede gia' l'EMG dei 30-100 ms precedenti e seguenti; un
   lead time appreso ha senso per un decoder causale, che non e' questo. E' una deroga, non un'applicazione.
3. **Conseguenza sul compito a valle:** la stima della posa su emg2pose, per gli utenti dello split di pretraining, non e' piu' un compito a
   valle «pulito» per un encoder R3; lo resta sul test ufficiale per fasi nuove, con l'eccezione delle 540 registrazioni il cui lato non
   sollecitato sta nel pretraining (gia' dichiarata per l'EMG il 09/10) e che con R3 portano in piu' gli angoli di quel lato.

**Scelte di Claude dentro R3 (per il `critico`):** (i) **testa lineare**, non MLP: il criterio e' la leggibilita' lineare del readout medio, e
una testa lineare sulla media dei latenti chiede esattamente quella; un MLP potrebbe nascondere la posa in combinazioni non lineari che la sonda
non legge. (ii) **Tutte le patch** con target valido, visibili e nascoste, pesate alla pari: sulle patch nascoste il termine e' in parte
predittivo dal contesto (in parte, per la fuga attraverso il filtro del loader attuale: i token visibili entro ~100 ms da una regione nascosta
portano una traccia del contenuto nascosto, `docs/pilot_obiettivi_v1.md` §6 punto 1, uguale per tutti i bracci); la maschera resta quella di A
perche' le ancore la richiedono. (iii) **Peso 1,0 sul termine standardizzato**, fissato ora dalle grandezze misurate: sul pannello fisso di R0
(`fixed_eval.jsonl` di `PILOT-R0-A-30M-s0`) la somma delle cinque ancore vale 29,8 al passo 0, 10,17 al 500, 8,01 al 1.000, 7,23 al 1.500 e 4,35 al
40.000, cioe' **6,0 -> 2,0 -> 0,87 una volta pesata per 0,2**; in `metrics.jsonl` (un record per passo) la somma delle ancore nei passi 1-100 di R0
vale in media 22,0 (4,40 pesata), 1,58 pesata nei passi 500-1.300. **La testa di posa si inizializza a zero** (pesi e bias): predice la media
standardizzata, quindi il termine parte esattamente da 1 per elemento e scende se la testa impara. Con peso 1,0 la posa pesata vale al passo 500
circa la meta' delle ancore pesate (al piu' 1,0 contro 2,0) e lungo il run resta dello stesso ordine di grandezza (ancore pesate 0,87 al passo
40.000); con 0,2 sarebbe stata un decimo del totale, e un esito negativo sarebbe stato confondibile con un termine troppo debole. **Cancello
congelato per il collaudo dbg** (§6, punto 7), letto da `metrics.jsonl`, termine di posa pesato (x 1,0) contro somma delle ancore pesate (x 0,2),
solo sui passi con conteggio di posa > 0: (1) **scala all'avvio:** media dei passi 1-100, posa pesata >= 0,1 x ancore pesate (atteso ~0,23: posa
~1 contro 4,40 nei passi 1-100 di R0; e' un controllo che la standardizzazione non abbia una scala sbagliata, non una parita'); (2) **tetto:**
media dei passi 500-1.300, posa pesata <= 2 x ancore pesate (divergenza o scala sbagliata; con le ancore di R0 il tetto e' 3,15); (3)
**discesa:** media del termine di posa sui passi 1.201-1.300 **<= 0,9 x** quella dei passi 500-599 (il margine del 10% distingue una discesa dal
rumore per passo, che e' grande: in R0 la somma delle ancore fra i passi 1.201 e 1.300 oscilla fra 4,7 e 10,5); (4) conteggio di posa > 0 in
almeno il 90% dei passi 1-1.300. **Nessun limite inferiore sui passi 500-1.300:** una testa che impara fa scendere il rapporto, e non e' un
difetto. Fuori dal cancello non si lancia e il peso si ridecide con Simone, registrato. (iv) **Standardizzazione per giunto sui campioni
validi dello split di pretraining;** per lato o insieme si decide **dopo l'ingest intero**, su tutto lo split (`pose_stats.json` le calcola per lato
e insieme), con questo criterio scritto ora: **per lato se per almeno un giunto |media sinistra - media destra| supera 0,5 deviazioni standard
congiunte, o il rapporto fra le deviazioni standard dei due lati supera 1,5;** altrimenti insieme (la stessa definizione dei 20 angoli per le due
mani e' un fatto da verificare: il repository non specchia gli angoli per lato e il paper inverte il segno solo dell'EMG della mano sinistra,
fatto 28). (v) **Mano non sollecitata tenuta** (sopra). (vi) **Niente velocita' angolare** come secondo target in questa ondata: e' un'opzione,
non la prova minima.

**Fuori da questo preventivo:** R4 (R2 + posa); A + posa (JEPA acceso); sonda non lineare (MLP) sulle feature gia' estratte: possibile dopo, su CPU.

## 3. Disegno

- **Taglia, durata, semi:** identici al pilot (`docs/pilot_obiettivi_v1.md` §3): 30M (preset `sanity`), seme 0, LR 1e-4, batch 32 su 4 GPU,
  manifest v1.1 al 100% (`--nested 1`), regole della finestra 1, allarme sul rango `--erank-min 38.4` (firma del 23/10), loader attuale. Tronco a
  2 epoche = 49.046 passi, WSD D14 (warmup 1.962, decay lineare dal passo 39.237), checkpoint tenuto al passo 19.618; ramo a 1 epoca dal
  19.618 a 24.523 passi. Due endpoint per run. Piano: `results/infra/pilot_posa/piano.json` (`scripts/wsd_chain.py`, con R3 e R3c; le opzioni
  della posa entrano nel piano col codice, il pianificatore le convalida contro il trainer).
- **Riferimento gia' fatto:** nessun nuovo R0. R0-e2 e i suoi rapporti `pool` esistono (sul cluster
  `$WORK/wearusfm_runs/results/pilot_obiettivi_20261024/R0-e2/`, copia sul Mac `~/wearusfm_local/pilot_obiettivi_20261024/R0-e2/`); R0-e1
  arriva coi blocchi B del pilot (sonde in coda). Loader attuale, con la fuga attraverso il filtro dichiarata nel pilot (§6, punto 1): vale
  per R0, R3 e R3c allo stesso modo. **Condizioni perche' il confronto regga, tutte da verificare e registrare prima di leggere:**
  (a) il codice nuovo lascia il percorso di default **bit per bit identico** (test, come per R2 e R5: stesse perdite, stesso stato, stessi batch
  e stesso pannello fisso senza le opzioni nuove); (b) **estrazione, sonda e confronto danno lo stesso risultato col codice di R3.** La garanzia
  sono **tre controlli d'uguaglianza**, non un diff vuoto, perche' l'estrazione importa file che R3 deve toccare: (b1) `extract_channel_readouts`
  su finestre fisse con un checkpoint fisso (riferimento prodotto con 705ec04 e salvato nel test) da' `pool`, `dec` e `local` **identici bit per
  bit** col codice di R3, e un modello con la testa di posa codifica identico agli stessi pesi senza testa (la testa non sta nel percorso di
  `encode`); (b2) la sonda sullo stesso rapporto da' la stessa S e zero predizioni diverse (il controllo d'uguaglianza gia' previsto da
  `channel_readout.py --stage equality`); (b3) il confronto del pilot R1 contro R0 su P1, rifatto col codice di R3 sugli stessi rapporti,
  riproduce `results/pilot_obiettivi_20261024/confronti/criterio_R1_vs_R0_e2_p1_pool.json` identico (delta, intervallo, per dataset). Accanto, come
  documentazione, il `git diff` fra 705ec04 e il deploy di R3 sulla **chiusura degli import** dei tre stadi (ricavata il 26/10 per via statica: 40
  file, cioe' `scripts/channel_readout.py`, `scripts/arms_compare.py`, 8 di `harness/`, 16 di `model/`, `data/pretraining_loader.py`,
  `data/processed.py`, `data/rvq_codes.py`, `data/virtual_montage.py`, `training/jepa.py`, `training/masking.py`, `ingest/common.py`,
  `metadata/schema.py`, `metadata/taxonomy.py`, `preprocessing/canonical_view.py`, piu' gli script caricati dinamicamente da `channel_readout.py` e
  `arms_compare.py`, fra cui `calibration_select.py`, `arms_plan.py`, `calibration_probe.py`, `probe_p1.py`, e `training/run.py` che questi
  importano dentro le funzioni; elenco rifatto al deploy e allegato al registro): **vuoto su tutti i file tranne quelli che R3 deve toccare e che
  stanno nella chiusura**, cioe' `data/pretraining_loader.py` (campo `pose` di `PretrainBatch`, `LoaderConfig.pose_root`), `training/jepa.py`
  (termine `pose`, `JEPAConfig.pose_weight`), `model/fm.py` (campo `FMConfig.pose_head` e attributo) e `training/run.py` (opzioni e pannello
  fisso), per i quali il diff si limita alle aggiunte della posa e **le funzioni che estrazione e confronto usano restano identiche**
  (`pretraining_loader._filter`, `jepa.hidden_queries`, `WearUsFM.encode` e `decode`, le funzioni di `run.py` importate da `calibration_select.py`
  e `arms_plan.py`): si verifica col diff per funzione e coi tre controlli. Gli altri file che R3 cambia (`scripts/train_jepa.py`,
  `scripts/arms_collaudo.py`, `scripts/wsd_chain.py`, `model/pose_head.py` nuovo, `data/collate.py` se serve, l'ingest) stanno fuori dalla chiusura
  e sono coperti dal test (a); (c) stesso manifest, scale, finestre, seme; (d) **ambiente del cluster:** il launcher di R3 registra `module list` e `pip
  freeze` nel log del job e nel config del run; per R0 si ricostruiscono i moduli dal log del job del pilot, e il venv e' lo stesso
  `$WORK/venv_wearusfm`, condiviso e senza una fotografia al 25/10: **la (d) e' dichiarata, non dimostrabile a posteriori per R0**; il `pip freeze`
  di R3 fa da fotografia da qui in avanti. Un controllo d'uguaglianza della (b) che fallisce, o un diff fuori dai quattro file dichiarati o dentro
  le funzioni che devono restare identiche, **blocca la lettura**: si riporta a Simone e si decide prima di leggere, non dopo.
- **Perche' due epoche anche qui:** come nel pilot; la pendenza 1 -> 2 di R3 accanto a quella di R0 (e di R3c) dice se il bersaglio nuovo impara
  piu' lentamente.

## 4. Valutazione

Identica al pilot (`docs/pilot_obiettivi_v1.md` §4), con lo stesso strumento e gli stessi comandi, piu' i monitor della posa:

1. **S interna P1 e P2** del teacher post-decay, readout `pool`: e' il criterio (A1: 2 estrazioni + 2 sonde per run, confronto con R0-e2).
   Accanto, descrittivo, `dec` per canale a k 768 (A2: 2 sonde per run).
2. **Endpoint a 1 epoca e pendenza 1 -> 2** (B: 2 estrazioni + 4 sonde per run), descrittivi, con R0-e1 del pilot.
3. **Confronti descrittivi di attribuzione, se R3c c'e':** R3 contro R3c e R3c contro R0, ai due endpoint, `pool` e `dec`, stessa chiamata del
   criterio, letti come descrittivi (fuori dal criterio, dichiarato).
4. **Diagnostiche di non collasso** e allarme firmato (38,4 / 0,05 / 3 controlli): un run fermato dall'allarme non passa.
5. **Loss come monitor:** ancore (target fisici uguali a R0), termine di posa e suo conteggio in `metrics.jsonl` a ogni passo e sul pannello
   fisso ogni 500 passi; **descrittivo nuovo, senza job in piu':** errore assoluto medio della posa, de-standardizzato nell'unita' degli angoli,
   per giunto, sui batch del pannello fisso ai pannelli piu' vicini agli endpoint (passi 24.500 e 49.000; il pannello gira ai multipli di 500),
   dai log del trainer; i passi senza finestre di posa restano fuori dalle medie. Dice se la testa impara; non entra nel criterio.
6. **Riferimenti esterni** sulla stessa validazione: baseline classica (P1 14,30, P2 67,48), A storico (18,34 / 81,92), R0 del pilot (18,84 / 81,22).

## 5. Regola di lettura (testo unico, da copiare identico in `docs/decisioni.md`; DA FIRMARE da Simone prima di qualunque job)

«**Regola di lettura della seconda ondata (R3).** Criterio: per ciascun compito (P1, P2) della validazione interna, all'endpoint a 2 epoche
(teacher di `checkpoint_postdecay.pt` del tronco, run completo senza STOP), col readout `pool` estratto e sondato con `scripts/channel_readout.py`
(`--stage extract`, poi `--stage probe --layer pool`), il confronto accoppiato di R3 con **R0 del pilot degli obiettivi allo stesso endpoint**
(`PILOT-R0-A-30M-s0`, `checkpoint_postdecay.pt`, SHA-256 registrato in `results/pilot_obiettivi_20261024/provenienza_criterio.json`) con
`scripts/channel_readout.py --stage compare --single-seed --models` (soggetti ricampionati insieme, 1.000 ricampionamenti, seme 20261009,
intervallo dei percentili, `arms_compare.s_comparison`). R3 **passa** se in P1 **e** in P2 il delta e' >= +1 punto col limite inferiore
dell'intervallo al 95% > 0. Due letture del criterio senza correzione per confronti multipli: dichiarato. Un solo seme: il criterio dell'accordo
fra i semi non si applica e il rapporto lo dichiara (`seeds` 1). Un run fermato dall'allarme sul rango o incompleto non passa. Provenienza: prima
di leggere si verifica che i due checkpoint siano i `checkpoint_postdecay.pt` dei run giusti (R0 del pilot e R3, tronco a 2 epoche), che
`arms_compare.run_status` dia «completo» per entrambi, che lo SHA di R0 coincida con quello registrato dal pilot e che le condizioni di
comparabilita' del §3 del protocollo (commit della firma) siano verificate: percorso di default bit per bit identico senza le opzioni della posa
(test); i tre controlli d'uguaglianza superati (estrazione identica bit per bit da un checkpoint di riferimento, sonda con la stessa S e zero
predizioni diverse, confronto R1 contro R0 del pilot riprodotto identico col codice di R3); `git diff` fra 705ec04 e il deploy di R3 sulla
chiusura degli import di estrazione, sonda e confronto vuoto fuori dai quattro file dichiarati nel §3 e, dentro quelli, limitato alle aggiunte
della posa con le funzioni usate dall'estrazione e dal confronto identiche; stesso manifest, scale, finestre e seme; ambiente registrato nel log
del job di R3. Tutto si scrive nel registro; una condizione che manca blocca la lettura. **Conferma:** se R3 passa si lanciano R3 seme 1 e R0 seme 1 (stesso disegno,
tronco e ramo), e si rilegge con la regola dei bracci intera (due semi: `compare --models` senza `--single-seed`); se anche la conferma passa, la
configurazione candidata al default della finestra 1 e' R3 cosi' com'e' (ancore 0,2 + posa 1,0, senza JEPA) e la decisione e' di Simone.
**Descrittivi, fuori dal criterio:** `dec` per canale a 768, l'endpoint a 1 epoca, la pendenza 1 -> 2, i confronti con R3c (R3 contro R3c, R3c
contro R0) se R3c e' stato lanciato, l'errore di posa sul pannello fisso, le loss e le diagnostiche. R3 non passa -> si registra e il default non
cambia; la mossa successiva la decide Simone. **Non si decide qui:** K, taglie, LR della campagna, default della finestra 1, R4, A + posa.»

## 6. Prerequisiti, nell'ordine (nessun job prima della firma e del via; ogni job da un checkout di campagna con `CODE_DIR`, mai da `$WORK/wearusfm`)

1. **Firma di Simone:** regola §5, deroghe 1-2 e conseguenza 3 del §2, scelta sulla mano non sollecitata, con o senza R3c.
2. **Codice (Claude), con test CPU prima della review:**
   - ingest: `scripts/ingest_emg2pose_pose.py` + `scripts/slurm/ingest_emg2pose_pose.sbatch` (launcher con `lib_code_dir.sh` e `CODE_DIR`, non
     `cd $WORK/wearusfm` come il launcher dell'ingest EMG), stesso accesso casuale al tar dell'ingest EMG (`src/wearusfm/ingest/emg2pose.py`),
     tutte le 25.253 registrazioni, per registrazione `pose_200hz_f16.npy` + maschera di validita' per blocco + `pose_meta.json` (campioni
     validi, non finiti, righe tutte a zero e loro blocchi con lunghezze, qualunque campo o maschera presente nel file, minimo e massimo per
     giunto, varianza della posa, `side`, `moving_hand`, `split`) e, **per dimensionare il margine senza assumere l'ipotesi delle rampe**, due
     misure: il profilo di |differenza prima| degli angoli in funzione della distanza dal bordo di ogni blocco di zeri, fino a 200 ms,
     confrontato con la stessa distribuzione lontano dai bordi (il margine e' la distanza oltre la quale il profilo torna a quella
     distribuzione), e il conteggio dei minimi simultanei dei 20 giunti senza righe nulle (le «V» di un frame fallito isolato interpolato); modalita'
     collaudo (`--limit N --both-sides`, stratificato per lato e per `moving_hand`), `--skip-existing` e budget di tempo per la ripresa; alla fine
     `pose_stats.json` (media e deviazione per giunto sui campioni validi dello split `pretraining`, per lato e insieme). Il job scrive in
     `$WORK/wearusfm_runs/results/passo2/` (fuori dal repo); il riepilogo si copia sul Mac e si committa in `results/passo2/`;
   - loader: `LoaderConfig.pose_root`; `PretrainBatch.pose` per campione `(P, 20)` float32 e maschera `(P, 20)`, o `None`; lettura in memoria
     mappata per chiave di sessione; una sessione di emg2pose senza file di posa e' un errore, non un salto silenzioso;
   - modello: `FMConfig.pose_head`, `PoseHead` = LayerNorm + Linear sui latenti mediati per istante, **valutata su tutte le finestre del
     micro-batch** con la maschera che azzera il contributo delle finestre senza posa: nessun parametro resta fuori dal grafo, senza
     `find_unused_parameters` (oggi acceso solo per RVQ e keep);
   - loss: `training/jepa.py`, termine `pose` nei `losses` e nei `terms`, `loss_counts` con numeratore e conteggio `pose`; `JEPAConfig.pose_weight`;
     in `metrics.jsonl` il conteggio di posa del passo accanto al termine (un passo con conteggio 0 si riconosce, non si legge come posa = 0);
     `_fixed_eval_losses` (`training/run.py`) con il termine di posa e l'errore assoluto medio per giunto de-standardizzato;
     `scripts/arms_collaudo.py`: la formula della composizione della loss include il termine di posa (oggi `total = wj x jepa + wa x ancore`) e
     i passi con conteggio di posa 0 non contano per il termine; il cancello del §2 (iii) implementato come criteri congelati del collaudo;
   - launcher: `--pose-root`, `--pose-weight`, `--pose-unprompted`; config e summary del run con sha256 di `pose_stats.json` e l'ambiente;
     `wsd_chain` convalida le opzioni;
   - test: default bit per bit identico senza le opzioni; i due test d'uguaglianza del §3 (b); target per patch = media dei campioni a 200 Hz col
     centro nella patch, con sfasamento della griglia; perdita = media sugli elementi validi per finestra, elementi non validi esclusi, finestre
     senza posa escluse dal conteggio; contratto delle loss (numeratori e conteggi uguali a `loss_counts`; batch intero = micro-batch per perdita,
     gradienti, ottimizzatore ed EMA); un micro-batch senza finestre di posa fa un passo DDP completo con gradiente nullo sulla testa, e **un
     passo intero senza posa** (conteggio globale 0 su tutti i processi) resta finito, senza STOP, col conteggio 0 nel log; la testa legge solo i
     latenti del proprio istante; dal loader vero su un albero sintetico di posa. Suite completa verde.
3. **Review del `critico`** su protocollo e codice (ingest compreso: calcola il target); correzioni; verifica di GPT dopo, non bloccante.
4. **Deploy e checkout di campagna:** `deploy/pilot-posa-<data>`, `campaign_checkout.sbatch` (job CPU), SHA completo, albero pulito.
5. **Collaudo dell'ingest** (dal checkout, **40 registrazioni, 20 per lato, stratificate per `moving_hand`**: pochi minuti): riporta il massimo
   assoluto per giunto (radianti attesi: pochi valori sopra ~3,2), la frazione di campioni non finiti e di righe tutte a zero (atteso ~12,7%), il
   profilo di |differenza prima| attorno ai blocchi di zeri e i minimi simultanei senza righe nulle (§6, punto 2), campi o maschere presenti nei
   file, statistiche per lato, varianza per registrazione. **Con quei numeri si congela in `decisioni.md` la regola di validita' dei campioni**
   (nucleo del §2 piu' il margine misurato e la regola per le «V»), firmata da Simone; la standardizzazione per lato o insieme si decide dopo
   l'ingest intero col criterio gia' scritto nel §2 (iv). Se la regola richiede codice nuovo: **review del `critico` sul diff**, poi secondo
   deploy e secondo checkout (nel preventivo).
6. **Ingest intero** (CPU, `lrd_all_serial`, 2 core, 4 h di tetto, piu' un job di ripresa se serve) e `pose_stats.json`; report sul cluster e
   copia sul Mac.
7. **Collaudo dbg di R3** (1.300 passi, 30 min, 4 A100; **criteri congelati prima**): gli 8 di `arms_collaudo.py` con la formula estesa; il
   cancello del §2 (iii) (scala all'avvio, tetto, discesa, conteggio di posa > 0 nel 90% dei passi, da `metrics.jsonl`); errore di posa
   de-standardizzato riportato; **s/passo <= 0,81** (la soglia del pilot: con 12 h 45 di tetto e 49.046 passi il tronco finisce entro il limite sotto ~0,87 s/passo,
   e la soglia tiene il margine). Sopra 0,81 non si lancia con questo piano: si rigenera con un job di scorta (`spare_jobs` 1, +51 GPU-ora di
   tetto) e lo si registra. Se R3c c'e', il suo collaudo e' quello di ancore-only del 21/10 (61607561, superato con lo stesso disegno a 30M):
   non si ripete.
8. **Training:** 2 job per run con `scripts/submit_job.py` (tronco; ramo in `afterok`), solo col via di Simone dopo il collaudo.
9. **Valutazione:** blocchi A1, A2, B come nel pilot (comandi dallo stesso schema di `comandi_valutazione.txt`, alimentatore e attese con gli
   agenti leggeri); provenienza e condizioni del §3 prima della lettura; registro, `decisioni.md`, `results/pilot_posa/`.

**Stato del codice (26/10/2026, dopo la firma; commit d9e8fa6, 1bf33fd, b41b40c, 71c0a93, dfb6d3e sul ramo di Claude):** fatto il punto 2 per
intero, in attesa del `critico` (punto 3). Dove sta: riferimento d'uguaglianza (b1) in `tests/cpu/readout_reference.py` +
`tests/cpu/data/readout_reference_705ec04.npz` (scritto PRIMA della testa, codice di `src/wearusfm/` identico a 705ec04) e
`test_readout_reference.py`; ingest in `src/wearusfm/ingest/emg2pose_pose.py`, `scripts/ingest_emg2pose_pose.py`,
`scripts/slurm/ingest_emg2pose_pose.sbatch` (`CODE_DIR`, 2 core, 4 h, budget 3 h 40; collaudo `--limit 40 --both-sides`); target per finestra in
`src/wearusfm/data/pose_targets.py` (`PoseStore`, `window_pose`) e nel loader (`LoaderConfig.pose_root`, `pose_unprompted`,
`PretrainBatch.pose`); testa in `src/wearusfm/model/pose_head.py` (`FMConfig.pose_head`); termine e conteggi in `training/jepa.py`
(`pose_terms`, `pose_time_valid`, `JEPAConfig.pose_weight`); trainer in `training/run.py` (`_pose_tensors`, `_Micro.pose`, `pose_stats_sha256`
nel config, pannello fisso con `pose_mae`, `pose_mae_std`, `pose_mae_j00..j19`, `pose_windows`); launcher `--pose-root`, `--pose-weight`
(default 1,0), `--pose-unprompted on|off`; collaudo `scripts/arms_collaudo.py --arm R3` (pesi con la posa, `posa_configurata`, i quattro criteri
del cancello, `ritmo` <= 0,81); `scripts/slurm/train_ddp.sbatch` scrive `module list` nel log e `pip_freeze_<job>.txt` nella cartella del run.
Piano con le opzioni della posa in `results/infra/pilot_posa/piano.json` (convalidato da `wsd_chain.py`). Tre dettagli d'implementazione da
sapere: la validita' della posa e' per blocco, e la maschera (P, 20) la ripete sui 20 giunti; l'errore di posa de-standardizzato del pannello
fisso usa la deviazione per giunto **dell'insieme** anche quando la standardizzazione e' per lato (esatto se insieme, approssimato se per lato:
il valore standardizzato `pose_mae_std` e' quello esatto); il margine della regola di validita' entra nell'ingest come `--zero-margin-samples`
(default 0 = solo righe a zero), da congelare dopo il collaudo. Test CPU nuovi: 33 (riferimento 3, ingest 9, target 6, obiettivo 6, collaudo 3,
piu' i flag); suite completa in corso al momento di questa nota.

**Precisazioni dopo la review del `critico` sul codice (26/10, sera; voce in `decisioni.md`):** validita' del termine su tutte le patch esistenti
(anche gli istanti senza canali visibili: il §2 (ii) vale alla lettera); SHA delle statistiche nel checkpoint, nel `summary.json` e in
`pose_provenance.json` (non nel config, che resta l'identita' del run; ripresa e ramo lo verificano); ambiente nel log e nel `summary.json`;
deviazione congiunta per lato = sqrt((s_sx^2 + s_dx^2) / 2); nella chiusura degli import del §3 (b) `_config_with_defaults` e' toccata e
dichiarata, `pose_targets` e `pose_head` si caricano solo coi rami della posa; `--skip-existing` dell'ingest salta solo a regola uguale.
Test CPU della posa: 44 in sei file (riferimento 4, ingest 9, target 6, obiettivo 7, training 5, collaudo 3), piu' i flag.

**Precisazione (26/10, sera, PRIMA del rapporto del collaudo; voce in `decisioni.md`):** la regola che trasforma il profilo dei bordi in M e'
congelata in `decisioni.md` e nel codice (`margin_from_collaudo`): precondizioni (15 per lato, zeri fra 2% e 40%, radianti, nessun file fallito);
M = `lag_to` dell'ultimo bin fuori tolleranza (0,25 sul rapporto con la media lontana, bin con almeno 200 campioni); M = 0 se il profilo e' piatto;
«non determinato» se non rientra entro 200 ms (profilo da allungare, non M a occhio); «V» tollerate fino allo 0,1% delle righe non nulle, sopra serve
una regola in piu'. Il §6, punto 5, resta: il valore di M lo firma Simone.

**Precisazione (27/10, dopo il collaudo 62364572 e la review del `critico`; voci in `decisioni.md`):** la regola v1 del 26/10 ha dato «non
determinato» (profilo piatto a ~1,78 volte il valore lontano da ~125 a 200 ms, tutti i 40 bin fuori dal 25%). E' stata scritta una regola di riserva
R1-R5, **post hoc, informata da quel profilo** (non pre-registrata): scatta solo se la v1 e' non determinata o da' M > 400 campioni; R1 plateau =
media dei bin 301-400; R2 persistenza a due lati (bin 401-1.000 fra 0,80 e 1,25 volte il plateau, nessun bin oltre 400 sopra 1,25, bin contati fino
ad almeno 2.000 campioni), altrimenti la riserva non si applica e vale la v1 (se M > 400 ci si ferma); R3 M3 = ultimo bin entro 300 sopra 1,10
volte il plateau; R5 M5 dal profilo per altezza del salto (terzili, alto / basso > 1,25); M = max(M3, M5). Provvisorio sul primo campione (2 utenti
soli): M3 = 200 campioni. Il profilo lungo (2 s) e un campione con almeno 10 utenti richiedono un secondo deploy e checkout (§6, punto 5) e due
job CPU **non coperti dall'autorizzazione anticipata**: via di Simone da chiedere. Il valore di M lo firma Simone dopo la verifica di GPT.

## 7. Costi (preventivo `results/infra/pilot_posa/preventivo.json`; training da `wsd_chain.py`, s/passo 0,75 misurato sul pilot e su ancore-only, slack 1,15)

| Voce | Job | GPU-ora attese / tetto | Core-ore attese / tetto |
|---|---|---|---|
| Checkout di campagna (secondo checkout se la regola di validita' cambia il codice) | 1 (+1) | 0 | 0,2 / 1,0 (+0,2 / 1,0) |
| Ingest della posa: collaudo su 40 registrazioni (30 min) + intero (4 h) + ripresa (4 h), 2 core | 3 | 0 | 7,4 / 17,0 |
| Collaudo dbg di R3 (1.300 passi, 30 min) | 1 | 1,3 / 2,0 | |
| R3: tronco a 2 epoche (12 h 45 di tetto, un nodo) + ramo a 1 epoca (2 h 15) | 2 | 47,0 / 60,0 | |
| R3: estrazioni GPU, 2 endpoint x (P1 + P2) | 4 | 1,4 / 5,0 | |
| R3: sonde CPU, 2 endpoint x (pool P1, pool P2, dec P1, dec P2), 8 core, una alla volta (~8 h 50 di parete) | 8 | 0 | 70,8 / 256,0 |
| **Prima ondata, solo R3** | **19 (+1)** | **~49,7 / 67,0 GPU-ora = ~400 / 536 ore locali** | **~79 / 275** |
| R3c (controllo, consigliato): tronco + ramo, 4 estrazioni, 8 sonde | 14 | 48,4 / 65,0 | 70,8 / 256,0 |
| **Prima ondata con R3c** | **33 (+1)** | **~98,1 / 132,0 GPU-ora = ~785 / 1.056 ore locali** | **~149 / 531** |
| Job di scorta del tronco di R3, solo se il collaudo da' s/passo > 0,81 | +1 | +0 / +51,0 | |
| Conferma, solo se R3 passa: R3 seme 1 + R0 seme 1 (tronco e ramo ciascuno) + valutazione dei due | 4 + 24 | 94,0 / 120,0 + 2,8 / 10,0 | 141,6 / 512,0 |

Basi: estrazioni dal pilot (sacct del 25/10: P1 12-14 min, P2 ~28 min su 1 A100; tetti 1 h e 1 h 30); sonde dal pilot (pool P1 64-68 min,
pool P2 15-17 min da `watch_pilot_eval_pool_e2.txt`; dec P1 2 h 26 - 2 h 29 da `watch_pilot_eval_dec_e2.txt`, le prime due di A2; dec P2 stimata
36 min; il consuntivo `sacct` della valutazione del pilot arriva a code vuote); ingest dal consuntivo dell'ingest EMG (12.755 s in tre job, uno
fermato dal budget di 9.600 s: un solo job da 4 h ha ~11% di margine, da cui il job di ripresa nel tetto). Ore locali: 1 GPU-ora = 8 ore
locali. **Voce di budget: finestra 1, famiglia «Ablation a 30M»** (stima ~2.100 GPU-ora in `docs/finestra1_fattibilita.md`), la stessa dei
bracci W1ARM e del pilot: speso finora dalla famiglia ~530 GPU-ora (bracci ~321 + JEPA-only ~10 + valutazione dei bracci ~15; pilot: collaudi 2,29,
training 176,5, valutazione ~5,6 attese); residuo indicativo della finestra 1 ~5.300 GPU-ora (i ~5.500 della proposta del pilot meno ~184),
**da confermare col consuntivo prima della firma**. Disco: 12,2 GB di posa (422,68 h x 200 Hz x 20 giunti x float16, piu' le maschere) e 3,2 GB
di checkpoint per run su `$SCRATCH`, nel giro di rinfresco dei 21 giorni (prossimo 12/11).
**Calendario dal via:** codice e test ~1 giorno; `critico`; deploy + checkout (~30 min di coda); collaudo dell'ingest (~15 min) e firma della
regola di validita'; ingest intero ~4 h sulla seriale (dietro le sonde B2 del pilot, che la occupano fino all'alba del 27/10); collaudo dbg
(~30 min); tronco ~10 h 30 + ramo ~1 h; valutazione ~1 giorno sulla seriale (R3 e R3c si alternano, una sonda alla volta): **4-5 giorni** con
R3 solo, ~5-6 con R3c (i due training in parallelo su due nodi, le sonde in fila).

## 8. Rischi dichiarati

- **Specializzazione al polso.** La posa esiste solo su emg2pose (bracciale a 16 canali). EPN-612 (P2) e' anch'esso un bracciale, NinaPro (P1) e'
  l'avambraccio: un guadagno solo su P2 sarebbe coerente con un encoder che impara il polso, non la mano. Il criterio «P1 **e** P2» lo esclude
  dal passaggio; i descrittivi per dataset lo mostrano.
- **Segnale di posa su un quarto del tempo campionato**, con un 12,7% di frame senza posa valida e un 23% di registrazioni a mano non
  sollecitata: il cancello del §2 (iii) ferma il lancio se il termine ha una scala sbagliata, diverge o non scende di almeno il 10% fra i passi
  500-599 e 1.201-1.300; non garantisce che la testa impari bene, lo dice solo il run intero.
- **Fatti raccolti ma non ancora verificati sui file:** unita' (radianti dai commenti del codice), rampe non nulle ai bordi dei blocchi di zeri
  (ipotesi), stessa definizione degli angoli per le due mani (non verificabile dalle fonti). Li misura il collaudo dell'ingest prima della
  regola di validita'; le fonti stanno in `docs/fatti_da_verificare.md`, n. 28, stato `raccolto` (non «verificato da SB»).
- **Attribuzione.** Senza R3c un passaggio di R3 non distingue «la posa aiuta» da «togliere il JEPA aiuta»; con R3c si distingue, in modo
  descrittivo.
- **Fuga attraverso il filtro** (loader attuale, pilot §6 punto 1): i token visibili vicino a una regione nascosta portano una traccia del
  contenuto nascosto; uguale per R0, R3 e R3c, ma rende meno netta la lettura «predittiva» del termine di posa sulle patch nascoste.
- **Un seme, 30M, due epoche:** gli stessi limiti della prima ondata; il pilot vede effetti da un punto in su.
- **Comparabilita' con R0:** regge solo con le condizioni (a)-(d) del §3; una che manca blocca la lettura.

## 9. Cosa NON fa questo pilot

Non tocca K, il tokenizer, il loader delle finestre, il manifest, i target degli altri dataset; non ingerisce etichette di gesto; non usa la posa
del test ufficiale di emg2pose; non lancia R4 ne' A + posa; non cambia il default della finestra 1 (lo puo' fare solo la conferma a due semi,
e la decisione e' di Simone). Se R3 non passa, restano sul tavolo, nell'ordine dell'evidenza: sonda non lineare per canale sulle feature gia'
estratte (CPU), analisi dei pesi e token da 100-200 ms (`docs/pilot_obiettivi_v1.md` §8), e un riesame del principio §6.1 piu' ampio di questa
deroga.
