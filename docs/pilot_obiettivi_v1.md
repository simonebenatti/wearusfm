# Pilot degli obiettivi di pretraining — `pilot_obiettivi_v1`

**Stato: regola di lettura (§5) e scostamento da D10 per R5 FIRMATI da Simone il 24/10/2026** («Firmo la regola di lettura del pilot (testo del
§5) e lo scostamento da D10 per R5. Via al deploy deploy/pilot-obiettivi-20261024, al checkout di campagna, al gate D8 sul banco da 150 ms e al
collaudo dbg di R2 e R5 (tetto 4 GPU-ora). I 4 training aspettano il mio via dopo gate e collaudo.»), dopo due review del `critico` (24/10,
rilievi tutti corretti). La regola congelata sta in `docs/decisioni.md` (voce della firma); i 4 training attendono il via dopo gate e collaudo. Origine: `docs/diagnosi_pretraining_20261023.md` §3.2-3.3 e
`docs/sintesi_pretraining_20261024.md` (il readout per canale ha escluso il readout come causa principale; *interpretazione:* il collo di
bottiglia piu' probabile e' nell'obiettivo).
Decisioni di Simone gia' prese: braccio R5 («tipo 150ms con overlap di 125.ms. che ne pensi? va esplorata?» -> «si ottimo!», 23/10); durata
(«scelgo le 2 epoche per tutti i 4 run», 23/10); preparazione («scrivi la sintesi e prepara il pilot degli obiettivi (codice R2, pianificatore a
2 endpoint, gate D8 per R5, regola di lettura, critico); nessun job senza il mio via», 24/10).

## 1. Domanda

Un obiettivo che chiede la struttura alla scala del movimento (regione nascosta, non patch da 25 ms: R2 nella loss, R5 nel tokenizer), o che
toglie il percorso query -> target (R1, target (a) di D11), produce un encoder che una sonda lineare legge meglio di A a pari esposizione?
Misura: S della validazione interna (P1 e P2) col readout medio `pool`, lo stesso di tutta la storia del progetto.

## 2. Bracci (prima ondata)

| Braccio | Cosa cambia rispetto ad A (JEPA target b + ancore 0,2) | Codice |
|---|---|---|
| **R0** riferimento | A rifatto a 2 epoche col ramo di decay a 1 epoca: stesso seme 0, LR 1e-4, manifest, loader e sampler di C1 s0 e dei bracci | pronto (e' A) |
| **R1** target (a) | `JEPAConfig.target = "a"`: target = uscita per canale dell'encoder locale del teacher allo stesso (canale, istante), non il decoder a query (D11; il braccio di controllo di v10 §5.6 mai eseguito) | `--jepa-target a` (23/10), test di default invariato |
| **R2** target per regione | il target JEPA di ogni **regione nascosta** (una maschera estratta: tubo, slab o maschera spaziale) e' la LayerNorm della media del teacher sui suoi token; lo studente predice la media delle sue uscite sugli stessi token; l'errore quadratico della regione e' attribuito a ciascuno dei suoi token e ridotto come oggi (per finestra), cosi' **le quote di D10 restano quelle per token** e cambia solo l'unita' del target; ancore invariate per token | **fatto il 24/10**: `generate_mask_regions` (numero della maschera per token), `PretrainBatch.region`, `JEPAConfig.region_target`, `jepa.region_means`, `--jepa-region-target`; test: default bit per bit identico, regione per token = perdita per token, perdita = media per token dell'errore della regione, contratto delle loss (numeratori e conteggi uguali a `loss_counts`; batch intero = micro-batch per perdita, gradienti, ottimizzatore ed EMA), dal loader vero |
| **R5** finestra 150 ms, passo 25 ms | il kernel del front-end vive su 150 ms centrati sulla patch; la griglia dei token resta a 25 ms (sequenza, maschere, ancore, RVQ invariati); basi scalate (156 armoniche, 120 nodi, omega 36); **la perdita conta solo i token interamente nascosti** (nessuna patch visibile entro 3 patch per lato); le maschere da 1-2 patch non producono token interi: `share_short` 0, quote 0,294 / 0,235 / 0,471 (**scostamento dichiarato da D10, firmato da Simone il 24/10/2026**) | fatto il 23/10 (`FMConfig.window_ms`, `core_hidden`, `--window-ms`); il 24/10: `--mask-shares 0:0.294:0.235:0.471`, e `loss_counts` corretto (prima non applicava il raggio: il contratto delle loss avrebbe fermato il run al primo passo; test) |

R2 e R5 provano la stessa ipotesi (unita' meno rumorosa, alla scala dell'inviluppo) in due posti: R2 nella loss, col tokenizer di sempre; R5
nel tokenizer. Seconda ondata (non in questo preventivo, decisione di Simone separata): R3 ancore + testa di posa su emg2pose senza JEPA; R4 =
R2 + posa. Richiedono l'ingest degli angoli (`joint_angles` dal tar di emg2pose, 462 GB) e un cambio di principio rispetto a v10 §6.1.

**Scelte di Claude dentro R2 (il `critico` del 24/10 le ha giudicate difendibili; la (v) e' la sua correzione):** (i) la LayerNorm si applica
alla media del teacher grezzo (non alla media dei token normalizzati): il target ha sempre scala unitaria, come per token; (ii) la predizione
e' la media delle uscite grezze del decoder: lo studente puo' scegliere la norma; (iii) una regione e' una maschera estratta, non una componente
connessa: due tubi sovrapposti restano due regioni (i token in comune vanno al primo); (iv) il log per tipo di maschera e per topologia pesa
ogni token con l'errore della sua regione; (v) **ogni regione pesa quanto i suoi token**: la prima stesura faceva la media alla pari fra le
regioni, e il `critico` ha misurato col generatore vero che cosi' il 70% della perdita sarebbe finito sulle maschere corte da 1-2 patch (regioni
da 2-6 token) e il 2% su quelle spaziali (0,5-4 s): l'opposto dell'ipotesi, e uno scostamento da D10 non dichiarato. Con la pesatura per token
le scale delle regioni sono quelle delle maschere di D10 (corte 15%, medie 25%, lunghe 20%, spaziali 40% dei token).

## 3. Disegno

- **Taglia, durata, semi:** 30M (preset `sanity`), seme 0, LR 1e-4, batch 32 su 4 GPU, manifest v1.1 al 100% (`--nested 1`), regole della
  finestra 1, soglia assoluta dell'allarme sul rango `--erank-min 38.4` (firma del 23/10, ambito «tutti i run della finestra 1»), loader
  attuale (vedi «fuga»). **Tronco: 2 epoche** = 49.046 passi, WSD D14 (warmup 1.962 = 2% dei passi a D = 98.094, come C1 e i bracci; decay
  lineare sul 20% finale, dal passo 39.237), checkpoint tenuto al passo 19.618. **Ramo a 1 epoca, come D14 lo definisce** (decay da 0,8 E a
  E): dal checkpoint 19.618, totale 24.523 passi, decay su 4.905 passi: e' esattamente lo schedule di C1 s0 (la prima stesura metteva il ramo
  a 1,25 epoche, da 24.523 a 30.654: corretto dal `critico`). Due endpoint validi per run (teacher post-decay a 1 e a 2 epoche), 53.951 passi
  di calcolo (2,2 epoche). Il pianificatore (`scripts/wsd_chain.py`) non va esteso: tronco con `keep_checkpoints` e ramo con `branch_of` sono
  il caso gia' previsto (`_check_branch`, `with_wsd`).
- **Controllo di coerenza, descrittivo:** l'endpoint a 1 epoca di R0 ha lo stesso schedule, seme, dati e codice di C1 s0 (S P1 18,21): e' una
  replica (a meno del non determinismo di DDP e delle differenze di codice fra 72cf24a e il deploy del pilot, dichiarate al deploy).
- **Perche' due epoche:** A e' piatto da 1 a 4 epoche (18,21 -> 18,34); i bracci nuovi possono imparare piu' lentamente (R2 ha meno unita' di
  perdita per finestra; R5 conta solo i token interi): con due endpoint si legge la pendenza 1 -> 2 accanto a quella di R0.

## 4. Valutazione

Le loss **non** sono il criterio: fra obiettivi diversi non sono confrontabili (target a e b sono grandezze diverse, la media per regione
cambia l'unita', R5 conta solo i token interi). A ciascuno dei due endpoint di ogni run:

1. **S interna P1 e P2** del teacher post-decay, encoder congelato, sonda lineare della calibrazione, **readout `pool`**: e' il criterio.
   Strumento unico: `scripts/channel_readout.py` (`--stage extract`, poi `--stage probe --layer pool`), che riproduce la sonda della
   calibrazione (controllo d'uguaglianza del 23/10: stessa S e zero predizioni diverse su P1, A s0; sulla P2 il controllo non e' stato fatto:
   un job CPU di ~30 min lo chiuderebbe, proposto fra i prerequisiti). Accanto, **descrittivo**, `dec` per canale a k 768 (stesso strumento,
   protocollo del readout per canale): e' l'unico readout in cui un cambiamento a montaggio condiviso (EPN-612, +1,4 punti il 24/10) si
   vedrebbe prima.
2. **Pendenza 1 -> 2 epoche** di S accanto a quella di R0: descrittiva.
3. **Diagnostiche di non collasso** del trainer ogni 500 passi (rango effettivo di studente e teacher, rapporto di collasso da query, varianza
   fra campioni), con l'allarme firmato il 23/10 (soglia assoluta 38,4, rapporto 0,05, 3 controlli): un run fermato dall'allarme non passa.
4. **Loss come monitor**, non per decidere: JEPA per tipo di maschera e per topologia (per R2 i contatori `jepa_regions` e
   `jepa_hidden_tokens`, il cui rapporto e' il numero di token per regione, nel trainer DDP; per R5: `core_hidden_fraction`), ancore (target
   fisici uguali fra i bracci), pannello fisso ogni 500 passi.
5. **Riferimenti esterni** sulla stessa validazione: baseline classica (P1 14,30, P2 67,48) e A storico (18,34 / 81,92).

## 5. Regola di lettura (testo unico, identico in `docs/decisioni.md`; FIRMATA da Simone il 24/10/2026, prima di qualunque job)

«**Regola di lettura del pilot degli obiettivi.** Criterio: per ciascun braccio R_i (R1, R2, R5) e per ciascun compito (P1, P2) della validazione
interna, all'endpoint a 2 epoche (teacher di `checkpoint_postdecay.pt` del tronco, run completo senza STOP), col readout `pool` estratto e
sondato con `scripts/channel_readout.py` (`--stage extract`, poi `--stage probe --layer pool`: la sonda della calibrazione, controllo
d'uguaglianza del 23/10 su P1, A s0), il confronto accoppiato con R0 allo stesso endpoint con `scripts/channel_readout.py --stage compare --single-seed
--models` (soggetti ricampionati insieme, 1.000 ricampionamenti, seme 20261009, intervallo dei percentili, `arms_compare.s_comparison`). R_i
**passa** se in P1 **e** in P2 il delta e' >= +1 punto col limite inferiore dell'intervallo al 95% > 0. Sei letture del criterio (3 bracci x 2
compiti) senza correzione per confronti multipli: dichiarato. Un solo seme: il criterio dell'accordo fra i semi della regola dei bracci non si
applica e il rapporto lo dichiara (`seeds` 1). Un run fermato dall'allarme sul rango (38,4 / 0,05 / 3 controlli) o incompleto non passa.
Provenienza: il rapporto di confronto registra percorso e SHA-256 dei due checkpoint; prima di leggere si verifica che siano i
`checkpoint_postdecay.pt` dei run giusti (R0 e R_i, tronco a 2 epoche) e che `arms_compare.run_status` dia «completo» per entrambi, e lo si
scrive nel registro. **Conferma:** per ogni braccio che passa si lanciano il suo seme 1 e, una volta sola, R0 seme 1 (stesso disegno, tronco
e ramo), e si rilegge con la regola dei bracci intera (due semi: `compare --models` senza `--single-seed`); se piu' bracci passano anche la
conferma, si registra l'ordine per delta medio di P1 e P2 e la scelta del default e' di Simone. **Descrittivi, fuori dal criterio:** `dec` per
canale a 768, l'endpoint a 1 epoca, la pendenza 1 -> 2, le loss e le diagnostiche. Nessun braccio passa -> si registra e il default non
cambia; la seconda ondata la decide Simone. **Non si decide qui:** K, taglie, LR della campagna, default della finestra 1.»

## 6. Prerequisiti

1. **Fuga attraverso il filtro** (`results/rami_pilota/fuga_preprocessing_20261023.json`): il passa-banda del loader agisce prima della
   maschera, quindi i token visibili entro ~100 ms da una regione nascosta portano una traccia del contenuto nascosto; vale per A e per tutti i
   bracci allo stesso modo. Si usa il **loader attuale** (confrontabile con C1 s0 e coi bracci); «mascherare prima del filtro» e' un'ablation
   separata, dopo.
2. **Gate D8 riaperto per il banco da 150 ms** (R5): `scripts/slurm/gate_d8.sbatch --window-ms 150` dal checkout di campagna (CODE_DIR), stesse
   definizioni e soglie firmate il 01/10 (errore relativo <= 5% per famiglia, sonda <= 55%), job CPU sulla seriale (tetto 2 core x 2 h = 4
   core-ore, attesa ~1; 0 GPU-ora); il JSON dell'esito registra il banco (`frontend.window_ms`, `bank`). Comandi in
   `results/infra/pilot_obiettivi_20261024/comandi_gate_d8.txt`. NON SUPERATO -> R5 non parte, gli altri tre si'.
3. **Collaudo su dbg di R2 e R5** (1.300 passi l'uno, 30 min, `boost_qos_dbg`, piano `collaudo.json`: 2,6 / 4,0 GPU-ora): misura i s/passo
   (il preventivo usa 0,7 misurati su A; con `TIME_LIMIT_S` 40.800 e nessun job di scorta il tronco regge fino a 0,81 s/passo) e prova sul
   nodo il contratto delle loss con le regioni e coi token interi. Se R2 o R5 superano 0,81 s/passo, il piano passa a `spare_jobs` 1 per
   quel run prima del lancio.
4. **Firma dello scostamento da D10** per R5: quote delle maschere `share_short` 0, 0,294 / 0,235 / 0,471 (`--mask-shares`), **e** la
   ripartizione della perdita che ne segue, perche' la perdita conta solo i token interi (misura del `critico` del 24/10 col generatore vero,
   raggio 3, 160 patch, 40 semi; quota dei token nascosti -> quota dei token interi): media 15,8 -> 10,2%, slab 19,6 -> 14,1%, spaziale 52,3 ->
   62,3% su capgmyo; emg2pose 17,9 -> 11,9 / 23,1 -> 18,1 / 46,7 -> 56,4; DB2 16,0 -> 9,7 / 20,9 -> 16,3 / 48,3 -> 57,8; token interi / nascosti
   ~0,83. Le maschere spaziali pesano di piu', le corte spariscono: e' parte dell'ipotesi di R5, ma e' uno scostamento da D10. **Firmato da
   Simone il 24/10/2026** (quote e ripartizione della perdita; `docs/decisioni.md`).
5. **Controllo d'uguaglianza sulla P2 (proposta, facoltativo):** `pool` di A s0 sulla P2 con `channel_readout.py` contro `s_p2.json` storico
   (un job CPU di ~30 min, 0 GPU-ora), per chiudere l'uguaglianza dello strumento del criterio anche sulla P2.
6. **Review del `critico`** su protocollo e codice (fatte: due giri il 24/10, rilievi corretti); regola di lettura in `decisioni.md` (firmata il
   24/10); poi deploy `deploy/pilot-obiettivi-20261024`, checkout di campagna, gate D8 e collaudo (autorizzati il 24/10; ordine: checkout, poi
   gate e collaudo di R2 insieme, collaudo di R5 dopo l'esito del gate), e il via di Simone ai 4 training.
7. **Valutazione:** le estrazioni e le sonde si sottomettono a run finiti, coi launcher del readout per canale (`channel_readout_extract.sbatch`,
   `channel_readout_probe.sbatch --layer pool` e `--layer dec`); il confronto sul Mac dai rapporti copiati, come il 23-24/10.

## 7. Costi (preventivo `results/infra/pilot_obiettivi_20261024/preventivo_training.json`, s/passo 0,7 misurato a 4 GPU, slack 1,15)

| Voce | Job | GPU-ora attese / tetto |
|---|---|---|
| Collaudo dbg di R2 e R5 (1.300 passi, 30 min l'uno) | 2 | 2,6 / 4,0 |
| 4 tronchi a 2 epoche (12 h di tetto l'uno, un nodo) | 4 | 156,4 / 192,0 |
| 4 rami di decay a 1 epoca (2 h 15 l'uno) | 4 | 19,2 / 36,0 |
| Gate D8 per il banco da 150 ms (CPU seriale) | 1 (+1 checkout) | 0 (~1 core-ora, tetto 4) |
| Estrazioni GPU: 8 endpoint x (P1 + P2) | 16 | ~5,6 / 20,0 |
| Sonde CPU: 8 endpoint x (pool P1, pool P2, dec P1, dec P2) | 32 seriali (10 in coda alla volta) | 0 (~270 core-ore attese, tetto 1.024) |
| **Prima ondata** | | **~184 / 252 GPU-ora = ~1.472 / 2.016 ore locali** |
| Conferma: seme 1 del braccio che passa + R0 seme 1 (tronco e ramo ciascuno) | 4 (+8 valutazione) | ~88 / 114 GPU-ora (= 704 / 912 ore locali; +2,8 / 10 di valutazione) per il primo braccio, +44 / 57 per ogni altro |

Ore locali: 1 GPU-ora = 8 ore locali (convenzione del progetto). Voce di budget: finestra 1 (tetto firmato il 16/10: 6.000 GPU-ora,
calibrazione compresa). Spese registrate finora in `decisioni.md`: calibrazione ~149,5 (su 220, C4 compresa); bracci: 4 run sani ~321 (244,2
da sacct per tre run + A s0 ~77), JEPA-only s0 e s1 ~10 (5,79 + 4,22), valutazione ~15 (readout per canale compreso); in tutto ~496:
**residuo indicativo ~5.500 GPU-ora**, da confermare col consuntivo prima della firma. Dentro la finestra 1 la voce e' la famiglia «Ablation a 30M»
(~2.100 GPU-ora stimate in `docs/finestra1_fattibilita.md`), oppure una voce nuova decisa da Simone. Calendario: i 4 tronchi in parallelo su 4
nodi (~12 h), i rami dietro (~1 h 10 di calcolo, 2 h 15 di tetto), valutazione ~1-2 giorni sulla seriale. Disco: 12,8 GB di checkpoint su
`$SCRATCH/wearusfm_runs/runs/pilot_obiettivi/`.

## 8. Cosa NON fa questo pilot

Non tocca K ne' l'identita' di canale; non ingerisce etichette; non cambia il loader. Se nessun braccio passa e la seconda ondata non e' voluta,
la mossa successiva e' sul tokenizer (token da 100-200 ms sopra le patch, lettura non lineare per canale), con l'analisi dei pesi come primo
passo a costo zero.
