# Pilot degli obiettivi di pretraining: esiti della campagna `pilot-obiettivi-20261024`

Protocollo `docs/pilot_obiettivi_v1.md` (regola di lettura §5 firmata da Simone il 24/10/2026, prima di qualunque job); codice sul cluster
705ec0465420e80d5d818a8bbc7acc3a92c06efc (checkout di campagna `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pilot-obiettivi-20261024`).
Quattro bracci a 30M, seme 0: R0 = A rifatto, R1 = `--jepa-target a`, R2 = `--jepa-region-target`, R5 = `--window-ms 150`; tronchi a 2 epoche
(49.046 passi) e rami `-e1` a 1 epoca dal checkpoint 19.618. Training finito il 25/10 (8 job COMPLETED per `max_steps`, nessun allarme; 176,5
GPU-ora, `results/infra/pilot_obiettivi_20261024/consuntivo_training.json`). Rapporti delle sonde e predizioni fuori fold restano fuori da Git
(`~/wearusfm_local/pilot_obiettivi_20261024/`, stesso albero del cluster); qui stanno i controlli di provenienza, i confronti, le letture e il
riepilogo delle sonde. Comandi: `results/infra/pilot_obiettivi_20261024/comandi_valutazione.txt`.

- `provenienza_criterio.json`: controllo §5 PRIMA di leggere (26/10/2026, 00:23): per R0, R1, R2, R5 i rapporti `pool` P1 e P2 vengono dal
  `checkpoint_postdecay.pt` del tronco giusto (percorso nel rapporto), con lo stesso SHA-256 nelle due estrazioni, codice 705ec04, OOF integri;
  `arms_compare.run_status` = «completo» per i 4 tronchi (`summary.json` con `stopped: max_steps` a 49.046 passi, nessun file STOP); i 4
  checkpoint sono distinti. Lo `sha256sum` indipendente sul login node non e' stato eseguito (fuori dai comandi che leonardo-ops ammette): lo
  SHA del rapporto e' calcolato dal job di estrazione sul file stesso, e P1 e P2 sono due estrazioni separate dello stesso file.
- `confronti/criterio_<R>_vs_R0_e2_<p>_pool.json`: le 6 uscite di `scripts/channel_readout.py --stage compare --single-seed --models`
  (riferimento R0, candidato R_i, stesso readout `pool`; soggetti ricampionati insieme, 1.000 ricampionamenti, seme 20261009, intervallo dei
  percentili; `seeds` 1: il criterio dell'accordo fra i semi non si applica, dichiarato).
- `criterio.json`: la lettura per braccio e compito e l'esito del criterio. `riepilogo_sonde.json`: le sonde (S, S per dataset, k effettivo,
  rango effettivo delle feature, SHA del checkpoint e del codice, durata); per ora le 8 del criterio, i descrittivi si aggiungono a fine coda.

## Criterio (§5): R_i passa se in P1 **e** in P2 delta >= +1 punto col limite inferiore dell'IC95 > 0

| Braccio | Compito | S R0 | S R_i | Delta [IC95] | Lettura |
|---|---|---|---|---|---|
| R1 (`--jepa-target a`) | P1 | 18,84 | 18,08 | -0,76 [-1,25; -0,28] | regola non superata |
| R1 | P2 | 81,22 | 79,95 | -1,27 [-2,63; +0,15] | regola non superata |
| R2 (`--jepa-region-target`) | P1 | 18,84 | 18,57 | -0,27 [-0,72; +0,16] | regola non superata |
| R2 | P2 | 81,22 | 81,77 | +0,55 [-1,53; +2,66] | regola non superata |
| R5 (`--window-ms 150`) | P1 | 18,84 | 18,31 | -0,53 [-1,07; -0,02] | regola non superata |
| R5 | P2 | 81,22 | 81,58 | +0,35 [-2,14; +2,68] | regola non superata |

Punti percentuali di S della validazione interna (P1: NinaPro DB2/DB3/DB6; P2: EPN-612, UCI-EMG). Sei letture senza correzione per confronti
multipli, dichiarato nella regola. **Nessun braccio passa il criterio**: nessuna delle sei letture ha delta >= +1 con limite inferiore > 0.
Nessuna lettura e' «peggiora» secondo la regola speculare (delta <= -1 con limite superiore < 0). Per dataset (delta [IC95]):

- R1: P1 DB2 -0,96 [-1,65; -0,31], DB3 +0,24 [-0,34; +0,81], DB6 -1,46 [-3,21; +0,17]; P2 EPN-612 -0,97 [-1,47; -0,50], UCI -1,57 [-4,31; +1,24].
- R2: P1 DB2 -0,48 [-1,08; +0,06], DB3 -0,07 [-0,86; +0,69], DB6 +0,27 [-1,14; +1,67]; P2 EPN-612 -0,28 [-0,77; +0,14], UCI +1,38 [-2,63; +5,59].
- R5: P1 DB2 -1,22 [-1,95; -0,51], DB3 +0,13 [-0,42; +0,63], DB6 +1,25 [-0,70; +3,18]; P2 EPN-612 -2,45 [-2,97; -1,92], UCI +3,15 [-1,78; +7,80].

Conseguenza fissata dalla regola: si registra e il default non cambia; nessuna conferma col seme 1 da lanciare; la seconda ondata la decide
Simone. *Interpretazione (non e' una lettura della regola):* R1 sta sotto R0 in tutti e due i compiti (su P1 l'intervallo intero sotto lo zero,
su P2 il delta medio oltre un punto ma l'intervallo tocca lo zero): il bersaglio A da solo non e' piu' leggibile del bersaglio di default, pur con
un rango effettivo del teacher piu' alto (163 contro 130 a fine run, `metrics.jsonl`). R2 e R5 non si distinguono da R0: delta piccoli, di segno
opposto nei due compiti, intervalli che includono lo zero (su P1 R5 ha l'intervallo appena sotto lo zero); su R5 i dataset si muovono in direzioni
diverse (DB6 e UCI su, DB2 ed EPN-612 giu'), coerente con un banco a 150 ms che cambia la risoluzione temporale piu' che la qualita' media.

## Diagnostiche dei tronchi a fine run (descrittive, `runs/*/metrics.jsonl`, passo 49.046)

| Run | erank teacher | collapse ratio teacher | s/passo | GPU max (GiB) | durata |
|---|---|---|---|---|---|
| R0 `PILOT-R0-A-30M-s0` | 130,4 | 2,90 | 0,77 | 17,5 | 10 h 02 |
| R1 `PILOT-R1-targetA-30M-s0` | 163,0 | 0,59 | 0,74 | 17,5 | 9 h 39 |
| R2 `PILOT-R2-regione-30M-s0` | 131,1 | 3,37 | 0,77 | 17,5 | 10 h 03 |
| R5 `PILOT-R5-finestra150-30M-s0` | 134,0 | 4,22 | 0,74 | 18,1 | 10 h 13 |

Soglie dell'allarme (38,4 / 0,05 / 3 controlli) mai toccate. Il rapporto di collasso di R1 e' su un'altra scala perche' il bersaglio A ha
un'altra scala: non si confronta con gli altri tre (solo con la soglia).

## Descrittivi (fuori dal criterio): `dec` per canale a k 768, endpoint a 2 epoche (letti il 26/10/2026, 15:10)

Stessa chiamata del criterio (`compare --single-seed --models`, riferimento R0), readout `dec` per canale a 768 componenti
(`confronti/descrittivo_<R>_vs_R0_e2_<p>_dec.json`, `descrittivi.json`). Sei letture senza correzione; **non decidono nulla**: servono a capire
il meccanismo (Simone, 26/10: «I descrittivi servono a capire il meccanismo, non a scegliere retroattivamente un vincitore»). S assolute `dec`
di R0: P1 17,38 (`pool` 18,84), P2 83,14 (`pool` 81,22): come nel readout per canale, a montaggio condiviso (P2) il decoder per canale sta sopra il
readout medio, su P1 sotto.

| Braccio | Compito | S R0 `dec` | S R_i `dec` | Delta [IC95] | Per dataset (delta [IC95]) |
|---|---|---|---|---|---|
| R1 (`--jepa-target a`) | P1 | 17,38 | 17,20 | -0,19 [-0,73; +0,36] | DB2 -0,97 [-1,64; -0,28], DB3 +0,17 [-0,32; +0,81], DB6 +2,39 [+0,31; +4,85] |
| R1 | P2 | 83,14 | 82,49 | -0,65 [-1,47; +0,21] | EPN-612 -1,62 [-2,14; -1,12], UCI +0,33 [-1,27; +2,00] |
| R2 (`--jepa-region-target`) | P1 | 17,38 | 17,54 | +0,16 [-0,21; +0,57] | DB2 -0,07 [-0,51; +0,39], DB3 +0,37 [-0,23; +0,94], DB6 +0,79 [-0,47; +2,50] |
| R2 | P2 | 83,14 | 82,50 | -0,65 [-1,86; +0,63] | EPN-612 +0,23 [-0,16; +0,64], UCI -1,52 [-3,93; +0,99] |
| R5 (`--window-ms 150`) | P1 | 17,38 | 17,70 | +0,32 [-0,17; +0,82] | DB2 +0,05 [-0,61; +0,75], DB3 +0,20 [-0,72; +1,06], DB6 +1,57 [+0,74; +2,54] |
| R5 | P2 | 83,14 | 82,29 | -0,85 [-2,38; +0,72] | EPN-612 -1,96 [-2,43; -1,48], UCI +0,26 [-2,68; +3,45] |

*Interpretazione (non e' una lettura della regola):* col decoder per canale nessun braccio si stacca da R0 nemmeno in modo descrittivo (tutti i
delta aggregati entro un punto, intervalli che includono lo zero). Il quadro per dataset e' lo stesso del `pool`: R1 e R5 perdono su EPN-612
(intervallo intero sotto lo zero) e guadagnano su DB6 (R5 +1,57 [+0,74; +2,54], R1 +2,39 [+0,31; +4,85]: intervalli sopra lo zero, ma su 8 soggetti).
Lo scarto di R1 da R0 e' piu' piccolo con `dec` che con `pool` su P1 (-0,19 contro -0,76). Nessun segnale che un readout per canale cambi
l'esito del criterio.

Diagnostiche di non collasso e tempi delle 16 sonde a 2 epoche in `riepilogo_sonde.json` (`dec` P1 134-149 min, `dec` P2 24-28 min).

## Descrittivi ancora in coda

Endpoint a 1 epoca (`pool` e `dec`), pendenza 1 -> 2 epoche, R0-e1 contro C1 s0 (18,21 su P1) come controllo di coerenza; consuntivo `sacct` della
valutazione (16 estrazioni, 32 sonde) a code vuote.
