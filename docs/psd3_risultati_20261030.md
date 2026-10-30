# PSD-3: risultati della matrice B/R3c × pool/dec × logistica/MLP (30/10/2026)

**Stato:** ESEGUITO il 29-30/10/2026 su mandato di Simone («ti autorizzo a partire», 29/10) e prompt di GPT
`docs/prompt_claude_psd3_dopo_sonde_20261029.md` (73b6fba). Protocollo scientifico di GPT: `docs/psd3_readers_20261029.md`
(codice d4092b8); lettura descrittiva, nessuna soglia di adozione, nessun via automatico a R3 completo. Questo documento
e' la consegna di Claude richiesta dal §3 del prompt. Autore: Claude; verifica: GPT; firma: Simone.

**Identita' dell'esecuzione.** Checkout di campagna `pilot-psd3-20261029b`, codice `2235c15106a61066c035da95648971219e34dce7`
(deploy `deploy/pilot-psd3-20261029b`); piano `b55b6562f3b7…`, ricetta `c7b47289f667…` unica per i 20 shard e per il confronto;
protocollo `psd3-frozen-readers-20261029-v1`, confronto `psd3_paired_subject_bootstrap_v1`. Checkpoint congelati: B `96f87035520e291e…`,
R3c `ee78d2a962643222…`. Ambiente: Python 3.11.6, numpy 1.26.4, scipy 1.12.0, scikit-learn 1.4.1.post1, torch 2.2.0a0; 4 thread
BLAS/OpenMP/torch in tutti gli shard. Job SLURM: checkout 62769337; inventario 62769884; benchmark 62770045, 62770053, 62770089,
62770102; shard B 62822106-62822129 (10); shard R3c 62871535-62871562 (10); confronto 62946283. Tutti COMPLETED 0:0.

**File.** Piano e comandi: `results/infra/psd3_20261029/{piano_job.json,comandi.txt}`. Risultati nel repo (nessun array di feature o
predizioni): `results/infra/psd3_20261029/{compare.json,report_completo.json,report_completo.md,inventory.json,sacct_psd3_20261030.txt}`.
Shard completi (predizioni OOF, audit per fit, curve): `$WORK/wearusfm_runs/results/psd3_20261029/shards/*.json` (57 MB) e copia in
`~/wearusfm_local/psd3_20261029/shards/` sul Mac. Rapporto generato con `scripts/psd3_report.py` (70447ff, 64600eb); i contrasti
vengono soltanto dal `compare.json` di GPT (`harness/psd3_compare.py`), con controllo incrociato del punto: scarto massimo 0.

## 1. Esito del controllo delle sonde e adeguamenti a PSD-3

Dettagli in `docs/sonde_convergenza_20261029.md` §8. A 12.000 iterazioni nessun fit al limite (280 record; i 18 fit prima al limite
convergono per criterio fra 3.160 e 4.823 iterazioni); B−R3c P1 −0,553 [−1,105; −0,017] (era −0,556 [−1,104; −0,026]),
P2 −1,172 [−2,956; +0,651] (era −1,233 [−2,997; +0,576]); differenza dei delta P1 +0,004 [−0,002; +0,013], P2 +0,061 [+0,001; +0,120].
La lettura congiunta di B19618 resta valida. Regola §1.3 del prompt: nessun warning residuo, quindi **nessun controllo mirato e nessun
emendamento di `max_iter`** in PSD-3. Nessuna altra correzione alla ricetta: viste, PCA, split, griglie, epoche, semi e metriche sono
quelli del protocollo. Unico adeguamento, operativo e congelato prima dei blocchi 3-5 (decisioni, 29/10): tetti di walltime per dataset
(DB2 4 h, EPN612 3 h, DB6 2 h, DB3 2 h, UCI 30 min; 4 core e 30 G per shard), entro il residuo e senza toccare la ricetta.

## 2. Esecuzione

- 27 job sulla QoS seriale (`lrd_all_serial`), 0 GPU-ora; sequenza: checkout e inventario (29/10 17:31), 4 benchmark (17:42),
  blocco 3 = 10 shard B (29/10 23:45 -> 30/10 06:40), blocco 4 = 10 shard R3c (06:42 -> 13:50), confronto (17:02, 18 s).
  Gli shard sono girati uno alla volta: la coda li ha trattenuti per il limite di memoria per utente (`QOSMaxMemoryPerUser`, 30 G
  richiesti ciascuno), non per i core.
- Nessun TIMEOUT: l'uso massimo del tetto e' il 42 % (R3c DB2 dec, 1:41:36 su 4 h). MaxRSS massimo 4,3 GB (R3c DB6 dec) contro 30 G
  richiesti: la richiesta di memoria era sovradimensionata di circa sette volte (vedi §9).
- Elapsed per shard (B / R3c): DB2 pool 1:27:40 / 1:26:21, DB2 dec 1:34:12 / 1:41:36, DB3 pool 25:05 / 25:29, DB3 dec 36:23 / 38:18,
  DB6 pool 31:29 / 29:31, DB6 dec 32:56 / 33:13, EPN612 pool 49:21 / 51:08, EPN612 dec 59:02 / 1:00:07, UCI pool 0:54 / 0:59,
  UCI dec 0:54 / 0:55. I due encoder costano lo stesso: la differenza e' nella vista (dec piu' lenta) e nel dataset.
- Secondi di fit sommati sui quattro fold, per cella: la MLP domina (DB2 4.436-4.697 s, EPN612 2.372-2.522 s, DB6 1.616-1.755 s,
  DB3 1.127-1.213 s, UCI ~40 s) contro la logistica (DB2 dec 1.134-1.311 s, pool 550-568 s; DB3 dec 1.022-1.061 s; EPN612 dec 974-1.014 s;
  UCI ~1 s) e la PCA (1-75 s).
- Bootstrap del confronto: 2.000 campioni di soggetti per dataset, seme 20261029, stesso campione per tutte le celle e i semi; in nessun
  dataset ci sono estrazioni con classi mancanti o con una sola classe; nessuna predizione fuori dalle classi osservate.
- Campione: DB2 32 soggetti, 38.544 finestre OOF, 49 classi; DB3 8 soggetti, 9.834, 49 classi; DB6 8 soggetti, 33.961, 7 classi;
  EPN612 314 soggetti, 47.100, 6 classi; UCI 19 soggetti, 364, 6 classi.

## 3. BAcc OOF per dataset e cella (punti percentuali, IC95 percentile per soggetto)

Logistica (C scelto in validazione, riaddestrato sul solo train):

| dataset | B/pool | B/dec | R3c/pool | R3c/dec |
|---|---:|---:|---:|---:|
| ninapro_db2 | 17,37 [15,50; 19,35] | 18,30 [16,24; 20,45] | 18,27 [16,22; 20,37] | 18,43 [16,04; 20,82] |
| ninapro_db3 | 2,69 [1,81; 3,82] | 2,64 [1,96; 3,54] | 2,43 [1,99; 3,06] | 3,06 [2,69; 3,50] |
| ninapro_db6 | 33,73 [30,62; 36,68] | 30,57 [27,88; 33,03] | 34,38 [31,79; 37,06] | 31,82 [29,63; 34,09] |
| epn612 | 76,36 [75,01; 77,71] | 75,95 [74,58; 77,37] | 80,35 [79,01; 81,66] | 81,40 [80,04; 82,79] |
| uci_emg | 77,15 [69,45; 84,50] | 78,77 [69,68; 86,38] | 77,82 [69,03; 85,18] | 75,44 [65,21; 83,79] |

MLP 128→32→classi (media delle BAcc dei due semi, non ensemble; fra parentesi seme 0 / seme 1):

| dataset | B/pool | B/dec | R3c/pool | R3c/dec |
|---|---:|---:|---:|---:|
| ninapro_db2 | 16,74 [14,97; 18,56] (16,74 / 16,74) | 17,44 [15,58; 19,37] (17,55 / 17,32) | 17,68 [15,72; 19,66] (17,48 / 17,89) | 17,65 [15,46; 19,82] (17,61 / 17,70) |
| ninapro_db3 | 2,52 [1,96; 3,25] (2,47 / 2,57) | 3,26 [2,92; 3,68] (3,10 / 3,43) | 2,67 [2,09; 3,31] (2,23 / 3,11) | 2,94 [2,36; 3,69] (3,22 / 2,66) |
| ninapro_db6 | 32,75 [29,37; 35,84] (32,54 / 32,97) | 29,73 [27,09; 32,34] (29,92 / 29,55) | 33,16 [30,28; 35,94] (33,34 / 32,99) | 30,26 [28,45; 32,36] (30,13 / 30,38) |
| epn612 | 77,25 [75,93; 78,59] (77,38 / 77,12) | 76,77 [75,35; 78,18] (76,76 / 76,77) | 80,70 [79,33; 82,02] (80,72 / 80,68) | 81,38 [80,01; 82,73] (81,44 / 81,32) |
| uci_emg | 77,65 [70,32; 84,06] (78,42 / 76,88) | 76,15 [68,05; 83,11] (76,90 / 75,39) | 79,08 [71,72; 85,89] (79,97 / 78,19) | 76,77 [67,89; 84,18] (78,11 / 75,43) |

Caso: DB2 e DB3 2,04 (49 classi), DB6 14,3 (7), EPN612 e UCI 16,7 (6). DB3 e' al livello del caso in tutte le celle (2,4-3,3; con 8
soggetti e 49 classi i limiti inferiori degli IC sono 1,8-2,9). UCI ha IC larghi 14-18 punti (19 soggetti, 364 finestre). I due semi
della MLP differiscono al massimo di 1,5 punti (UCI), di meno di 0,3 su EPN612.

## 4. Aggregati P1 e P2

P1 = 8/13 DB2 + 3/13 DB3 + 2/13 DB6; P2 = 1/2 EPN612 + 1/2 UCI. IC95 dal bootstrap congiunto.

| cella | P1 | P2 |
|---|---:|---:|
| B/pool/logistica | 16,50 [15,24; 17,82] | 76,75 [72,89; 80,54] |
| B/pool/MLP | 15,92 [14,74; 17,17] | 77,45 [73,79; 80,82] |
| B/dec/logistica | 16,58 [15,18; 17,95] | 77,36 [72,74; 81,21] |
| B/dec/MLP | 16,06 [14,75; 17,31] | 76,46 [72,39; 80,09] |
| R3c/pool/logistica | 17,09 [15,72; 18,47] | 79,09 [74,78; 82,85] |
| R3c/pool/MLP | 16,60 [15,25; 17,97] | 79,89 [76,12; 83,43] |
| R3c/dec/logistica | 16,94 [15,43; 18,46] | 78,42 [73,29; 82,47] |
| R3c/dec/MLP | 16,20 [14,80; 17,59] | 79,07 [74,64; 82,84] |

## 5. Contrasti appaiati fissati (differenze in punti, IC95 percentile appaiato per soggetto)

Gli IC che non contengono lo zero sono segnati con `*`; sono una lettura descrittiva, senza correzione per molteplicita' (14 contrasti per
dataset e per aggregato, 98 in tutto) e senza soglia di adozione, come fissato dal protocollo.

### 5.1 B − R3c (per vista e lettore)

| | pool/logistica | pool/MLP | dec/logistica | dec/MLP |
|---|---:|---:|---:|---:|
| ninapro_db2 | −0,91 [−1,51; −0,27] * | −0,95 [−1,57; −0,27] * | −0,12 [−0,91; +0,61] | −0,22 [−0,82; +0,40] |
| ninapro_db3 | +0,27 [−0,89; +1,58] | −0,15 [−0,88; +0,68] | −0,42 [−1,03; +0,24] | +0,32 [−0,22; +0,70] |
| ninapro_db6 | −0,65 [−1,89; +0,46] | −0,41 [−1,74; +1,03] | −1,25 [−2,98; +0,50] | −0,52 [−2,40; +1,48] |
| epn612 | −3,99 [−4,69; −3,30] * | −3,45 [−4,05; −2,88] * | −5,45 [−6,05; −4,85] * | −4,61 [−5,14; −4,03] * |
| uci_emg | −0,67 [−3,85; +2,37] | −1,43 [−3,71; +0,83] | +3,34 [−0,15; +7,08] | −0,62 [−4,17; +2,98] |
| **P1** | −0,60 [−1,10; −0,10] * | −0,68 [−1,14; −0,18] * | −0,37 [−0,95; +0,16] | −0,14 [−0,63; +0,35] |
| **P2** | −2,33 [−4,00; −0,72] * | −2,44 [−3,65; −1,27] * | −1,05 [−2,78; +0,84] | −2,62 [−4,37; −0,78] * |

### 5.2 dec − pool (per modello e lettore)

| | B/logistica | B/MLP | R3c/logistica | R3c/MLP |
|---|---:|---:|---:|---:|
| ninapro_db2 | +0,94 [+0,13; +1,78] * | +0,70 [−0,19; +1,53] | +0,15 [−0,89; +1,25] | −0,03 [−0,72; +0,70] |
| ninapro_db3 | −0,05 [−1,45; +1,13] | +0,75 [+0,01; +1,40] * | +0,64 [+0,29; +0,93] * | +0,27 [−0,36; +1,02] |
| ninapro_db6 | −3,16 [−4,44; −1,98] * | −3,02 [−3,95; −2,01] * | −2,56 [−3,99; −0,95] * | −2,91 [−4,41; −1,05] * |
| epn612 | −0,40 [−1,11; +0,27] | −0,48 [−1,06; +0,09] | +1,05 [+0,42; +1,63] * | +0,68 [+0,13; +1,22] * |
| uci_emg | +1,62 [−2,06; +5,06] | −1,50 [−4,27; +0,96] | −2,39 [−7,18; +2,26] | −2,31 [−7,52; +2,74] |
| **P1** | +0,08 [−0,55; +0,67] | +0,14 [−0,45; +0,69] | −0,15 [−0,84; +0,58] | −0,41 [−0,93; +0,14] |
| **P2** | +0,61 [−1,25; +2,41] | −0,99 [−2,40; +0,30] | −0,67 [−3,08; +1,67] | −0,82 [−3,53; +1,69] |

### 5.3 MLP − logistica (per modello e vista)

| | B/pool | B/dec | R3c/pool | R3c/dec |
|---|---:|---:|---:|---:|
| ninapro_db2 | −0,63 [−1,14; −0,10] * | −0,87 [−1,31; −0,41] * | −0,59 [−1,09; −0,04] * | −0,77 [−1,37; −0,18] * |
| ninapro_db3 | −0,18 [−1,03; +0,60] | +0,62 [−0,17; +1,21] | +0,24 [−0,20; +0,60] | −0,12 [−0,46; +0,31] |
| ninapro_db6 | −0,98 [−2,97; +0,91] | −0,84 [−1,93; +0,16] | −1,22 [−3,15; +0,83] | −1,56 [−3,54; +0,11] |
| epn612 | +0,89 [+0,41; +1,38] * | +0,81 [+0,32; +1,33] * | +0,35 [−0,13; +0,80] | −0,02 [−0,51; +0,47] |
| uci_emg | +0,50 [−2,13; +2,94] | −2,63 [−5,59; +0,69] | +1,26 [−2,58; +4,93] | +1,33 [−1,86; +4,40] |
| **P1** | −0,58 [−1,03; −0,11] * | −0,52 [−0,89; −0,16] * | −0,49 [−0,95; −0,02] * | −0,75 [−1,24; −0,27] * |
| **P2** | +0,70 [−0,64; +1,97] | −0,91 [−2,42; +0,76] | +0,80 [−1,15; +2,66] | +0,66 [−0,90; +2,18] |

### 5.4 Interazione (MLP − logistica in dec) − (MLP − logistica in pool), per modello

| | B | R3c |
|---|---:|---:|
| ninapro_db2 | −0,24 [−0,80; +0,30] | −0,19 [−0,97; +0,61] |
| ninapro_db3 | +0,80 [−0,37; +1,90] | −0,36 [−0,86; +0,37] |
| ninapro_db6 | +0,15 [−1,20; +1,52] | −0,35 [−1,95; +1,04] |
| epn612 | −0,08 [−0,75; +0,55] | −0,37 [−0,99; +0,24] |
| uci_emg | −3,13 [−6,23; +0,08] | +0,07 [−4,01; +4,46] |
| **P1** | +0,06 [−0,42; +0,52] | −0,25 [−0,82; +0,31] |
| **P2** | −1,60 [−3,17; +0,05] | −0,15 [−2,20; +2,05] |

## 6. Convergenza e selezione

- **Logistica.** 480 fit candidati (20 celle × 4 fold × 6 C) e 80 riaddestramenti. Al limite di 12.000 iterazioni: 2 candidati, entrambi
  C = 100 in B/DB3/dec (fold 0 e 2), **non selezionati** (in quei fold e' scelto C = 0,001, convergente a 291 e 336 iterazioni); nessun
  riaddestramento al limite. Iterazioni massime per C: 0,001 -> 1.053; 0,01 -> 1.574; 0,1 -> 2.826; 1 -> 3.773; 10 -> 6.398; 100 -> 12.000.
  C scelti sugli 80 fold-cella: 0,001 ×28, 0,01 ×19, 0,1 ×9, 1 ×12, 10 ×8, 100 ×4 (regolarizzazione forte prevalente: lo spazio PCA128 e'
  gia' compresso). Nessun fit al limite entra nelle celle lette: il punto 3 del §1 del prompt non si applica.
- **MLP.** 60 epoche fisse (budget, non prova di convergenza), lambda scelte: 0,01 ×49, 0,001 ×20, 0 ×11; semi 0/1 conservati entrambi.
  Divario medio train − validazione 28,3 punti di BAcc (sovra-adattamento su tutti i dataset piccoli: DB3 train 39-66 contro val 2-3;
  UCI train 94-100 contro val 77-82). Curve della train loss (media su fold e semi della lambda scelta) alle epoche 1 / 10 / 30 / 60:

| cella | ep. 1 | ep. 10 | ep. 30 | ep. 60 | train BAcc | val BAcc |
|---|---:|---:|---:|---:|---:|---:|
| B/epn612/pool | 1,466 | 0,546 | 0,426 | 0,380 | 89,0 | 76,5 |
| B/epn612/dec | 1,438 | 0,552 | 0,484 | 0,464 | 88,9 | 76,1 |
| R3c/epn612/pool | 1,447 | 0,420 | 0,308 | 0,261 | 92,7 | 80,0 |
| R3c/epn612/dec | 1,377 | 0,462 | 0,411 | 0,398 | 91,8 | 80,5 |
| B/ninapro_db2/pool | 3,903 | 2,757 | 2,508 | 2,421 | 36,8 | 19,7 |
| B/ninapro_db2/dec | 4,101 | 2,430 | 2,243 | 2,185 | 45,0 | 19,5 |
| R3c/ninapro_db2/pool | 3,913 | 2,708 | 2,470 | 2,395 | 39,2 | 20,0 |
| R3c/ninapro_db2/dec | 4,104 | 2,308 | 2,107 | 2,042 | 49,0 | 19,6 |
| B/ninapro_db3/pool | 4,094 | 3,487 | 2,822 | 2,531 | 39,2 | 2,7 |
| B/ninapro_db3/dec | 4,559 | 2,990 | 2,071 | 1,728 | 56,2 | 3,1 |
| R3c/ninapro_db3/pool | 4,095 | 3,503 | 2,846 | 2,559 | 39,7 | 2,8 |
| R3c/ninapro_db3/dec | 4,523 | 2,854 | 1,823 | 1,433 | 61,5 | 2,1 |
| B/ninapro_db6/pool | 1,837 | 1,105 | 0,980 | 0,933 | 71,7 | 35,4 |
| B/ninapro_db6/dec | 1,876 | 0,900 | 0,747 | 0,693 | 78,5 | 32,0 |
| R3c/ninapro_db6/pool | 1,846 | 1,026 | 0,884 | 0,831 | 75,7 | 36,0 |
| R3c/ninapro_db6/dec | 1,871 | 0,874 | 0,731 | 0,681 | 80,2 | 33,2 |
| B/uci_emg/pool | 2,001 | 1,633 | 1,037 | 0,521 | 93,6 | 77,9 |
| B/uci_emg/dec | 2,260 | 1,499 | 0,597 | 0,216 | 99,5 | 81,9 |
| R3c/uci_emg/pool | 1,928 | 1,574 | 0,991 | 0,483 | 95,5 | 77,2 |
| R3c/uci_emg/dec | 2,149 | 1,402 | 0,530 | 0,174 | 99,5 | 80,9 |

  La loss di train scende ancora fra l'epoca 30 e la 60 in tutte le celle (di poco su EPN612 e DB2, molto su UCI e DB3): la dose del lettore
  e' un budget fisso, non uno stato convergente. Con batch 512 il numero di passi per epoca varia di due ordini di grandezza fra i dataset
  (UCI: train di ~220-250 finestre, un passo per epoca, 60 passi in tutto; EPN612: ~60 passi per epoca). Nessun gradiente non finito.
- **Avvisi di metrica.** 26 per cella nelle quattro celle di DB3 (sommati su fold, logistica e MLP), zero negli altri dataset: soggetti
  tenuti fuori senza alcune classi, come gia' visto nelle sonde, identici per encoder e vista. Nessuna predizione fuori dalle classi
  osservate nel confronto.

## 7. Varianza spiegata dalla PCA128 (train-only, media dei quattro fold; cumulata a 16 / 32 / 64 / 128 componenti)

| dataset | B/pool | B/dec | R3c/pool | R3c/dec |
|---|---|---|---|---|
| ninapro_db2 | 85,3 / 94,9 / 98,4 / 99,6 | 81,9 / 90,6 / 94,9 / 97,4 | 86,5 / 95,2 / 98,6 / 99,6 | 77,4 / 89,1 / 95,1 / 97,7 |
| ninapro_db3 | 90,7 / 96,4 / 98,9 / 99,7 | 88,0 / 93,7 / 96,6 / 98,3 | 92,2 / 96,9 / 99,1 / 99,7 | 85,9 / 92,9 / 96,7 / 98,5 |
| ninapro_db6 | 88,4 / 95,8 / 98,6 / 99,6 | 87,3 / 93,4 / 96,6 / 98,4 | 89,4 / 95,9 / 98,8 / 99,6 | 84,8 / 93,0 / 96,8 / 98,6 |
| epn612 | 95,6 / 98,6 / 99,6 / 99,9 | 93,2 / 96,9 / 98,6 / 99,4 | 96,1 / 98,7 / 99,6 / 99,9 | 91,8 / 96,4 / 98,5 / 99,4 |
| uci_emg | 92,7 / 97,6 / 99,4 / 99,9 | 90,3 / 95,5 / 98,2 / 99,5 | 93,4 / 97,8 / 99,5 / 99,9 | 89,0 / 95,2 / 98,2 / 99,6 |

La vista pool (768 dimensioni) e' quasi interamente rappresentata (99,4-99,9 %); la vista dec (C × 384, fino a 4.608 dimensioni in DB2)
perde il 0,4-2,6 % della varianza, di piu' su DB2 (97,4-97,7 %). La PCA puo' eliminare direzioni discriminanti a bassa varianza: per dec
la compressione e' piu' severa, e il confronto dec−pool ne risente in modo non quantificato qui.

## 8. Confronto indicativo con le sonde pool768 (baseline distinte, non un contrasto appaiato)

Le sonde storiche (logistica su pool768, fase B a 12.000 iterazioni) danno S P1 B 16,995, R3c 17,547; P2 B 79,613, R3c 80,785. Le celle
pool/logistica di PSD-3 (PCA128) danno P1 B 16,50, R3c 17,09; P2 B 76,75, R3c 79,09: la compressione a 128 costa circa 0,5 punti in P1 e
1,7-2,9 punti in P2 a entrambi gli encoder. Ricette diverse (griglia, standardizzazione, dimensione): e' un'indicazione dell'ordine di
grandezza del costo della PCA, non un contrasto fissato; non si attribuisce alla MLP nulla che dipenda da questo preprocessing.

## 9. Costi (consuntivo)

| voce | job | core-ore | elapsed |
|---|---:|---:|---:|
| checkout di campagna | 1 | 0,01 | 18 s (2 core) |
| inventario cache | 1 | 0,01 | 24 s (2 core) |
| benchmark (B, fold 0, 4 celle) | 4 | 0,34 | 5 min (4 core) |
| shard B (blocco 3) | 10 | 27,86 | 6,97 h |
| shard R3c (blocco 4) | 10 | 28,51 | 7,13 h |
| confronto | 1 | 0,02 | 18 s (4 core) |
| **totale** | **27** | **56,75** | **~14,2 h di coda** |

0 GPU-ora. Tetto approvato 195,33 + 0,33 di checkout: consuntivo al 29 %. Residuo CPU-only della campagna: 432,14 − ~55 (sonde)
− 56,75 (PSD-3) ≈ 320 core-ore. Due note operative: (a) i 30 G richiesti per shard erano sette volte il MaxRSS misurato (≤ 4,3 GB) e sono
la ragione per cui la QoS ha fatto girare un solo shard alla volta: con ~8 G si sarebbero sovrapposti piu' shard; (b) il confronto e'
partito tre ore dopo la fine del blocco 4 perche' il limite di 10 job in coda per utente non permetteva di accodarlo con `afterok` insieme
al blocco 4, e l'operatore attendeva un messaggio: per le prossime catene il messaggio di fine blocco va sottoscritto (vedi registro).
Fonte: `results/infra/psd3_20261029/sacct_psd3_20261030.txt`.

## 10. Limiti

- Unico pretraining per encoder: gli IC sono condizionali ai checkpoint e ai lettori selezionati, non coprono la variabilita' del
  pretraining ne' della procedura di selezione. I semi 0/1 della MLP misurano la sensibilita' dell'ottimizzazione.
- PCA128 train-only: capacita' d'ingresso comune, ma direzioni discriminanti a bassa varianza possono essere perse, in modo diverso per
  pool e dec (§7); il costo indicativo rispetto a pool768 e' di 0,5-2,9 punti (§8).
- Dose del lettore: MLP 32 unita', 60 epoche fisse con loss ancora in discesa, lambda su tre valori; nessuna ricerca su architettura,
  dimensione o epoche (per protocollo). Un esito negativo della MLP vale per questa dose.
- dec e pool cambiano anche strato e natura della rappresentazione: dec−pool non isola la media dei canali; nessuna delle due viste
  conserva l'asse temporale (cache gia' mediate nel tempo).
- Statistica descrittiva: 98 contrasti senza correzione per molteplicita'; gli asterischi del §5 non sono decisioni. Con 2.000 repliche,
  un IC percentile ha un'incertezza propria dell'ordine di 0,05-0,1 punti sui limiti.
- DB3 al caso in ogni cella (peso 3/13 in P1) e UCI con IC di 14-18 punti (peso 1/2 in P2): P1 e' di fatto letto su DB2, P2 su EPN612.

## 11. Conclusioni sostenibili (lettura descrittiva, nelle categorie fissate dal protocollo)

1. **B − R3c.** Lo svantaggio di B19618 rispetto a R3c nella classificazione interna, visto con le sonde pool768, si ripresenta con la
   logistica PCA128 (P1 pool −0,60 [−1,10; −0,10]; P2 pool −2,33 [−4,00; −0,72]) e **non viene recuperato** ne' dal lettore non lineare
   (P1 pool/MLP −0,68 [−1,14; −0,18]; P2 dec/MLP −2,62 [−4,37; −0,78]) ne' dalla vista che conserva i canali (P2 dec/MLP sopra; P1 dec
   compatibile con zero, −0,37 e −0,14). La sorgente e' EPN612 (−3,5 … −5,5 punti in tutte e quattro le celle, IC stretti) e, in P1,
   DB2 pool (−0,9); DB6, DB3 e UCI sono compatibili con zero. Nella vista dec di DB2 il divario si chiude (−0,12 e −0,22).
2. **MLP − logistica.** In P1 la MLP e' **sotto** la logistica in tutte e quattro le celle (−0,49 … −0,75, IC sotto lo zero), guidata da
   DB2; in P2 e' compatibile con zero (EPN612 B +0,8/+0,9 con IC sopra lo zero, R3c nullo, UCI largo). L'esito «favorevole alla MLP»
   previsto dal protocollo come indicazione di accessibilita' non lineare nelle viste compresse **non si osserva**, con i limiti di dose,
   epoche e PCA dichiarati.
3. **dec − pool.** Negli aggregati tutti e otto i contrasti sono compatibili con zero (P1 da −0,41 a +0,14; P2 da −0,99 a +0,61). Per dataset
   il segno dipende dal dataset: DB6 perde 2,6-3,2 punti con dec in tutte le celle; DB2 guadagna con B/logistica (+0,94 [+0,13; +1,78]);
   EPN612 guadagna solo con R3c (+1,05, +0,68). La «motivazione a una successiva ablazione che isoli il pooling» prevista dal protocollo
   **non emerge** dagli aggregati; emerge semmai che dec non e' una vista uniformemente migliore.
4. **Interazione.** Tutti i 14 contrasti compatibili con zero (P2 B −1,60 [−3,17; +0,05], tutto da UCI): il vantaggio relativo della MLP
   non cambia fra le due viste.
5. **Categoria del protocollo: «nessun guadagno»**: le due ipotesi (lettura piu' flessibile; vista che conserva i canali) escono ristrette
   per queste viste, questi lettori e questa dose. La regressione e l'inseguimento della posa restano aperti, non toccati da questa prova.
   Per le regole del prompt, **nessun esito di PSD-3 autorizza R3 completo, modifiche del FM o un braccio di nuove loss.**

## 12. Domande aperte

- Perche' il divario B−R3c e' concentrato su EPN612 (314 soggetti, 6 classi, BAcc piu' alta) ed e' insensibile a lettore e vista, mentre su
  DB6 e UCI e' nullo e su DB2 compare solo in pool? E' una proprieta' del pretraining di B19618 (obiettivo di posa) sul tipo di segnale di
  EPN612, non di come lo si legge.
- La MLP sotto la logistica in P1 e' effetto della dose (32 unita', 60 passi su UCI, loss in discesa) o della rappresentazione? Non e'
  decidibile senza cambiare la ricetta, che il protocollo non consente in questa prova.
- Quanto della differenza dec−pool su DB6 (−3 punti) e' la compressione PCA piu' severa della vista dec (98,4-98,6 % contro 99,6 %) e quanto
  e' lo strato? Serve un controllo che separi la dimensione dall'origine della vista.
- Il costo indicativo di PCA128 rispetto a pool768 (0,5 punti in P1, 1,7-2,9 in P2) e' lo stesso per i due encoder? Qui e' simile in P1 e
  maggiore per B in P2, ma e' un confronto fra ricette diverse.

## 13. Proposta successiva (per GPT, che decide; firma di Simone)

Nessun lancio e' proposto come automatico. In ordine di costo:

1. **Chiusura descrittiva di PSD-3** con questa lettura nel registro e in `docs/decisioni.md` (fatto in questa consegna, in attesa della
   verifica di GPT): nessun guadagno da lettore non lineare o vista dec a encoder congelato, svantaggio di B su EPN612 confermato.
2. **Controllo del costo della PCA** (solo se GPT lo ritiene utile): celle logistica su pool768 nella stessa pipeline PSD-3 (stessi fold,
   stesso bootstrap, contrasto appaiato PCA128−pool768), senza MLP: ordine di 10 core-ore CPU, 0 GPU-ora. Trasformerebbe il confronto
   indicativo del §8 in un contrasto fissato e direbbe quanto PSD-3 e' limitato dalla compressione.
3. **Passo temporale** gia' previsto come distinto dal protocollo (lettore su feature con asse temporale, cache T × 384 della posa o nuova
   inferenza dai checkpoint congelati): da disegnare e preventivare da GPT; non e' un'estensione di questa prova.

Operativo, senza costi: ridurre la memoria richiesta per shard (30 G -> 8 G) nelle prossime campagne seriali e sottoscrivere il segnale di
fine blocco dell'operatore, cosi' da sovrapporre gli shard e non perdere ore di coda fra un blocco e il successivo.
