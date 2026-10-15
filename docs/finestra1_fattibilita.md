# Finestra 1: fattibilita', costi e calendario dei job (stima di Claude, 14/10/2026)

Punto 6 della numerazione unica (registro condiviso, `c4a0302`: Claude «memoria/throughput/LR-batch, costi e launcher»), sulla lista dei
bracci della bozza di GPT (`docs/finestra1_job_plan.md` nel suo worktree, non ancora firmata). Simone: «fattibilità dei job (6)».
**Stima, non preventivo firmato:** i numeri vanno sostituiti con le misure della prova `w1_gpu_dati_v1` (C) e coi passi per epoca ricalcolati
sul corpus e sulle quote correnti (nota di GPT) prima della firma (entro il 25/10 secondo la bozza di GPT).

## Basi della stima

| Grandezza | Valore | Fonte |
|---|---|---|
| Passi per epoca, batch 32 | ~23.150 (D = 4 epoche ~92.600 passi) | `decisioni.md`, warmup del WSD (09/10); da ricalcolare |
| s/passo 30M, DDP 4 GPU, batch 32, regole della finestra 1, tutti i dataset | **0,73** (misurato) | collaudo 60150683 (12/10) |
| s/passo 100M e 300M, DDP 4 GPU, batch 32 | ~1,32 e ~2,51 (stimati) | 0,73 x rapporti della sonda di memoria (8 finestre/GPU, caso peggiore Hyser 256 canali: 2,9/1,6 e 5,5/1,6) |
| Memoria per GPU, 8 finestre, caso peggiore | 19,8 / 31,7 / 50,4 GiB (30M / 100M / 300M) | sonda 60150685: niente accumulo nemmeno al 300M |

GPU-ora per epoca (un nodo, 4 GPU): **30M ~18,8; 100M ~34,0; 300M ~64,6**. Tempo di calendario per epoca su un nodo: 4,7 / 8,5 / 16,1 h.

## Costo per famiglia di job (bracci e semi della bozza di GPT)

| Famiglia | Contenuto | Run equivalenti | GPU-ora (stima) |
|---|---|---|---|
| Ablation a 30M (W1-OBJ, TARGET, KEEP, ID, K, LOCAL, BOTTLENECK, NEIGHBOR) | riferimento + 13 bracci, 2 semi, D = 4 epoche | 28 run x 4 epoche | **~2.100** |
| Calibrazione LR/batch 30–100M | griglia da fissare (segnaposto: 30M 3 LR x 2 batch x 1 epoca; 100M 2 x 2 x 0,5) | | ~180 |
| Pilot sulle epoche a 100M (W1-EPOCH) | tronco a 8 epoche + rami di decay, ~9,4 epoche per seme, 2 semi | | **~640** |
| Asse dei soggetti (W1-SUBJECT) | 12,5/25/50/100% a E = 4 nominali, 100M e 300M, 2 semi (il 100M al 100% condiviso col pilot) | 7,5 epoche piene per seme e taglia | **~1.210** (100M ~240, 300M ~970) |
| Controllo d'esposizione (25% agli stessi passi del 100%) | 4 epoche piene per seme; taglie da firmare | | ~270 (solo 100M) – ~790 (100M e 300M) |
| Valutazione P1/P2 dei checkpoint | ~60-65 checkpoint, estrazione e sonde sui nodi GPU come nel 2C | | ~200 |
| Prove su dbg prima della finestra | protocollo GPU/dati (<= 14) e collaudi dei launcher | | ~30 |
| **Totale** | | | **~4.600–5.200 GPU-ora (37.000–41.200 ore locali)** |

Contro il budget: il piano stima ~6.000–7.000 GPU-ora del FM a novembre e ~70.000–90.000 disponibili; la quota mensile dell'account e'
~20.500 GPU-ora (piano operativo §5). **Il costo sta nel budget**; il limite e' il calendario.

## Calendario

- **Nodo-ore:** ~1.100–1.250 (training su nodi interi). Con N nodi in parallelo: N = 4 -> ~12-13 giorni; N = 8 -> ~6-7; N = 16 -> ~3. La
  bozza di GPT vuole i risultati entro il 15/11 col lancio dal 01/11: **con 4 nodi non c'e' margine** per code, errori o run di conferma.
- **Limite dell'account da verificare:** il 14/10 alcune sonde del 2C sono rimaste in coda per `MaxNodePerAccount` con ~13 job da un quarto di
  nodo in esecuzione. Il tetto di nodi per account (condiviso con gli altri utenti del progetto) va letto (`sacctmgr`/`sshare`, comandi
  di sola lettura non ammessi oggi a leonardo-ops: serve il via di Simone) prima di firmare il calendario.
- **Catene da 24 h:** 30M a 4 epoche ~19 h (un job); 100M a 4 epoche ~34 h (2 job), tronco del pilot a 8 epoche ~68 h (3 job); 300M al 100%
  ~65 h (3 job). Ripresa esatta e rami di decay dal checkpoint pre-decay sono gia' nel trainer. **Cammino critico: tronco del pilot e 300M al
  100%, ~3 giorni ciascuno senza attese**: vanno lanciati per primi (come prevede il piano: «prima i job grandi»).
- **Valutazione:** la partizione CPU ammette 10 job in coda e 8 CPU in esecuzione per utente: ~60 checkpoint in serie richiederebbero
  settimane. Va fatta sui nodi GPU (come nel 2C) o con un accordo diverso; costo incluso sopra.
- **Dati e disco:** processato e checkpoint su `$SCRATCH`: rinfresco entro il 22/10 e poi il 12/11 (purge a 40 giorni). Checkpoint con
  optimizer ~0,45 / ~1,7 / ~4,7 GB (30M / 100M / 300M), piu' copie pre/post-decay e della fase stabile.

## Cosa manca perche' i job siano «pronti»

1. Misure: secondi per passo e memoria a 100M/300M in DDP coi dati veri, attesa sui dati <= 2%, equivalenza NCCL/bf16 (protocollo
   `w1_gpu_dati_v1`, da approvare).
2. Passi per epoca ricalcolati da manifest, quote e durate correnti; batch e picco del LR dalla calibrazione (che cambiano passi e throughput).
3. Bracci non ancora implementati (bozza di GPT): «soli target espliciti», «waveform MSE», identita' «solo geometria» e «gating della
   topologia», bottleneck a 2 blocchi; da implementare e collaudare entro il 31/10, con proprietari da concordare. Gia' configurabili: K,
   strati dell'encoder locale, vicini (`k_neighbors`), target a/b, keep.
4. Manifest annidati per soggetto (D16) e stratificazione: da verificare che esistano e siano congelati.
5. Launcher (miei): catena DDP da 24 h con ripresa e tempi per rung; tronco e rami del WSD (`branch_from`); catena di valutazione dei
   checkpoint sui nodi GPU; sottomissione in blocco con `submit_job.py` e dipendenze; ciascuno collaudato su dbg entro il 31/10.
6. Lettura del tetto di nodi dell'account e delle QoS (sopra).

## Aggiornamento con le misure della prova `w1_gpu_dati_v1` (15/10/2026)

s/passo misurati in DDP a 4 GPU col batch 32: **30M 0,707; 100M 1,390; 300M 2,769** (le stime erano 0,73 / ~1,32 / ~2,51). GPU-ora per
epoca: **18,2 / 35,8 / 71,2**. Il batch 64 costa per finestra quanto il 32 (nessun risparmio di throughput dal batch piu' grande).
Totale ricalcolato coi bracci della bozza di GPT: ablation ~2.040, pilot ~670, asse soggetti ~1.320 (100M ~250, 300M ~1.070), esposizione
~290 (solo 100M) – ~860 (100M e 300M), calibrazione ~180, valutazione ~200, prove ~30: **~4.730–5.300 GPU-ora**. Memoria: 300M a 8
finestre per GPU 28,7 GiB, a 16 (in 2 micro-batch) 37,1 GiB: nessun limite. Attesa sui dati al 30M 2,4-2,8% (sopra il 2% di D6a), sotto
il 2% a 100M e 300M.

Col prefetch (prova `w1_prefetch_v1`, 15/10/2026) il 30M scende a 0,701 s/passo (mediana, batch 32; media 0,706 contro 0,714) e l'attesa
sui dati a 0,67%: la stima della finestra 1 non cambia in modo apprezzabile (~1% sulle ablation al 30M).

**Passi per epoca dal corpus (15/10/2026, `docs/manifest_annidati_d16.md`):** 24.528 col batch 32 (finestra media 3,77 s) invece di ~23.150:
D = 4 epoche = 98.112 passi. I costi dei run a D salgono del ~6%: la stima della finestra 1 diventa **~5.000–5.600 GPU-ora** (da ~4.730–5.300).
Asse dei soggetti: le frazioni reali delle ore sono 13,8 / 26,1 / 51,0% (passi per epoca 3.412 / 6.403 / 12.478).

## Calendario con 8 nodi (15/10/2026; Simone: «puoi usare anche 8 nodi in parallelo ... io ti autorizzo se cineca lo consente»)

Simulazione `scripts/finestra1_calendario.py`: run della bozza di GPT (ablation 30M 28 run a D, pilot 100M con i rami, asse dei soggetti
100M e 300M, controllo d'esposizione), passi per epoca dal corpus, s/passo misurati col prefetch, +3% per avvio, valutazioni e
checkpoint; calibrazione (segnaposto) prima di tutto; i rami partono quando il tronco ha superato il loro passo; i run piu' lunghi per
primi. Le pause di coda fra i job di una catena non sono contate.

| Nodi | Controllo solo 100M (~5.100 GPU-ora) | Controllo 100M e 300M (~5.700 GPU-ora) |
|---|---|---|
| 4 | 13,4 giorni | 15,0 giorni |
| 8 | **6,8 giorni** | **7,6 giorni** |
| 12 | 4,5 giorni | 5,3 giorni |

Il run piu' lungo e' il tronco del pilot (78 h, 4 job da 24 h): con 8 nodi il calendario e' limitato dal riempimento dei nodi, non dalla
catena piu' lunga. Lancio dal 01-03/11 -> training finito verso il 09-11/11 con 8 nodi, margine per code, errori e run di conferma prima del
15/11; con 4 nodi nessun margine. Il limite dell'account non e' ancora letto: il 15/10 nella coda dell'account c'era un solo altro utente
con un nodo; il 14/10 alcune sonde aspettavano per `MaxNodePerAccount`.

## Aggiornamento dopo l'accordo del 15/10/2026 (QoS normal, fino a 8 nodi; `scripts/finestra1_calendario.py`)

Passi dai manifest derivati con un solo floor sul totale: D = 4 epoche col batch 32 = **98.094** passi al 100% (49.926 / 25.628 / 13.625 a
50 / 25 / 12,5%). Calibrazione C1-C4 dell'accordo al posto del segnaposto (157 GPU-ora con avvii; `docs/calibrazione_c1_c4_piano.md`), in
sequenza prima dei run che ne usano il LR. Il **tempo di calcolo** e le **attese** sono separati: `attesa per job` = coda prima di ogni job da
24 h (una catena da k job la paga k volte), che dipende anche dal consumo degli altri 16 utenti del progetto (il 14/10 un altro utente ha
usato 2.655 GPU-ora in 30 ore) e dalla priorita' legata alla quota mensile; le attese delle decisioni di GPT fra le fasi della calibrazione
sono a parte (non stimate).

| Nodi (QoS normal) | Attesa per job | Controllo solo 100M (~5.070 GPU-ora) | Controllo 100M e 300M (~5.700 GPU-ora) |
|---|---|---|---|
| 8 | 0 h | 7,2 giorni | 8,0 giorni |
| 8 | 1 h | 7,8 giorni | 8,6 giorni |
| 8 | 3 h | 8,9 giorni | 9,8 giorni |
| 4 | 0 h | 13,6 giorni | 15,2 giorni |
| 4 | 3 h | 16,3 giorni | 18,1 giorni |

Calibrazione: ~20-32 h di calcolo e code piu' le decisioni; se si fa entro fine ottobre la campagna parte il 01/11 e con 8 nodi finisce
verso il 09-12/11 (prima del 15/11 anche con 3 h di attesa per job); con 4 nodi no. Gli 8 nodi sono ammessi dalla QoS normal (sacctmgr,
15/10); la loro disponibilita' reale dipende dalla coda.

## Calibrazione C1-C4 con la valutazione (15/10/2026)

Dal pianificatore (`scripts/calibration_plan.py`, `docs/calibrazione_c1_c4_selezione.md`): training 160,2 GPU-ora attese / 208,0 tetto;
estrazioni per la selezione 2,6 / 6,5 (1 GPU per checkpoint: 30 min al 30M, 1 h al 100M, 2 h al 300M di tetto); sonde e selezione su CPU
seriale fino a ~199 core-ora. **Totale ~163 / 214,5 GPU-ora.** Calendario: C1 in un'ondata (3 nodi, ~5 h) piu' valutazione (~1,5 h), poi
decisione e via di Simone per ogni fase: ~4 ondate da ~6-7 h di calcolo piu' code e decisioni, 2-3 giorni. La lista completa dei job della
finestra 1 (nucleo prioritario, altri bracci, parti mancanti) dipende dai bracci scientifici del punto 5, non decisi qui.
