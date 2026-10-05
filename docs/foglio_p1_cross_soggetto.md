# Foglio di firma — P1, sonda cross-soggetto sul ramo sparso (bozza di AG del 05/10/2026; da firmare)

**Perche'.** D12 (firmata il 04/10) usa due metriche primarie per scegliere E dal pilot; P1 e' definita solo a grandi linee: «sonda lineare
sull'encoder congelato, accuratezza bilanciata sui soggetti di test della classe A (split firmati)». Per un numero servono dataset, etichette,
finestre, soggetti della sonda e il modo di combinare i dataset. Proposte qui sotto, marcate; codice gia' scritto e provato su dati sintetici
(`src/wearusfm/harness/p1_cross_subject.py`, `scripts/probe_p1.py`, job `scripts/slurm/probe_p1.sbatch` + `probe_p1_probe.sbatch`).

| Punto | Proposta di AG | Alternativa / nota |
|---|---|---|
| 1. Dataset | **NinaPro DB2, DB3 (amputati), DB6 (10 sessioni in 5 giorni)**: i soli della classe A con soggetti di test negli split v1 **e** etichette allineate all'EMG | Camargo: etichette non ingerite; Zhang 2026: etichette nel tempo del video (`labels_aligned_to_emg: false`); DB4 e DB7 sono interi nel pretraining |
| 2. Etichette | `restimulus`; **il riposo (0) non e' una classe**, il riempimento (-1) nemmeno. DB2 e DB3 numerano i movimenti 1-49 attraverso i tre esercizi, DB6 le 7 prese con codici fino a 11 (letto sui dati processati il 05/10) | riposo come classe (comune nei paper NinaPro); escluso per coerenza con UCI-EMG |
| 3. Finestre | **1 s senza sovrapposizione**, ciascuna dentro un tratto con una sola etichetta e dentro un esercizio o una sessione (come UCI-EMG) | 200-300 ms (lo standard di NinaPro per la latenza): il modello ha visto contesti da 1-4 s |
| 4. Soggetti della sonda | la sonda si addestra sui **soggetti di pretraining** del dataset (l'encoder ne ha visto l'EMG, mai le etichette); C scelto sul **10% di loro** (per eccesso, almeno 1, seme 0); il numero si legge sui **soggetti di test** degli split v1. DB2: 28 / 4 / 8; DB3: 7 / 1 / 3; DB6: 7 / 1 / 2 | sonda addestrata solo sui soggetti di test (leave-one-subject-out fra loro): meno dati, ma senza soggetti visti dall'encoder |
| 5. Classi | una sonda per dataset; le finestre di test di classi assenti dal train si tolgono e si contano (DB3: gli amputati non hanno fatto tutti i movimenti) | — |
| 6. Ingresso e feature | come P2 (`harness/fm_features.py`): filtro del pretraining, scala per **sessione** dalle sue prime 10 finestre, montaggio vero della sessione, feature = media del backbone + media dell'encoder locale (scelta di AG gia' in uso per P2) | — |
| 7. Un numero | **P1 = media delle accuratezze bilanciate dei tre dataset, pesata per i soggetti di test** (8, 3, 2: ogni soggetto conta uguale); si riporta anche la media a pesi uguali | pesi uguali: piu' peso agli amputati e al multi-giorno, ma DB3 e DB6 hanno 3 e 2 soggetti di test e l'errore standard cresce |
| 8. Errore standard | bootstrap sui soggetti di test, **stratificato per dataset** (1.000 ricampionamenti), poi combinato sui 2 seed come firmato in D12 | — |

**Da dichiarare:** i soggetti di test sono pochi (13 in tutto; 3 amputati, 2 in DB6), quindi P1 avra' un errore standard largo e la regola di D12
(Δ₄→₈ > 2 SE) scattera' difficilmente su P1: *lettura di AG,* in pratica decidera' piu' spesso P2. DB3 e DB10 potrebbero avere amputati in comune (non
dichiarato, gia' scritto in D9).

**Costo** (stima, da misurare al primo lancio): ~120.000 finestre da 1 s (DB2 ~60.000, DB6 ~45.000, DB3 ~15.000), lettura di ~9 GB.
Estrazione su `boost_qos_dbg`, **al piu' 0,5 GPU-ora = 4 ore locali** per checkpoint; sonde su `lrd_all_serial`, 0 GPU-ora. Nel pilot (E in {1,
2, 4, 8}, 2 seed, checkpoint pre- e post-decay) sono ~16 valutazioni, **<= 8 GPU-ora**. **Proposta:** un primo lancio sul checkpoint finale del
sanity, come collaudo sui dati veri (<= 0,5 GPU-ora, budget del passo 5: usate 0,58 + i run della replica in corso, ~32).

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo
