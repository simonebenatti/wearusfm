# Fatti da verificare

Registro dei "fatti da verificare" di [`docs/fm_emg_reference_v10.md`](fm_emg_reference_v10.md)
§12. Un fatto qui dentro è un **compito**, non un dato: non entra nel documento di riferimento
come premessa finché non passa da `raccolto` a `verificato da SB`.

Formato per riga: **fatto** · **fonte** (URL o percorso esatto) · **citazione esatta** ·
**stato**.

Stati possibili: `raccolto` (un subagent — FACT o LIC — ha trovato e citato la fonte) →
`verificato da SB` (Simone ha controllato la fonte di persona ed è d'accordo) oppure
`non trovato` (nessuna fonte ufficiale lo riporta: resta un compito aperto, non si inferisce).

---

## Dalla v10 §12, elenco originale

| # | Fatto | Fonte | Citazione esatta | Stato |
|---|---|---|---|---|
| 1 | Frequenze native di NinaPro DB8 (~1111 Hz) e DB10 (~1926 Hz); presenza di amputati in DB8 e DB10; varianti del montaggio NinaPro (DB6 a 14 elettrodi, DB8 a 16); disposizione interna delle fasce di GRABMyo | — | — | da raccogliere |
| 2 | Sovrapposizione di soggetti fra i DB NinaPro | — | — | da raccogliere |
| 3 | Conteggio dei soggetti del corpus (~600) e delle ore dopo la rimozione di DB9 | — | — | da raccogliere |
| 4 | Dal codice del repo NeuroRVQ: banda di filtraggio usata nel pretraining del tokenizer (20–90 Hz?), normalizzazione attesa in ingresso, se il transformer mescola canali diversi | — | — | da raccogliere |
| 5 | Sovrapposizione fra i soggetti visti dal tokenizer NeuroRVQ e i soggetti di test di §8; licenza del checkpoint | — | — | da raccogliere |
| 6 | Lista degli 11 muscoli di Camargo 2021 | (a) nomi delle colonne della table MATLAB in `/leonardo_work/IscrB_WearUsFM/data/raw/camargo2021/Subjects_Part1_AB06-AB14.zip` -> `AB06/10_09_18/levelground/emg/levelground_ccw_fast_01_01.mat` (decompresso in memoria via leonardo-ops, 23/09/2026); (b) https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/ (pagina del laboratorio degli autori, HTML grezzo letto il 23/09/2026); (c) `/leonardo_work/IscrB_WearUsFM/data/raw/camargo2021/README.txt` | (a) colonne, in ordine: `gastrocmed`, `tibialisanterior`, `soleus`, `vastusmedialis`, `vastuslateralis`, `rectusfemoris`, `bicepsfemoris`, `semitendinosus`, `gracilis`, `gluteusmedius`, `rightexternaloblique`. (b) "EMG electrodes are placed on the following muscle groups:" Gastroc medialis, Tibialis Anterior, Soleus, Vastus medialis, Vastus lateralis, Rectus femoris, Biceps femoris, Semitendinosus, Gracilis, Gluteus Medius, Right external Oblique (stesso ordine). (c) "emg - Electromyography from 11 muscles. Sampled at 1000Hz, bandpass filtered (20-400Hz)." (nessun nome di muscolo). **Nota:** l'obliquo esterno e' un muscolo del tronco, non dell'arto inferiore come dice v10 §4.5 | raccolto e firmato (D7b, 24/09/2026) — il paper (DOI 10.1016/j.jbiomech.2021.110320) non e' in open access su Europe PMC e non e' stato letto, ma le due fonti indipendenti concordi sono state ritenute sufficienti |
| 7a | Kaifosh et al.: presenza e forma delle curve di scaling nel numero di partecipanti | https://www.nature.com/articles/s41586-025-09255-w | pagina dietro paywall (redirect a idp.nature.com), non letta | da raccogliere |
| 7b | Kaifosh et al., dataset "Discrete Gestures": accesso, licenza, dimensioni | https://github.com/facebookresearch/generic-neuromotor-interface (repo verificato via API GitHub, non solo WebFetch) | download via `python -m generic_neuromotor_interface.scripts.download_data --task discrete_gestures --output-dir ~/emg_data` (anche `--small-subset` per un sottoinsieme di test); "The dataset and the code are CC-BY-NC-4.0 licensed"; 51,4h train + 6,2h val + 6,4h test, 100 partecipanti (80/10/10), .hdf5, 2 kHz | raccolto — in attesa della firma di Simone (D4) |
| 8 | Orientamento della fascia documentato, dataset per dataset, per tutti gli anelli e le fasce (serve alla funzione atlante di §4.5) | — | — | da raccogliere |
| 9 | `flash_attn_varlen_func` in `cineca-ai/4.3.0`: import verificato (flash_attn 2.5.5) **e** prova funzionale forward + backward in bf16 su A100 reale, nella forma d'uso prevista per il Perceiver (cu_seqlens_q fisso a K=64, cu_seqlens_k variabile C ∈ {8,...,256}) | sessione corrente, job SLURM 58134850 su Leonardo (`boost_qos_dbg`, COMPLETED in 20s) via leonardo-ops; script `bench/verify_flash_attn_varlen.py` | output `results/step0/flash_attn_varlen_check_58134850.json`: `"status": "OK"`, `"forward_output_shape": [448, 8, 64]`, `"forward_output_dtype": "torch.bfloat16"`, `"backward_grads_finite": true`, `"gpu_name": "NVIDIA A100-SXM-64GB"` | raccolto — evidenza completa, in attesa della firma di Simone (D4) |
| 10 | MFU reale sul Booster, per sostituire la stima di §10.5 | — | — | da raccogliere |

## Dalla dataloader_bench_spec.md §13, aggiunte rev. 2

| # | Fatto | Fonte | Citazione esatta | Stato |
|---|---|---|---|---|
| 11 | Hyser: 256 canali a 2048 Hz, 20 soggetti, due sessioni in giorni diversi | PhysioNet (pagina ufficiale) | vedi dataloader_bench_spec.md §13 | raccolto — **le 4 griglie da 64 (due lato flessorio, due estensorio) vengono da fonte secondaria**, da verificare sulla pagina ufficiale o sul paper |
| 12 | putEMG: 24 elettrodi in 3 fasce da 8, a 45°, primo elettrodo di ogni fascia sull'ulna, numerazione oraria; 5120 Hz; registrazione monopolare; licenza CC BY-NC 4.0 | pagina del dataset e paper ufficiali | vedi dataloader_bench_spec.md §13 | raccolto — da firmare (D4) |
| 13 | `flash_attn`: interfaccia espone maschera causale, finestra scorrevole e pendenze ALiBi; nessun argomento per un bias additivo arbitrario | repository ufficiale flash-attention | vedi dataloader_bench_spec.md §13 | raccolto |
| 14 | CapgMyo-DBa: disposizione fisica della griglia 128 canali (assunta 8x16 in `src/wearusfm/ingest/capgmyo.py`, dal paper Geng et al. 2016 - non verificato in sessione, nessun README nel dataset scaricato) | non verificato da fonte ufficiale in questa sessione | — | da raccogliere |
| 15 | UCI-EMG (Lobov et al.): frequenza di campionamento nativa, 1 kHz | Lobov et al., "Latent Factors Limiting the Performance of sEMG-Interfaces", Sensors 2018, 18(4):1122, DOI 10.3390/s18041122, https://pmc.ncbi.nlm.nih.gov/articles/PMC5948532/ | "the data flow x(t)∈ℝ8 was divided into 200 ms overlapping time windows at a 100 ms step (t=0,1,2,… is the discrete time with the sampling rate of 1 kHz)" | raccolto — in attesa della firma di Simone (D4) |
| 16 | GRABMyo: gli header WFDB dichiarano 32 canali, ma 4 (U1-U4) sono "unused channels" (non EMG, marcano solo la posizione tra i 4 anelli di elettrodi); i 28 canali EMG reali sono organizzati in 4 anelli distinti: ring1=F1-F8 (avambraccio), ring2=F9-F16 (avambraccio), ring3=W1-W6 (polso), ring4=W7-W12 (polso) | PhysioNet, pagina ufficiale del dataset (https://physionet.org/content/grabmyo/1.1.0/) | "There are 32 signals/channels ... 4 channels remain unused. ... The unused channels are listed as {U1-U4} ... are provided to distinguish the rings of electrode setup." | raccolto — in attesa della firma di Simone (D4) |
| 17 | putEMG: mappatura TRAJ_GT -> nome gesto (0=Idle, 1=Fist, 2=Flexion, 3=Extension, 6=Pinch index, 7=Pinch middle, 8=Pinch ring, 9=Pinch small; -1=pausa/transizione, filtrata da `filter_transitions`, non un gesto) | codice sorgente reale, repo GitHub biolab-put/putemg_examples, file `shallow_learn.py` righe 142-151 e `biolab_utilities.py` (funzione `filter_transitions`), clonato temporaneamente e ispezionato via leonardo-ops il 24/09/2026 | `gestures = {0: "Idle", 1: "Fist", 2: "Flexion", 3: "Extension", 6: "Pinch index", 7: "Pinch middle", 8: "Pinch ring", 9: "Pinch small"}` | raccolto — in attesa della firma di Simone (D4). Nota: un primo tentativo basato su un riassunto della pagina web ufficiale aveva nomi e stato -1 sbagliati (diceva "relax" invece di "pausa/transizione filtrata") - corretto verificando il codice sorgente letterale, non un riassunto |
| 18 | putEMG: ordine dei 24 canali EMG nelle 3 fasce (assunto EMG_1-8=fascia1, EMG_9-16=fascia2, EMG_17-24=fascia3 in `src/wearusfm/ingest/putemg.py` - non verificato da fonte ufficiale in questa sessione, solo l'ordine piu' naturale dato lo schema di naming) | non verificato da fonte ufficiale in questa sessione | — | da raccogliere |
