# Review di Claude — esito del collaudo train-only 30M/300M (job 63526191, readout di GPT ab2eb4a) — 4 novembre 2026

**Verdetto sul readout di GPT: CORRETTO.** Tutte le letture numeriche di `docs/reviews/scaling_preflight_results_20261104.md`,
della voce di registro e di `decisioni.md` (ab2eb4a) sono riprodotte dai quattro report di rank e dagli otto stream con codice mio;
lo script di GPT rieseguito produce un JSON identico a quello committato. Due frasi da precisare (RNG e clipping), nessun errore di
lettura.

**Verdetto sul collaudo: CONFORME al piano approvato** (`claude_review_scaling_300m_preflight_20261104.md`, befab4a + 2f3207b):
dose 64 nativi + 64 stress su quattro rank, 512 coppie appaiate, LR nativo e di stress esatti, scadenza 1.500 s rispettata (523,6 s),
gate di memoria 0,9 superato con margine (13,11 e 28,41 GiB contro 57,08), un solo modello residente anche per misura, codice eseguito
identico ad afb644e e al checkout di campagna, nessun checkpoint scientifico, proxy di produzione 33,52 GiB prespecificato prima del
lancio e sotto il limite: ripiego `accum_steps=2` non attivato.

**Verdetto sui gate residui: il training scientifico a 5.000 NON è pronto.** S3 (63527446) è terminato COMPLETED 0:0 alle 21:09:51 con
`report.json` scritto, ma la sua lettura e validazione non fanno parte di questa review; launcher, readiness e controllo di 34b878a
non sono ancora rivisti; manca la review finale della catena con i pin reali e il via di Simone.

## 0. Perimetro e metodo

- Richiesta della sessione madre con l'approvazione di Simone («Sì, chiedila»): rilettura del risultato del collaudo eseguito e letto
  da GPT, contro la review approvata del piano. Solo review: nessun job, deploy o modifica ai rami di GPT. Checkout detached temporaneo
  di `origin/codex/scaling-300m-20261104` nello scratchpad (punta 34b878a; i file del readout sono identici a 3cc3e09, diff vuoto),
  rimosso alla fine. Su Leonardo solo letture: `sacct`, `squeue`, `scontrol show config`, `ls`, `sha256sum` dei 13 file remoti,
  `git rev-parse`/`status` del checkout di campagna, coda del log del job.
- Letti per intero: readout, `verify_scaling_preflight_63526191.py`, `report.json` (590 righe, compattato), `independent_verification.json`,
  `collection_manifest.json`, `remote_sha256.txt`, `preflight_plan.json`, ricevuta di sottomissione, ricevuta rifiutata,
  `submission_amendment.json`, `preflight_scontrol_initial.txt`, `preflight_log_initial_tail.txt`, `resource_policy.json` a 05d4bf0,
  le voci di registro e `decisioni.md` di ab2eb4a, il protocollo (S3/S4) a 70cbd85 e a 34b878a.
- Eseguiti: lo script di GPT sugli artefatti committati (`~/wearusfm_local/logs/verify_preflight_63526191_rerun.json`: uguale a
  `independent_verification.json`); una ricostruzione indipendente mia, con codice diverso, dai quattro `rank{i}.json.gz` e dagli otto
  stream (`~/wearusfm_local/logs/claude_recompute_preflight_63526191.txt`).

## 1. Provenienza, sottomissione e identità del codice

- **Catena temporale (CET).** Consegna 05d4bf0 16:33 → review di Claude befab4a 16:53 e seguito 2f3207b 17:04 → piano di esecuzione
  congelato 70cbd85 17:11 (contiene la formula del proxy di memoria e il ripiego accum 2 / batch 32) → checkout 63525937 18:29:42–18:30:18
  (COMPLETED 0:0, 16 s, `login08`, `lrd_all_serial`, 2 CPU, 0,008889 core-ore) → prima `sbatch` 18:32:45 **rifiutata** → seconda `sbatch`
  18:35:41 → job 63526191 18:35:59–18:45:05 (COMPLETED 0:0, 546 s, `lrdn1813`, `boost_usr_prod`/`boost_qos_dbg`, 32 CPU, 480G, 4 GPU,
  limite 30 min, commento `wearusfm:SCALING-300M-preflight-20261104`) → readout ab2eb4a 18:51.
- **Rifiuto della prima sottomissione.** L'argv aveva `--dependency=afterok:63525937`; SLURM ha risposto «Job dependency problem» perché
  il job di checkout, finito alle 18:30:18, era già stato purgato dal controller (`MinJobAge = 30 sec`, `DependencyParameters =
  kill_invalid_depend`; `scontrol show job 63525937` dava «Invalid job id specified»). Nessun job creato (`squeue` e `sacct` vuoti per quel
  nome), ricevuta rifiutata conservata, `submission_amendment.json` registra prove e motivo; la seconda `sbatch` ha argv identico senza
  la dipendenza, con checkout verificato prima (HEAD 05d4bf0, tree pulito). Non è un rilancio alla cieca. `sacct` di oggi mostra un solo
  job per ciascun nome logico. GPT ha codificato la lezione nel gate del checkout di S3 («sacct COMPLETED 0:0 … no expired afterok»).
- **Codice.** `report.code_sha` = 05d4bf0 = `PREFLIGHT_CODE_SHA` di `scaling_train.py` = `code_sha` di `execution_plan.json` = HEAD del
  checkout `scaling_300m_preflight_20261104_05d4bf0` (tree pulito oggi). I sorgenti eseguiti (`scaling_preflight.py`, il launcher,
  `lib_code_dir.sh`, `train_jepa.py`, `src/`) sono identici fra 05d4bf0 e la punta 34b878a (diff vuoto); l'identità con afb644e era già
  verificata nella review del piano. Input pinnati nel report uguali al contratto (configurazione T100-G `dd68e900…` = `config_30m`,
  piano sorgente `13ed48b6…`, manifest, scale, finestre, statistiche di posa): lo script di GPT lo impone e passa.
- **Costi.** Allocati 4 GPU × 546 s = 0,606667 GPU-ore e 32 core × 546 s = 4,853333 core-ore, contro i tetti 2,0 e 16,0; checkout
  0,008889 contro 0,3333 core-ore. Tutto dentro il budget pubblicato.

## 2. Integrità degli artefatti

- 13 file (report, 4 report di rank, 8 stream JSONL): gli hash in `remote_sha256.txt` coincidono con quelli calcolati da me oggi sui
  file remoti (`sha256sum` sul login node alle 21:05), con gli originali decompressi nel repo e con `collection_manifest.json`
  (dimensioni, hash degli originali e delle copie compresse). La cartella remota contiene solo questi 13 file: nessun `.pt`.
- Report: protocollo `scaling-300m-preflight-20261104-v1`, `complete`, `paired_data_masks_queries_targets`, `one_model_resident`,
  `scientific_checkpoints_written: false`, 128 batch per braccio, 64 + 64, 4.096 finestre per braccio, world 4, `plan_sha256` uguale al
  piano pinnato, runtime torch 2.2.0a0+git6c8c5ad / CUDA 12.1 / A100-SXM-64GB, `wall_seconds_max_rank` 523,6 s.

## 3. Ricostruzione indipendente (codice mio, non lo script di GPT)

- **Appaiamento.** 1.024 record, 128 per taglia e rank, passi 1…128 consecutivi, fasi 64/64, `scientific_update` sempre falso. Per
  ciascuna delle 512 coppie rank/passo le sei impronte dei dati (batch, segnali processati, maschere, target di ancoraggio, target di
  posa, origini) e le sette chiavi del supporto sono identiche fra 30M e 300M; le impronte differiscono fra rank (ogni rank il suo
  shard). LR per passo 1e-4·(j+1)/1962 nei primi 64 (5,097e-8 al passo 1, 3,262e-6 al 64) e 1e-4 esatto in tutti i 64 di stress, uguale
  fra le taglie. Stream = record + `arm`/`rank`/`fingerprint`. `clip_factor = min(1, 1/(‖g‖+1e-6))` riprodotto. Tutti i valori finiti;
  ogni gruppo diagnostico (frontend, proiezione, posa, QKV e MLP del primo e dell'ultimo blocco) ha gradiente e update > 0. Sorgenti
  invariate durante il run; 16 dataset di pretraining, 1.024 finestre per rank e braccio, nessun EPN-612 né UCI. Impronta banco+gain
  `5819ee30…` identica fra taglie; stati iniziali diversi; parametri 28.362.013 / 292.054.847.
- **Tempi.** Somma dei massimi fra rank di `forward_backward_s` e `clip_optimizer_ema_s`: i 128 valori per taglia coincidono con
  `normal_update_rank_max_seconds` del report. Media sui 112 passi misurati (esclusi i primi 8 di ogni fase): **0,70785 s (30M) e
  2,76668 s (300M)**, rapporto **3,908563**; medie e p90 per fase uguali al report entro 1e-12 (0,71088/0,70483 e 2,78312/2,75023 s). Le
  diagnostiche (circa 19–20 s per rank e taglia) restano fuori dai tempi; durata per taglia 127,1 s e 389,5 s per rank.
- **Memoria.** Picco allocato 14.078.980.096 B = 13,112072 GiB e 30.500.367.872 B = 28,405681 GiB; riservato 21,0703 e 43,8984 GiB;
  memoria del dispositivo 68.099.571.712 B = 63,4227 GiB, quindi gate 0,9 = **57,0804 GiB** (il valore reale, non i 64 GiB nominali della mia
  tabella). Baseline del primo update per rank: 0,347–0,355 GiB (30M) e 3,334–3,343 GiB (300M), contro 0,317 e 3,264 GiB attesi per tre
  copie FP32 (studente, teacher, gradienti); baseline massime 0,593 e 5,534 GiB con i momenti di AdamW (attesi 0,528 e 5,440). Un 30M
  rimasto residente avrebbe alzato la baseline del 300M a circa 3,9 GiB: **un solo modello residente è confermato dalla misura**, non solo
  per costruzione (era il mio F2).
- **Clipping e loss.** Clipping attivo in ogni update delle quattro celle (`fraction_clipped` 1,0). Mediane norma/fattore: 30M 17,33 e
  0,0578 (nativo), 4,626 e 0,2162 (stress); 300M 12,71 e 0,0787 (nativo), 8,326 e 0,1201 (stress); nello stress la norma del 300M sta fra
  3,68 e 19,39, quella del 30M fra 1,90 e 9,32. Loss totale media: 5,866 → 3,402 (30M) e 4,749 → 3,400 (300M) fra le fasi; dentro ogni
  fase cala dai primi 8 agli ultimi 8 passi misurati (nativo 7,02 → 4,86 e 6,68 → 3,68; stress 3,80 → 3,22 e 3,79 → 3,23) e i massimi
  dello stress (4,45 e 4,60) stanno sotto quelli nativi: nessuna divergenza a LR 1e-4 su pesi giovani (era il mio F1).
- **Proxy e proiezioni.** 28,405681 × max(1, 15,473959/13,112072) = 28,405681 × 1,180130 = **33,522417 GiB < 57,080402 GiB**; il ripiego
  accum 2 / batch 32 non si attiva. `seconds_per_step = max(3883,03/5000 × 3,908563, 2,76668) = 3,035413 s`, lo stesso di
  `production_memory_resource_policy.json`; 5.000 passi → ceil a 900 s di (5000 × 3,035413 × 1,2 + 1200) = 19.800 s = 22,0 GPU-ore di tetto;
  500 passi → 3.600 s, come nel piano S3. Il rapporto storico provvisorio di `resource_policy.json` a 05d4bf0 (3,9165, 15.208 s) è coerente
  con quello misurato.

## 4. Letture di GPT, una per una

| Affermazione di GPT | Verifica |
|---|---|
| COMPLETED 0:0, 18:35:59–18:45:05, 546 s; 0,606667 GPU-ore, 4,853333 core-ore; checkout 0,008889 core-ore | `sacct`: uguale |
| Parametri 28.362.013 / 292.054.847; 128 update per rank; 4.096 finestre | report e rank: uguale |
| Picco allocato 13,1121 / 28,4057 GiB; riservato 21,0703 / 43,8984 GiB | ricalcolato: uguale |
| Tempo medio su 112 passi 0,70785 / 2,76668 s; rapporto 3,9086 | ricalcolato: 0,7078507 / 2,7666787, 3,908563 |
| Norma mediana al picco LR 4,6261 / 8,3264; fattore di clip 0,21617 / 0,12010 | ricalcolato: uguale |
| Clipping in ogni update; nessun NaN/Inf; nessun gruppo senza update | ricalcolato: uguale |
| 13 SHA remoti uguali; 1.024 record, 512 coppie; LR esatto; identità iniziale banco/gain | ricalcolato: uguale |
| Proxy 33,522417 GiB < 57,080402; ripiego accum 2 non attivato | ricalcolato: uguale; formula e ripiego scritti in 70cbd85 (17:11), prima del lancio |
| Nessun checkpoint scientifico; S3 deve arrivare a 500 passi da zero | report e cartella remota: uguale; coerente con le cadenze 500 del config (review 5) |
| Aggiornamento dopo la review: 2.275 passati + B19618 preesistente anche alla base 6d80e8d | coerente con la mia registrazione |

Precisazioni, non errori:

- **P1 — RNG del modello identico fra le taglie.** `rng_after_sha256` (stato dei generatori CPU e CUDA dopo l'update) è uguale fra 30M e
  300M in 512/512 coppie, mentre è diverso fra passi (128 valori distinti) e fra rank (0/128 uguali). Spiegazione: l'unica operazione
  casuale nel passo è il dropout del muscolo (`channel_identity.py:58-61`), estratto dal generatore globale su una forma che dipende dal
  batch e non dalla larghezza; i blocchi transformer sono senza dropout e i rami spettrali (assenti qui) usano un generatore separato.
  Le due taglie hanno quindi visto le stesse maschere di dropout: l'appaiamento è più forte di quanto dica la frase `rng_scope` del
  report («different model shapes and dropout consume different model RNG», ripresa dalla mia review del piano). Il report è congelato;
  basta correggere la frase nel readout.
- **P2 — Frase ambigua sul clipping.** «Nel 300M è più forte nella fase di stress» vale rispetto al 30M nella stessa fase (fattore 0,120
  contro 0,216), non rispetto alla fase nativa del 300M (0,079, più forte). Riformulare.
- **P3 — Gate di memoria.** GPT usa correttamente 0,9 × 63,4227 GiB = 57,0804 (memoria rilevata), come chiede il protocollo; i 57,6 GiB
  della mia tabella erano il valore nominale.

## 5. Gate residui prima dei 5.000 (stato alle 21:10 CET)

- **S3 (63527446):** COMPLETED 0:0, 20:42:52–21:09:51 (26 min 59 s, `lrdn2103`, 1,8 GPU-ore allocate su 4 di tetto). Nella cartella
  `wearusfm_runs/results/scaling_300m_20261104/production_memory_v1/` ci sono `launch.json`, `rank_0…3.json`, `report.json` (9.505 B) e
  `training.log`; `rank_0.json` riporta `returncode 0`, codice 0d5f0f9, picco allocato 31.794.550.272 B = 29,61 GiB e riservato 37,30 GiB su
  63,42. La lettura formale (quattro rank, `inspect_evidence`, `validate_report`, memoria reale contro il proxy 33,52 e il gate 57,08) **non è
  fatta qui**: spetta a GPT, con review successiva su richiesta.
- **Launcher e readiness (34b878a, 20:57):** `scaling_train.sbatch`, `scaling_readers.sbatch`, `scaling_evaluation.sbatch`, i loro test,
  `evaluation_readiness.json` (`measured_benchmarks_300m5000: false`, controllo richiesto fino all'11/11), `evaluation_resource_envelope.json`,
  `control_verification_20261104.json`, `scientific_runtime_freeze.json`: non rivisti da Claude.
- **Elenco di GPT in `readers_memory_verification.json`** (`scientific_5000_ready: false`): evidenza S3 verificata; piani e launcher di
  estrazione e lettura; retention e pin del controllo 30M; review finale che leghi gate e risorse reali. Più il via di Simone, che questa
  review non dà.

## 6. Facoltativi

- **F-a** Correggere nel readout le frasi di P1 e P2.
- **F-b** Nello script di verifica aggiungere la distinzione delle impronte fra rank e il controllo della baseline di memoria (sezione 3):
  rendono misurabili sharding e modello unico residente invece di dichiararli.
- **F-c** Il gate «niente `afterok` scaduto» già scritto per S3 vale per ogni sottomissione a catena del submitter: con `MinJobAge` 30 s
  una dipendenza su un job CPU breve è quasi sempre già purgata.

## 7. Limiti

Nessuna esecuzione su GPU; nessuna lettura degli artefatti di S3 oltre allo stato SLURM, all'elenco dei file e a `rank_0.json`; 34b878a
non rivisto. Le misure del collaudo riguardano il solo passo di training: loader, pannelli, valutazione fissa e checkpoint restano
fuori, come dichiarato da GPT e da me.
