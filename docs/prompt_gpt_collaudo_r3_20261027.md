# Prompt per GPT (responsabile scientifico dal 26/10/2026): perche' il collaudo dbg di R3 non supera `posa_discesa`?

Richiesta di Simone (27/10/2026, dopo la lettura del cancello): «come mai fallisce? vediamo se chatgpt lo risolve, dammi il prompt per lui per
provare a capire cosa non va nel collaudo dbg».

## Contesto in breve

- **R3** (seconda ondata del pilot degli obiettivi, `docs/pilot_posa_v1.md` firmato il 26/10, emendamento 1 `docs/emendamento_pilot_posa_v1.md`):
  ancore con peso 0,2 (JEPA 0) piu' una **testa lineare di posa** con peso 1,0: `LayerNorm(384) -> Linear(384, 20)`, inizializzata a zero, sulla
  **media per istante dei latenti del backbone** (`latent_mean`, la parte del backbone del readout `pool_p1p2`); bersaglio = i 20 angoli
  articolari di emg2pose, standardizzati per giunto con le statistiche del train (`pose_stats.json`), per blocchi di 10 campioni (200 Hz),
  mappati sulle patch da 200 ms; righe non valide = zeri della cinematica inversa piu' un margine M = 280 campioni per lato (firmato da Simone
  il 27/10). Perdita = errore quadratico medio standardizzato sugli elementi validi, media per finestra e poi fra le finestre (`pose_terms`).
- **Collaudo dbg** (passo [5] della catena, job SLURM 62429050, 27/10 14:52-15:10): 1.300 passi, 4 A100, batch 32 (8 per processo), lr 1e-4,
  schedule WSD con `total_steps = steps_at_d = 1.300`, codice c5e42cb (checkout `pilot-posa-20261026`), `--pose-grad-every 100` (E4). Criteri
  congelati del cancello (`scripts/arms_collaudo.py --arm R3`, §2 (iii) e §6 punto 7 del protocollo): **13 su 14 OK, `posa_discesa` NO**.
  Esito: `results/infra/pilot_posa/collaudo_r3_dbg_62429050.json`; lettura di Claude con le opzioni: `docs/decisioni.md`, voce «COLLAUDO dbg di
  R3 (job 62429050) LETTO». **R3 [6] non parte** senza il via di Simone (fuori dal cancello non si lancia).

## I dati (tutti nel checkout di Claude, `results/infra/pilot_posa/collaudo_r3_dbg_62429050/`)

`metrics.jsonl` (1.300 record: `step`, `total`, `pose`, `pose_windows`, le 5 ancore, `jepa`, `lr`, `grad_norm`, `t_step_s`, `windows`; ogni 100
passi `shared_grad_pose`, `shared_grad_anchors`, `shared_grad_cos` = E4 sul micro-batch finale del processo 0), `fixed_eval.jsonl` (pannello
fisso: 3 valutazioni ai passi 0 / 500 / 1.000 su 32 finestre di posa fisse: `pose`, `pose_mae_std`, `pose_mae` in radianti, `pose_mae_j00..j19`),
`fixed_eval_set.json`, `config.json`, `summary.json`, `pose_provenance.json`, `ls_run_dir.txt` (i checkpoint restano sul cluster).

| Che cosa | Valore |
|---|---|
| Perdita di posa pesata, media per blocchi di 100 passi (1-100 ... 1.201-1.300) | 1,08 1,07 1,02 1,03 1,00 0,99 1,04 0,96 1,01 1,05 1,05 1,08 1,01 (escluso il passo 119, senza posa) |
| Criterio `posa_discesa`: media 500-599 -> media 1.201-1.300 (richiesto <= 0,9 x) | 0,991 -> 1,014 (+2,3%) |
| Deviazione della perdita di posa dentro un blocco di 100 passi; errore standard della media | 0,20-0,28; 0,02-0,03 |
| Pannello fisso ai passi 0 / 500 / 1.000: perdita; MAE standardizzata; MAE in radianti | 1,075 / 1,060 / 1,048; 0,871 / 0,854 / 0,865; 0,350 / 0,343 / 0,350 |
| Finestre di posa per passo (`pose_windows`, conteggio GLOBALE sui 4 processi, su 32 per passo) | 7,2 in media (~1,8 per processo); 99,9% dei passi ne ha almeno una; 9.332 viste in tutto |
| Somma delle ancore pesate: passi 1-50 -> 1.251-1.300; sulle finestre del criterio (500-599 -> 1.201-1.300); `total` | 24,3 -> 6,4; 7,93 -> 6,45 (-18,7%); 5,87 -> 2,24 |
| `grad_norm` (prima del clip a 1,0), media primi / ultimi 100 passi | 12,5 / 4,4 |
| lr ai passi 1 / 500 / 1.000 / 1.100 / 1.200 / 1.300 | 1e-7 / 5e-5 / 1e-4 / 7,7e-5 / 3,9e-5 / 3,8e-7 (warmup `max(1.000; 2%)` = 1.000 passi, piu' rapido di quello di R3 = 1.962; plateau a 1e-4 nei passi 1.000-1.041; decadimento dal 1.042) |
| E4 ai passi 500 ... 1.300: norma del gradiente della posa / norma di quello delle ancore sui moduli condivisi | 25%, 146%, 15%, 0*, 46%, 27%, 0*, 328%, 81% (*ultimo micro-batch senza posa) |
| E4: coseno fra i due gradienti | fra -0,80 e +0,39, senza segno sistematico (E4 e' il gradiente locale di un solo micro-batch del processo 0, ~1,8 finestre di posa) |
| Ritmo | 0,711 s/passo (soglia 0,81) |

Per confronto, le stesse 1.300 iterazioni nel collaudo dbg del pilot (R0/R1/R2/R5, 24/10) hanno fatto scendere le ancore allo stesso modo; la posa
non c'era.

## Che cosa i dati escludono gia'

- Scala della standardizzazione: la perdita parte da ~1,0-1,08, cioe' la varianza dei bersagli standardizzati con una testa a zero (criterio
  `posa_scala_avvio` 0,275 >= 0,1 e `posa_tetto` 0,695 <= 2 OK).
- Configurazione: testa presente (+8.468 parametri rispetto a R3c), albero della posa e SHA di `pose_stats.json` uguali a quelli dell'ingest
  intero (757f2aee...), peso 1,0; conteggio delle finestre di posa stabile.
- Flusso del gradiente: nelle misure con posa nel micro-batch il termine produce gradiente sui moduli condivisi dell'ordine di quello delle
  ancore (E4, locale al processo 0), quindi la perdita e' collegata ai parametri; `clip_grad_norm_` a 1,0 non riduce il passo di AdamW, che e' invariante alla scala del gradiente.
- Le ancore imparano, nessun collasso, nessun allarme, nessun NaN.

## Ipotesi che restano (non distinguibili col solo collaudo)

- **H1, passi e learning rate.** AdamW con lr <= 1e-4 (pieno solo al passo 1.000, poi in decadimento), weight decay 0,05, ~7 finestre di posa
  per processo per passo, segnale EMG -> posa debole e rumoroso per una testa lineare: il gradiente e' dominato dal rumore e i pesi fanno un
  cammino casuale attorno allo zero. Prevede una discesa lenta ma visibile con molti piu' passi a lr pieno.
- **H2, accessibilita' lineare.** La posa (lenta, in gran parte differenze di postura fra utenti e fra sessioni) non e' leggibile linearmente
  dalla media per istante dei latenti di un encoder addestrato sulle ancore (spettro e ampiezza su 200 ms). Prevede che anche una ridge
  offline dai latenti congelati (E3 dell'emendamento) non batta il costante.
- **H3, difetto nella catena dei bersagli sui dati veri** che i test CPU su dati costruiti non vedono: allineamento temporale fra la finestra
  EMG (con i margini del loader) e i blocchi di posa; mappatura blocchi -> patch per centro (`window_pose`); lato o registrazione sbagliati;
  maschera di validita' che, con M = 280, toglie proprio i tratti dinamici vicino ai buchi; `pose_unprompted`. Prevede che nessun predittore
  lineare, nemmeno da feature semplici dell'EMG, batta il costante entro l'utente, mentre emg2pose mostra che la posa e' predicibile dall'EMG.
- **H4, riduzione DDP del termine.** Contratto «numeratori locali / conteggi globali su processi e micro-batch» (`summary.json`): verificare
  che per la posa dia davvero la media sulle ~28 finestre di posa del passo e non un gradiente per finestra scalato di un fattore grande.

## Codice da leggere (checkout di Claude, ramo `worktree-claude-training-engine`, ultimo commit)

- `src/wearusfm/model/pose_head.py` (testa, `latent_mean`); `src/wearusfm/training/jepa.py`: `pose_time_valid`, `pose_terms`, e il blocco
  `if cfg.pose_weight > 0` in `_jepa_compute` (la predizione e' calcolata sempre; `losses["pose"]`, `losses["pose_windows"]`).
- `src/wearusfm/training/run.py`: `_pose_tensors` (bersaglio, maschera, deviazioni del micro-batch), ottimizzatore (`AdamW`, `grad_clip`),
  `wsd_warmup_steps` / `with_wsd` (schedule), `_shared_grad_stats` (E4), il pannello fisso (`pose_std_log`), `pose_provenance.json`.
- `src/wearusfm/data/pose_targets.py`: `PoseStore` (statistiche, `per_side`, `pose_unprompted`), `PoseStore.window`, `window_pose` (blocchi ->
  patch per centro, maschera); `src/wearusfm/data/pretraining_loader.py` (geometria delle finestre, `patch_ms`, margini, `read_window`).
- `src/wearusfm/ingest/emg2pose_pose.py` (ingest della posa, regola di validita', `already_done`); `scripts/train_jepa.py` (`--pose-root`,
  `--pose-weight`, `--pose-unprompted`, `--pose-grad-every`).
- Test esistenti: `tests/cpu/test_pose_objective.py`, `tests/cpu/test_pose_targets.py`, `tests/cpu/test_pose_eval.py`, `tests/cpu/test_arms_collaudo.py`.
- Valutazione fuori campione E1-E3 (ancora da eseguire): `src/wearusfm/harness/pose_eval.py`, `scripts/pose_eval.py`,
  `scripts/slurm/pose_eval_extract.sbatch` (1 A100, 40 min di tetto per estrazione).

## Che cosa ti chiedo

1. **Diagnosi.** Quale fra H1-H4 (o altro) e' coerente con TUTTI i numeri sopra? Che cosa, in `metrics.jsonl` o nel pannello fisso, le
   distinguerebbe gia' ora (per esempio: andamento per giunto, relazione fra perdita di posa e lr, varianza della perdita contro il numero di
   finestre, le finestre fisse dell'`fixed_eval_set.json`)? Ricalcola quello che ti serve dai file.
2. **Rilettura della catena dei bersagli** (`PoseStore.window`, `window_pose`, `_pose_tensors`, `pose_terms`, `_jepa_compute`) cercando un difetto
   che i test non coprono: un offset di una o piu' patch, la maschera, il lato, la standardizzazione per giunto, `pose_unprompted`, la riduzione
   DDP. Se trovi qualcosa, indica righe e un test che lo dimostri; il codice lo corregge Claude (ruoli).
3. **Un conto a tavolino per H1:** con ~28 finestre di posa per passo, una testa lineare a zero su 384 dimensioni normalizzate, AdamW a lr 1e-4
   (pieno solo per ~100 passi qui) e un rapporto segnale/rumore r del bersaglio lineare, quanti passi servono per una discesa del 10%? Il
   criterio congelato (discesa >= 10% fra i passi 500-599 e 1.201-1.300, con warmup di 1.000 passi su 1.300) era realistico?
4. **Controlli a basso costo, con preventivo, SENZA lanciarli** (dopo la review interna le opzioni in decisioni.md sono: (A) E1+E2+E3 sul
   checkpoint del collaudo 62429050, ~1,3 GPU-ora + 16 core-ore di tetto; (A2) regressione CPU da feature grezze per H3, 8 core-ore; (B) collaudo
   lungo con la ricetta esatta di R3, riprendibile; giudica anche queste): (a) sul Mac, un test con dati costruiti che imiti una registrazione vera (finestre
   del loader, blocchi a 200 Hz, margine) e misuri in quanti passi una testa lineare impara a lr 1e-4; (b) sul cluster, CPU, <= 1 h, 0 GPU-ora: per
   esempio una ridge dalle feature semplici dell'EMG (RMS per canale per patch) alla posa standardizzata su 20 registrazioni del train, entro
   l'utente: se nemmeno quella batte il costante, la catena dei bersagli e' sospetta (H3); (c) l'E3 dell'emendamento su R3c (ridge dai latenti
   congelati: 2 estrazioni GPU da 40 min di tetto + rapporto CPU, ~1,3 GPU-ora di tetto) per H2. Di' quale ordine e quali esiti separano H1, H2,
   H3.
5. **Criterio per un eventuale collaudo lungo** (opzione B di decisioni.md: 5.000 passi con lo schedule vero, warmup 1.962, plateau a 1e-4, ~5
   GPU-ora di tetto): quale criterio di discesa congeleresti PRIMA, e su quali passi?
6. **La tua raccomandazione** fra le opzioni (A) E3 su R3c, (B) collaudo lungo, (C) peso della posa piu' alto (cambia la definizione firmata), (D)
   fermare R3, o una combinazione, con costi e ordine.

## Vincoli

- Nessun job senza il via di Simone; non toccare i checkout di campagna; il cluster non e' raggiungibile da te: solo i file nel checkout e
  nei `results/`. Le soglie firmate (M = 280, regola del §5, 0,81 s/passo) non si riscrivono: se proponi un criterio nuovo, e' una proposta
  per Simone, dichiarata come tale.
- Rispondi nel registro condiviso (`docs/claude_GPT_shared`, sezione 1, voce tua) con: diagnosi, righe di codice sospette (se ce ne sono),
  controlli proposti con preventivo, criterio proposto per il collaudo lungo, raccomandazione.
