# Catene WSD della finestra 1: launcher, pianificatore e collaudo (Claude, 15/10/2026)

Simone, 15/10/2026: «sì, vai coi launcher delle catene WSD». Punto 6 della numerazione unica (launcher di Claude).

## Come funziona una catena

- **Un run = una catena di job** su un nodo intero (DDP a 4 GPU, `scripts/slurm/train_ddp.sbatch`), collegati con `--dopo` di
  `submit_job.py` (afterok). Ogni job riprende dal checkpoint della cartella del run (ripresa esatta: stream, generatori, optimizer, EMA) e si
  ferma da solo `margin_min` minuti prima della fine del job (`TIME_LIMIT_S`), salvando: l'uscita e' 0 e la catena prosegue.
- **Arresti:** un allarme, una perdita o dei gradienti non finiti scrivono STOP e i job successivi escono subito; un errore del job (codice
  diverso da 0, TIMEOUT di SLURM) ferma la catena per dipendenza. Nessun retry automatico.
- **Job di scorta** (`spare_jobs`): se il run e' gia' completo (`summary.json` con `stopped` = `max_steps` e almeno i passi chiesti) il job
  esce subito con 0, senza caricare modello e dati.
- **Rami del WSD** (D14: rami di decay del pilot sulle epoche, continuazione a 2D): `BRANCH_FROM` = checkpoint con nome della fase stabile
  del tronco (`checkpoint_step<N>.pt`, da `--keep-checkpoints`) oppure, per la continuazione a 2D, il pre-decay (`checkpoint_predecay.pt`,
  quando il ramo parte dall'inizio del decay del tronco: 80% di D, decay sul 20% finale di 2D, stesso warmup). Il primo job del ramo dipende dal job del tronco che ha sicuramente scritto
  quel checkpoint, con la stima pessimista dei passi per job (s/passo x `slack`); se il checkpoint non c'e' ancora esce con 3 e la catena del
  ramo si ferma (non parte mai da zero). Dopo il primo salvataggio il ramo riprende dalla propria cartella. Il trainer accetta un ramo solo se
  cambiano soltanto schedule e max_steps e il checkpoint sta nella fase stabile di tronco e ramo (`_check_branch`, gia' provato bit per bit
  sulla CPU in `tests/cpu/test_training_wsd.py`).
- **Durata dei job:** 24 h (QoS `normal` di `boost_usr_prod`, quella dei job lunghi gia' girati) per i run che ne richiedono piu' d'uno; un
  run che sta in un job (i rami) chiede solo il tempo che gli serve con la stima pessimista, arrotondato al quarto d'ora (meno tetto, piu'
  backfill).

## Pianificatore: `scripts/wsd_chain.py`

Dal piano JSON (formato nella docstring dello script) ai comandi di `submit_job.py`, uno per riga e nell'ordine (prima i tronchi, poi i
rami; in un ciclo `while read` ogni comando con `< /dev/null`, perche' il suo `ssh` legge lo stdin e consumerebbe le righe successive),
piu' la tabella del preventivo (job, durata, dipendenze, GPU-ora attese e tetto, spazio dei checkpoint). Non sottomette nulla: i
comandi li esegue leonardo-ops dopo l'autorizzazione. Prima di generare i comandi controlla, col codice del trainer: D14 (`with_wsd`,
`wsd_warmup_steps`, `_check_wsd`), checkpoint dei rami fra quelli tenuti dal tronco e nella fase stabile di tronco e ramo, campi ereditati
dal tronco, s/passo misurato per ogni coppia preset/batch, niente virgole negli argomenti (sbatch `--export` le usa come separatore: i
`--keep-checkpoints` si separano con `:`), SHA completo.

Esempio (pilot a 100M, un seme, 23.150 passi per epoca provvisori; s/passo misurato col prefetch 1,39):

| Run | Da | Passi | Decay da | Job (+scorta) x durata | Dopo | GPU-ora attese / tetto |
|---|---|---|---|---|---|---|
| tronco | — | 185.200 | 148.160 | 4 (+1) x 24 h | — | 290 / 480 |
| E1 | tronco @ 18.520 | 23.150 | 18.520 | 1 (+1) x 3 h | tronco j01 | 8 / 24 |
| E2 | tronco @ 37.040 | 46.300 | 37.040 | 1 (+1) x 5,25 h | tronco j01 | 15 / 42 |
| E4 | tronco @ 74.080 | 92.600 | 74.080 | 1 (+1) x 9,25 h | tronco j02 | 30 / 74 |

**Ingressi ancora aperti** (non decisi qui): passi per epoca da manifest, quote e durate correnti (le stime usano ~23.150 col batch 32);
batch e picco del LR dalla calibrazione; lista dei job firmata. Il pianificatore li prende come dati del piano.

## Collaudo proposto (`w1_catena_v1`, da autorizzare)

Scopo: provare sul cluster la meccanica della catena coi tempi veri, che sulla CPU non si puo': uscita al limite di tempo e ripresa per
dipendenza, job di scorta, ramo dal checkpoint con nome dopo il job giusto del tronco, prefetch acceso. Non e' una misura scientifica.

**Piano:** 30M (preset `sanity`), regole della finestra 1, tutti i dataset, batch 32, `--prefetch`, QoS `normal` con job da 15 minuti
(margine 3, avvio 2; dbg ammette solo 2 job sottomessi, la catena ne ha 5). D = 1.250 passi (warmup 1.000, il minimo di D14).
- **Tronco:** 1.400 passi (decay da 1.120), checkpoint tenuto a 1.100: con ~735 passi per job (pessimista; ~845 nominali) 2 job + 1
  di scorta.
- **Ramo:** da 1.100 a 1.375 passi (decay da 1.100), dopo il job 2 del tronco: 1 job + 1 di scorta.

**Criteri (congelati qui, prima del lancio):**
1. Tronco: il job 1 finisce COMPLETED con `stopped` = «limite di tempo»; il job 2 riprende e finisce a 1.400 passi (`max_steps`); il job di
   scorta esce con «gia' completo» senza torchrun. `metrics.jsonl` con i passi 1-1.400 senza buchi ne' duplicati; `lr` di ogni passo
   uguale a `_lr_at` dello schedule (warmup, fase stabile, decay); presenti `checkpoint_step0001100.pt`, `checkpoint_predecay.pt` (1.120),
   `checkpoint_postdecay.pt` (1.400).
2. Ramo: parte dopo il job 2 del tronco, `branch.json` con l'origine (passo 1.100 e SHA-256 del checkpoint), passi 1.101-1.375, `lr` uguale
   a `_lr_at` del ramo; job di scorta «gia' completo».
3. Nessuno STOP, nessun job FAILED/TIMEOUT.
Descrittivo: tempi di avvio e di ripresa, s/passo. L'uguaglianza ramo = run da zero e' provata sulla CPU, non su GPU (bf16).

**Risorse e costo:** 5 job da un nodo (4 GPU) da 15 minuti su `boost_usr_prod`, QoS `normal`: **tetto 5 GPU-ora (40 ore locali)**, stima
~2 GPU-ora (tronco ~17 minuti di passi piu' due avvii, ramo ~5 minuti, scorte ~1 minuto); checkpoint di campagna su `lrd_all_serial` (0,17
core-ora). Dal budget «preparazione dell'infrastruttura» (usate 7,71 su 25).

Precisazione prima di qualunque risultato (15/10/2026): l'origine del ramo il trainer la scrive in `branch.json` nella cartella del
ramo, non in `summary.json`; il criterio 2 e' lo stesso.

## Dopo il collaudo (15/10/2026)

Collaudo `w1_catena_v1` superato (`results/infra/w1cat_20261015.json`, `docs/decisioni.md`). Aggiunta al pianificatore la continuazione a 2D
dal pre-decay (test `test_continuation_to_2d_starts_from_the_predecay_checkpoint`). I piani veri della finestra 1 useranno i passi per epoca
dal corpus (24.528 col batch 32, `docs/manifest_annidati_d16.md`) e, per l'asse dei soggetti, `--nested` negli `args` dei run.
