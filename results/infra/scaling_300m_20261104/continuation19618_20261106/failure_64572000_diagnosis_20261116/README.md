# Fallimento dell'estrazione p1 64572000 (16/11/2026) e diagnosi

Job `SCALING-300M-eval19618-p1-20261106` = 64572000: `FAILED 1:0` su lrdn0248 dopo 31 min 22 s (09:49:49-10:21:11), dal
checkout `scaling_300m_evaluation19618_20261106_v1` a `df2de35`. L'estrazione era finita: il sottoprocesso ha scritto le tre
cache ed e' uscito con codice 0 («estrazione finita in 1743 s», `failed_job/p1.log`). Il job si e' fermato dopo, nel controllo
d'identita' delle uscite (`scaling_evaluation19618.execute` → `tokenizer_preflight.file_identity`):
`ValueError: input changed while hashing: .../evaluation19618_v1/T100-G-300M/p1/ninapro_db6.npz` (`slurm_logs/`).
Il controllo confronta inode, dimensione, mtime e ctime prima e dopo lo sha256; il messaggio non dice quale campo e' cambiato.
`ninapro_db2` e `ninapro_db3` hanno superato il controllo e la verifica e hanno la loro provenance; `ninapro_db6` non ha
provenance, `receipt.json` manca. La cartella resta com'e' (`evaluation_chain_plan.md` §7).

| Cartella | Contenuto |
|---|---|
| `slurm_logs/` | log SLURM del job 64572000 (traceback) |
| `failed_job/` | `p1.log` del sottoprocesso d'estrazione |
| `readings/` | `sacct` delle tre estrazioni; `stat` delle cache a 19.618 (p1, pose, p2) e delle p1 a 5.000 |

Tempi (granularita' di un secondo su Lustre). Scrittura: nascita e ultima modifica del file. Controllo: dalla provenance
precedente alla provenance del file, o alla fine del job.

| Cache | Byte | Scritta | Controllo | Distanza | Esito |
|---|---|---|---|---|---|
| 19.618 p1 `ninapro_db2` | 1.934.763.898 | 10:05:14-10:05:15 | 10:20:59-10:21:07 | ~16 min | superato |
| 19.618 p1 `ninapro_db3` | 493.636.450 | 10:08:32-10:08:33 | 10:21:07-10:21:08 | ~13 min | superato |
| 19.618 p1 `ninapro_db6` | 2.191.580.650 | 10:20:57-10:20:59 | 10:21:08-10:21:11 | ~9 s | **fallito** |
| 19.618 p2 `epn612` | 1.693.610.834 | 11:30:34-11:30:35 | 11:31:00-11:31:07 | ~25 s | superato |
| 19.618 pose `val` | 5.305.413.139 | 11:49:52-11:52:52 | 11:52:52-11:53:49 | subito, scrittura in 3 min | superato |
| 5.000 T100-G-300M p1 `ninapro_db6` | 2.191.580.598 | 13:19:24-13:19:25 | 13:19:33-13:19:40 | ~8 s | superato |
| 5.000 T100-G-30M p1 `ninapro_db6` | 939.641.974 | 13:02:50 | 13:02:55-13:02:59 | ~5 s | superato |

Lettura (interpretazione, non verificata): gli attributi di un file grande appena scritto sono cambiati durante il suo hash,
senza processi che scrivessero. I 2,19 GB sono stati scritti in circa 2 s, quindi nella cache del nodo; la causa plausibile e'
lo scarico differito di quei dati da parte del client Lustre, che aggiorna gli attributi letti da `stat` mentre l'hash corre.
E' una corsa fra lo scarico e il controllo d'identita', non un cambiamento dei dati o degli ingressi scientifici. La stessa
situazione (stesso file, stessa dimensione, ~8 s) e' passata a 5.000: la corsa non e' sistematica. Non verificati: il campo
cambiato e il contenuto di `ninapro_db6.npz`, che la verifica (`verify_classification_cache`) non ha mai letto.

Conseguenze. La catena si ferma (§7). pose 64571999 e p2 64572001 sono `COMPLETED 0:0` con ricevuta, non ancora raccolte:
`collect extractions` e il cancello degli stadi successivi chiedono le tre estrazioni `COMPLETED 0:0`. Un rilancio e' fuori dal
pacchetto rivisto (§1) e richiede una decisione di Simone, un ID nuovo e un piano con i suoi pin (§7).

Consuntivo delle tre estrazioni (1 GPU e 8 CPU ciascuna, `billing=8`): 15.179 s di GPU = 4,216389 GPU-ore, 33,731111 core-ore
(pose 7.330 s = 2,036111; p1 1.882 s = 0,522778; p2 5.967 s = 1,6575). Tetto dello stadio 7 GPU-ore (56 ore locali): residuo
2,783611 GPU-ore.

Recuperi possibili (decide Simone):

- A. Rilancio unico della sola p1 dallo stesso checkout congelato `df2de35` e con lo stesso piano d'estrazione, ID nuovo
  `SCALING-300M-eval19618-p1-r1-20261116`, cartella nuova `.../evaluation19618_v1/T100-G-300M/p1-r1`. Modifica solo lato Mac in
  `scripts/scaling_plans19618.py` (ID e cartella della p1 per cancello, righe, raccolta e piani dei lettori), con test.
  Costo atteso ~0,52 GPU-ore, tetto 2 (stadio al massimo 6,216389 su 7: nessun emendamento). Rischio dichiarato: i tempi del
  controllo restano gli stessi e la corsa puo' ripetersi; in quel caso stop.
- B. Correzione del controllo (scaricare il file, per esempio con `os.fsync`, prima della prima lettura degli attributi) in un
  nuovo checkout del codice di valutazione, poi riestrazione della p1. Toglie la corsa, ma cambia il codice congelato: la p1
  avrebbe un commit di estrazione diverso da pose e p2 e i lettori dovrebbero accettarne due. Piu' codice e una review.
- Esclusa: adottare le cache gia' scritte con un job di sola verifica, che scriverebbe nella cartella fallita (§7) con codice
  nuovo.
