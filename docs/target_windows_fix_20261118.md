# Target RMS e spettro rapido: correzioni del 18/11/2026 sera

**Dal 24/11/2026** la ricetta storica completa ha due opzioni in più, quelle del tokenizer: `--window-ms 25 --tokenizer-activation identity`.
`historical_args` scrive ora le quattro opzioni che mancano, e un piano nuovo le scrive tutte: `docs/tokenizer_default_100ms_20261124.md`.
Il resto di questa pagina descrive la consegna del 18/11.

Seguito di `5333c5c` (RMS a 100 ms) e `b4b0963` (spettro rapido a 100 ms), integrati su `main` da GPT. Simone, dopo la verifica
di Claude: «quei 3 risolvili», poi «gli altri se riesci risolvili tu» e, sulle bande, «magari sii un po elastico con le bande, se
servono 125 invece di 100 va bene lo stesso». Sul terzo punto: «sul punto 3 passiamo a 100/100».

## 1. Spettro rapido a 125 ms

A 100 ms il supporto e' di `round(0,1 fs)` campioni e la griglia FFT di `fs/round(0,1 fs)` Hz. A 1926 e 2048 Hz il secondo bin
cade a 19,96 e 19,98 Hz, appena sotto il bordo dei 20 Hz, e la banda 20-33,6 Hz contiene un bin solo; a 1000 e 2000 Hz ne
contiene due. Per lo stesso segnale la banda 1 risultava piu' bassa nei dataset a 2048 Hz (Hyser, CSL-hdemg, GRABMyo) e a
1926 Hz (NinaPro DB10): −0,61 in log sul segnale del test (toni nella banda, codice di `b4b0963`), con un valore che dipende
dallo spettro, perche' la banda sommava un bin invece di due. A 50 ms lo scarto era +0,004 sullo stesso segnale, a 125 ms e'
+0,001; nessun training aveva ancora usato la versione a 100 ms.

A 125 ms la griglia FFT e' di 8 Hz (7,99 a 1926 Hz) a ogni frequenza del corpus, e ogni banda ha gli stessi bin:

| Banda (Hz) | 20-33,6 | 33,6-56,5 | 56,5-94,9 | 94,9-159,4 | 159,4-267,9 | 267,9-450 |
|---|---:|---:|---:|---:|---:|---:|
| Bin a 1000, 1024, 1200, 1926, 2000, 2048, 2148, 4000, 4096, 5120, 10240 Hz | 2 | 3 | 4 | 8 | 14 | 23 |
| Bin a 100 ms, 2000 Hz | 2 | 2 | 4 | 6 | 11 | 18 |
| Bin a 100 ms, 2048 Hz | 1 | 2 | 4 | 6 | 11 | 19 |

| Bersaglio | Supporto | Passo |
|---|---:|---:|
| Log RMS | 100 ms | 25 ms |
| Log-potenze rapide, sei bande (`ms_fast`) | **125 ms** (storico 50 ms; 100 ms ritirato) | 25 ms |
| Forma spettrale, cinque bande | 200 ms | 25 ms |
| Log-inviluppo | 500 ms | 25 ms |
| Log-potenze lente, 24 bande | 500 ms | 100 ms |

- Nuovo default `loader.ms_fast_window_ms = 125`, versione `step1_psd_power_shape_energy_fast125_v3`.
- Ricetta storica: `--ms-fast-window-ms 50`, versione `step1_psd_power_shape_energy_v1`, identica bit per bit a prima.
- 100 ms e `step1_psd_power_shape_energy_fast100_v2` sono rifiutati con un messaggio esplicito («ritirato», «ritirata»): la
  finestra dalla configurazione del loader e dalla CLI, la versione (`RETIRED_TARGET_VERSIONS`) da `run.train` e dalla firma
  della calibrazione.
- Stessa Hann simmetrica, stessa densita' monolatera, stesse sei bande, arrotondamenti del nuovo RMS: lunghezza del supporto
  con `round` (meta' al pari: 268 campioni a 2148 Hz), inizi delle finestre con i mezzi campioni verso l'alto. A 200 Hz il
  supporto e' di 25 campioni, passo 5, overlap 20; restano disponibili le tre bande sotto 100 Hz.
- Ai bordi i supporti incompleti restano mascherati. Non cambiano teste, uscite, pesi della loss, gli altri bersagli,
  la baseline delle sonde (50/500 ms) e il frontend dei rami spettrali (50/500 ms).

## 2. Strumenti delle campagne storiche

Il contratto chiedeva che riprese e controlli storici passassero esplicitamente la ricetta storica (`--rms-window-ms 25`,
`--ms-fast-window-ms 50`). Gli strumenti che ricostruiscono run nati prima del 18/11 dai comandi o dai piani registrati non lo
facevano, e ricalcolavano la ricetta con i default nuovi. Sulla suite completa a `b4b0963` fallivano 16 test, tutti per questo
motivo; altri strumenti non coperti dai test sarebbero falliti alle guardie se usati da `main`.

Correzione:
- `train_jepa.historical_args(argv)` aggiunge `--rms-window-ms 25 --ms-fast-window-ms 50` se mancano, senza toccare valori gia'
  scritti; `run.with_historical_targets(cfg)` fa lo stesso per i costruttori che non passano dalla riga di comando.
- Le configurazioni salvate dei run storici, che non hanno i due campi, si confrontano dopo `run.with_legacy_target_windows`:
  aggiunge solo quei due campi (assente = 25/50), quindi un altro campo mancante resta un errore, come prima del 18/11.
  `run._config_with_defaults` resta alla ripresa: completa anche altri campi storici e nei confronti esatti li avrebbe allentati
  (lo ha mostrato il test del preflight del tokenizer che toglie `tokenizer_activation`).
- Dodici strumenti corretti: `calibration_select`, `consolidation_long_common`, `logsquare_evaluation`,
  `logsquare_initialization_diagnostic`, `logsquare_preflight`, `logsquare_train`, `scaling_evaluation`,
  `scaling_production_memory`, `scaling_train`, `tokenizer_evaluation`, `tokenizer_preflight`, `tokenizer_train`. Inoltre
  `step1_ab` (quindi `step2c` e `branch_pilot`), `eval_wavelet_vs_gabor` e `scripts/slurm/spectral_step1_extract.sbatch`: il
  launcher dell'estrazione, legato allo split storico dello Step 1, passa all'estrazione la ricetta storica come `step1_ab` la
  passa al training.
- Nei costruttori di comandi (`logsquare_train`, `scaling_train`, `scaling_production_memory`, `tokenizer_train`) le due
  opzioni entrano anche nel comando. Ogni confronto con un config salvato di un run storico passa da
  `run.with_legacy_target_windows`: pin del confronto, config degli endpoint, rapporto della memoria e `state['config']` del
  checkpoint in `tokenizer_train.checkpoint_identity`.
- `historical_args` riconosce anche le abbreviazioni accettate da argparse (`--ms-fast 125`) e la forma `--opzione=valore`, con
  la regola di `wsd_chain.free_arg_dests`: un valore gia' scritto non viene mai sovrascritto.
- `consolidation_continue.py` resta invariato: un test lo vuole identico byte per byte alla versione congelata `17dd455`, e la
  sua campagna gira con il codice congelato.
- Limite preesistente, non cambiato dal ramo: le versioni su `main` di `tokenizer_evaluation`, `scaling_train`,
  `scaling_evaluation`, `scaling_production_memory` e `consolidation_long_common` non validano gli artefatti veri di prima del
  04/11, ai quali mancano anche `model.tokenizer_logsquare` e `model.tokenizer_logsquare_eps` (solo `tokenizer_preflight` li
  completa). Era cosi' gia' a `e114a78`; quelle campagne girano con il proprio codice congelato. Con gli artefatti T100-LQ
  (04-05/11) i confronti corretti tornano uguali: verificato su `config_T100-LQ.json` e `completed_5000/config.json` veri.
- I checkout congelati e i lanciatori congelati non cambiano. Il test del lanciatore congelato B19618 ora verifica che, con la
  CLI di oggi, la ripresa senza ricetta storica venga rifiutata e con la ricetta storica torni compatibile. Il test della
  diagnostica log-quadratica fa girare il codice congelato `63eba9a` con la propria libreria, come in produzione.
- Un piano nuovo scrive le due finestre in modo esplicito (`--rms-window-ms`, `--ms-fast-window-ms`). Senza, il trainer
  lanciato dal piano (`wsd_chain`, `train_ddp.sbatch`) usa i default nuovi, mentre `calibration_select` lo legge come storico
  (25/50) e rifiuta il run solo dopo il training. La regola non e' ancora controllata in modo meccanico prima del lancio.
- Regola per i piani futuri: un braccio nuovo confrontato a un solo fattore con bracci storici usa la ricetta storica esplicita,
  oppure il piano dichiara i due fattori in piu'.
- `--rms-window-ms 25` da solo, con lo spettro rapido nuovo, resta accettato: e' una variazione a un fattore legittima della
  ricetta nuova (per esempio un'ablazione della finestra RMS). Per riprodurre un run storico serve la coppia, cioe'
  `historical_args`.
- `scripts/eval_wavelet_vs_gabor.py` calcola i target con la ricetta storica, per riprodurre l'esito congelato del 13/10.

## 3. Linea 3, confronto contrastivo

Simone: «sul punto 3 passiamo a 100/100». Nel riquadro di domanda era stata scelta l'opzione «restano 25/50», ma il messaggio
scritto subito dopo dice il contrario: vale il messaggio scritto. I tre training nuovi C0, CN e CR della linea 3 usano quindi la
ricetta nuova, oggi RMS 100 ms e spettro rapido 125 ms. La linea 3 e' di GPT, che deve:
- integrare `main` nel ramo della linea 3 e rifare audit, calibrazione, benchmark e λ con la ricetta nuova;
- emendare il protocollo: C0 contro T100-G differisce anche per RMS e spettro rapido;
- correggere `contrastive_prepare.config_from_dict`, che dopo l'integrazione di `main` reinterpreterebbe le ricette storiche
  senza i due campi con i default nuovi: leggere i config storici con `run.with_legacy_target_windows` e scrivere in modo
  esplicito la ricetta nuova dei bracci C0, CN e CR.

## Verifiche

`tests/cpu/test_target_windows_fix.py`:
- stessi bin per banda a 125 ms su undici frequenze di campionamento, e il difetto a 100 ms;
- banda 1 del codice di produzione coerente fra 1926, 2000 e 2048 Hz su un segnale sintetico;
- rifiuto di 100 ms e della versione `fast100_v2`;
- `historical_args` (anche con abbreviazioni e `=`), `with_historical_targets`, `with_legacy_target_windows` (solo i due campi:
  un altro campo mancante resta un errore) e guardia della ripresa.

Aggiornati i test dello spettro rapido (oracolo FFT indipendente a 125 ms) e quelli degli strumenti storici. Le fixture che
sintetizzano config storici li scrivono senza i due campi, come i config veri, cosi' i test esercitano i confronti normalizzati.
Esito della suite CPU completa nel registro. Nessun job, nessun lancio, nessun checkout o piano congelato modificato.
