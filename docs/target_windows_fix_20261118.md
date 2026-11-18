# Target RMS e spettro rapido: correzioni del 18/11/2026 sera

Seguito di `5333c5c` (RMS a 100 ms) e `b4b0963` (spettro rapido a 100 ms), integrati su `main` da GPT. Simone, dopo la verifica
di Claude: «quei 3 risolvili», poi «gli altri se riesci risolvili tu» e, sulle bande, «magari sii un po elastico con le bande, se
servono 125 invece di 100 va bene lo stesso». Sul terzo punto: «sul punto 3 passiamo a 100/100».

## 1. Spettro rapido a 125 ms

A 100 ms il supporto e' di `round(0,1 fs)` campioni e la griglia FFT di `fs/round(0,1 fs)` Hz. A 1926 e 2048 Hz il secondo bin
cade a 19,96 e 19,98 Hz, appena sotto il bordo dei 20 Hz, e la banda 20-33,6 Hz contiene un bin solo; a 1000 e 2000 Hz ne
contiene due. Per lo stesso segnale la banda 1 risultava spostata di circa −0,3 in log nei dataset a 2048 Hz (Hyser, CSL-hdemg,
GRABMyo) e a 1926 Hz (NinaPro DB10). A 50 ms il problema non c'era; nessun training aveva ancora usato la versione a 100 ms.

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
- `100` e `step1_psd_power_shape_energy_fast100_v2` sono rifiutati con un messaggio esplicito.
- Stessa Hann simmetrica, stessa densita' monolatera, stesse sei bande, stesso arrotondamento dei mezzi campioni verso l'alto
  del nuovo RMS. A 200 Hz il supporto e' di 25 campioni, passo 5, overlap 20; restano disponibili le tre bande sotto 100 Hz.
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
- Le configurazioni salvate dei run storici, che non hanno i due campi, si confrontano dopo `run._config_with_defaults`
  (campo assente = 25/50).
- Strumenti corretti: `calibration_select`, `consolidation_continue`, `consolidation_long_common`,
  `logsquare_initialization_diagnostic`, `logsquare_preflight`, `logsquare_train`, `scaling_evaluation`,
  `scaling_production_memory`, `scaling_train`, `step1_ab` (quindi `step2c` e `branch_pilot`), `tokenizer_evaluation`,
  `tokenizer_preflight`. Nei costruttori di comandi (`consolidation_continue`, `logsquare_train`, `scaling_train`,
  `scaling_production_memory`) le due opzioni entrano anche nel comando.
- I checkout congelati e i lanciatori congelati non cambiano. Il test del lanciatore congelato B19618 ora verifica che, con la
  CLI di oggi, la ripresa senza ricetta storica venga rifiutata e con la ricetta storica torni compatibile. Il test della
  diagnostica log-quadratica fa girare il codice congelato `63eba9a` con la propria libreria, come in produzione.
- Un piano nuovo scrive le due finestre in modo esplicito (`--rms-window-ms`, `--ms-fast-window-ms`): senza, `calibration_select`
  e gli altri strumenti lo leggono come storico (25/50).
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
  senza i due campi con i default nuovi: normalizzare con `run._config_with_defaults`, o scrivere i campi espliciti.

## Verifiche

`tests/cpu/test_target_windows_fix.py`:
- stessi bin per banda a 125 ms su undici frequenze di campionamento, e il difetto a 100 ms;
- banda 1 del codice di produzione coerente fra 1926, 2000 e 2048 Hz su un segnale sintetico;
- rifiuto di 100 ms;
- `historical_args`, `with_historical_targets`, normalizzazione e guardia della ripresa.

Aggiornati i test dello spettro rapido (oracolo FFT indipendente a 125 ms) e quelli degli strumenti storici. Esito della suite
CPU completa nel registro. Nessun job, nessun lancio, nessun checkout o piano congelato modificato.
