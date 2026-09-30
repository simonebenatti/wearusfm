# Inventario del corpus (dai riepiloghi di ingest, 30/09/2026)

Generato da `scripts/corpus_inventory.py`. Ore = durata dei dati ingeriti (`hours_total` dei report), soggetti = soggetti/utenti distinti del report.

## Ingeriti

| Dataset | Soggetti | Ore | fs nativa (Hz) | Canali | Topologia | Ruolo |
|---|---|---|---|---|---|---|
| CapgMyo-DBa | 18 | 0.4 | 1000 | 128 | griglia HD | pretraining |
| GRABMyo | 43 | 21.3 | 2048 | 28 | 4 anelli | pretraining |
| putEMG | 44* | 21.4 | 5120 | 24 | 3 anelli | pretraining |
| CSL-hdemg | 5 | 6.0 | 2048 | 168 | griglia HD | pretraining |
| Camargo 2021 | 22 | 20.1 | 1000 | 11 | sparso (arto inferiore) | pretraining |
| Kaifosh (Discrete Gestures) | 100 | 63.9 | 2000 | 16 | anello (Meta) | pretraining |
| NinaPro DB2 | 40 | 28.8 | 2000 | 12 | anello 8 + 4 mirati | pretraining |
| NinaPro DB3 (amputati) | 11 | 7.6 | 2000 | 12 | anello 8 + 4 mirati | pretraining |
| NinaPro DB4 | 10 | 7.6 | 2000 | 12 | anello 8 + 4 mirati | pretraining |
| NinaPro DB6 | 10 | 19.7 | 2000 | 16 | anello 8 + 6 distali (+2 vuote) | pretraining |
| NinaPro DB7 (2 amputati) | 22 | 13.4 | 2000 | 12 | anello 8 + 4 mirati | pretraining |
| **Totale ingerito** | **325** | **210.3** | | | | |

\* soggetti da documentazione, non da un ingest.

Attenzione: la somma dei soggetti NON e' il numero di persone distinte: la sovrapposizione fra i DB NinaPro (fatto n. 2) non e' verificata, e Kaifosh conta utenti diversi da quelli di emg2pose/emg2qwerty solo per costruzione degli identificatori, non per verifica.

## Non ancora ingeriti (stime dichiarate, NON misure)

| Dataset | fs (Hz) | Canali | Topologia | Soggetti | Ore | Fonte della stima | Stato |
|---|---|---|---|---|---|---|---|
| NinaPro DB5 | 200 | 16 | 2 anelli (Myo) | 10 | ~8 h | parser pronto; 10 soggetti x ~0,8 h misurati in locale sui file reali | in coda (job 59054445) |
| Hyser | 2048 | 256 | 4 griglie 8x8 | 20 | n.d. | 76 GB di `*_raw_*`: ore da calcolare a ingest fatto | in coda (job 59054442) |
| emg2qwerty | 2000 | 32 | 2 anelli (Meta, sx+dx) | n.d. | ~346 h | documentazione (dataset_access.md): ~346 h, 1.136 file | parser scritto |
| emg2pose | 2000 | 16 | anello (Meta) | 193 | n.d. | 25.253 registrazioni di 193 utenti (CSV di metadati) | parser scritto |
| Zhang 2026 | 2000 + 4000 | 8 | anatomico vs equidistante | 64 | n.d. | 1.245 file CSV, 35 GB | serve una decisione di schema |

Solo harness (fuori dal pretraining): EPN-612 (200 Hz, 8 canali), UCI-EMG (1 kHz, 8 canali). Esclusi: NinaPro DB1 (100 Hz, inviluppo), DB8 e DB10 (raw non scaricato), DB9 (solo cinematica).
