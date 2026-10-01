# Inventario del corpus (dai riepiloghi di ingest, 01/10/2026)

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
| NinaPro DB5 | 10 | 8.8 | 200 | 16 | 2 anelli (Myo) | pretraining (fuori da vista canonica e ancora RVQ) |
| Hyser | 20 | 20.2 | 2048 | 256 | 4 griglie 8x8 | pretraining |
| emg2qwerty | 108 | 346.1 | 2000 | 32 | 2 anelli (Meta, sx+dx) | pretraining |
| NinaPro DB8 (2 amputati) | 12 | 8.5 | 2000 (banda 0-555) | 16 | 2 anelli da 8 (amputati: sparsi) | pretraining |
| Zhang 2026 | 62 | 24.9 | 2000 (4 canali da 4000) | 8 | sparso: 8 muscoli / 8 attorno all'avambraccio | pretraining |
| **Totale ingerito** | **537** | **618.8** | | | | |

\* soggetti da documentazione, non da un ingest.

Attenzione: la somma dei soggetti NON e' il numero di persone distinte: la sovrapposizione fra i DB NinaPro (fatto n. 2) non e' verificata, e Kaifosh conta utenti diversi da quelli di emg2pose/emg2qwerty solo per costruzione degli identificatori, non per verifica.

## Non ancora ingeriti (stime dichiarate, NON misure)

| Dataset | fs (Hz) | Canali | Topologia | Soggetti | Ore | Fonte della stima | Stato |
|---|---|---|---|---|---|---|---|
| emg2pose | 2000 | 16 | anello (Meta) | 193 | n.d. | 25.253 registrazioni di 193 utenti (CSV di metadati) | parser provato su 2 file veri (30/09) |

Solo harness (fuori dal pretraining): EPN-612 (200 Hz, 8 canali), UCI-EMG (1 kHz, 8 canali). Esclusi: NinaPro DB1 (100 Hz, inviluppo), DB9 (solo cinematica). DB10 (MeganePro): download dei soli .mat di MDS1/2/4 in corso (01/10/2026), parser da scrivere.
