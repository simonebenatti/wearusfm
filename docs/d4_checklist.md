# D4 — Checklist di firma dei fatti (aggiornata il 30/09/2026)

Scadenza D4: 11/10/2026. Il registro completo, con fonti e citazioni, è `docs/fatti_da_verificare.md`. Questa pagina
serve a firmare in fretta: per ogni fatto dice **cosa controllare** e **cosa può cambiare una decisione**.
Regola: «raccolto» non vuol dire «verificato». Firmare = aver guardato la fonte e essere d'accordo.

## 1. Raccolti da fonte, da firmare
| # | Fatto | Cosa guardare | Cosa cambierebbe |
|---|---|---|---|
| 7b | Kaifosh «Discrete Gestures»: CC-BY-NC-4.0, 100 partecipanti, 2 kHz | README di `facebookresearch/generic-neuromotor-interface` | uso non commerciale: ok per ricerca, non per ridistribuire |
| 9 | `flash_attn_varlen_func` funziona in bf16 su A100 (job 58134850) | `results/step0/flash_attn_varlen_check_58134850.json` | il Perceiver del piano |
| 12 | putEMG: 24 elettrodi, 3 fasce da 8, 5120 Hz, CC BY-NC 4.0 | pagina/paper putEMG | montaggio nello schema |
| 15 | UCI-EMG: 1 kHz | Lobov et al. 2018, Sensors 18(4):1122 | solo harness |
| 16 | GRABMyo: 28 canali EMG in 4 anelli, U1–U4 non EMG | pagina PhysioNet | schema a 4 gruppi |
| 17 | putEMG: mappa TRAJ_GT → gesto | `putemg_examples/shallow_learn.py` righe 142–151 | etichette dell'harness |
| 19 | NinaPro DB5: 2 Myo da 8, secondo ruotato di 22,5° — **FIRMATO il 30/09** | ninapro.hevs.ch/instructions/DB5.html | anelli di DB5 |
| 20 | Bracciale Meta sEMG-RD: 16 canali differenziali, 20–850 Hz, 2 kHz — **FIRMATO il 30/09 (senza i dettagli di PMC12818089)** | paper emg2qwerty (arXiv 2410.20081) e repo del dataset | banda effettiva, tutti i dataset Meta |
| 21 | NinaPro DB2/3/4/6/7: 12 o 14 elettrodi Delsys/Cometa, 2 kHz — **FIRMATO il 30/09** | pagine ufficiali DB2…DB7 | montaggi misti |
| 22 | DB7: amputati = soggetti 21 e 22; DB6: 16 colonne di cui 2 vuote — **FIRMATO il 30/09** | pagine DB6 e DB7 | anatomia nominale |
| 11 | Hyser: 256 canali, 4 griglie 8×8 (ED/EP/FD/FP), 2048 Hz | `readme.txt` del dataset | griglie HD |

## 2. Da guardare con attenzione prima di firmare
- **Licenze «NoDerivatives»:** NinaPro DB4 e DB5 sono CC BY-ND 4.0. **Decisione di Simone 30/09: si includono.** (AG non ha valutato la
  compatibilità legale: è una decisione d'uso, non una verifica.)
- **DB2 «CC0»:** indicata dal fact-checker come default di Dryad, non come citazione: **non verificata**. **Licenze mancanti:** DB3, DB6, DB7
  (nessuna trovata). **Decisione di Simone 30/09: si includono senza problemi.** Hyser ODC-By 1.0 (attribuzione).
- **Frequenza di rete** (50 o 60 Hz) è assunta in tutti i sidecar dei dataset ingeriti dopo GRABMyo: nessuna fonte
  la dichiara. Ha effetto solo su notch e augmentation.

## 3. Parziali o aperti
| # | Stato | Nota |
|---|---|---|
| 4 | NeuroRVQ: transformer mescola i canali (sì); normalizzazione in ingresso (assente nel codice d'esempio); **banda di pretraining non trovata** (20–400 Hz nel codice, 20–90 Hz nel paper solo per il downstream) | decisa per progetto la vista a 20–400 Hz (`decisioni.md`) |
| 5 | licenza del codice NeuroRVQ = CC BY-NC 4.0; **licenza dei pesi e sovrapposizione dei soggetti non verificate** | Simone 30/09: verificare la licenza dei pesi «non serve»; il fatto resta non verificato |
| 1 | frequenze di DB8 (~1111 Hz) e DB10 (~1926 Hz): **il raw è vuoto** (download mai riuscito) | download **autorizzato da Simone il 30/09** (`decisioni.md`), da fare col cluster; poi verifica su file veri |
| 2 | sovrapposizione soggetti fra DB NinaPro | **non calcolabile dagli ingest**: gli ID soggetto sono per-database (s01 di DB2 non dice nulla su s01 di DB3); serve una fonte (pagine ufficiali o paper). Correzione del 30/09: prima c'era scritto il contrario |
| 3 | conteggio soggetti e ore del corpus | ora calcolabile dai report di ingest in `results/passo2/` |
| 8 | orientamento delle fasce, dataset per dataset | non trovato per i bracciali Meta, DB5 (solo rotazione relativa), DB6, DB7 |
| 14, 18 | ordine/griglia di CapgMyo e putEMG | assunti nel codice, non verificati |
| 7a, 10 | curve di scaling di Kaifosh (paywall) e MFU reale sul Booster | non fatti |

## 4. Decisioni che aspettano te (oltre alle firme)
- **D5b:** presa il 30/09: **l'ancora RVQ non si scarta** (deroga consapevole: V3 e V4 non passano) e **si restringe** ai dataset sotto V2. **Codici firmati il 30/09** (solo livello 0, rami scelti con una regola congelata): `docs/proposta_ancora_rvq.md`. Le misure aspettano il cluster.
- **Misura del QC** (soglia assoluta contro relativa, 7 ingest): autorizzata il 30/09, parte quando il cluster torna.
- **Zhang 2026:** schema per un CSV con EMG a 2000 Hz e 4000 Hz più ACC/GYRO a ~74–148 Hz.
- **NinaPro DB8 e DB10:** riscarico autorizzato il 30/09; parte quando il cluster torna.
- **Cancellazione degli originali** già copiati su scratch: un comando (`scripts/slurm/delete_verified_raw.sbatch`).
