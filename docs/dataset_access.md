# Dataset EMG Superficiale - Accesso e Condizioni

Verifica delle condizioni d'accesso, formato file e dimensione per 11 dataset pubblici di EMG superficiale.

| Dataset | URL | Registrazione richiesta | Formato file | Dimensione approssimativa |
|---------|-----|------------------------|--------------|--------------------------|
| NinaPro DB1-DB10 | https://ninapro.hevs.ch/instructions/DB1.html (DB1 come esempio) | Sì: login "reviewers" con password "rev2019" sul sito ufficiale | MATLAB .mat | non trovata |
| putEMG | https://biolab.put.poznan.pl/putemg-dataset/ | Nessuna: accesso libero via cloud storage (chmura.put.poznan.pl) | HDF5, CSV | non trovata |
| CapgMyo | https://figshare.com/articles/dataset/Data_from_Gesture_Recognition_by_Instantaneous_Surface_EMG_Images_CapgMyo-DBa/7210397 | Nessuna: accesso libero | MATLAB .mat | 1.31 GB |
| CSL-hdemg | http://www.csl.uni-bremen.de/CorpusData/download.php?crps=cslhdemg | Sì: richiesta email con informazioni di contatto e affiliazione | ZIP (contenuto non specificato sulla pagina ufficiale) | >2 GB |
| Hyser (HD-sEMG) | https://physionet.org/content/hd-semg/2.0.0/ | Nessuna: open access PhysioNet | WFDB (.dat, .hea) | 135.2 GB (compressed) / 142.8 GB (uncompressed) |
| EPN-612 | https://zenodo.org/records/4421500 | Nessuna: accesso libero | ZIP | 5.5 GB |
| UCI-EMG (Lobov) | https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures | Nessuna: accesso libero | Testo (.txt) | 16.9 MB |
| GRABMyo | https://physionet.org/content/grabmyo/1.1.0/ | Nessuna: open access PhysioNet | WFDB (.dat, .hea) | 9.1 GB (compressed) / 9.4 GB (uncompressed) |
| emg2pose | https://github.com/facebookresearch/emg2pose | Nessuna: accesso libero via S3 | HDF5 | 431 GB (full dataset) |
| emg2qwerty | https://github.com/facebookresearch/emg2qwerty | Nessuna: accesso libero via S3 (repository archived, read-only) | HDF5 | ~346 ore di registrazione (1,136 file) |
| Camargo 2021 (Lower Limb Biomechanics) | https://data.mendeley.com/datasets/fcgm3chfff/1 (Part 1); https://data.mendeley.com/datasets/k9kvm5tn3f/1 (Part 2); https://data.mendeley.com/datasets/jj3r5f9pnf/2 (Part 3) | Nessuna: accesso libero (CC BY 4.0) | non trovata | non trovata |

## Note

- **Colonna "Registrazione richiesta"**: "Nessuna" indica download/accesso libero; "Sì" indica che è richiesto login, email, o accordo
- **NinaPro DB1-DB10**: Le diverse sotto-versioni (DB1-DB10) sono accessibili dallo stesso sito con lo stesso meccanismo di autenticazione (login reviewers)
- **Hyser e GRABMyo**: Ospitati su PhysioNet, open access con licenza Creative Commons Attribution 4.0
- **emg2qwerty**: Repository GitHub è archived (read-only) dal 1º agosto 2026
- **Camargo 2021**: Dataset in 3 parti separate su Mendeley Data; include EMG, IMU, goniometri e dati di motion capture
- Informazioni marcate come "non trovata" non sono disponibili sulla pagina ufficiale del dataset/repository

