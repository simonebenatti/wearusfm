# Dataset EMG Superficiale - Accesso e Condizioni

Verifica delle condizioni d'accesso, formato file e dimensione per 11 dataset pubblici di EMG superficiale.

| Dataset | URL | Registrazione richiesta | Formato file | Dimensione approssimativa |
|---------|-----|------------------------|--------------|--------------------------|
| NinaPro DB1-DB10 | https://ninapro.hevs.ch/instructions/DBn.html (n=1..10) | **Corretto il 21/09/2026: nessuna registrazione.** Verificato scaricando un file reale da DB2 (`https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s1.zip`, >10MB, nessun login incontrato). Le pagine di istruzioni linkano direttamente agli zip per soggetto | ZIP (contenuto .mat) | non trovata |
| putEMG | https://biolab.put.poznan.pl/putemg-dataset/ | Nessuna: accesso libero via cloud storage (chmura.put.poznan.pl), WebDAV pubblico (vedi nota) | HDF5, CSV | 712 record totali (264 emg_gestures + 448 emg_force), 44 partecipanti; cartella Data-HDF5 completa ~30.9GB, sottoinsieme emg_gestures stimato ~11GB |
| CapgMyo | https://figshare.com/articles/dataset/Data_from_Gesture_Recognition_by_Instantaneous_Surface_EMG_Images_CapgMyo-DBa/7210397 | Nessuna: accesso libero | MATLAB .mat | 1.31 GB |
| CSL-hdemg | http://www.csl.uni-bremen.de/CorpusData/download.php?crps=cslhdemg | Sì: modulo di registrazione sulla pagina, link di download personali inviati via email | ZIP, spezzato in 5 parti (schema `split`: partaa..partae) da concatenare con `cat` prima di estrarre | **12,9 GB** (verificato scaricando ed estraendo, 22/09/2026) |
| Hyser (HD-sEMG) | https://physionet.org/content/hd-semg/2.0.0/ | Nessuna: open access PhysioNet | WFDB (.dat, .hea) | 135.2 GB (compressed) / 142.8 GB (uncompressed) |
| EPN-612 | https://zenodo.org/records/4421500 | Nessuna: accesso libero | ZIP | 5.5 GB |
| UCI-EMG (Lobov) | https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures | Nessuna: accesso libero | Testo (.txt) | 16.9 MB |
| GRABMyo | https://physionet.org/content/grabmyo/1.1.0/ | Nessuna: open access PhysioNet | WFDB (.dat, .hea) | 9.1 GB (compressed) / 9.4 GB (uncompressed) |
| emg2pose | https://github.com/facebookresearch/emg2pose | Nessuna: accesso libero via S3 | HDF5 | **462.824.048.640 byte (~431 GiB)** - Content-Length verificato dalla sorgente S3 e confermato sul file scaricato, 23/09/2026 |
| emg2qwerty | https://github.com/facebookresearch/emg2qwerty | Nessuna: accesso libero via S3 (repository archived, read-only) | HDF5 | ~346 ore di registrazione (1,136 file); **308.382.645.571 byte (~287 GB)**, Content-Length verificato dalla sorgente e confermato sul file, 23/09/2026 (ri-scaricato su $SCRATCH dopo la cancellazione del 22/09 per liberare quota $WORK) |
| Camargo 2021 (Lower Limb Biomechanics) | https://data.mendeley.com/datasets/fcgm3chfff/1 (Part 1); https://data.mendeley.com/datasets/k9kvm5tn3f/1 (Part 2); https://data.mendeley.com/datasets/jj3r5f9pnf/2 (Part 3) | Nessuna: accesso libero (CC BY 4.0) | non trovata | non trovata |

## Note

- **Colonna "Registrazione richiesta"**: "Nessuna" indica download/accesso libero; "Sì" indica che è richiesto login, email, o accordo
- **NinaPro DB1-DB10**: nessuna registrazione necessaria (corretto il 21/09/2026 — la
  riga precedente di questa tabella, e `docs/fm_emg_reference_v10.md` §2.2, riportavano
  erroneamente "account e accettazione dei termini"; il nome file cambia per DB e va
  letto dalla pagina di istruzioni di ciascun DB, non indovinato: vedi
  `scripts/slurm/download_ninapro.sbatch`)
- **Hyser e GRABMyo**: Ospitati su PhysioNet, open access con licenza Creative Commons Attribution 4.0
- **emg2qwerty**: Repository GitHub è archived (read-only) dal 1º agosto 2026
- **Camargo 2021**: Dataset in 3 parti separate su Mendeley Data; include EMG, IMU, goniometri e dati di motion capture
- Informazioni marcate come "non trovata" non sono disponibili sulla pagina ufficiale del dataset/repository
- **CSL-hdemg**: i 5 file scaricati (`part_a.zip`...`part_e.zip`) NON sono zip
  indipendenti — è un singolo archivio spezzato con `split` (nomi lato server
  `partaa`..`partae`, il classico schema a due lettere). `file` sui singoli pezzi
  segnala "data" o addirittura riconoscimenti bizzarri (es. "DIY-Thermocam raw data")
  perché il contenuto compresso in mezzo a un archivio ha entropia alta e nessun
  header riconoscibile — **non è corruzione**. Vanno riuniti con
  `cat part_a.zip part_b.zip part_c.zip part_d.zip part_e.zip > csl_hdemg.zip`
  prima di validare o estrarre.
- **Quota $WORK esaurita, 22-23/09/2026**: la project quota di IscrB_WearUsFM su $WORK e'
  1TB soft (condivisa da 11 utenti), non i "100% pieno" mostrati inizialmente da `df -h`
  (che sulla mount Lustre condivisa riporta numeri sbagliati/in ritardo - usare invece
  `lfs quota -hp <project-id> $WORK`, vedi CLAUDE.md). Il superamento ha causato due
  fallimenti reali (`OSError: Disk quota exceeded`): Hyser fermato a 119GB su ~143GB
  (cancellato, da riscaricare) ed emg2pose fermato a 311GB su 431GB (il messaggio di
  riepilogo del vecchio script diceva erroneamente "completo": non verificava il
  risultato, solo un `echo` fisso - corretto in `download_emg2pose_full.sbatch`).
  Liberato spazio cancellando **emg2qwerty (288GB, riscaricabile da S3)** invece di
  emg2pose, su richiesta esplicita di Simone (emg2pose ha priorita' piu' alta per il
  progetto). Soluzione strutturale: scoperto uno scratch personale non condiviso,
  `$SCRATCH` (vedi CLAUDE.md), usato ora per Hyser e il completamento di emg2pose invece
  di $WORK. Nessuna richiesta di aumento quota a CINECA inviata per ora (non piu'
  urgente con $SCRATCH disponibile).
  **Chiusura, 23/09/2026**: tutti e tre i dataset coinvolti sono ora completi e
  verificati byte-per-byte contro il Content-Length della sorgente: Hyser (143GB,
  $SCRATCH), emg2pose (462.824.048.640 byte, $SCRATCH), emg2qwerty (308.382.645.571
  byte, $SCRATCH, ri-scaricato su richiesta di Simone dopo il completamento di
  emg2pose). GRABMyo resta su $WORK (9.5GB, gia' completo prima della crisi).
- **putEMG, downloader ufficiale rotto, 23/09/2026**: il repo `putemg-downloader`
  clonato in `$WORK/data/raw/putemg/putemg-downloader/` usa un URL Nextcloud vecchio
  stile per leggere `records.txt` (`.../s/<token>&files=records.txt`, senza "?" -
  404 su Nextcloud moderno). Il link di condivisione pubblico
  (`https://chmura.put.poznan.pl/s/45NY5snj0U4tgQz`) e' comunque valido: fix trovato
  usando il **WebDAV pubblico standard di Nextcloud**
  (`https://chmura.put.poznan.pl/public.php/webdav/`, username = token di
  condivisione, password vuota), che funziona sia per elencare (`records.txt`,
  PROPFIND) sia per scaricare i singoli file. Vedi `scripts/slurm/download_putemg.sbatch`,
  che sostituisce lo script del repo per il nostro caso d'uso (solo `emg_gestures`,
  solo HDF5 - niente `emg_force`/CSV/video/depth).

