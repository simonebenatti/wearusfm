# Audit GPT indipendente delle ricevute classificazione — 5 novembre 2026

**Esito: PASSATO, nessun problema rilevato** nelle quattro ricevute originali e nelle dieci provenienze raccolte in `results/infra/scaling_300m_20261104/classification_completed_20261105/` del worktree scaling. La conclusione include la **nuova P2 300M**, che la review di Claude consegnata alle 13:41 dichiarava ancora in corso. Questi controlli sono di GPT e non vengono attribuiti a Claude.

## Integrità e identità

- Ricalcolati SHA256 e lunghezza **dopo decompressione gzip** delle quattro ricevute; ricalcolati SHA256 e lunghezza delle dieci provenienze. **14/14** coincidono con `collection_manifest.json`, SHA `2f0e30d3ac0409cdc1c75d3d08c27733a147002cb1c7205c2cd82129e867ffa9`. Verificati percorsi relativi/locali e corrispondenza ai percorsi remoti dichiarati.
- Le quattro `receipt_summary.json` sono riduzioni fedeli degli originali: campi presenti identici, `checkpoint_guard` differente solo per l'omissione dichiarata degli `states`, SHA/bytes della ricevuta sorgente corretti. L'audit ha usato gli **originali completi**, non soltanto questi riepiloghi.
- Tutte: protocollo `scaling-evaluation-20261104-v1`, `complete=true`, `checkpoint_unchanged=true`, `postload_recipe_passed=true`, `stage=extract-classification`, **teacher a 5.000**, estrazioni da `aef665ba7ea66fc80291846487907bb3ef9107df`, piano SHA `8b358951182f02dc6af8fafd2621f2dc1ae2eb98939e67c5c914ecc09933528d`.
- Checkpoint 30M SHA `49dd518b5eea632a16bc7ff88ab0fb013f30d61458164c9521867572c83daa80`; checkpoint 300M SHA `37279995d0d484e241b032eb9b6b23a31c16a5947a0e2cf0c575da0cb70641aa`. Pin di checkpoint/config/receipt/summary, dose, codice del training e configurazioni modello coincidono con il piano e le copie del training. Snapshot5000 storica corretta per il 30M.
- Verificati per ogni ricevuta i pin dei sei input, confronto, contratto, **103 sorgenti scientifici** e **cinque primitive di estrazione**; SHA corrispondenti anche ai byte locali. Ricostruiti i comandi primitivi, includendo la risoluzione corretta del percorso `splits` relativo dentro il checkout congelato: argv scientifici identici, interpreter e controller Leonardo corretti.
- **Guadagno 548,6939187567021 identico nelle quattro ricevute**, finito e positivo; policy `exact saved buffer; no evaluation-data calibration`. Il guard dichiara controllo post-caricamento del teacher e del guadagno; la review locale non ricarica i checkpoint.
- Runtime identico in tutte: Python3.11.6, NumPy1.26.4, Torch2.2.0a0+git6c8c5ad, CUDA12.1, devicecuda, un thread Torch/OMP/MKL/OpenBLAS.

| Ricevuta originale | SHA256 non compresso |
|---|---|
| 30M P1 | `4634ac05f7689cefa4c3ee0aa24848324d8a6738eb8576b6b9e971d9364986f5` |
| 30M P2 | `0cf0d507a0be62f6af1c0645368568eec9b6b2e5e4715efbd8d69b8dfe027fae` |
| 300M P1 | `03abef4652024b9e0a78f52f9115eb028b7cb32a47db14f2b3625e12c57518fd` |
| **300M P2** | **`172aa79b27bf3783d89c89f4213913f9547011473c71101bd7f14c85e58e00eb`** |

## Parità dei cinque dataset fra 30M e 300M

Per ogni dataset coincidono **numero di finestre/utenti, input_sha256, splits_sha256, ricetta di estrazione, ordered_rows_sha256, fold_roles_sha256, reference_cache_sha256 e dtype**. I riferimenti coincidono con quelli congelati del piano. P1 usa lo split firmato; P2 usa l'identità congelata dello split interno. Gli inventari dei dati grezzi e i loro stati registrati sono identici fra i due bracci: 360 file P1 e 684 P2, tutti attestati invariati durante ogni estrazione.

| Dataset | Finestre | Utenti | Pool30M | Pool300M |
|---|---:|---:|---:|---:|
| NinaPro DB2 | 38.544 | 32 | 768 | 1.792 |
| NinaPro DB3 | 9.834 | 8 | 768 | 1.792 |
| NinaPro DB6 | 33.961 | 8 | 768 | 1.792 |
| EPN612 | 47.100 | 314 | 768 | 1.792 |
| UCI EMG | 364 | 19 | 768 | 1.792 |

Ogni provenienza coincide con l'output della propria ricevuta per path/hash/bytes della cache e per hash della provenienza stessa. Il pool è float32 in tutti i casi; la larghezza cambia come previsto da D384 a D896. La ricetta è teacher, media sui canali/patch secondo il protocollo congelato, senza nuova calibrazione.

## Nuova P2 300M

Ricevuta completa di **248.449 byte**, elapsed interno **5.888,139705 s** (98,14 min), inferiore ai150 minuti della cella; questo tempo interno non sostituisce `sacct` nel consuntivo.

- EPN612: cache SHA `0c327ceb51930458aa61720981a999b2532d03a93fed7f3fa6a40b08d71b8dea`, 1.693.610.782 byte, shape pool `[47100,1792]`; provenienza SHA `5e4c567b4cad56d528f8d16451af58c4b70528123d2a6ca006c9f596df01d7a7`.
- UCI: cache SHA `39e121da65eb98b3334d74525e4b8ccd791c8b9055bbf9a8685522314f703ce8`, 13.061.126 byte, shape pool `[364,1792]`; provenienza SHA `97e515f62b8bad7b2aa0508f8ead8dcc949514ab929e54954c300b22f90ea82a`.
- EPN612 ordered rows SHA `c472630351616e4a2ba0ebb7dddf98e370dc08ad311b7f462d1333513902a67c`, fold SHA `20cac9e3a35ab37bd4b95d14a08fbc457b2cd244ad13e7d7305aa3e049a91743`: entrambi esattamente uguali al30M.
- UCI ordered rows SHA `43c626313385a604c8e46eeaa626d18b5abac38a37a23dfa04029673cfd22b5c`, fold SHA `1889d4291142c093437de15e3ec0747d1c139020e07ce4ebd877a546a20d99d8`: entrambi esattamente uguali al30M.

Il verificatore frozen confronta gli array di etichette/utenti con dtype e uguaglianza esatta, compreso EPN612 testuale; non applica `isfinite` alle etichette. La finitezza è richiesta al pool numerico. Nessuna ricomparsa del precedente errore di guardia sulle etichette testuali nella catena esaminata.

## Limiti e seguito

Audit **solo locale**: nessun SSH, nessun job, nessuna modifica al repository. Sono stati verificati i byte delle ricevute/provenienze contro il manifest e la loro coerenza con codice/piano; le cache `.npz` e i checkpoint non sono locali e non sono stati riaperti/rihashati da questo revisore. Supporto, labels, finitezza e uguaglianza del gain sono attestazioni del controller congelato, con i vincoli riletti nel codice. L'inventario raw registra stat dei file, non un nuovo hash integrale di ogni sorgente; le provenienze riportano anche l'impronta dell'input usato.

Nessun problema che imponga rigenerare queste quattro estrazioni. I lettori CPU devono comunque rileggere e validare le cache effettive e le ricevute pinnate; prima dei fit restano necessari i benchmark e il preventivo misurato. Nessuna lettura scientifica del trasferimento è prodotta da questo audit.

Aggiornamento prima della consegna: root ha pubblicato la nuova lettura remota `remote_sha256.txt` (SHA `0e6edb7ff52b86971c39f2bd405dcfd0763de898767750026a3baee8ee93bef1`). Ho riconciliato localmente **24/24 voci**: quattro ricevute, dieci provenienze e dieci cache coincidono con manifest e ricevute originali. Questo aggiunge la concordanza degli hash remoti raccolti da root; non implica che questo revisore abbia eseguito SSH o riaperto le cache.
