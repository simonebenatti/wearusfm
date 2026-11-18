# SEEDS: review indipendente dell'acquisizione e dell'audit

**18/11/2026 — FAVOREVOLE all'acquisizione e all'audit strutturale, dopo la correzione del limite4h.** Reviewer: subagente GPT `seeds_comparison_plan`, diverso dagli autori degli script. Nessun lancio, commit o modifica ai sorgenti effettuato dal reviewer. Il protocollo scientifico è stato rettificato nel file assegnato; questa review **non qualifica né autorizza i fit FM/FT/TEMPONet**.

## Perimetro e controlli effettivamente eseguiti

Lettura integrale di `scripts/seeds_download.py`, `scripts/seeds_inventory.py`, `scripts/seeds_audit.py`, dei due launcher `scripts/slurm/seeds_download.sbatch` e `seeds_audit.sbatch`, del piano `docs/seeds_acquisition_execution_20261118.md`, della policy risorse e della ricevuta dei limiti scheduler. Letto inoltre il contratto `lib_code_dir.sh` utilizzato dai launcher e i test di audit per verificare l'indipendenza delle fixture.

Rieseguiti **70test**, tutti verdi in6,38s:

```text
~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider \
  tests/cpu/test_seeds_download.py tests/cpu/test_seeds_inventory.py tests/cpu/test_seeds_audit.py
```

`bash -n` sui due launcher: superato. Ricalcolati direttamente dal manifest:5.880file,59.737.275.276byte,25soggetti, nodi `wa3qk`, `yp4vq`, `e7m5k`; file massimo21.996.433byte. Questa riconta è sui metadati congelati, non sui60GB scaricati. L'audit locale di due file comunicato dall'autore non viene presentato come audit completo o come una nuova esecuzione di payload del reviewer.

## Rilievo bloccante trovato e chiuso prima del freeze

**R1 — walltime download8h incompatibile con la seriale.** La prima versione richiedeva08:00:00 su `lrd_all_serial`, mentre i vincoli documentati indicavano4h. Segnalato al coordinatore prima del freeze. La lettura live registrata in `results/infra/seeds_20261118/scheduler_limits.json`,18/11 ore13:54UTC, conferma `MaxTime=04:00:00` e QoS `MaxWall=04:00:00`.

Correzione riletta: launcher download04:00:00, policy240min, piano coerente; tetto totale **12,1667core-ore richieste, zeroGPU** (checkout1CPU×10min, download2CPU×4h, audit2CPU×2h). La proposta8h non è stata sottomessa. Il tetto non è una previsione di completamento: velocità e latenza OSF non sono ancora misurate sul trasferimento completo. Un timeout richiede diagnosi; non si introduce un rilancio automatico per completare il volume.

**Nessun altro difetto materiale bloccante trovato nel pacchetto di acquisizione corretto.**

## Perché la catena è adeguata a questa fase

- L'inventario attraversa i componenti ufficiali e congela percorso, dimensione, SHA256/MD5, versione e sorgente; controlla collisioni e paginazione, registra materiali non acquisiti. Non confonde la root dei primi dieci soggetti con l'intero dataset.
- Il downloader valida tutti i percorsi prima della rete, rifiuta traversal/symlink e collisioni, vincola la destinazione a `external_benchmarks/seeds` con marker di esclusione dal pretraining. Scrive `.part`, verifica size/hash, pubblica senza sovrascrivere tramite link atomico. Un file completo corrotto è un errore, non una riparazione silenziosa.
- Massimo quattro trasferimenti; retry limitati a errori di trasporto/429/5xx, non a checksum errati. Dopo il primo errore non vengono pianificati nuovi file; i trasferimenti già in corso possono finire. Ricevuta progressiva e byte di rete conservati. Il controllo dello spazio include5GiB oltre ai byte ancora da scrivere.
- L'audit rilegge hash e MATLAB, valida identità e combinazioni pubblicate, frequenze, forme, nomi/ordine dei canali, valori finiti, calibrazioni e durate. Un campione non può diventare `full_dataset_pass`; il launcher finale usa l'intero manifest e `--check-duplicates`.
- Il test sui target fuori0–1 ne conferma la conservazione. Non si convertono letture del guanto in gradi. L'audit dichiara esplicitamente che differenze di lunghezza inferiori a un campione del guanto non dimostrano allineamento fisiologico e che i due helper non sono equivalenti.
- I launcher richiedono il checkout congelato tramite `CODE_DIR`/`EXPECTED_SHA` e `code_dir_setup`, che verifica hash, pulizia e sola lettura. Output e ricevute sono inWORK, payload inSCRATCH dedicato; nessun accesso in scrittura a loader, checkpoint o output delle altre campagne. La catena proposta usa il submitter/ledger comuni e dipendenze `afterok`.

## Limiti dichiarati, non ostacoli al download

1. **Audit dei duplicati a livello di file.** `--check-duplicates` confronta gli SHA256 dei MATLAB completi. Non prova assenza di segnali identici dentro file con header/metadati diversi. Per una futura affermazione di assenza di duplicati fra benchmark e pretraining serve un controllo del contenuto/corpus separato. La review non pretende che il controllo attuale lo abbia già fatto.
2. **Byte attesi e traffico non sono la stessa quantità.** `download_bytes_max` limita i byte dichiarati della selezione; i retry di trasporto possono aumentare `network_bytes`. Il tetto di risorse è sul job, e la ricevuta permette di rendicontare entrambi. Non chiamare59,7GB un limite assoluto al traffico di rete.
3. **Policy di esclusione.** Radice isolata e marker impediscono inserimenti accidentali in questa catena, ma non costituiscono da soli una prova storica contro tutti gli alias del corpus. Il futuro gate di non contaminazione del benchmark resta esplicito.
4. **Audit strutturale e test scientifico.** Ispezionare integrità, range e metadati dei file di tutti i25soggetti è appropriato per l'acquisizione. Non autorizza tuning, grafici di predizioni o selezione del metodo temporale sui cinque soggetti test.
5. **Riserva per la calibrazione EMG.** La sola dataYYYYMMDD e la randomizzazione dei movimenti non ricostruiscono un prefisso cronologico. Il protocollo ora richiede una riserva deterministica dichiarata se l'ordine non è disponibile. Nessuna statistica del guanto test viene usata per scegliere quella riserva.

## Rettifiche scientifiche trasferite nel protocollo preparatorio

Copertura dei25soggetti confermata nei metadati dei tre componenti; audit del payload pendente. Guanto acquisito60Hz/distribuito256Hz e valori normalizzati esterni a0–1 ammessi. Allineamento finale **da congelare dopo audit degli helper**, non bin6/7campioni già assunti: MATLAB interpola fra endpoint, Python usa FFT e troncamento. Il campione32.678/4.085 non ha ratio8 esatto. LaE_dyn centrata dopo ownership delle finestre misura solitamente segmenti di circa1s con contesto4s, quindi non coincide con la vecchia metrica emg2pose4s.

Restano fuori da questa review: scelta finale della TEMPONet densa, implementazione SEEDS, scelta di LR/dose, costi dei fit, collaudo134canali e confronto predittivo. I70test di acquisizione non verificano questi elementi.

## Hash delle versioni lette

| File | SHA256 |
|---|---|
| `scripts/seeds_download.py` | `8820d0fc1d1e2fa99c366ac4366ed1f36fd4d98690da1cdd7961b4ca0bcb68eb` |
| `scripts/seeds_inventory.py` | `452a5b9ee19a378d4a49be459c4a3485a05fb02a3c7f204fafde2178bbc256d3` |
| `scripts/seeds_audit.py` | `2dafb060ea67e562ce599c76be36f5f548e1e92daa12fb2095652740b2785e5b` |
| `scripts/slurm/seeds_download.sbatch` | `c5954c72538c9233ffd725c5a1db6e7df967de5cbab24ff64ee6adfc2da773a1` |
| `scripts/slurm/seeds_audit.sbatch` | `c4276c4bcf92366085abf903256d6837ab8eec506956f2e1af8e4db8ed1cdfa6` |
| `results/infra/seeds_20261118/source_manifest.json` | `fcfb6308b5155fdd14e63109b200a2208c88326c602e8a963c746b76e14856d7` |
| `results/infra/seeds_20261118/resource_policy.json` | `16dbe6072cce01733e95f47b18ba92f4ec9fd406c8c3de07a5d71756cb8a9296` |

Il coordinatore può congelare/pubblicare questo pacchetto corretto e proseguire con la sola catena di acquisizione già autorizzata. Dopo l'esecuzione servono ricevute complete, statoSLURM, checksum e rapporto strutturale; questa review preventiva non ne anticipa l'esito.
