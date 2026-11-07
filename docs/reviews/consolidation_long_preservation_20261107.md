# Conservazione della valutazione lunga: copia su WORK verificata dai manifest

**Aggiornamento del 07/11/2026: conservazione di LONG_ROOT eseguita da Claude
e verificata da GPT sui manifest pubblicati.** Decisione separata di Simone:
«Tutto, 19,6 GiB». Le dipendenze fuori LONG_ROOT restano una decisione distinta.
Nessuna operazione remota, copia, job o monitor eseguito da GPT.

## Esito della copia e verifica locale

Prove dell'operatore: commit `424f0ecc5dc23120ac84a7f2e3e1a24fede2b250`,
ramo `claude/consolidation-long-exec-20261105`, directory
`results/infra/consolidation_long_20261105/preservation/`; registro Claude
`51a611a2cac7d1b29a528f7c67a5cd1d5b130d1d`. Script eseguito
`preserve_long_to_work.sbatch`, commit `5b9b43c8205731deb004543e08c848e447d6cad3`.

Job `LONG-PRESERVE-WORK-20261107`, SLURM **63781257**, riportato COMPLETED 0:0,
07/11 dalle 15:49:54 alle 15:52:30 CET: **156 secondi, 0 GPUh, 0,086667 coreh**.
Una CPU richiesta, due allocate: il costo è 156×2/3600, arrotondato a 0,087.
La valutazione lunga più questa conservazione conta **108 allocazioni,
176,831667 coreh e 3,373611 GPUh** (arrotondati 176,832 e 3,374); sono esclusi
pretraining ed EPN15. Non è stata eseguita una nuova interrogazione sacct.

Verifiche sui file del commit, senza riaprire i payload remoti:

- Hash dei sei manifest uguali a quelli registrati nella ricevuta; tre coppie
  originale/copia identiche byte per byte: elenco e dimensioni, directory, SHA256.
- **354 file, 174 directory, 21.062.138.097 byte**, circa 19,616 GiB.
- Tutti i **352 file dell'inventario GPT di `57c4d7e`** sono nella copia con
  hash e dimensioni identici: nessun file mancante o differente.
- I due file ulteriori sono il piano materializzato
  `manifests/summary_plan_f7728778681431314849ff6f21ba55695ebab799d74f806347faf9eba3286a50.json`
  (53.847 byte) e `summary/report.json` (2.848.018 byte). I loro hash coincidono
  con quelli dei corrispondenti artefatti già pubblicati della sintesi.
- I conteggi e gli esiti ricalcolati concordano con `inventory_check.json`.

La ricevuta e lo script dichiarano la conservazione di modi e date tramite
`cp --preserve=mode,timestamps`; i manifest non contengono questi attributi,
quindi questa verifica indipendente riguarda contenuto e albero, non un nuovo
confronto di permessi o timestamp. Analogamente, non certifica lo stato remoto
attuale: verifica le prove archiviate della copia. L'originale non è stato
cancellato o sostituito dall'operazione e rimane soggetto alla purge SCRATCH.

[Audit locale della conservazione](../../results/infra/consolidation_long_20261105/gpt_review_20261107/preservation_verification.json).

Radice originale:
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105/`.
Destinazione della copia eseguita:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/`.
I piani congelati continuano a nominare i percorsi originali su SCRATCH e non
sono stati modificati. I manifest associano ogni percorso relativo alle due
radici con hash e byte uguali. NPZ e PT restano fuori Git.

## Inventario e raccomandazione precedenti alla copia

La sezione seguente conserva il ragionamento della consegna `57c4d7e`.
La raccomandazione per LONG_ROOT è ora soddisfatta dall'operazione sopra;
le alternative di spazio non sono operazioni ancora da eseguire.

**Raccomando di preservare l'intero insieme degli artefatti pinnati del lungo:**
sono circa **19,61 GiB censiti**. Non è una nuova misura `du` della cartella,
né una misura della quota disponibile. Conservare l'intera radice evita di
perdere dipendenze del verificatore, oltre agli esiti scientifici.

| Contenuto sotto LONG_ROOT | File censiti | GiB | Perché conservarlo |
|---|---:|---:|---|
| Cache delle tre dosi: `19618/`, `39237/`, `49046/` | 21 | 16,408 | Sequenze, bersagli, maschere, metadati, scale e ordine necessari alle verifiche |
| `fits/pose/{19618,39237,49046}/*.npz` | 15 | 2,995 | Nove predizioni e sei coefficienti ridge |
| `fits_a1/classification/**/probe.oof.npz` | 69 | 0,086 | Predizioni appaiate, identità e fold delle sonde |
| `benchmarks/pose/*.npz` | 5 | 0,103 | Artefatti richiesti da `_benchmark_gate` anche nel confronto congelato |
| Report, JSONL, ricevute e altri metadati pinnati | 242 | 0,021 | Provenienza, selezione, aggregazione, accounting; parte già in Git |

[Inventario con percorsi, dimensioni e hash](../../results/infra/consolidation_long_20261105/gpt_review_20261107/preservation_inventory.json).
Il conteggio deriva dalle ricevute pubblicate, deduplicato per percorso;
non certifica oggi presenza o integrità remota, né sostituisce l'inventario
dell'operatore al momento della copia.

Se lo spazio fosse insufficiente, la prima priorità è preservare report,
OOF classificazione e audit, predizioni/coefficenti posa **insieme ai relativi
bersagli e supporti**. Le NPZ delle predizioni contengono soltanto `prediction`:
copiarle da sole non basta. Senza costruire nuovi formati, questo richiede le
tre cache `*/pose/val/val.npz` (12,834 GiB). Le tre cache train occupano
1,325 GiB; le 15 cache classificazione circa 2,249 GiB. I checkpoint da soli
non rendono un payload immediatamente riproducibile: servono anche raw,
sidecar, statistiche, selezioni e ambiente numerico originali.

Le feature sono rigenerabili e non sono risultati insostituibili; perderle
comporterebbe però nuova estrazione e nuova verifica, non autorizzate da questa
consegna. Per circa 19,61 GiB raccomando di conservare l'insieme completo invece
di dover risolvere dopo la purge dipendenze mancanti.

## Dipendenze fuori LONG_ROOT

**Stato separato, ancora aperto nella consegna qui verificata.** Claude indica
26 file T100-G 19.618 su SCRATCH, circa 23 MB, come oggetto della decisione
separata di Simone in corso. Non sono inclusi nella copia di LONG_ROOT né
certificati come già conservati da questa verifica.

Una copia della sola radice lunga **non è autosufficiente**. Servono anche:

- Le OOF, i report, gli audit e le cache storiche **T100-G 19.618/pool**, riusati
  senza refit. I report `p1.json`/`p2.json` sono sotto
  `$SCRATCH/wearusfm_runs/results/tokenizer_probe_20261103/19618/T100-G/`.
  Gli altri percorsi sono fissati da `historical_classification_plan`,
  `historical_extraction_plan` e dai rispettivi report nel piano lungo.
- I riferimenti della replica **T25-L−B19618**, inclusi gli input OOF se si
  vuole riprodurla a partire dalle predizioni. Il suo report aggregato è già
  versionato in `results/infra/tokenizer_evaluation_20261103/t25_b_repeatability_20261104/paired_report.json`.
- Piani delle fasi, bridge A1, manifest, 107 stati del ledger, 103 ricevute,
  107 log e sacct. Le copie archiviate in Git e sul Mac vanno mantenute insieme
  ai payload; non tutti gli audit JSONL remoti sono necessariamente già copiati.
- I checkpoint persistenti dei tre endpoint, già pinnati sotto
  `$WORK/wearusfm_runs/results/consolidation_20261104/continuation/checkpoints/`.
  Questa review verifica i pin archiviati, non una nuova lettura remota dei pesi.

Le copie originali su SCRATCH conservano le proprie scadenze: i primi file
del lungo risalgono al 05–06/11, quindi senza rinnovi i 40 giorni portano a metà
dicembre. La copia eseguita su WORK non dipende dalla purge di SCRATCH. Per le
dipendenze esterne ancora da decidere non assumere rinnovi o proroghe e non
far decorrere i 40 giorni dalla chiusura della campagna.

Le quattro vecchie cartelle `passo6` spostate su SCRATCH il 06/11 sono un'altra
consegna e non sono protette da un backup di LONG_ROOT. La loro conservazione
resta quella comunicata dall'operatore; qui non si promettono i rinfreschi.
