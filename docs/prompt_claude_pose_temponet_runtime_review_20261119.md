# Review operativa TCN frozen, dopo la convergenza documentale

Claude, la tua review `ad9f5141711ba60ea1014bd9ba842af168ff0a37` ha chiuso il
protocollo v3. Ho applicato i tre ritocchi testuali richiesti senza nuovo giro
sul disegno. Ti consegno ora l'implementazione e il piano operativo.

Ramo: `codex/pose-temponet-implementation-20261119`.
Commit dei contenuti: quello pinnato in
`results/infra/pose_temponet_execution_20261119/delivery_pin.json`.
Base del diff: `0e56c8b1c67f8257e800f6cd51bafacac28971b4`.
Stato operativo: **REVIEW_REQUIRED**. Nessun job o deploy della prova frozen.

Leggi anzitutto:
- `docs/pose_temponet_execution_20261119.md`;
- `results/infra/pose_temponet_execution_20261119/plan.json` e README;
- `scripts/pose_temponet/` e i due launcher `pose_temponet*.sbatch`;
- `tests/cpu/test_pose_temponet_*.py`, ricevute e log dei test nella cartella della consegna.

L'entry point reale è `PYTHONPATH=src:scripts python -m pose_temponet`:
prepare-plan, inspect, release-plan, resource-plan, materialize-plan, run, commands.
I comandi di invio sono generati per il submitter comune; il pacchetto non
sottomette automaticamente. Sono specificate 24 allocazioni: 18 GPU e 6 CPU-only,
con checkout incluso nel preflight. I sedici walltime neurali restano da misurare.

Verifica le parti operative previste dalla tua `ee118a25`, senza riaprire il
disegno già convergente:

1. Architetture e loss: cinque lettori, parametri/RF/crop, ramo affine diretto,
   inizializzazione e RNG; stessa MSE per finestra e supporto. FM e runtime raw
   invariati. FP16 host, FP32 per batch, determinismo e precisione espliciti.
2. Ottimizzazione e controlli: fold per utente, selezione MSE pooled e checkpoint
   della stessa coppia, estensioni finite, allarmi e S-2; distinzione y/y*, ridge
   inner e WLS, gate riferimenti e sei controlli prima dei fit reali. Rifit dei
   tre semi solo dopo il blocco delle selezioni, quindici stati prima del held.
3. Provenienza: otto input pinnati, produttori storici separati dal lettore;
   parità R-1 prima dei fit e dei contrasti; ricevute complete, identità numerica,
   scritture esclusive, catena delle dipendenze e guardie sulle fasi. SHA del
   codice rivisto distinto dal checkout successivo di soli metadati, sorgenti
   verificati byte per byte. Le review sono artefatti effettivi con commit/hash;
   il verdetto favorevole resta responsabilità dell'operatore che lo legge.
4. Risorse: benchmark soltanto train, p95 su dieci epoche dopo warmup, L13 alla
   dimensione piena e tutti i pattern, 1.936 combinazioni per fold; costo massimo
   con tutte le estensioni, memoria e I/O, 180 s per job, gate globale prima dei
   fit. Tetti invariati 24 GPUh, 80 coreh CPU-only, 272 coreh totali. Nessuna
   misura CUDA locale spacciata per stima A100; valori da acquisire col benchmark.
5. Lettura: cinque IC99 appaiati, media degli accumulatori dei tre semi,
   descrittivi/allarmi direzionali/candidatura prescritti, tutti i giunti,
   nessun contrasto con celle incomplete. Fallimenti distinti da risultati
   scientifici, nessun retry automatico.

Verifiche eseguite: **204 test nuovi passati**, compreso l'entry point pubblico
e una catena simulata fino ai quindici stati bloccati. **125 test storici passati**.
Un test storico fallisce: `test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned`
presume un vecchio piano non ancora rivisto, mentre il piano committato è già
convergente. Riprodotto identico in un export isolato della base `0e56c8b1`;
prova in `baseline_failure.json`. Non ho cambiato la campagna chiusa per farlo
passare. `bash -n` e `git diff --check` puliti. Nessun test CUDA o accesso ai dati
reali dichiarato in questa consegna.

La review matematica indipendente ha inoltre riprodotto WLS/ridge con una
soluzione separata, pesi della loss, accumulatori e cinque bootstrap dalle
predizioni grezze. Non sostituisce i controlli positivi da eseguire sulle cache.

Pubblica un verdetto sul pacchetto con eventuali blocchi e prove riproducibili,
commit e push verificati, voce nelle due sezioni del registro. Indica il commit
dei sorgenti che hai effettivamente verificato. Se una correzione è solo
documentale, distinguila da un cambio del runtime o della ricetta.

Nessun lancio/deploy da parte tua. L'operatore unico resta la chat
**2-TCN head launcher**. Dopo review favorevole e gate, la catena sul teacher
30M congelato a 49.046 è già autorizzata: non serve una nuova approvazione
di Simone per lo stesso calcolo. Comunicazione con te tramite Simone.
