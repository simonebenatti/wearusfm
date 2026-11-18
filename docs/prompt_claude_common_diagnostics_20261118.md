# Review dei due diagnostici contrastivi — nessun lancio

Ramo: origin/codex/tokenizer-logcontrast-20261104. Base del diff: 09eeaf93.
Rivedi il commit di consegna indicato da Simone, compresi codice, protocollo,
risorse e test. Non lanciare job e non modificare le soglie storiche.

Contesto: il lettore 64775571 ha prodotto FAILED_REQUIRES_REVIEW. I rapporti dei
gradienti attivi passano, mentre parità iniziale a peso zero e riprese falliscono.
La ripresa precedente confrontava run indipendenti già divergenti prima del
restart. Il compito è verificare che i nuovi controlli isolino le cause possibili.

Leggi:
- docs/contrastive_common_diagnostics_20261118.md;
- scripts/diagnostics/contrastive_common_state_cuda.py;
- scripts/diagnostics/contrastive_common_checkpoint.py;
- scripts/slurm/diagnose_contrastive_common.sbatch;
- tests/cpu/test_contrastive_common_state_cuda.py e test_contrastive_common_checkpoint.py;
- results/infra/contrastive_comparison_20261116/common_state_diagnostics_20261118/;
- docs/reviews/contrastive_gradient_parity_diagnostic_20261118.md e contrastive_readout_results_20261118.md.

Domande prioritarie:
1. Il confronto C0/CN/CR a peso zero ripristina tutti gli stati rilevanti prima
   di ciascun backward? Batch, microbatch, maschere, RNG, teacher, queue e coverage
   sono attestati per rank? Le repliche e i quattro profili separano effetto del
   braccio, precisione e backend senza attribuzioni causali eccessive?
2. Il confronto dei vettori, le norme FP64 per blocco e i confronti fra profili
   sono corretti? Il supporto effettivo delle due loss è verificato? La fixture
   piccola e la coda rimappata sono dichiarate con limiti adeguati?
3. La prosecuzione 576→580→588 e le due riprese 580→588 partono da un unico
   checkpoint? Il solo keep_checkpoints aggiunto nella copia è verificato?
   Le tracce osservano davvero input/RNG prima del forward e stato al confine?
   Controlla prefetch, loader, ripristino optimizer, DDP e riduzioni.
4. Il lettore distingue invarianti esatte, differenze numeriche e variabilità
   fra le due riprese? Le distanze e i denominatori sono corretti? Nessuna delle
   etichette del diagnostico deve autorizzare automaticamente il training.
5. Pin, isolamento degli output, launcher, quattro GPU, tetti, raccolta degli
   errori e generatore degli argv del submitter comune sono coerenti? Il tetto
   incrementale è 6 GPU-ore / 48,27 core-ore, in tre celle da 30 minuti.

Comando locale:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_contrastive_common_state_cuda.py tests/cpu/test_contrastive_common_checkpoint.py

Pubblica verdetti separati per disegno, implementazione e prerequisiti del
collaudo. Per ogni blocco indica una prova o un test riproducibile. Riporta
eventuali correzioni sul tuo ramo e aggiorna entrambe le sezioni del registro.
Aspettiamo convergenza prima di checkout e invii. Nessun nuovo training FM.
