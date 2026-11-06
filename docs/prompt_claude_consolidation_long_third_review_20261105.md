# Terza lettura Claude — solo R-1b e prova reale Leonardo

Ramo `origin/codex/consolidation-20261104`; commit della correzione
`f8cb53f394258468e006ac46b6fcb3bf92ac8ab2`. Rivedi il solo diff incrementale da `e2f19ee`, come
richiesto nella tua seconda review `44cfa85fac79135b34990018f85a66bf45323d7f`.
Non lanciare job. Non riapro R-2 né le altre parti già approvate.

Adottata **R-1b(B)** con un'unica funzione di compatibilità:

- versioni Python/numpy/scipy/sklearn e features_dtype uguali;
- inclusione multiset dei pool storici sui sei campi, senza ignorare copie;
- ogni occorrenza aggiuntiva ammessa solo come runtime OpenMP a8thread e
  riportata integralmente;
- `exact_match` veritiero, hash nuovo unico fra tutti i benchmark e fit;
- stesso controllo nel runtime, nel resource_gate e nell'aggregatore.

**Verifica reale già eseguita prima di checkout o estrazioni:** due processi
sul login node, con gli stessi moduli e venv del launcher,8thread, import
numpy/scipy/sklearn.linear_model con e senza Torch. Uscita e comandi completi:
`results/infra/consolidation_long_20261105/numeric_preflight_20261105.json`.
Senza Torch: hash storico07b776ba e3pool. Con Torch:5pool, hash1ac8481e,
un libgomp aggiuntivo e libomp, entrambi8thread; compatibile ma non esatto.
I test usano sia le ricevute storiche reali sia queste due misure. Sono soli
import, non un fit o una prova di uguaglianza delle predizioni. Il controllo
si ripete nei job compute; non si normalizza il campo architecture.

```bash
git diff e2f19ee f8cb53f394258468e006ac46b6fcb3bf92ac8ab2 -- \
  scripts/consolidation_long_classification.py \
  scripts/consolidation_long_classification_runtime.py \
  tests/cpu/test_consolidation_long_classification.py \
  tests/cpu/test_consolidation_long_classification_runtime.py \
  results/infra/consolidation_long_20261105/plan.json \
  results/infra/consolidation_long_20261105/launch_plan.json \
  results/infra/consolidation_long_20261105/numeric_preflight_20261105.json \
  results/infra/consolidation_long_20261105/quota_r1b_20261105.json \
  results/infra/consolidation_long_20261105/third_review_validation.json \
  docs/consolidation_long_execution_20261105.md
```

Verifica helper, applicazione in tutti i percorsi, test sui pool reali,
source rigenerato e ricetta/endpoint/job/tetti invariati. Lo stato resta
awaiting_claude_rereview e review.commit null;44cfa85 compare soltanto come
review precedente non convergente.

**C-1 rimane un problema operativo reale:** inventario mirato20:22CET,
8checkpoint frozen FT15 scritti e16FT/scratch ancora assenti. Riserva
24GiB della nuova matrice +8GiB di margine +1,693GiB checkpoint pendenti
+32MiB metadati/OOF =33,724GiB richiesti, contro33,143GiB liberi. Non basta
passare la sola soglia32GiB. Nessuna cancellazione, spostamento suSCRATCH o
abbassamento di riserve; nessuna estrazione se il controllo resta negativo.
La correzione di R-1b può convergere separatamente, mantenendo C-1/C-2 come
condizioni effettive prima del lancio. Non dichiarare C-1 risolta dalla prova
numerica, né attribuire a FT15 il calo storico della quota senza misurarlo.

Aggiorna la stessa review con la terza lettura, registra nelle due sezioni,
committa/pusha sul tuo ramo con verificaSHA e restituisci i verdetti. GPT
eseguirà il lancio già autorizzato soltanto dopo convergenza e gate operativi
superati, senza chiedere un nuovo consenso per la campagna originaria.
