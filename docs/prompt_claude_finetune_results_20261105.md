# Review dei risultati EPN a cinque epoche — consolidamento

Lavora soltanto come revisore sul tuo ramo: nessuna modifica ai runtime,
nuovo fit, inferenza held o sottomissione. Riferimento: la consegna che introduce
questo file su `origin/codex/consolidation-20261104`, successiva a 3fd0859.
Le tue review 2d157cc/f70dc72 restano approvate; non serve rifare la review dei
lanci. L'owner ha raccolto tutti i 24 fit COMPLETED 0:0.

Rileggi:

- `scripts/consolidation_ft_report.py`, nuovo aggregatore di sola lettura,
  e `tests/cpu/test_consolidation_ft_report.py` (15 test).
- `docs/reviews/finetune_results_20261105.md` e la review indipendente associata.
- `results/infra/consolidation_20261104/final_20261105/`: report, manifest,
  ricevute/candidati/curve, hash remoti, accounting e ricalcolo indipendente.
- Il protocollo congelato F-R2/SCRATCH_RULE; V-R1 non è applicata al solo EPN.

Chiedo una review mirata di cinque punti:

1. Provenienze, selezione LR/epoca esclusivamente sulla validation,
   C0 senza held e C1 con una sola predizione held, pairing dei tre regimi.
2. Macrorecall OOF pooled e bootstrap appaiato per soggetto, 2.000 repliche,
   seed 20261102, nessuno scarto; IC condizionati a modelli/fold, seme unico.
3. Lettura FT−scratch **+1,403 pp [0,930;1,837]** come vantaggio alla dose
   fissata, senza affermare convergenza o vantaggio finale universale.
4. F-R2 scatta ([4,5,5,5]); frozen70,575% è sotto logistica storica82,577%.
   Verifica che non si venda il +23,191 come guadagno sul miglior frozen.
5. Consuntivo dei 24 fit **68,271667 GPUh /546,173333 coreh**; con entrambi
   i benchmark e i tre checkout **68,523611 GPUh /548,221111 coreh**.
   Checkout iniziale condiviso contato una volta; continuazione esclusa.

OOF esterne al repo: WORK/wearusfm_runs/results/consolidation_20261104/
finetuning/epn/foldN/REGIME/candidate1/held.npz. Sul Mac sono in
`/private/tmp/wearusfm-ft-results-20261105/epn/`. Nessun nuovo calcolo GPU
necessario; l'aggregazione locale impiega meno di un secondo.

Esprimi un parere sulla futura proposta FT/scratch a 15 epoche dalle
inizializzazioni originali, non una ripresa dai cinque, senza avviarla.
Preventivo e approvazione di Simone saranno un passaggio successivo.
Matrice lunga a due momenti/profondità resta consegna separata.

Scrivi verdetto, eventuali rilievi e limiti nel tuo ramo, aggiorna entrambe
le sezioni del registro, committa/pusha e restituisci SHA verificato.
Non modificare le soglie o scegliere nuovi candidati alla luce delle OOF.
