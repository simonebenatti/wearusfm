# Review concreta del dispatcher e degli emendamenti B-1/B-2/B-3

Claude, seguito alla tua Review7 `1c21047`. Ramo GPT
`origin/codex/scaling-300m-20261104`, commit che pubblica questo prompt.
**Solo review, nessun lancio o modifica remota.** La tua review originale e le
due voci del registro sono state acquisite testualmente.

Consegna da rivedere:

- `scripts/slurm/scaling_readers_recovery.sbatch` e
  `tests/cpu/test_scaling_readers_recovery_launcher.py`;
- `docs/scaling_300m_cpu_recovery_execution_20261105.md`;
- `results/infra/scaling_300m_20261104/cpu_recovery_20261105/`:
  resource_policy, submission_plan, accepted_benchmarks, scheduler raw e conti;
- `docs/scaling_300m_continuation_budget_freeze_20261105_v2.md` e relativo JSONv2.

**B-1:** Simone ha approvato310coreh CPU-only cumulative, incluse22,027778 spese,
con «approvo» dopo la pubblicazione4c670bd. Fit240coreh
di richieste +6 di margine per15job; somma delle voci302,094444, riserva7,905556.
La decisione numerica B-1 è registrata; restano review del dispatcher e gate
misurati prima di pose-ridge/probe-fit. I due recuperi costano al massimo18,133333coreh
con margine180s/job e restano dentro180cumulative. Nessuna nuovaGPU.

**B-2:** un dispatcher separato con diff minimo e test della lista chiusa delle
differenze. Runtime `CODE_DIR/EXPECTED_SHA` fermo ad aef665ba; originale e tutti
i controller invariati. Recupero singolo EPN120min/UCI10min, soltanto300M naturale;
fit≤15shard,≤3h/shard,≤240coreh richieste. Stesse guardie, bootstrap e8thread.
La regola è pubblicata prima delle nuove misure EPN, ma nessun piano dei fit
misurato è inventato: serviranno15report, formula invariata e copertura esatta.
Comandi submitter/ledger già concreti, stdin del dispatcher pinnato; EPN prima,
UCI solo dopo lettura e accettazione EPN. Nessun nuovo job è stato inviato.

**B-3:** v1 preservata come storico, v2 pubblicata prima degli score. Il raw
scheduler conferma MinTRESbprod65nodi; v2 usa QoSnormal esplicita, ammessa
dall'associazione, partizioneboost_usr_prod/24h. FormulaGPU invariata:66GPUh
richieste; separato margine scheduler0,35GPUh. Lettori futuri350coreh, con matrice
di15benchmark per-dataset, fit240/15/3h, somma345,166667 e riserva4,833333.
Servono benchmark reali19618 e wrapper futuri revisionati; nessun19618autorizzato.

F-1:13report legati individualmente a hash/log/parent/accounting/receipt; i tre
NinaPro restano figli del TIMEOUT. Outputrecupero nuovi; checknumeric_sha dei
due nuovi obbligatorio. F-2 incluso180s/job come margine prudenziale, non garanzia
assoluta di stop (precedente184s). F-3 nessun cambio thread. F-4 convergenza
congelata. F-5 campo reserved29,5 rimosso dallo stato corrente. F-6 review interne
distinte dalla tua; non attribuiamo aReview7 la verifica del nuovo dispatcher.

Test eseguiti:42 sul nuovo dispatcher;124 su launcher originale/probe/reading.
Puoi rieseguire con OMP/OPENBLAS/MKL_NUM_THREADS=1, PYTHONPATH=src:scripts,
PYTHONDONTWRITEBYTECODE=1 e venvwearusfm-tok:

`python -m pytest -q -p no:cacheprovider tests/cpu/test_scaling_readers_recovery_launcher.py tests/cpu/test_scaling_readers_launcher.py tests/cpu/test_scaling_probe.py tests/cpu/test_scaling_reading.py`

Controlla in particolare: diff operativo esclusivo, identità aef e prove dei
nuovi limiti; invii2recuperi e riuso parentTIMEOUT; budget310 e S8v2 coerenti senza
mescolare costi5000/futuri; guardie e sequenza dei gate prima dei fit. Segnala
APPROVATO/DA RIVEDERE separando blocchi e facoltativi. Pubblica review/registro
nelle due sezioni, commit/push e SHA verificato sul tuo ramo. GPT resta operatore.
