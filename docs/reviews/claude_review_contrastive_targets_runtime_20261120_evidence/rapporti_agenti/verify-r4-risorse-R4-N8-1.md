# verify-r4-risorse-R4-N8-1 — scettico n. 1 sul rilievo R4-N8

Inizio: 2026-11-20 03:21:38
Copia: rev_targets/clone, HEAD 0efa4d45; codice fbfde797

## Riproduzione (clone temporaneo n8v1_clone a 0efa4d45)

### Parte (1) — "piani rigenerati identici" non attestato: RIPRODOTTO, NOTA valida
- Ricevuta `validation_final/local_validation.json`: `runs` = 1 (solo pytest, 29 file, rc 0, 465 passati); nessun
  esito di `contrastive_targets_plan.py --check`. `amendment_20261119/verification.json` attesta `bash_n` sui tre launcher
  (dichiarazione, senza log) ma nulla sui piani. Prontuario :204-206 mette "piani rigenerati identici" accanto a
  "Ricevuta con comandi/hash in validation_final/". Produttore: `contrastive_local_validation.py:44-56`, una sola esecuzione.
- Prova che nessun test confronta i byte su disco: preflight_plan.json alterato nel clone (checkout seconds_each 600,
  core_hours_cap 999) -> `pytest tests/cpu/test_contrastive_targets_plan.py` = `3 passed`; `--check` =
  `ValueError: stale generated plan: .../preflight_plan.json`. Nessun test referenzia il preflight reale (grep).
- Affermazione vera: `--check` alla punta rc 0 (source 40d39f56..., 17 allocazioni); generazione senza --check ->
  `git status` vuoto; sha256 plan b03c0bc6..., preflight e4b39591... = delivery_pin.
- Attenuanti (nessun effetto operativo): `contrastive_release.py:117-123` rifiuta un preflight con `source_sha256` o
  `science_plan.sha256` non correnti (provato: generatore mutato -> preflight 40d39f56 vs digest 5fe57953, condizione
  della riga 121 vera; `--check` fallisce); `:126-128` lega i byte dei due piani al commit rivisto. Si esegue cio' che la
  review approva, non cio' che il generatore produrrebbe. Resta un difetto di tracciabilita' della prova: NOTA.

### Parte (2) — 360 s senza UnkillableStepTimeout: CONFUTATA come difetto
- File archiviato :7-10: KillWait 180, OverTimeLimit 3 min, UnkillableStepTimeout 120.
- Sforamenti reali dei TIMEOUT archiviati: 64652559 ElapsedRaw 3787 vs 3600 -> +187 s; 63593508 5584 vs 5400 -> +184 s
  (probe_benchmark/sacct_all.txt:5). Cioe' OverTimeLimit + pochi secondi: KillWait non compare nell'Elapsed contabilizzato.
  Il margine 360 s e' dichiarato "cautelativo" (prontuario :105-107) e supera di ~173 s lo sforamento osservato; anche
  aggiungendo 120 s si resta a ~307 s < 360. UnkillableStepTimeout regola il drain del nodo dopo SIGKILL.

### Parte (3) — checkout fuori digest, override procedurali: FATTI VERI, NON DIFETTO
- Vero: `contrastive_prepare.py:192-197` e `contrastive_release.py:58-64` non includono campaign_checkout.sbatch; default
  `#SBATCH --cpus-per-task=1 --mem=2G --time=00:10:00` (campaign_checkout.sbatch:5-7); nessun gate sacct per i checkout
  (quello dell'audit c'e': contrastive_release.py:187-189, 8 CPU e 120 min).
- Non difetto: lo script e' passato su stdin dal Mac, non gira dal checkout congelato, invariato da 4fcc1dfd e assente dal
  diff 438c8d65..fbfde797; l'integrita' del codice eseguito la impongono i launcher (in digest) via lib_code_dir.sh
  (EXPECTED_SHA, checkout pulito). Prassi identica al 17/11 (audit_submission_plan_20261117.json: override nel comando).
- Costi: piano 6x4x240 s = 1,6 coreh; default 6x1x600 = 1,0 (meno); caso misto 6x4x600 = 4,0 = esattamente
  1,6 + margine checkout 6x4x360 = 2,4 dell'inviluppo dichiarato (202,4 coreh): nessun superamento dell'inviluppo.
  Sono tetti, non consumi: checkout storici 11-106 s. Il "fix" richiesto e' gia' la procedura dichiarata (prontuario
  :167-173 con i tre override; :178-179 comandi definitivi post-review; `mandatory_sbatch_overrides` testato).

### Verifiche positive
- `contrastive_local_validation.py --verify` sulla ricevuta: `{"verified": true, "all_passed": true}`.

## Esito
Rilievo NON confutato nel suo nucleo (parte 1, NOTA documentale riprodotta). Parte 2 confutata; parte 3 vera nei fatti
ma non un difetto (procedura dichiarata, costo entro l'inviluppo). Gravita' reale: NOTA.

Fine: 03:28:49. Clone temporaneo cancellato; copia di sola lettura pulita.
