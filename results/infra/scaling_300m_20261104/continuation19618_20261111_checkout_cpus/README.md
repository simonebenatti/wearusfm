# Linea 1 — Stadio B: checkout a 2 CPU (11/11/2026). Proposta: decide Simone

## Che cosa e' successo

- Il training del recupero v3 (job 63979977) e' `COMPLETED 0:0` al passo 19.618; prove e piano d'estrazione sono in `df2de35`.
- Il cancello del checkout `code` e' passato (9 controlli su 9). Riferimento `deploy/scaling_300m_evaluation19618_20261106_v1`
  pubblicato a `df2de35`; checkout `SCALING-19618-code-checkout-20261106` = job 64194074, `COMPLETED 0:0` in 34 s.
- `collect checkout-code` lo rifiuta: «allocated CPUs or elapsed time outside the published job». Nessun file scritto.
- Causa: `lrd_all_serial` assegna core interi da due thread (`login13`, `ThreadsPerCore=2`). Il checkout ha chiesto 1 CPU e ne ha
  avute 2 (`scheduler_reading_64194074.txt`, `sacct_64194074.txt`). Il freeze v2 scrive gia' 2 CPU per checkout; il codice dello
  stadio B ne pubblicava 1 per far stare cinque checkout negli 0,8 core-ore congelati. I test usavano contabilita' finte a 1 CPU.
- Lo stadio B e' fermo prima delle estrazioni: nessuna GPU usata dopo il training.

## Correzione proposta (commit che contiene questo file)

- `scripts/scaling_plans19618.py`: i checkout dello stadio B si pubblicano a 2 CPU, come li assegna Slurm e come li chiede gia' il
  checkout del training. La riga dei checkout della tabella dei budget ha come tetto 4/3 core-ore con margine.
- Regola nuova per una review fatta dopo un checkout (`rereview_check`): copre le modifiche successive a quel checkout solo se
  nessun job del cluster le legge. Ammessi lo strumento del Mac `scripts/scaling_plans19618.py`, i test CPU, `docs/` e i file
  nuovi della campagna. Restano vietati il codice dei job, i launcher, le configurazioni e la riscrittura di un documento gia'
  distribuito. `release_check` non cambia; dopo la review valgono le regole di prima.
- Test: la contabilita' reale del job 64194074 e' accettata a 2 CPU e rifiutata a 1; la review successiva al checkout `code` e'
  accettata per lo strumento e rifiutata per il codice dei job, un documento distribuito o le split; l'inviluppo v3 pubblicato si
  ricostruisce identico con i checkout a 1 CPU, cioe' com'era il 09/11.

## Budget (`budget_amendment.json`, da `make_amendment.py`, verificabile con `--check`)

- Quattro checkout da 1 a 2 CPU, 300 s piu' 180 s di margine: +8/15 = 0,533333 core-ore.
- Solo CPU: da 349,384444 a 349,917778 su 350, margine 0,082222. GPU-ore invariate: 66,731111.
- Riga dei checkout dello stadio B: da 0,8 a 4/3. Stadio dei checkout nell'inviluppo: da 4/3 a 28/15.

## Che cosa non cambia

- Nessun job nuovo e nessun retry. Il checkout 64194074 resta quello della catena; il codice che gira sul cluster resta `df2de35`.
- Gli altri tre checkout si chiedono a 2 CPU. I checkout successivi usano `--reviewed` uguale al commit di questa correzione.
- I builder degli inviluppi v2 e v3 leggono la tabella dello stadio B dal codice: il loro `make_envelope.py --check` ora
  trova i checkout a 2 CPU. Si rifa' sul commit `44d3adf`, o con i checkout a 1 CPU come in `test_scaling_recovery19618.py`.

## Seguito, dopo la decisione

`collect checkout-code`, commit e push verificato, `gate extractions` e le tre righe, come in `evaluation_chain_plan.md` §5 e §9.
