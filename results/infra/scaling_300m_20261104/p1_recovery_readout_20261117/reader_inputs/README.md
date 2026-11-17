# Ingressi verificati per i lettori — P1-r2

Questi documenti contengono i pin effettivi delle sette cache 300M, unite alle
sette cache del controllo 30M congelato. I costruttori e gli schemi dei lettori
esistenti li accettano; nessuna cache o ricevuta e' stata riscritta.

- `mixed_lineage.json`: P1 da job 64620434 / codice e4ebdbfa, posa da 64571999
  e P2 da 64572001 / codice df2de35. Checkpoint 300M comune d7ebe9d8, passo 19.618.
- `pose_plan.json`, `probe_plan.json`: piani generati dal join esplicito
  `mixed_candidate_caches`, poi da `pose_plan` / `probe_plan`, con i riferimenti
  30M originali. Il riproduttore e l'audit sono in `../reviews/`.

**Stato: ingressi preparati e validati localmente; nessun rilascio operativo o
job dei lettori eseguito in questa consegna.** Non usare il vecchio
`build_reader_plans()` direttamente: punta ancora al P1-r1 fallito e pretende
un solo commit produttore. Non cambiare i pin o gli esiti storici per aggirarlo.

Prima degli invii, il controller deve usare questo join e questi percorsi,
pubblicare i piani e superare i controlli operativi congelati (checkout, pin,
ledger, uscite assenti, coda, quota). Restano validazione completa sui nodi di
calcolo e benchmark train-only prima dei fit, budget derivato da tempi/RSS,
stop al primo errore o timeout. Nessuna nuova loss, dose o training FM.

La nuova guardia riguarda soltanto la finalizzazione degli output P1. Le guardie
dei lettori restano invariate. La lettura integrale aggiuntiva delle tre cache
P1 viene verificata dal Mac; i payload preesistenti posa/P2 e controllo 30M
saranno riletti dai validatori congelati dello stadio successivo.
