# Scettico 2 su R4-M1 (r4-risorse)

Stato: in corso. Copia a 0efa4d45, diff 438c8d65..fbfde797.


## Progressi (20/11 03:0x)

Riprodotto in proprio:
- `git diff 438c8d65 fbfde797 -- results/infra/contrastive_targets_20261118/preflight_plan.json`: tolto
  `"publish and present resource proposal to Simone"`; caps 14/137,066667 -> 18/169,6. Confermato.
- Generatore `scripts/contrastive_targets_plan.py:114-115`: nuova lista a tre voci. Confermato.
- `decisioni.md:7842` e `7846`: tetto 14 GPUh / 129,3333 coreh del collaudo 16/11, via di Simone 17/11. Confermato.
- `decisioni.md:8292`: "Preventivo proposto ... 18 GPUh / 169,6 coreh richieste". Nessuna decisione di Simone
  sul nuovo tetto nel repository (grep "18 GPU|169,6" in docs/: solo proposte/richieste). Confermato.
- Codice: `contrastive_release.make_job` e `contrastive_job.build_command` leggono dal preflight solo
  cells (id/mode/arm/repeat/seconds) e fresh_root; nessun campo resources letto (grep). Confermato,
  ma NON e' una regressione: `scripts/submit_job.py` registra solo `--gpu-ore` dichiarate (righe 206-207, 308),
  il dispatcher a 438c8d65 non leggeva il preflight, e nessuno script ha mai letto `approval_authority`
  o `submission_authorized` (grep su scripts/ e src/).

## Controprova principale

Il requisito NON e' sparito dal pacchetto: il piano scientifico legato dalla review
(`plan.json`, sha256 b03c0bc6..., = `science_plan_sha256` del delivery_pin e del preflight) contiene,
invariato da 0efdc29c a 0efa4d45:
- `scientific_decisions.authorization`: "Simone authorizes launches after convergence; this file is not a deployment release"
- `scientific_decisions.target_windows.execution`: "Prepare and present new costs before any launch; no release in this file"
(`plan.json:321` e `:328` a fbfde797). Entrambe le frasi sono nate in 0efdc29c insieme alla voce
tolta dal preflight (`git log -S` su entrambe le stringhe: stesso commit 0efdc29c). La voce del preflight
era un doppione; fbfde797 ha tolto la copia e conservato quella nel piano scientifico, con portata
"before any launch" (piu' ampia di "before first submission").

## Altre controprove

- Le frasi del pacchetto citate dal rilievo indicano condizioni NECESSARIE, non sufficienti: prontuario:140
  «pubblicano pin solo dopo le ricevute», «Per calibrazione servono»; CLAUDE.md:3 «calibrazione condizionata a sacct»;
  decisioni.md:8303 «calibrazione e seguito richiedono le rispettive ricevute». Nessuna dice «solo il sacct».
  Il blocco CLAUDE.md del 19/11 «prevale sulle assegnazioni operative storiche» (ruoli), non sulle regole di budget.
- Il pacchetto nega esplicitamente ogni autorizzazione del nuovo tetto: preflight `submission_authorized: false`,
  `historical_unspent_not_new_authorization`; prontuario:88 «Tetto richiesto», :99-101 «non equivale a una nuova
  autorizzazione»; decisioni.md:8292 «Preventivo proposto». Nessun testo lascia credere che 18/169,6 siano approvati.
- Restano in vigore le regole generali: CLAUDE.md:150-157 (autonomia sotto 5 GPUh, l'approvazione della proposta
  comprende il lancio), :266-269 (superare il budget di un passo richiede decisione registrata); decisioni.md:3445-3449
  («ogni tranche arriva con protocollo, preventivo e via di Simone»).
- Nessun cancello di budget nel codice per le fasi di collaudo e' il disegno storico, non un difetto nuovo:
  `contrastive_run.py:157` (gia' a 438c8d65) chiede `budget_decision_pin` solo per lo stage training; il collaudo da
  14 GPUh del 17/11 e' partito senza pin di budget, col via di Simone registrato in decisioni.md:7846.
- «Il blocco di review basta a costruire tutti i 17 job» e' inesatto: sonda in clone temporaneo (test mio,
  preflight con audit+calibrazione, cap 99999, nessun prerequisito): `make_job` della calibrazione con il solo binding
  e' rifiutato; con ricevute audit valide passa senza alcun artefatto di budget. 23 passed (sonda + 22 test di
  test_contrastive_release.py). I 6 checkout non passano da make_job; i job GPU richiedono anche la release
  esecutiva APPROVED_FOR_PREPARATION scritta a parte per fase.
- Numeri del rilievo esatti (ricalcolati dalle celle): 18 GPUh / 169,6 coreh, inviluppo 21,6/202,4, storico 6,086667,
  residuo 7,913333/71,505556, cumulato 24,086667; il cumulato supera 14 al secondo full (CN).
- `contrastive_targets_plan.py --check` in clone temporaneo: exit 0 (piani rigenerati identici, compresa la frase
  «Prepare and present new costs before any launch»).

## Esito

CONFUTATO come difetto introdotto da fbfde797. Il prerequisito di presentare i costi nuovi non e' stato tolto dal
pacchetto: resta, con portata «before any launch», nel piano scientifico legato dalla review. Anche la base chiedeva
solo di «presentare», non un'approvazione; l'approvazione del nuovo tetto viene dalle regole generali ancora in vigore,
e il pacchetto dichiara apertamente che il residuo storico non autorizza nulla.
Gravita' se lo si volesse tenere: NOTA (chiarezza): nella review scrivere che il blocco contrastive-release-review
lega l'identita' del pacchetto e non il budget, e che la decisione di Simone su 18/169,6 (sopra il tetto 14/129,333
del 17/11) va registrata in decisioni.md prima dei lanci, come chiede plan.json:328.
Non verificato: eventuali via di Simone dati in chat e non registrati; contenuto del futuro prompt per 3-launcher;
saldo reale di progetto.
