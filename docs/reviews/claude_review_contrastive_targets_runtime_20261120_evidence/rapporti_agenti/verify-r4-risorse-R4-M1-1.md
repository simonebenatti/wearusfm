# Verifica scettica n.1 del rilievo R4-M1 (r4-risorse)

Inizio: 2026-11-19 14:24


Ripresa: 2026-11-20 (la sessione precedente si era fermata dopo l'intestazione). Copia a 0efa4d45, diff 438c8d65..fbfde797.

## Passo 1: riproduzione della rimozione del prerequisito
- RIPRODOTTO: `git diff 438c8d65 fbfde797 -- results/infra/contrastive_targets_20261118/preflight_plan.json` toglie
  "publish and present resource proposal to Simone" (e anche "check ledger, quota, maintenance, available balance; no duplicates",
  sostituito da "fresh ledger/quota/balance and pending writes checked by 3-launcher"). Generatore fbfde797
  scripts/contrastive_targets_plan.py:114-115 identico al JSON.
- Numeri del piano (preflight resources): richiesto 18 / 169,6; storico effettivo 6,086667 / 57,827778; tetto storico 14 / 129,333333;
  residuo "historical_unspent_not_new_authorization" 7,913333 / 71,505556; cumulato 24,086667 / 227,427778; inviluppo 21,6 / 202,4.
- decisioni.md (fbfde797) :7842 "Tetto 14 GPU-ore e 129,3333 core-ore"; :8292 "Preventivo proposto ... 18 GPUh / 169,6 coreh richieste";
  :8301-8304 "Dopo review si prepara la release audit; calibrazione e seguito richiedono le rispettive ricevute".
- `git grep` a 0efa4d45 di 169,6 / 18 GPUh / 21,6 / 202,4: solo proposta (registro, decisioni, prontuario, prompt, risposta). Nessuna decisione di Simone.

## Passo 2: cerco motivi per confutare
- Piste di confutazione esaminate, nessuna regge:
  1. Prerequisito spostato altrove? No. Prontuario fbfde797 (73-119 costi, 121-146 rilascio, 181-187 fasi), risposta R-3, CLAUDE.md
     (blocco nuovo righe 1-4), delivery_pin ("requested_resources"), verification.json, registro: il preventivo e' solo "proposto/richiesto";
     nessun passo di approvazione di Simone prima delle fasi GPU. Il blocco CLAUDE.md dice: "calibrazione condizionata a sacct COMPLETED 0:0 entro 7200s".
  2. Cancello nel codice? No. contrastive_job.build_command (fbfde797:62-86) legge dal preflight solo id/mode/arm/repeat/seconds e fresh_root;
     contrastive_release.make_job (215-243) e validate_review_binding (106-134) non leggono resources, submission_authorized ne' la lista dei
     prerequisiti; submit_job.py registra solo la stima --gpu-ore. Prova: rev_runtime/tmp/sk1_r4m1/probe_budget_gate.py (fixture come
     test_contrastive_release.py) con preflight gpu_hours_cap=0, submission_authorized=false e prerequisito "registered Simone decision":
     make_job produce {"mode":"calibration","status":"APPROVED_FOR_PREFLIGHT"}.
     Il vecchio codice non aveva un cancello di budget per il collaudo: lo garantiva solo la procedura, cioe' la voce tolta.
  3. Autorizzazione di Simone gia' esistente? No. «sul punto 3 passiamo a 100/100», «Ti affido l'implementazione ... e la preparazione del
     pacchetto per 3-launcher», «Gli invii e i controlli su Leonardo spettano a 3-launcher» non nominano un tetto. CLAUDE.md chiude il blocco
     target con "Nessun lancio implicito". faa4a8d7 (copiata identica): "Il tetto proposto in precedenza di 14 GPU-ore ... va ricalcolato" e
     "proposta da discutere prima degli invii". L'autorizzazione del 16/11 vale "per lo stesso piano" (decisioni:7832). Il piano vecchio
     aveva "approval_authority: Simone explicit instruction 2026-11-17"; il nuovo non ha un campo analogo.
     Precedenti di prassi: "Confermo il tetto" (decisioni 7211-7214) e il 17/11 «ti do fiducia» col tetto 14 presentato.
  4. Il vecchio item diceva solo "presentare", e la consegna stessa lo presenterebbe? Il preventivo e' pubblicato, ma decisioni:8292 lo
     chiama "proposto". CLAUDE.md chiede "una decisione registrata, non una conferma al volo". La review 94f5094 R-3 era intitolata
     «Preventivo da completare prima della presentazione a Simone»: presupponeva il passaggio che fbfde797 toglie.
  5. Il primo invio e' l'audit, CPU, 0 GPUh, entro l'autonomia: e' vero, e il rilievo lo dice. Il vuoto riguarda le fasi GPU: full (6 GPUh)
     e reload (8 GPUh) superano ciascuna la soglia di 5 GPUh anche prese da sole.
- Ricalcolo autonomo: GPU 9x4x0,5=18; core 16+9x32x0,5+8+6x4x240/3600=169,6; margine 0,1 h per allocazione -> +3,6/+32,8 = 21,6/202,4;
  storico 5,284444+4x722/3600=6,086667 e 51,371111+(140+23104)/3600=57,827778; residuo 7,913333/71,505556; cumulato 24,086667/227,427778.
  `contrastive_targets_plan.py --check` a 0efa4d45: exit 0, stessi valori. pytest test_contrastive_targets_plan.py + test_contrastive_release.py: 25 passati.
- Imprecisioni minori del rilievo, che non lo indeboliscono: il tetto 14/129,333 e' quello del collaudo della coorte 25/50
  (CONTRASTIVE-v2-20261116), non dell'intera linea. Per la coorte nuova nessun tetto e' registrato, ed e' piu' forte. "Basta a costruire tutti
  i 17 job" vale per le 11 celle: i 6 checkout non usano il binding; le celle GPU richiedono anche release APPROVED_FOR_PREPARATION scritte a
  mano e le ricevute, ma nessuno di questi elementi e' un'approvazione di budget.

## Esito
NON CONFUTATO. Gravita' reale: MINORE. Non BLOCCANTE: non supera nulla subito, perche' il primo rilascio e' il solo audit. Fra le fasi
c'e' un passaggio umano: release e checkout per fase. La correzione e' documentale e procedurale, senza cambio di codice: decisione
registrata di Simone sul tetto 18/169,6 (inviluppo 21,6/202,4) prima della release della calibrazione. Nella review va scritto che il
blocco lega l'identita' del pacchetto, non il budget. Non NOTA: la catena documentata
(prontuario, CLAUDE.md, decisioni) porta a spendere 18 GPUh senza decisione registrata.
Pulizia: basetemp e repo di fixture cancellati; resta solo lo script. Copia di sola lettura pulita (git status vuoto).
