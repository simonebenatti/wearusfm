# Verifica scettica n. 2 del rilievo G-1 (r7-governance)

Avvio: 2026-11-20 03:27. Copia: HEAD 0efa4d45, sola lettura.

## Fatti riprodotti (comandi miei)

- faa4a8d7 (non antenato di fbfde797: `git merge-base --is-ancestor` = no) si intesta «Decisione proposta a Simone» (science doc r.3),
  chiude «proposta da discutere prima degli invii» (r.52); voce decisioni «Proposta per Simone, ... training e invii sospesi»;
  registro «Preventivo e release restano da approvare». Confermato con `git show faa4a8d7 -- docs/decisioni.md docs/claude_GPT_shared`.
- fbfde797 copia il documento scientifico byte per byte (diff vuoto) ma non la voce di decisioni.md di faa4a8d7 (grep rc=1);
  dichiara «attua la decisione scientifica faa4a8d7» (decisioni.md:8278-8279), plan.json:368 `science_decision_commit`,
  prontuario r.4-6, prompt Claude r.8-9.
- Nessun commit successivo a faa4a8d7 sul ramo logcontrast nel deposito oggetti della copia (fsck --unreachable, ordinati per data:
  l'ultimo commit di linea 3 e' faa4a8d7 alle 11:57; fbfde797 alle 13:52).
- Citazioni letterali di Simone aggiunte in 438c8d65..fbfde797 (grep su tutto il diff, esclusi log/json): solo «Ti affido
  l'implementazione dell'emendamento RMS100/fast125 e la preparazione del pacchetto per 3-launcher» e «Gli invii e i controlli su
  Leonardo spettano a 3-launcher». Nessuna su preventivo, profilo, gate o ripresa.
- preflight 438c8d65:191 «publish and present resource proposal to Simone» tolto; fbfde797:315-319 tre prerequisiti senza Simone.
  plan.json@fbfde797:328 conserva «Prepare and present new costs before any launch».
- Codice: `decision_pin` compare solo in contrastive_run.py:160 (validate_measured_budget, chiamata solo per stage training, r.215-220).
  make_job scrive APPROVED_FOR_PREFLIGHT (contrastive_release.py:236). Il blocco: provato che una chiave `conditions` o uno status
  diverso da APPROVED vengono rifiutati («review is not an approved target-cohort package review").
- Precedenti: 17/11 via di Simone al collaudo 14 GPUh «per questo giro» (decisioni:7846); 16/11 review Claude 0ab9d98 §6
  «Budget: serve una decisione di Simone» e «calibrazione e benchmark dei tre bracci sono una prova sola ai fini della soglia di 5
  GPU-ore»; review Claude 94f5094 R-3 «Preventivo da completare prima della presentazione a Simone».

## Prova dinamica (clone temporaneo g1v2_clone a 0efa4d45, poi cancellato)

Review sonda committata nel clone con il blocco APPROVED e gli hash del delivery_pin (source 40d39f56…, plan b03c0bc6…,
preflight e4b39591…, reviewed_commit fbfde797…) e, nel testo fuori dal blocco, «il preventivo da 18 GPUh resta da firmare da
Simone». `contrastive_release.py bind-review` riesce (status APPROVED). `make_job` produce APPROVED_FOR_PREFLIGHT per l'audit
con il solo binding e, con ricevute d'audit fabbricate (report con source corrente, sacct COMPLETED 0:0 3900 s 8 CPU 120 min,
accounting v1), anche per CALIB, CUDA, BENCH-CN, RELOAD-CR-B: nessun campo di decisione/budget/Simone. La condizione scritta nel
testo della review e' ignorata. Il dispatcher (contrastive_job.build_command) per `cuda` non legge nemmeno una release
(r.119-128); calibrazione/full/resume chiedono una release APPROVED_FOR_PREPARATION che nessun builder produce (scritta da
GPT/operatore). Launcher sbatch e submit_job.py: nessun controllo di budget o approvazione.

## Tentativi di confutazione

1. Autorizzazione permanente «Simone authorizes launches after convergence» (plan.json:321; decisioni:7832 del 16/11).
   Non regge: decisioni:7832 la limita a «lo stesso piano»; il via del 17/11 e' «per questo giro» (14 GPUh). Il pacchetto stesso
   la esclude per la nuova coorte: plan.json:328 «Prepare and present new costs before any launch»; amendment 18/11 titolo
   «Preventivo da presentare prima degli invii» e «L'autorizzazione dei vecchi invii non viene copiata nel nuovo piano»; intestazioni
   N-2 «release sotto non autorizzano nuovi invii».
2. Autonomia del 28/10 sotto 5 GPU-ore (CLAUDE.md:149-157): ogni cella e' 2 GPUh, ma il collaudo e' una prova sola (review Claude
   0ab9d98 §5 «calibrazione e benchmark dei tre bracci sono una prova sola ai fini della soglia di 5 GPU-ore»; divieto di
   frazionare); anche le sole fasi full (6) e reload (8) superano 5. Serve l'approvazione della proposta da parte di Simone.
3. Inoltro dei prompt da parte di Simone a 3-launcher come via di fatto (18/11 diagnostici). Non e' un cancello del pacchetto:
   prontuario e prompt prevedono, dopo la convergenza, release audit e poi fasi sulle sole ricevute; GPT ha gia' tentato l'invio
   diretto a 3-launcher (decisioni:8058); un inoltro non e' «decisione registrata» (CLAUDE.md:267-269).
4. Blocco «Linea3, incarico del 19/11» in CLAUDE.md (fbfde797:1-3) che chiama faa4a8d7 «decisione»: scritto da GPT nello stesso
   commit sotto review, non citazione letterale. L'unica citazione («Ti affido l'implementazione … e la preparazione del pacchetto
   per 3-launcher») copre implementazione e preparazione, coerente con faa4a8d7 «proposta da discutere prima degli invii».
5. Nessun commit successivo in deposito con decisioni di Simone (fsck, ultimo commit linea 3 faa4a8d7 11:57, fbfde797 13:52).

Esito: non confutato.

## Limiti e correzioni del rilievo

- CLAUDE.md:183 («nessun job senza il via di Simone») e' in parte superato da :147-157 (28/10); la conclusione regge sulla soglia
  per prova e sull'approvazione della proposta.
- «Criteri non firmati» (:177) e' la parte piu' debole: nella linea 3 i criteri dei diagnostici (db66b507) sono stati adottati
  dopo la review di Claude senza firma letterale di Simone (pacchetto richiesto da Simone e prompt inoltrato da lui).
- «Supera l'ultimo tetto approvato 14 GPUh»: quel tetto era «per questo giro» (vecchia coorte). Piu' esatto: nessun tetto
  approvato per la nuova coorte; la proposta 14 GPUh di 438c8d65 doveva essere presentata a Simone e non risulta approvata.
- La mancanza di pin di decisione per le fasi preparatorie c'era gia' nella base 438c8d65 (preparazione approvata da Claude in
  94f5094 con R-3 «Preventivo da completare prima della presentazione a Simone»). In fbfde797 cambiano la rimozione del
  prerequisito testuale, il blocco binario come chiave di release e la dicitura «attuata».
- La prima release e' solo audit (CPU, 16 + 0,27 core-ore). L'effetto riguarda le fasi GPU (18 GPUh, inviluppo 21,6), che lo stesso
  binding sblocca appena esistono le ricevute.

## Gravita'

BLOCCANTE per l'emissione del blocco contrastive-release-review, non per l'approvazione del codice. Senza una decisione registrata
di Simone, la catena rilasciata dal blocco supererebbe il limite di autonomia per prova e girerebbe senza un tetto approvato
(18 GPUh / 169,6 core-ore, inviluppo 21,6 / 202,4). Rimedio senza codice: una decisione di Simone in decisioni.md, con
citazione letterale e data, su preventivo, profilo e cancelli, prima del blocco. In alternativa un pin di budget anche per le fasi
preparatorie (cambia il digest e richiede una nuova review).

Pulizia: clone temporaneo e cartella di lavoro g1v2 cancellati; copia di sola lettura pulita a 0efa4d45. Chiuso: 2026-11-20 03:41.
