# Catena di valutazione della finestra 1: orchestratore (Claude, 15/10/2026)

Accordo GPT-Claude del 15/10/2026 (`docs/accordo_operativo_finestra1_20261015.md`, «Interfaccia della catena di valutazione»): GPT possiede
harness, estrazioni, sonde, confronti e i loro launcher, e scrive il piano di valutazione; Claude possiede `scripts/evaluation_chain.py` e
`tests/cpu/test_evaluation_chain.py` e, per la campagna W1, e' l'unico responsabile delle sottomissioni di training e valutazione.

## Cosa fa

- Legge il piano (schema **provvisorio** `eval-plan-v0` nella docstring dello script, finche' GPT non pubblica
  `docs/evaluation_chain_contract.md`: si adatta a quello), lo valida e ordina i nodi (training -> guardia -> estrazione -> sonde -> confronto).
- **Guardia dei checkpoint** (`scripts/check_checkpoint.py`, `scripts/slurm/check_checkpoint.sbatch`, CPU, minuti): una per checkpoint, in
  afterok dopo il job di training che lo pubblica; verifica run finito (`max_steps`), nessuno STOP, checkpoint presente e non vuoto, passi
  minimi e, se dichiarato, SHA-256. Se fallisce esce con 3 e la valutazione che ne dipende non parte mai (nessun rilancio alla cieca).
- **Dipendenze:** un nodo che legge l'uscita di un altro deve dipenderne direttamente (il confronto parte solo con tutti gli output); piu'
  dipendenze diventano un solo `afterok:a:b` (`submit_job.py` accetta ora piu' `--dopo`).
- **Stato dal ledger del submitter:** `sottomesso` si salta; `ambiguo`, `duplicato`, `rifiutato`, `assente` bloccano il nodo e cio' che ne
  dipende (si chiariscono col submitter); dipendenze non ancora sottomesse -> il nodo aspetta la prossima chiamata (rieseguibile, nessun
  doppione). `limits` per coda ferma le sottomissioni al massimo indicato contando i job gia' in coda.
- **Default dry-run** (stato e comandi nell'ordine); `--submit` sottomette tramite `submit_job.py` (da leonardo-ops). Nessuna scrittura nei
  checkout: uscite assolute e fuori dal codice; `CODE_DIR` e SHA completo per ogni nodo.

Test: `tests/cpu/test_evaluation_chain.py` (piani sintetici, submitter vero con login node finto: validazione, guardie, ordine e dipendenze
multiple, attesa del training, blocco su stati ambigui, limiti di coda, sottomissione rieseguibile senza doppioni, guardia del checkpoint).

## Collaudo sul cluster `w1_valutazione_v1` (15/10/2026; Simone: «va bene procedi»)

Scopo: provare sul cluster guardia e catena (GPT: «guardie e catena di valutazione non risultano collaudate sul cluster»), coi checkpoint veri
del collaudo delle catene (`w1cat`, finiti stamattina) e nodi fittizi (`scripts/slurm/chain_dummy.sbatch`: controlla gli ingressi e scrive un
JSON, nessun calcolo) al posto di estrazioni, sonde e confronti di GPT. Nessun file di GPT toccato.

**Piano:** caso positivo sul checkpoint post-decay del tronco (`w1cat-T`, ultimo job della catena `w1cat-T-j03`, passi minimi 1.400):
guardia -> estrazione fittizia -> due sonde fittizie -> confronto fittizio. Caso negativo sul post-decay del ramo (`w1cat-B`, `w1cat-B-j02`)
con SHA-256 volutamente sbagliato; il nodo a valle sta in una coda di comodo con limite 0, cosi' non resta mai in coda con una dipendenza
impossibile. Uscite in `$WORK/wearusfm_runs/results/infra/w1val_1510/`.

**Criteri (congelati qui, prima del lancio):**
1. Positivo: la guardia T e' sottomessa SENZA afterok (sacct: `w1cat-T-j03` COMPLETED 0:0 da ore: la correzione del rilievo 2), finisce
   COMPLETED 0:0 col rapporto `ok`; estrazione, due sonde e confronto sono sottomessi nella stessa chiamata con afterok sui predecessori e
   finiscono COMPLETED 0:0 nell'ordine delle dipendenze; uscite presenti, quella del confronto con 2 ingressi.
2. Negativo: la guardia B finisce FAILED con codice 3 e il rapporto riporta lo SHA diverso; il nodo a valle non e' sottomesso alla prima
   chiamata; alla seconda risulta «bloccato» (guardia fallita secondo sacct) e nulla viene sottomesso.
3. Rieseguendo la catena dopo la fine: nessuna sottomissione in piu'.

**Risorse e costo:** checkout di campagna e fino a 6 job CPU su `lrd_all_serial` (2 guardie con tetto 20 minuti, 4 nodi fittizi con tetto
5): **0 GPU-ora, tetto ~1 core-ora**, stima pochi minuti.
