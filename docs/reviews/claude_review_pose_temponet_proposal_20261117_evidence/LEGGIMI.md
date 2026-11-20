# Prove della review della proposta TCN `c15aac9` (17/11/2026)

Materiale di lavoro della review `docs/reviews/claude_review_pose_temponet_proposal_20261117.md`. Non contiene dati da wearable,
cache o checkpoint: solo testi, script e piccoli risultati, ottenuti sul Mac con dati sintetici o con file del repository. Tutto
è stato eseguito su CPU, salvo alcuni micro-benchmark del costo sulla GPU del Mac (MPS).

- `rapporti_agenti/`: rapporti completi dei sei revisori (`review_<dimensione>.txt`) e dei tre verificatori avversari
  (`verify_<coppia>.txt`). Ogni voce porta la sua prova marcata ESEGUITO, CALCOLATO o LETTO. Gli ID dei rilievi (per esempio
  `selezione-1` o `architettura-controlli-N2`) sono quelli citati nella review.
- `rapporti_agenti/rilettura_<lente>.txt` e `rapporti_agenti/controllo_correzioni.txt`: la rilettura della bozza della review
  contro le fonti (lenti numeri, riferimenti e fedeltà) e il controllo delle correzioni. Le segnalazioni citano righe della bozza,
  non della versione committata.
- `script_agenti/<cartella>/`: script e uscite degli agenti, una cartella per revisore, verificatore o rilettore. Le uscite `.out` sono state
  rinominate `.out.txt` perché il repository ignora `*.out`. Esclusi le copie dei file del repository, le ricevute già versionate
  (`LONG-*.json`) e gli export di altri commit.

Avvertenze:
- Gli script leggono percorsi assoluti dello scratchpad di Claude: la copia della proposta (`gpt_c15aac9/`, rigenerabile con
  `git archive c15aac9`) e il worktree `claude-temponet` (`8917e2a`). Per rieseguirli vanno adattati i percorsi.
- Gli esperimenti sintetici (mondi «lineare», «temporale» e «V», micro-benchmark) mostrano meccanismi, non i numeri dei latenti
  veri.
- Un primo giro di agenti si è interrotto alle 09:26 per il limite d'uso del modello. Il secondo giro, i cui rapporti sono qui, è
  ripartito dai suoi script, dopo averli riletti e rieseguiti per i numeri che decidono i verdetti.
