# Prove della rilettura della revisione TCN `3e007c8` (18/11/2026)

Materiale di lavoro della review `docs/reviews/claude_review_pose_temponet_revision_20261118.md`. Contiene solo testi, script e
piccoli risultati: niente dati da wearable, cache o checkpoint. Tutto è stato eseguito sul Mac, su CPU, con conti in forma chiusa,
matrici casuali o file del repository: nessun esperimento sui latenti veri. Fa eccezione un solo comando in sola lettura su
Leonardo (`ls -l` di tre file su SCRATCH, `script_agenti/work_verifica_costo-consegna/ssh_ls_scratch_20261118.txt`).

- `rapporti_agenti/`: rapporti completi dei sei revisori (`review_<dimensione>.txt`) e dei tre verificatori avversari
  (`verify_<coppia>.txt`). Ogni voce porta la sua prova marcata ESEGUITO, CALCOLATO o LETTO. Gli ID dei rilievi (per esempio
  `controlli-1` o `V-ricetta-criteri-2`) sono quelli citati nella review. I sospetti S-A...S-H sono di Claude: li hanno ricevuti
  solo i verificatori, che li hanno giudicati come gli altri rilievi.
- I controlli della bozza prima del commit, in quattro passaggi. Le segnalazioni citano righe delle bozze di quel momento, non della
  versione committata.
  - `rapporti_agenti/rilettura_<lente>.txt`: rilettura della prima bozza contro le fonti, con tre lenti (numeri, riferimenti,
    fedeltà); l'elenco delle loro segnalazioni è in `rapporti_agenti/segnalazioni_rilettori.txt`.
  - `rapporti_agenti/controllo_<lente>.txt` e `verifica_controllo_<lente>.txt`: controllo della bozza corretta, con quattro lenti
    (correzioni applicate, nuovi errori, avvocato di GPT, coerenza) e un verificatore avversario per lente.
  - `rapporti_agenti/controllo_finale_<lente>.txt` e `verifica_controllo_finale_<lente>.txt`: ultimo controllo delle parti
    cambiate, con due lenti (applicazione dei rilievi, tenuta delle regole riscritte) e un verificatore per lente.
  - `rapporti_agenti/rilettura_finale.txt`: un rilettore sulle ultime modifiche.
- `script_agenti/<cartella>/`: script e uscite degli agenti, una cartella per revisore, verificatore, rilettore o controllore. Le
  uscite `.out` sono rinominate `.out.txt`, perché il repository ignora `*.out`. Restano fuori le ricevute già versionate
  (`LONG-*.json`), le copie di altri file del repository (`extraction_plan_19618.json`, `train_subset.provenance.json`), le liste
  dei file usate per il controllo d'integrità (`tree_3e007c8.tsv`, `snap_files.txt`, `tree_files.txt`) e le copie intermedie della
  bozza.

Avvertenze:
- Gli script leggono percorsi assoluti dello scratchpad di Claude: la copia della revisione (`gpt_3e007c8/`, rigenerabile con
  `git archive 3e007c8`) e i file di questo worktree. Per rieseguirli vanno adattati i percorsi.
- I conti sui controlli positivi (`work_controlli/`, `work_verifica_architettura-controlli/`) usano un modello in forma chiusa con
  parametri liberi: quota dinamica del segnale, inflazione dell'errore di stima. Mostrano un meccanismo, non i numeri dei latenti
  veri.
