# Verifica dei comandi (d1-comandi) — sequenza e brief della prova mirata InfoNCE al passo 545

## 0. INCIDENTE CAUSATO DA QUESTA VERIFICA (bloccante, da risolvere prima della consegna)

Durante la simulazione del passo 5 ho estratto per errore il blocco sbagliato della sequenza (il terzo blocco
`bash`, cioè il comando di invio del passo 4) e l'ho eseguito sotto zsh con un ssh finto in testa al PATH.
Il submitter condiviso è partito con il registro predefinito, quindi ha scritto nel **registro vero**:

- cartella `~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json`, stato `ambiguo`
  (storia: prenotato 15:33:41, in_sottomissione 15:33:41, ambiguo 15:33:42; stdout `finto-stdout`);
- due righe in `~/wearusfm_local/jobs/ledger.tsv` (righe 1891 e 1892: `prenotato` e `ambiguo`, agente claude,
  codice b58a66f6).

Nessun job è arrivato a Leonardo: la chiamata è stata ricevuta dal finto ssh, che ha scritto
`agents2/d1-comandi/ssh_argomenti.txt` (`leonardo` + riga `sbatch --parsable ...`) e il testo dello script su stdin.
Il lock d'operazione è stato rilasciato. Il worktree di release è rimasto pulito (`git status --porcelain
--untracked-files=all --ignored` vuoto).

Conseguenze se si consegna così: al passo 1 `grep -c "CN-RESTART545"` dà 2 (l'operatore si ferma, che è il
comportamento sicuro); al passo 4 il submitter rifiuterebbe con «già prenotato, stato 'ambiguo'. Non si
risottomette: usare --riconcilia». La riconciliazione interrogherebbe `sacct`/`squeue` veri, troverebbe vuoto e
porterebbe l'ID ad `assente` solo dopo due controlli a 10 minuti, poi servirebbe `--dopo-assenza-verificata`:
percorso non previsto dalla sequenza.

Non ho toccato il registro per ripararlo (fuori dalla mia cartella di scrittura, stato condiviso). Serve una
decisione di Simone: per esempio togliere la cartella dell'ID e le due righe 1891-1892 del ledger, oppure annotare
l'incidente e seguire la riconciliazione fino ad `assente`. Va deciso prima di consegnare il brief.

## 1. Rilievi sui testi

### I-1 (importante) Passo 1: `grep -c` esce con 1 quando il risultato atteso (0) è corretto
- Dove: sequenza riga 72; brief punto 1.
- Prova: `grep -c "CN-RESTART545" ~/wearusfm_local/jobs/ledger.tsv; echo "exit=$?"` prima dell'incidente ha
  stampato `0` ed `exit=1`. Lo strumento Bash della chat mostra «Exit code 1» per il blocco, e il brief dice
  «se un comando dà un errore inatteso: fermati».
- Correzione: scrivere «stampa 0 ed esce con codice 1: è l'esito atteso», oppure `grep -c ... || true`.
  Aggiungere anche `ls -d ~/wearusfm_local/jobs/CONTRAST-*RESTART545*` (deve fallire): il submitter rifiuta anche
  per la sola cartella dell'ID, senza riga nel ledger.

### I-2 (importante) Passi 3, 4 e 6: controlli su Leonardo senza comando esatto; `$WORK` e `$CODE_DIR` si espandono sul Mac
- Dove: sequenza righe 89-93 (passo 3), riga 111 (passo 4: `git -C "$CODE_DIR" rev-parse HEAD` e «stato pulito»),
  riga 147 (passo 6); brief punto 3.
- Prova: sul Mac `WORK` e `CODE_DIR` non sono impostate (`echo` → `nonimpostata`). Un `ssh leonardo git -C
  "$CODE_DIR" rev-parse HEAD` scritto in zsh arriva come `git -C  rev-parse HEAD` (fallisce: «cannot change to
  'rev-parse'»); `ssh leonardo lfs project -d "$WORK"` arriva senza percorso e legge il progetto della cartella
  sbagliata. In una nuova sessione ssh `CODE_DIR` non esiste nemmeno sul cluster. Il revisore stesso ha usato uno
  script con percorsi letterali (`leonardo/stato_pre_lancio.sh`).
- Correzione: dare i comandi pronti con apici singoli e percorsi letterali, per esempio
  `ssh leonardo 'git -C /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121 rev-parse HEAD;
  git -C ... status --porcelain --untracked-files=all'`, oppure riusare `stato_pre_lancio.sh` con `bash -s`.

### M-1 (minore) Il brief dice che `--agente claude` è «l'unica differenza voluta»
- Dove: brief riga 28. La sequenza (righe 108-109 e 115) dichiara anche l'invocazione diretta del submitter
  condiviso al posto di `mac_launcher_command.py`, lo SHA scritto per esteso e, al passo 5, `export RELEASE_SHA=…`
  e due `date`.
- Prova: diff fra il blocco del protocollo (righe 63-75 di `docs/contrastive_cn_restart_probe_20261121.md`) e il
  corpo dell'heredoc: solo le tre righe aggiunte. Il comando reso da `mac_launcher_command.py` è proprio
  `venv python + submitter condiviso + argomenti` con workdir il worktree, quindi la sostituzione è equivalente.
- Correzione: nel brief «le differenze volute sono quelle elencate nella sequenza».

### M-2 (minore) Passo 5: uscita attesa descritta come «un JSON»; stderr non vuoto atteso
- Dove: sequenza righe 140-141; brief punto 5.
- Prova: `contrastive_cn_restart.py` righe 149-152: prima gira `contrastive_run.py --validate-only`, che stampa
  una riga JSON (`contrastive_run.py` righe 324-327, con `"release_valid": true` ed `"execution_started": false`),
  poi il driver stampa il JSON indentato con `"new_binding_valid": true`. Lo stdout contiene quindi: data, riga
  JSON del runner, JSON del driver, data. Ogni ssh verso Leonardo scrive su stderr tre righe «WARNING: connection is
  not using a post-quantum key exchange algorithm» (`leonardo/*_20261127.err.txt`).
- Correzione: descrivere le quattro parti dello stdout e dire che le righe WARNING su stderr sono normali.

### M-3 (minore) Log del checkout: percorso non scritto e un solo file
- Dove: sequenza riga 110 e passo 7 riga 182.
- Prova: `campaign_checkout.sbatch` righe 9-10: `-o` ed `-e` puntano entrambi a
  `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_%j.out` (nessun `.err`). `--job-name` del
  submitter non cambia il nome del file (`%j` = ID numerico).
- Correzione: scrivere il percorso completo e precisare che i «due log .out e .err» del passo 7 sono quelli del job
  GPU; per il checkout c'è un solo file.

### M-4 (minore) `--script-stdin` relativo: il passo 6 non ripete «dalla radice del worktree di release»
- Dove: sequenza righe 97 e 149-155.
- Prova: `submit_job.py` riga 219 legge il percorso relativo alla cartella corrente. Dal main canonico
  `campaign_checkout.sbatch` ha gli stessi byte (sha256 `eaafc432…`), `cn_restart_probe.sbatch` manca: lettura
  fallita prima della prenotazione (riga 222), quindi nessun ID bruciato, ma un errore inatteso.
- Correzione: percorsi assoluti in `--script-stdin` o `cd` del worktree nello stesso comando.

### M-5 (minore) Riconciliazione citata senza il comando intero
- Dove: sequenza riga 159.
- Correzione: scrivere `/Users/simonebenatti/.venvs/wearusfm-tok/bin/python ".../submit_job.py" --riconcilia
  <ID logico>` (l'argomento è l'ID logico, non l'ID SLURM).

### M-6 (minore) Il controllo d'identità del submitter condiviso non si ripete al lancio
- Dove: sequenza passo 1. `mac_launcher_command.py` (righe 29-33) confrontava ogni volta i byte del submitter
  condiviso con quelli del commit; l'invocazione diretta salta questo controllo.
- Correzione: aggiungere al passo 1 `shasum -a 256` del submitter condiviso (atteso `814da4bc66db…`).

### N-1 (nota) Fork non letto nelle letture del 27/11
- `stato_pre_lancio.sh` controlla `fork/CN/checkpoint.pt` (assente), ma il piano usa
  `benchmark/CN/checkpoint_step0000544.pt`. Lo controlla comunque `--validate-only` (`checked_pin(plan["fork"])`)
  prima del job GPU.

### N-2 (nota) `module` dentro `bash -s`
- Il fatto verificato riguarda la shell non interattiva di ssh. Non ho potuto vedere se la funzione esiste anche
  nel `bash -s` figlio. Se manca, il passo 5 si ferma prima del job GPU: l'ID GPU non viene bruciato.

### N-3 (nota) La sequenza non è ancora nel repository
- `git status` nel worktree claude-training-engine: `?? docs/contrastive_cn_restart_probe_launch_20261127.md`.
  @SEQ_SHA@ richiede commit e push prima della consegna.

## 2. Verifiche positive fatte davvero
- Worktree di release a `b58a66f695c8744dfa5fe633829cafe13a5a1fc4`, pulito; il diff da `8c72655c` è il solo
  `release_binding.json` (+6 righe). Ref locale `origin/claude/cn-restart545-release-20261127` allo stesso SHA;
  review `76f5d0e7` antenata di `origin/worktree-claude-training-engine` (ref locale).
- Impronta del pacchetto ricalcolata come `package()`: `e9310d67…702d`; sha256 del piano `4f538cf1…853b`.
- Submitter del commit e submitter condiviso identici (`cmp`), sha256 `814da4bc…`.
- Prova a secco: le righe `sbatch` di chiamata_1 e chiamata_4 coincidono con i comandi del protocollo, a parte le
  differenze dichiarate; lo stdin coincide byte per byte con i due script del commit (sha256 `eaafc432…`,
  `a8c8b8b4…`); sbatch senza script come argomento legge stdin (nessun `/dev/stdin` necessario, nessun argomento).
- `cn_restart_probe.sbatch`: `$#`=0, `CODE_DIR` uguale a `$WORK/wearusfm_campaigns/cn-restart545-20261121` con
  WORK=/leonardo_work/IscrB_WearUsFM, `TORCH_DIST_INIT_BARRIER=1`, `TimeLimit=00:15:00` da `--time=00:15:00`;
  `validate_allocation("gpu")` chiede 1 nodo, 1 task, 32 CPU, 245760 MB (=240G) e 4 GPU: coincide con #SBATCH,
  argomenti e `plan.json` (`allocation`, `logical_ids`, `output`). Il nome del job non viene controllato, quindi
  `--job-name` del submitter non dà problemi.
- `campaign_checkout.sbatch` stampa `checkout di campagna pronto: $WORK/wearusfm_campaigns/cn-restart545-20261121
  @ <SHA>`: corrisponde alla stringa attesa.
- `--validate-only` non crea la cartella di uscita (`contrastive_run.py` righe 318-327; il driver fa `mkdir` solo in
  `--run`, riga 158), quindi il passo 5 non brucia il job GPU.
- Esiti del job: il confronto scrive `report.json` ed esce con 0 anche se `INCONCLUSIVE` (compare righe 402-406);
  timeout e fasi fallite scrivono `execution_failure.json` e danno FAILED (`checked_command`, `TimeoutError` è un
  `OSError`): coerente con la sequenza.
- Passo 5 simulato in zsh con ssh finto: ssh riceve `leonardo` e `bash -s`, lo stdin coincide con il corpo
  dell'heredoc, la tilde nelle ridirezioni si espande, `uscita: 0`. `bash -n` passa sul corpo e sui tre script.
- Refspec `SHA:refs/heads/deploy/...` provato su due repository locali con `updateInstead`: push accettato,
  `ls-remote` stampa lo stesso SHA, ramo estratto del destinatario invariato. Il remote `leonardo` è configurato,
  nessun hook locale né `core.hooksPath`.

## 3. Non verificato
- Push e `ls-remote` veri verso Leonardo e GitHub, certificato ssh, coda e quota al momento del lancio.
- Esistenza di `module` nel `bash -s` figlio; tempi reali della validazione sul login.
