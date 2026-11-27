# d3 - operatore che esegue alla lettera

Revisore d3, 27/11/2026. Tutto in sola lettura: nessun ssh, nessun submitter, nessuna scrittura nei worktree.

## Prima passata: solo brief e sequenza

1. Passo 1. Eseguo i quattro comandi. `grep -c "CN-RESTART545"` stampa `0` ma esce con codice 1. Il brief dice di fermarsi se un comando dà un errore inatteso. Qui ho un dubbio: il codice 1 è un errore? Non ricontrollo che il submitter condiviso coincida ancora con quello del commit (la tabella dice che coincideva, ma non c'è un comando per verificarlo al momento del lancio).
2. Passo 2. Faccio il push del riferimento deploy dal worktree di release, poi ls-remote. Chiaro.
3. Passo 3. Non ci sono comandi, solo un elenco. Devo scriverli io. Dubbio: `"$WORK"` va espanso sul cluster, quindi lo metto fra apici singoli dentro `ssh leonardo '...'`. Se lo scrivessi senza apici, `$WORK` sul Mac sarebbe vuoto. "Conservare l'uscita": dove? La cartella `~/wearusfm_local/cn_restart545_20261127` viene creata solo al passo 5. Non so come contare i "posti liberi" su dbg, né che cosa fare se sono esauriti.
4. Passo 4. Lancio il comando dalla radice del worktree di release. Ricavo il jobid dalla riga del submitter, che però non è descritta. Come aspetto la fine? Il comando squeue/sacct compare solo al passo 6. Il log ha il nome `campaign_checkout_<jobid>.out` ma manca la cartella. "Verificare poi `git -C "$CODE_DIR" rev-parse HEAD`": `CODE_DIR` non è definita sul Mac. Va eseguito via ssh? Con quale percorso? Se l'esito è `rifiutato` o `ambiguo`, che cosa faccio? La sequenza ne parla solo al passo 6.
5. Passo 5. Eseguo il blocco così com'è. Dubbio: l'uscita standard contiene una riga «codice: …», due date e forse più di un JSON. Lo stderr avrà gli avvisi di ssh: è un errore?
6. Passo 6. Rileggo quota e cartella di uscita senza comandi indicati, poi invio. Mi chiedo se devo ancora essere nella radice del worktree di release. Per aspettare non so ogni quanto rileggere la coda. Non so che cosa fare se il job resta in coda a lungo, né se il certificato scade (il brief dice 02:34 del 28/11).
7. Passo 7. Non ci sono comandi di copia: devo usare scp o rsync? Come escludo i `.pt`? Le sottocartelle si chiamano `ordinary/` e `warm/`? Non so se raccogliere anche le uscite della validazione, del passo 3, il log del checkout e gli state.json del registro. "Percorso proposto": è obbligatorio? In quale cartella del Mac scrivo? Nel worktree `claude-training-engine`, che è della chat revisore?
8. Passo 8. Non so in quale worktree o chat faccio il commit, né come calcolo il consuntivo (formula delle ore locali).

## Seconda passata: dubbi sciolti con le fonti

- **Il passo 5 non crea la cartella di uscita.** In `contrastive_cn_restart.py`, `run(check_only=True)` (righe 140-153) non chiama mai `mkdir`. Il controllo `contrastive_run.py --validate-only` esce prima delle `mkdir` (righe 316-327). Quindi la validazione non brucia il job GPU.
- **Le direttive della risorsa GPU stanno nello sbatch.** `cn_restart_probe.sbatch` ha `-A`, `-p boost_usr_prod`, `--qos=boost_qos_dbg`, `--gres=gpu:4`, `--nodes=1`, `--ntasks=1` e i log in `wearusfm_runs/logs/cn_restart_probe_%j.*`. Il submitter impone `--job-name=<ID>` e il log conserva il nome `cn_restart_probe_<jobid>` perché `-o` è letterale.
- **Il log del checkout.** È `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_<jobid>.out`; stdout e stderr finiscono nello stesso file (`campaign_checkout.sbatch`, righe 9-10). Contiene due righe: «checkout di campagna pronto: <DEST> @ <SHA>» e «files: …».
- **Il passo 5 copre già SHA e pulizia.** `code_dir_setup` verifica HEAD, pulizia, sola lettura e poi esegue `cd`. Il controllo di `git -C "$CODE_DIR"` al passo 4 è ridondante. Scritto così, sul Mac fallisce.
- **Lanciare dalla cartella sbagliata non brucia l'ID.** Su `main` e su `worktree-claude-training-engine`, `cn_restart_probe.sbatch` non esiste. `submit_job.py` legge lo script (`read_text`) prima di `reserve`, quindi senza prenotazione. `campaign_checkout.sbatch` è identico in tutti i rami (eaafc432).
- **Lo stdout del passo 5 è misto.** Contiene: data, «codice: … @ …», una riga JSON di `contrastive_run.py` (`recipe_valid`, `release_valid`), il JSON indentato del driver e di nuovo la data. Lo stderr contiene di sicuro l'avviso post-quantum di ssh (si vede nelle letture del 27/11).
- **Uscita del submitter.** Il submitter stampa `<ID>: sottomesso, job <N>` ed esce con 0. Per `rifiutato`/`ambiguo` esce con 3. Solo un nuovo invio sullo stesso ID mostra il suggerimento `--dopo-rifiuto` (righe 157-160 e 238).
- **Il preparatore che la sequenza toglie verificava una cosa in più.** `mac_launcher_command.py` controllava che il submitter condiviso coincidesse byte per byte con quello del commit, a ogni invio. Oggi coincidono: `814da4bc…` sia nel worktree di release sia nella copia condivisa.
- **Il ramo dichiarato non contiene la cartella di raccolta.** `worktree-claude-training-engine` (76f5d0e7) non contiene `results/infra/contrastive_targets_20261118/`. Il worktree del revisore ha due elementi non tracciati: la sequenza e `docs/reviews/…_evidence/rilettura_20261127/prova_a_secco_invii/`.
- **Il protocollo chiede controlli che la sequenza omette.** Nella sezione «Risorse e passaggio operativo» il protocollo chiede anche di verificare il «saldo netto degli impegni, billing corrente, quota e scritture concorrenti».

I rilievi classificati sono nell'oggetto strutturato restituito al workflow.
