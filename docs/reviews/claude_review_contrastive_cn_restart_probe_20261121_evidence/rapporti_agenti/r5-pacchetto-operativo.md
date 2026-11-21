# R5 - Pacchetto, binding, piano, lanciatore, tempi e costi (commit 9d34dc69)

Eseguito in un clone privato della copia (agents/r5-pacchetto-operativo/sim); il clone sotto review e' rimasto pulito.

## Eseguito
- `--package`: package_sha256 b0640ca6...dcfa3, plan_sha256 9d5811b6...d6 (coincidono con il blocco richiesto).
- 4 file di test: 32 passed in 13,3 s; `bash -n` del lanciatore ok.
- Simulazione del binding (sim_binding.sh / sim_binding.out): blocco corretto -> BOUND e require OK (anche con accenti UTF-8 nel testo,
  spazi finali, JSON indentato su piu' righe, copia in blockquote o indentata in aggiunta al blocco vero); rifiutati: blocco doppio,
  hash sbagliato, CRLF, esempio dentro una fence ```` (conta come secondo blocco), vecchio blocco contrastive-release-review, commit
  abbreviato, commit non antenato del ramo di Claude (CalledProcessError), file del pacchetto modificato, modulo di produzione
  modificato (source_digest), evidenza storica modificata (git diff). Accettati: test modificati e submit_job.py modificato (fuori
  dal pacchetto, non eseguiti sul cluster); binding contraffatto ma autoconsistente (require_binding non riautentica sul ramo di Claude).
  Rebind sopra un binding esistente: FileExistsError.
- Vecchio binding job['package_review']: validate_review_binding ritorna reviewed_commit fbfde797 (verify_git attivo).
- Timeout: subprocess.TimeoutExpired non e' catturato da run() -> nessun execution_failure.json (timeout_demo.py).

## Rilievi
- I-1 importante: contrastive_cn_restart.py:174 non cattura subprocess.TimeoutExpired.
- N-1 nota: il commit di review deve stare su refs/remotes/origin/worktree-claude-training-engine.
- N-2 nota: il documento di review non deve contenere un secondo blocco in una fence (anche annidata), ne' CRLF.
- N-3 nota: binding autoconsistente non riautenticato al runtime (per disegno).
- M-1 minore: manca il comando esplicito della validazione sul login (ambiente).
- M-2 minore: ledger --gpu-ore 0.59 contro tetto 1,0.
- N-4 nota: log .err separato da raccogliere; disco ~3 GB su WORK.

## Tempi
Due gambe storiche 205-265 s per job intero (32 passi ~99 s, primo passo 23,5 s di cui 18,5 di dati); stima diagnostico
450-620 s su ~830 s utili (900 - 25 - preambolo). Margine ~210-380 s.

## Costi
1,0 GPU-h e 8,0 ore locali + 0,2667 = 8,2667; inviluppo 1,4 / 11,8667; base 4,74 / 47,3322 (ricalcolata dai 17 sacct del repo);
cumulato 5,74 / 55,5989; residuo 12,26 / 114,0011; con inviluppo 6,14 / 59,1989.
