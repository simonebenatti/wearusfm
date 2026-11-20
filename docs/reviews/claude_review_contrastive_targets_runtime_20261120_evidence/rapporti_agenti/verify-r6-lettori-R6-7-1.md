# Verifica scettica n.1 di R6-7 (r6-lettori) — avvio
Rilievo: "'CPU e GPU devono terminare nei rispettivi limiti' verificato con sacct solo per l'audit" (NOTA).
Piano: rileggere prontuario 181-187, readout 201-273/392-526, job 129-159, run (timer), trainer (stopped), parity finish, release (sacct), launcher sbatch, piani.

## Esito: NON confutato. Gravita' confermata: NOTA

Prove riprodotte a fbfde797. La punta 0efa4d45 aggiunge solo delivery_pin.json.
1. Prontuario, righe 183-184: «La lettura richiede tutte le sette esecuzioni complete [...]; CPU e GPU devono terminare nei rispettivi limiti.»
2. sacct esiste solo per l'audit. `validate_audit_accounting` (release.py 155-192) e' chiamata da readout 421, job 81, run 195 e make_job 242.
   `git grep 'sacct|ElapsedRaw|TimelimitRaw'` su scripts/contrastive_*, slurm/contrastive* e src trova solo l'audit.
   Il piano ha `audit_sacct_gate` e "not_scientifically_accepted_runtime": "audit beyond7200s rejected even if COMPLETED", entrambi solo per l'audit.
3. `read_preflight` prende i parametri cohort, full, resumed, cuda, cpu_audit, calibration, calibration_run_dir, rank_exit e cpu_audit_sacct.
   Non c'e' alcun accounting GPU; la CLI ha solo `--cpu-audit-sacct`.
4. `_run_evidence` (201-273) ha un solo criterio di tempo: `stopped=='max_steps'` (207). Non legge `elapsed_s`.
   Prova (probe con l'helper `numeric_folder` dei test, in tmp): con elapsed_s None, 1799, 1980 e 99999 il run viene sempre accettato.
5. Timer. In run.py, riga 1243, `t_start` parte all'ingresso di `train()`. La riga 1609 controlla il tempo all'inizio di ogni passo:
   limita quando inizia l'ultimo passo, non quando il job finisce.
   Fuori dal timer restano:
   - il dispatcher, job.py 62-159 (validate_review_binding con git show, checked_file del fork);
   - il runner, run.py 290-364, su ogni rank: validate_release, poi validate_review_binding con verify_git=True (125 git show piu' 2 piani);
     validate_target_fork (hash e torch.load); prepare_reload (copia e hash); sha del checkpoint su ogni rank;
   - dopo R.train: save finale, `observer.finish` (parity 303-333: barrier, source_digest, scritture) e destroy_process_group.
   Misura locale: i 125 git show seriali richiedono 4,30 s su SSD (Mac), ripetuti nel dispatcher e in ciascuno dei 4 rank.
6. In job.py, riga 132, `expected_time = 1770`, e la riga 159 passa `--time-limit-s 1770`. Lo sbatch GPU chiede `--time=00:30:00` e non ha un timeout esterno.

Perche' NOTA e non di piu':
- **Gate formali.** Il gate scritto nel piano, il «senza estendere i gate» di decisioni e la R-4 di 94f5094 riguardano solo l'audit.
  La frase del prontuario e' piu' larga di quello che codice e piano impongono.
- **Effetti di un'eccedenza GPU.**
  - Entro OverTimeLimit piu' KillWait (al massimo 360 s per allocazione) resta nella riserva gia' inclusa nell'inviluppo di 21,6 GPUh.
  - Oltre quel margine il job finisce TIMEOUT.
  - Le misure restano complete e autenticate. Nessun tetto superato, nessun esito attribuito male.
- **Rischio pratico.** Per i reload (32 update) e' trascurabile. Per i full esiste solo se il ciclo finisce a ridosso dei 1770 s;
  la durata del nuovo profilo non e' ancora misurata.

In aggiunta: nemmeno le celle GPU di calibrazione e CUDA hanno una verifica sacct. La CUDA non ha neppure un timer interno:
ha solo il timeout NCCL di 10 minuti e la scadenza di 90 s del probe rank-exit.

Non verificato:
- se su Leonardo e boost_qos_dbg un job che finisce entro OverTimeLimit risulti davvero COMPLETED (il prontuario, righe 105-109, lo presuppone);
- i tempi reali dell'overhead su Lustre.
