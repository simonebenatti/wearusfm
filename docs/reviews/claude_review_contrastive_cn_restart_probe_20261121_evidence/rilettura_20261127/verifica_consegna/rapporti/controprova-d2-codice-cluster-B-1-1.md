# Controprova del rilievo B-1 (lente dei fatti)

Esito: il rilievo REGGE. Gravità corretta: bloccante.

## Verifiche fatte (sola lettura)
- `grep -n CN-RESTART545 ~/wearusfm_local/jobs/ledger.tsv`: righe 1891 (`prenotato`, 15:33:41) e 1892 (`ambiguo`, 15:33:42) per
  `CONTRAST-CHECKOUT-CN-RESTART545-20261121`, agente claude, codice b58a66f6, stessa riga di sbatch del passo 4. Nessuna riga per l'ID GPU.
- `~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json`: stato `ambiguo`, job_slurm null, stdout `finto-stdout`,
  stderr `finto-stderr`.
- `.../rev_probe2/agents2/d1-comandi/finto_ssh_bin/ssh` (15:33): stampa `finto-stdout` / `finto-stderr` ed esce 0. Quindi la prova a
  secco con ssh finto ha scritto nel registro vero; sul cluster non è partito nulla.
- `scripts/submit_job.py` riga 47: registro predefinito `~/wearusfm_local/jobs`; righe 141-161: con cartella esistente e stato `ambiguo`
  la prenotazione si rifiuta («Non si risottomette: usare --riconcilia»). Righe 13-17 e 49: `assente` solo dopo due controlli vuoti,
  il primo >= 10 min dal tentativo e l'ultimo >= 10 min dopo il primo; poi si risottomette solo con `--dopo-assenza-verificata`.
- Sequenza riga 100-105 (passo 4): comando senza `--registro`, quindi usa il registro vero e verrebbe rifiutato.
- Sequenza righe 38 («nessuna voce nel ledger per i due ID») e 42-43 («Il ledger vero non è stato toccato»): la seconda è oggi falsa.
- Brief riga 15: «nessuna riga con CN-RESTART545 nel ledger comune» come controllo del punto 1: oggi fallisce.

## Gravità
Seguendo i testi alla lettera il lancio si ferma al punto 1 o viene rifiutato al passo 4 (il rifiuto avviene prima di sbatch, quindi
nessun job parte e il tentativo non si brucia sul cluster, ma l'ID è bloccato finché non si riconcilia). Rientra in «il lancio
fallirebbe»: bloccante. Correzione proposta dal revisore ragionevole: decisione di Simone sulla pulizia (riconciliazione fino ad
`assente` e `--dopo-assenza-verificata` nel passo 4, oppure pulizia registrata), poi aggiornare righe 38, 42-43, passo 4 e punto 1 del brief.

## Non verificato
Non ho interrogato Leonardo (divieto): l'assenza del job sul cluster si deduce dallo ssh finto, non da sacct/squeue.
