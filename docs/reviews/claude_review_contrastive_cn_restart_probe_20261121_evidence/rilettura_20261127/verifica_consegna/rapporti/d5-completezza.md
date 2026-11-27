# d5 — Completezza della sequenza e del brief rispetto al protocollo di GPT e a decisioni.md

Fonti: sequenza `.claude/worktrees/claude-training-engine/docs/contrastive_cn_restart_probe_launch_20261127.md` (S),
brief `scratchpad/rev_probe2/brief_launcher.md` (B), protocollo `docs/contrastive_cn_restart_probe_20261121.md` al
commit di release (P, righe 31-87), `docs/decisioni.md` righe 8399-8419 (D), review 76f5d0e7 righe 1-201 (R).

## Tabella di corrispondenza (richiesta di P riga 43-45 -> S)

| Richiesta del protocollo | In S | Equivalente? |
|---|---|---|
| Assenza di prenotazioni | passo 1 (grep sul ledger) | parziale: non guarda le cartelle `~/wearusfm_local/jobs/<ID>/` (la prenotazione e' la cartella, submit_job.py riga 6) |
| Assenza di uscite preesistenti | passi 3 e 6 | si' |
| Saldo netto degli impegni | assente | no |
| Billing corrente | assente | no (precedente: `saldo -b` e AllocTRES billing nelle istantanee storiche) |
| Quota | passi 3 e 6, `lfs project -d` + `lfs quota -hp` | si', ma "4 GB oltre al checkout" senza "dopo gli impegni concorrenti" |
| Scritture concorrenti | assente (solo coda dell'utente) | no |
| Spazio per osservazioni e checkpoint | 4 GB | si' (numero del protocollo) |
| Slot debug | passo 3 | si'; manca lrd_all_serial per il checkout |
| Reservation | passo 3 | si'; non dichiara superata la pianificazione del 23/11 |
| Autenticazione SHA del deploy | passo 2 (ls-remote) | si' |
| Autenticazione riferimenti storici | implicita nel validatore (checked_pin) | si', non detto esplicitamente |
| Checkout COMPLETED 0:0, SHA, pulizia | passo 4 | si', ma `$CODE_DIR` non definito e senza comando sacct |
| Validazione sul login, stdout/stderr/ricevute | passo 5 | si' per salvarle; non entrano nella raccolta del passo 7 |
| Variabili del job GPU | passo 6 | si' (prova a secco conforme) |
| Nessun invio implicito dal validatore | implicito | non detto: non eseguire i comandi stampati |
| phase_timings.jsonl | passo 7 | si' |
| JSON delle interruzioni | passo 7 | si' |
| Inventari e impronte per rank | passo 7 | si' |
| Due file SLURM con SHA e accounting | passo 7 | si' |
| Tensori su WORK con impronte e percorsi | passo 7 | parziale: impronte dei checkpoint a 576 solo nel report |
| Lettura scientifica | passo 8 | si' |
| Recupero del solo lettore | passo 8 | testo stretto di D; omette caso del protocollo e la riserva di R minore 5 |

## Rilievi

Vedi l'oggetto strutturato restituito (stessi ID R1-R12).

## Nota sulla mia esecuzione

Ho eseguito per errore `git ls-remote origin | head -0` (head ha rifiutato l'argomento e la pipe si e' chiusa): una
lettura di rete verso GitHub, non push/fetch/pull. Nessun file nei worktree e' stato modificato.
