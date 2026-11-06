# Prove del coordinamento di Claude: letture dei job e copia del ledger

Questa cartella tiene nel repo due cose che prima stavano solo sul Mac di Simone, in `~/wearusfm_local`:

- `letture_leonardo/`: le letture di Leonardo fatte dalla chat di coordinamento di Claude (stato dei job, code, `sacct`,
  priorita' dei nostri job, quota di `$WORK`, riferimenti git del repo su Leonardo). Sono le letture citate nello specchietto
  `docs/coordinamento_review_claude_20261104.md` e nelle voci di coordinamento di `docs/claude_GPT_shared`. Il nome di ogni file
  porta data e ora della lettura.
- `ledger_copia.tsv`: copia intera del ledger comune delle sottomissioni, `~/wearusfm_local/jobs/ledger.tsv`, scritto da
  `scripts/submit_job.py` per Claude e per GPT. Campi: `quando`, `id`, `agente`, `campagna`, `stato`, `job_slurm`,
  `gpu_ore_stimate`, `codice_sha`, `comando`.

Decisione di Simone, 06/11/2026. Alla domanda di Claude «Metto nei commit anche le letture dei job e una copia del ledger?»:
«Sì, aggiungili». La domanda nasceva dalla sua «stiamo tenedo traccia di tutto coi commit?».

## Cosa vale e cosa no

- Sono copie per la tracciabilita'. Il ledger che vale resta quello sul Mac: `scripts/submit_job.py` non legge mai questa copia.
- La copia del ledger si sovrascrive a ogni raccolta: le versioni precedenti restano nella storia di git.
- Qui non ci sono dati da wearable: solo metadati dei job.
- La raccolta non lancia nulla e non scrive su Leonardo.

## Cosa viene tolto dalle copie

- Le righe dei job di un altro progetto dello stesso utente (nomi `mona_*`, `t1_*`, `t2_*`, percorsi `MONALISA`) e la riga di
  `saldo` di un account diverso da `IscrB_WearUsFM`. Simone, 05/11/2026: «che c'entrano, sono di un a ltro progetto».
- Le righe scritte dal client ssh del Mac: tre righe di avviso e, in alcune letture, una riga con il certificato ssh e
  l'indirizzo di posta di Simone.

`MANIFEST.tsv` dice, file per file, quante righe sono state tolte e riporta lo sha256 dell'originale e della copia.

## Cosa resta solo sul Mac

- `leonardo_priorita_63593128_20261105_2142_coda.txt`: la coda intera della partizione, cioe' job di altri utenti.
- `leonardo_audit_altri_*` e `leonardo_mona_account_*`: letture che riguardano l'altro progetto.
- `leonardo_work_inventario_*`: inventario di `$WORK`, che non e' una lettura dei job.
- Le letture rimaste senza contenuto dopo i tagli (tentativi di connessione andati a vuoto).
- Tutti gli altri file di `~/wearusfm_local/logs` e `~/wearusfm_local/jobs`: non sono letture del coordinamento.

## Come si aggiorna

Dalla radice del worktree di Claude, prima di ogni commit di coordinamento:

    ~/.venvs/wearusfm-tok/bin/python -I results/infra/coordinamento_claude/raccogli_prove.py

Lo script copia le letture nuove, riscrive la copia del ledger e il manifest. Si ferma se trova nel ledger righe dell'altro
progetto, oppure in un file una riga che somiglia a un segreto, a una chiave o a un indirizzo di posta: quel file non viene
copiato. Non cancella mai: una copia gia' nel repo resta anche se il sorgente sul Mac non c'e' piu'.
