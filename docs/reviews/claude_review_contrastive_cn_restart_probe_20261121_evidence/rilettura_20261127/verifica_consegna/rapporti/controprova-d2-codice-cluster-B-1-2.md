# Controprova del rilievo B-1 (ledger vero con l'ID del checkout in stato 'ambiguo')

Esito: il rilievo REGGE, gravita' bloccante.

## Verifiche fatte
- `grep -n CN-RESTART545 ~/wearusfm_local/jobs/ledger.tsv`: righe 1891 (prenotato, 15:33:41) e 1892 (ambiguo, 15:33:42) per
  CONTRAST-CHECKOUT-CN-RESTART545-20261121, codice b58a66f6. Nessuna riga per l'ID GPU.
- `~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json`: stato "ambiguo", job_slurm null, uscita 0,
  stdout "finto-stdout", stderr "finto-stderr".
- `agents2/d1-comandi/finto_ssh_bin/ssh` (15:33) stampa esattamente finto-stdout/finto-stderr ed esce 0: la scrittura viene da una
  prova a secco puntata sul registro vero. Nessun sbatch reale e' partito.
- scripts/submit_job.py righe 152-160: con stato 'ambiguo' `reserve` esce con "Non si risottomette: usare --riconcilia".
  Da 'ambiguo' si arriva ad 'assente' solo con due --riconcilia vuoti distanziati (righe 13-18, 293-296), poi --dopo-assenza-verificata.

## Tentativo di confutazione (lente delle conseguenze)
- L'operatore al passo 1 (sequenza riga 72, brief punto 1) trova 2 invece di 0: se segue i testi si ferma; se prosegue, il passo 4
  viene rifiutato dal submitter. In nessun caso il lancio avviene seguendo i testi alla lettera; la procedura di riconciliazione non
  e' nei due testi e un modello piccolo potrebbe improvvisare (ID nuovo, cancellazione dello stato) fuori dalle regole.
- Attenuante: il tentativo non e' davvero bruciato (nessun job Slurm), l'ID e' recuperabile con la riconciliazione registrata.
  Non basta per declassare: il lancio fallisce cosi' com'e' scritto.
- La sequenza alla riga 43 afferma "Il ledger vero non e' stato toccato": ora e' falso.

## Correzione
Far decidere a Simone il ripristino (due --riconcilia distanziati fino ad 'assente' e --dopo-assenza-verificata al passo 4, oppure
pulizia registrata), poi aggiornare righe 38, 43, 72, passo 4 e punto 1 del brief.
