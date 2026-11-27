# Controprova del rilievo I-1 (grep -c con codice 1)

Fatto: vero. Prova locale: `grep -c ZZZ file` stampa `0` ed esce con codice 1. Nella sequenza (riga 72) il grep e' l'ultima riga
del blocco del passo 1, quindi se l'operatore incolla il blocco in una sola chiamata Bash vede «Exit code 1» con uscita 0.

Conseguenze: il commento accanto dice «deve essere 0» e l'uscita stampata e' proprio 0; il caso peggiore e' che l'operatore
si fermi e chieda, cioe' un fermo prudente con ritardo. Nessun invio parte, nessun ID si brucia: il passo 1 precede tutti gli invii.
La seconda parte (cartella ~/wearusfm_local/jobs/<ID>): il submitter usa `mkdir` atomico come prenotazione (scripts/submit_job.py
righe 142-146) e rifiuta prima di sbatch, quindi anche la cartella orfana produce un rifiuto sicuro, non un tentativo bruciato.
Nota: oggi il ledger contiene gia' 2 righe CN-RESTART545 e la cartella CONTRAST-CHECKOUT-CN-RESTART545-20261121 esiste (dopo
l'incidente): non riguarda la validita' del rilievo.

Verdetto: regge, ma la gravita' corretta e' «minore» (non cambia l'esito; al piu' un fermo prudente). Correzione utile ma non necessaria:
scrivere «stampa 0 e codice di uscita 1».
