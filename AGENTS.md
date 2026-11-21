# WearUsFM su Leonardo

Per le chat scientifiche 1, 2 e 3 e i loro launcher, il via di Simone copre l’intera catena concordata. Un solo launcher per linea segue i passi approvati fino alla raccolta e pubblicazione dei risultati.

Prima di operare leggi la sezione «Un solo via per le tre chat e i loro launcher» di `CLAUDE.md`. Sul Mac condiviso il riferimento operativo è `/Users/simonebenatti/dev/wearusfm/CLAUDE.md`. Il codice scientifico e i piani si leggono dal commit fissato per la singola campagna.

Per ogni invio già rilasciato, prepara il comando con `python3 /Users/simonebenatti/dev/wearusfm/scripts/mac_launcher_command.py WORKTREE SHA -- <comando del piano>`. Il preparatore locale confronta il submitter con il commit congelato e restituisce `cmd` e `workdir`. Dopo i controlli previsti, passa questi due campi direttamente allo strumento di esecuzione. Mantieni separati preparazione, invio e raccolta. Usa le dipendenze già previste dal piano per i passi accodabili. Per un ID già inviato verifica la ricevuta e la partenza su Leonardo.
