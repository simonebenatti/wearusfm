# d4-regole: regole del progetto e autorizzazioni (27/11, letture fino alle 15:35)

## BLOCCANTE R0 - ledger comune inquinato da una prova finta
~/wearusfm_local/jobs/ledger.tsv contiene ora due righe per CONTRAST-CHECKOUT-CN-RESTART545-20261121
(agente claude, 15:33:41 prenotato, 15:33:42 ambiguo); ~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json
ha stdout "finto-stdout", stderr "finto-stderr", job_slurm null. E' una prova a secco con ssh finto eseguita SUL REGISTRO VERO
(non quella della cartella prova_a_secco_invii, che e' delle 15:26 su registro temporaneo). Effetti: il passo 1 (grep -c = 0) fallisce;
il submitter rifiuta l'ID ("gia' prenotato, stato 'ambiguo'. Non si risottomette: usare --riconcilia"); la riconciliazione sul
cluster troverebbe zero job e porterebbe ad "assente" solo dopo due controlli a 10 minuti, poi --dopo-assenza-verificata:
percorso non previsto da sequenza ne' brief. Serve decisione di Simone (pulizia documentata del registro o nuovo ID) prima della consegna.
Non ho toccato il ledger.

## Altri rilievi
Vedi l'oggetto strutturato (R1-R9).
