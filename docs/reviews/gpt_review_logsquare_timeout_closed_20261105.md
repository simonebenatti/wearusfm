# T100-LQ: R-TIMEOUT chiuso, solo diagnostico approvato

Review GPT del 05/11/2026 su `ff08483fd04aa5668d97ba54d6b4153a14ff33fe`,
diff da `92fa557`. Registro Claude consultato a `789249a` nelle due sezioni.
Due verifiche indipendenti: supervisore; launcher, pin, risorse e ingresso dei rank.

**APPROVATO per il solo diagnostico. R-TIMEOUT chiuso.**
Restano approvate A e B della review precedente. Nessuna modifica al criterio
scientifico, al guard, all'ingresso dei rank o al controller del training.

## Esito tecnico

Lo stesso supervisore opera nel genitore e nel figlio. Il timeout esterno copre
anche launcher e worker annidati; i gruppi separati osservati restano tracciati
dopo la morte del padre. La pulizia consente prima la terminazione ordinata di
torchrun, poi segnala i superstiti e applica SIGKILL entro una propria scadenza.
Errori della pulizia, superstiti o processi chiusi dopo un ritorno apparentemente
normale invalidano il contesto; log ed errori restano nel rapporto.

Il limite interno sottrae il tempo effettivo di preparazione, 15 secondi di pulizia
e 20 per il rapporto; con meno di 30 secondi disponibili torchrun non parte.
Mantengo le attese pubblicate **75 e 285 secondi**, senza allungarle: non hanno
ancora una misura cluster, ma un eventuale timeout verrà registrato come tale,
senza diagnosi scientifica, retry automatico o aumento implicito del limite.

Costo massimo confermato: `5 × (75 + 15) + (285 + 15) = 750 s`, più 150 secondi
di margine nell'allocazione da 900 secondi. **1 GPU-ora e 8 core-ore**, più
checkout da **0,333333 core-ore**. Nessun training incluso.

## Verifica indipendente

- I quattro file LQ e le due regressioni GPT contano **178 test**. Prima esecuzione
  sandbox: 156 passati, 22 impediti dal bind localhost o dalla lettura di `/bin/ps`.
  Riesecuzione mirata fuori sandbox: **23 passati, 76 deselezionati in 57,30 s**;
  include tutti i 22 casi impediti e un controllo già passato. Copertura complessiva:
  **178/178 verificati**, senza saltare le due regressioni originali.
- Review del supervisore favorevole; coperti worker in sessioni separate, timeout
  esterno durante attesa/pulizia interna, elastic, SIGTERM ignorato e tabella assente.
- Cinque pin verificati per SHA e byte, tutti i 102 sorgenti scientifici invariati.
  Guard, controller, ingresso rank, launcher training e `src/` identici a `92fa557`.
- Launcher coerente con 4 GPU, 32 CPU, 240G, un nodo, 900 secondi; sintassi shell verificata.
- In sola lettura su Leonardo: cartelle di checkout/output assenti, nessun job con
  i due ID logici, nuovo riferimento deploy assente, nessun hook attivo nel repo.
  Il ledger locale viene verificato nuovamente prima della sottomissione.

## Limiti accettati e precisazioni

La supervisione tramite tabella dei processi non è una garanzia universale contro
processi che si separano e diventano orfani tra due osservazioni. I worker ordinari
di torchrun restano figli del launcher; SLURM mantiene il contenimento esterno.
Gli intervalli 0,5/0,1 secondi sono nominali: includere il tempo di `/bin/ps`, quindi
non chiamarli limiti rigidi della finestra cieca. Dimenticare gruppi osservati vuoti
riduce il rischio di riuso dei numeri, senza escludere ogni gara fra snapshot e segnali.

Con tabella disponibile i segnali sono per PID; senza tabella esiste un fallback
al solo gruppo della radice ancora viva e non raccolta. Questa eccezione è ragionevole
e va descritta, senza promettere «mai segnali a gruppi». Queste precisazioni non
richiedono un cambiamento del codice per questa diagnostica SLURM limitata a 15 minuti.

## Rilascio operativo

Piano e risorse passano a `APPROVED_FOR_DIAGNOSTIC`; review e stato pubblicati prima
del checkout. Checkout nuovo `logsquare-initdiag-20261105` sul commit pubblicato,
ID `LOGSQ-initdiag-checkout-20261105` e `LOGSQ-initdiag-run-20261105`, submitter e
ledger comuni, nessun duplicato. Il job GPU attende il checkout riuscito.

L'autorizzazione operativa di Simone è già acquisita, comprese sottomissione diretta,
letture, raccolta, commit e push. Nessun monitor di completamento in background.
Il rapporto non autorizza il training: GPT deve leggere i tre esiti prefissati e
solo dopo preparare la nuova release, i pin e gli ID della catena successiva.
