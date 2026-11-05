# Review indipendente della preparazione EPN15 — 05/11/2026

Revisore `ft_walltime_critic`, sola lettura; non sostituisce la review esterna
di Claude. **Verdetto favorevole alla consegna, nessun rilievo bloccante.**
Esaminati piano, risorse, controller, launcher, aggregatore e test.

Verificati pin storici, 24 candidati completi a cinque epoche, sorgenti originari
invariati, formula `ceil(3*M*1,25*q+600+D)` e ricalcolo al gate. Risorse
6/16/16 ore, 24 split, nessun troncamento o riduzione automatica.
Inizializzazioni originali, dose completa15, griglie e selezione validation-only
coerenti; C0 senza held, C1 verifica C0 e valuta il solo candidato selezionato.
Nonfiniti come esiti espliciti; OOM/errori software propagati; dose parziale
ed epoche incoerenti rifiutate.

Aggregatore: snapshot Git concreto, hash del piano e dei sorgenti ricostruiti
senza eseguire codice storico; placeholder rifiutati. Raw e ricevute5 obbligatori,
risultati storici riprodotti; pairing esatto di righe/target/soggetti/fold fra
le sei celle. Una cella fallita impedisce le statistiche finali: nessun risultato
costruito su supporto ridotto. Diagnostica finale per tutti i regimi descrittiva,
senza estensioni automatiche o dichiarazioni di convergenza.

Il revisore ha rieseguito **86 test distinti**: 55 controller/regressioni in
4,86s e31 aggregatori in7,51s, oltre a `bash -n`. Ricalcolo indipendente del
bootstrap letterale con257campionamenti, classi sparse e attraversamento del
blocco128: i tre IC15−5 coincidono entro1e-12.
L'owner ha poi eseguito il pacchetto combinato con ulteriori test operativi:
**102 passati in12,31s**. Diff nullo sui runtime, launcher, piano e aggregatore5.

Due chiarimenti editoriali recepiti: `q` e `D` esplicitati nel protocollo;
12 combinazioni fold×regime chiamate `fold_regime_cells`, distinte dai24job.

Limiti: nessun risultato FT15 esiste ancora. I test dello snapshot usano un
archivio Git sintetico; il riferimento esecutivo reale sarà verificato dopo
pubblicazione e checkout. Quota corrente, benchmark CUDA, pin operativi e
consuntivi restano controlli prima delle relative fasi. IC descrittivi,
condizionati a modelli/fold e seme unico;15−5 cambia anche l'orizzonte cosine.
Nessuna modifica, SSH, sottomissione o commit da parte del revisore.
