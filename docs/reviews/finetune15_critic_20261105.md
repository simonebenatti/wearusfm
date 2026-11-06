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

## Rilettura dell'emendamento dopo Claude 0968bd1

Il medesimo critico ha rivisto il nuovo contrasto `transfer_gap15_minus_gap5`,
la policy TIMEOUT/NODE_FAIL e i limiti L1–L4 recepiti nell'analisi5.
**Favorevole, nessun difetto sostanziale.** Oracolo indipendente su sei soggetti
sbilanciati, 2.000 campionamenti e seed20261102: punto, IC e SE di Δ15−Δ5
riprodotti entro1e-12. Le sei celle condividono le estrazioni; non si
sottraggono estremi degli IC. Ricetta db975bd1… concorde nei quattro artefatti.

Rieseguiti dal critico32test in8,48s e bash-n; dall'owner il pacchetto completo
**107test in13,84s**, includendo regressioni5 e operazioni. Launcher con
no-requeue e rifiuto restart prima del setup; nessuna falsa ripresa da pesi
privi di optimizer/RNG. Secondo sguardo sul held e nessuna terza dose dichiarati.
Due refusi editoriali segnalati e corretti; nessun cambiamento di algoritmo
dopo questa rilettura. Provenienze/risorse di preparazione rigenerate dall'owner.
La review esterna del codice FT15 resta necessaria; Claude0968bd1 approva5.

## Verifica operativa indipendente dopo Claude ceaf95b e benchmark63579657

Il critico ha verificato in sola lettura locale: review copiata esattamente,
R2/R3 registrate, ricetta db975bd1 e sorgenti13de2f31 invariati,24ricevute
storiche valide. Report benchmark772bc487…/5018byte uguale alla raccolta
remota, codice02f3de8/piano76a6c87d… corretti, COMPLETED0:0/453s/1GPU8CPU.

Ricalcolo col controller e formula indipendente concordanti: q=1 per tutti,
D=4,161388635s; proiezioni17.551/50.701/51.027s sotto21.300/57.300/57.300,
tutti3split. Memoria host32,386GiB e GPU6,584GiB valide. Preventivo24fit
265,064444GPUh/2.120,515556coreh, benchmark0,125833GPUh/1,006667coreh.
Nessun blocco al pin, pubblicazione e secondo checkout. Prima dei fit restano
verifica checkout, quota fresca, assenza duplicati/output e dipendenzeC1→C0.
Nessuna SSH, modifica o sottomissione del critico; ownerGPT esegue.

## Audit indipendente delle 24 sottomissioni

Sola lettura locale su ricevute e ledger canonico: nessun rilievo.
24ID logici/24SLURM unici,12afterok esatti C1→proprioC0, CODE_DIR e
EXPECTED_SHA1359b924 coerenti, launcher/output/allocazioni e tetto fit
304GPUh/2432coreh corretti. Checkout immutabile e pin verificati; quota
35,23GiB con evidenza fresca per tutti gli invii. SHA e contenuti delle
24ricevute coincidono con il ledger salvo le redazioni dello stderr dichiarate;
una sola sequenza prenotato→in_sottomissione→sottomesso per ID, nessun retry.
I due percorsi del submitter contengono gli stessi byte (814da4bc…).
Snapshot delle13:15:17CET:12RUNNING/12PENDING per dipendenza; il critico
ha letto lo snapshot, senza eseguire una nuova SSH. Nessuna azione sui job.
