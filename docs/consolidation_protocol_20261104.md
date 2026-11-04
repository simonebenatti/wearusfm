# Consolidamento T100-G: proposta eseguibile da rivedere con Claude

Stato: **convergenza Claude raggiunta, preparazione dei lanci**. Simone ha assegnato a questa chat schedule
lungo e fine-tuning; il 300M appartiene al thread originale. Worktree esclusivo
`wearusfm-consolidation`, branch `codex/consolidation-20261104`, base `8ee6603`.
Nessuna scrittura nei checkout `wearusfm-gpt-science` e `wearusfm-scaling-300m`.
Il duplicato dell'agente 300M ha terminato senza modifiche o job. Un critico locale
legge soltanto questo pacchetto; non sostituisce la review esterna di Claude.

Ordine richiesto da Simone: preparazione → review Claude → correzioni e convergenza
→ pubblicazione dei riferimenti esecutivi → lanci tramite submitter/ledger comune.
La rilettura Claude `0b2623a` su `2dbd498` chiude la convergenza: i piani hanno
`status=claude_review_converged`. Corretto R-1: output FT, compresi i pesi dei
candidati, sotto WORK; `fits_status` convergente. N-2: quota reale pinnata nel
piano, con verifica dell'età subito prima degli invii. Benchmark da eseguire e
pinnare prima dei fit; matrice lunga e aggregatore restano da consegnare.
F-R2 e V-R1 convergenti senza modifiche alle soglie. Dettagli e costi degli invii
in `consolidation_execution_20261104.md`.
Niente monitor in background. Nessuna nuova autorizzazione per il lavoro approvato
dopo la convergenza, ma nessun lancio anticipato, neppure di checkout o benchmark.

## Domande, confronti e confini

| Attività | Cosa si fa | Risultato da leggere | Conclusione consentita |
|---|---|---|---|
| Dose del 30M T100-G | Stato completo da 19.618 a 49.046 | Posa e P1/P2 a 39.237 e 49.046, rispetto a 19.618 | Se più esposizione e completamento dello schedule aiutano questa ricetta |
| Due profondità | Perceiver prima del backbone e uscita dopo il backbone, stesso forward | Sonde di pari larghezza D, stesso supporto e split | Dove questi lettori accedono all'informazione; non perdita irreversibile se una sonda peggiora |
| Fine-tuning EPN | Teacher 19.618 congelato oppure adattato, e stessa architettura casuale | OOF appaiate su quattro fold di sviluppo | Vantaggio del pretraining/adattamento alla dose supervisionata fissata |

Non si lancia T100-L lungo in questa consegna: G49k contro L19k non misura l'effetto
della GELU a pari dose. Non si modifica loss di pretraining, seed, batch o dataset.
Log-quadratico, contrastiva e scaling non sono attività di questo worktree.

## 1. Continuazione, stato e risorse

Checkpoint sorgente teacher/student/optimizer/RNG comune:
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/tokenizer_20261103/T100-G/checkpoint_step0019618.pt`.
SHA256 `a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056`.
Ricevuta originale in `results/infra/tokenizer_execution_20261103/19618/T100-G/receipt.json`;
runtime originale `afb644ea10ecc2a716160b4d8a4904e805c366fb`.

`scripts/consolidation_continue.py` riusa il trainer senza modificarlo. Verifica
gli SHA dei sorgenti scientifici della campagna precedente e ricostruisce la
configurazione dagli stessi argomenti: cambia solo `max_steps`, directory di uscita
e limite operativo. Nuova cartella con `--branch-from` al primo segmento; stato
completo, non `--init-from`. Segmenti successivi riprendono il proprio checkpoint.
Prima e dopo: hash, configurazione, firma dati, optimizer finito, quattro RNG,
stream workers6/segment1200/world4, statistiche posa, nessun STOP. Fase completa
solo con `summary.stopped=max_steps` e numero di passi esatto; niente retry ciechi.

Lo schedule era già WSD49.046: warmup1.962, LR1e-4, decay dal39.237. Non si cambia
la curva e non si riavvia il warmup. Batch globale32, quattro GPU, accumulo1,
seme0, ancore0,2 + posa1, JEPA0. Il checkpoint originario resta in sola lettura.

| Segmento operativo | Passi aggiuntivi | GPU / CPU / RAM | Tempo massimo | Tetto GPUh |
|---|---:|---|---:|---:|
|19.618 → 30.000|10.382|4 / 32 / 480GiB|3h|12|
|30.000 → 39.237|9.237|4 / 32 / 480GiB|3h|12|
|39.237 → 49.046|9.809|4 / 32 / 480GiB|3h|12|
|Totale|29.428|un nodo per segmento|9h prenotate|36|

Il30.000 è solo un confine operativo per tenere le allocazioni entro3h, non un
endpoint selezionato dopo i risultati. Salva una copia esatta dello stato finale.
39.237 e49.046 conservano i checkpoint nativi predecay/postdecay. Stima dai11334s
osservati per14.618 passi di T100-G: **25,35GPUh**, circa6h20min senza coda; tetto
**36GPUh e288coreh allocate**. Deployment iniziale: 1CPU,2GiB,10min,0GPUh,0,167coreh.
I segmenti si concatenano soltanto a ricevute complete; SLURM0:0 da solo non basta.
Namespace `CONSOL-G-30000-20261104`, `CONSOL-G-39237-20261104`, `CONSOL-G-49046-20261104`.

RilieviC-M1/C-M2: [procedura operativa](consolidation_continuation_operations_20261104.md)
per timeout/hash/copia incompleti: niente retry cieco; recupero con piano revisionato
e stato parziale verificato, oppure sola verifica CPU se training già completo.
Copie persistenti in WORK di19.618,30.000,39.237,49.046 prima delle ricevute complete,
SHA verificato e date di purge stimate dai file reali. Gate quota Lustre di progetto
prima del lancio e delle copie; evidenza reale del04/11 17:04:12UTC pinnata nel piano.
Spazio aggiuntivo circa1,8GB, margine richiesto8GiB.

## 2. Sonde nei due momenti e ai due livelli

`harness/backbone_depth.py` osserva con un hook temporaneo il Perceiver nel forward
effettivo di `model.encode`, senza riscrivere l'encoder. Pre e post hanno forma
batch×patch×K×D. Pre include già tokenizer e local encoder; post include la LayerNorm
finale del backbone. K64, D384 qui; nessuna costante384/768 nel nuovo estrattore,
riusabile anche dal300M con D896. Il proprietario del codice comune è questo worktree.

Per classificazione: media FP32 sui64slot e sul tempo valido, D colonne per vista.
Per posa: media solo sugli slot, sequenza temporale conservata. Stessa maschera
`key_time_valid`, stessi ritagli, stesse finestre e bersagli. I due ritagli EPN da4s
si mediano soltanto per classificazione; un'eventuale cache temporale mantiene
un asse crop distinto, non mescola tempi diversi. Posa usa un singolo ritaglio
allineato ai bersagli esistenti: per posa si chiama **depth_readouts sugli input
del loader posa originale**, conservando scale di sessione, margini del filtro e
griglia; non si chiama extract_windows, il cui preprocessing è da classificazione.
Il pool storico2D resta separato: post-backbone
medio concatenato al locale medio. La metà post deve coincidere numericamente
con le prime D colonne della cache storica.

Matrice:19.618 riferimento,39.237 predecay,49.046 finale; teacher congelato; pre,
post e pool storico. Per19.618 pool/OOF già esistenti, estrazione nuova necessaria
solo per le viste mancanti. Stessi split e griglie convergenti delle ultime sonde,
scaler/PCA solo dove previsti e solo sul fit. Nessuna selezione del livello sul held.
Nuove estrazioni riusano lo stesso forward per entrambe le profondità.

Contrasti principali di dose sul pool storico:49.046−19.618, poi39.237−19.618 e
49.046−39.237. Per profondità: post−pre appaiato a ciascuna dose. Riportare tutti i
contrasti, non scegliere il migliore; il confronto profondità è diagnostico.
Classificazione: balanced accuracy EPN/UCI/DB2/DB3/DB6 e P1/P2, IC95% bootstrap
appaiato per soggetto,2000repliche seed20261102, stessa convenzione classi mancanti.
Posa: E_dyn, livello, correlazione, varianza predetta/target ed E2 sul supporto
comune; gli stessi bersagli/split delle valutazioni19.618. Non usare il pannello
train come evidenza su utenti esclusi. Seme unico e dati di sviluppo rimangono.

La [risposta alla review](reviews/consolidation_response_to_claude_20261104.md)
pre-registra lettura a due lati e scelta del riferimento per compito. Il controllo
T25-L/B si riporta con i propri IC, senza trasformare una coppia in una stima
universale del rumore. Soglie pratiche proposte:1pp classificazione,2% relativo
E_dyn posa, oltre a IC appaiato favorevole; da confermare con Claude prima delle
estrazioni, non applicate retroattivamente come verdetto.

La libreria comune e i controlli di parità sono in questa consegna. **I controller
di estrazione/fitting della matrice lunga non sono ancora in questa consegna**:
serviranno piani dedicati con pin dei checkpoint effettivi, adattatori alle cache
P1/P2/posa e benchmark CPU prima dei fit. La review di questa proposta non
autorizza a usare i vecchi valutatori hardcoded768 su cacheD. Non si dichiarano
queste valutazioni già in coda o già comprese nel costo del solo training.

Preventivo preliminare estrazioni: fino a12passaggi (tre checkpoint × posa train,
posa held, P1, P2), ciascuno1GPU/8CPU/100GiB fino45min: **tetto9GPUh/72coreh**.
È un limite di pianificazione da verificare col primo passaggio, non un tempo
osservato. Benchmark CPU separato8CPU/30GiB/15min,2coreh; matrice dei fit da
quantificare e pubblicare dopo misura, senza lanci automatici da questa stima.

## 3. Fine-tuning: prima prova EPN a19.618

EPN escluso dal pretraining, ma i soggetti delle sonde sono già sviluppo. Si riusa
la popolazione interna P2: fuori dall'unione dei test ufficiali dei semi**0,1,2**
(non42/123/456), quattro fold seed20261015, validation10% dei soggetti restanti.
Identità dei dati contro fingerprint storico; ruoli soggetto disgiunti. Classi
definite sul fit, nessuna eliminazione silenziosa di classi assenti nel fit.
Held letto per la predizione solo dopo selezione di LR ed epoca su validation.
Test ufficiali non valutati. Questo è sviluppo del protocollo, non conferma finale.

| Regime | Encoder | Testa | LR candidati |
|---|---|---|---|
|frozen|teacher19.618 immutato, eval|LayerNorm768 + Linear alle classi|testa1e-3,1e-2|
|finetune|teacher19.618, tokenizer→backbone aggiornabili|stessa testa, stessa inizializzazione|encoder1e-5,1e-4; testa100× =1e-3,1e-2|
|scratch|architettura identica, pesi casuali seed0|stessa testa, stessa inizializzazione|encoder1e-4,1e-3; testa10× =1e-3,1e-2|

Griglia della testa identica per indice nei tre regimi (F-B1); moltiplicatori
registrati nel piano, nella firma della ricetta e nei test (F-M4).

Decoder, ancore e testa posa non appartengono al percorso classificazione e non
ricevono aggiornamenti. «Fine-tuning completo» significa tutti i moduli dell'encoder
usati nel readout, incluso il tokenizer. Nessun teacher EMA downstream. La rete
casuale non eredita pesi/gain del teacher: costruzione nuova, calibrazione sintetica
prevista dal costruttore, stessa architettura; non si afferma che sia lo step0
bit-a-bit del run storico. Identica testa iniziale fra regimi, separata dal RNG FM.

Proposta da far validare a Claude: **5epoche complete**, batch8, seed0, AdamW
weight_decay0,01, clipping1, warmup fino100update (10% se dose più corta), cosine
fino alla dose prestabilita; nessuna augmentation aggiunta. Class-weight dal solo
fit, cross-entropy pesata con riduzione per esempio. Due LR per regime:24candidati
in12job fold×regime oppure24job nel ripiego preregistrato, selezione della migliore epoca su validation; pareggi prima
epoca e LR minore. Tutti completano la stessa esposizione supervisionata, nessun
early-stop dipendente dal held.5epoche sono uno screening a dose fissata, non una
garanzia di convergenza per la rete casuale. Curve loss/validation, norme gradiente
e budget effettivo vengono salvati; una sconfitta da zero ancora in discesa non
dimostra inferiorità del suo ottimo né chiude il confronto a dose maggiore.

F-R2: sui candidati scratch selezionati nei4fold completi, «ancora in miglioramento»
se almeno3fold hanno la prima epoca migliore alla5, oppure mediana del guadagno
validation5−4 ≥0,005 (0,5pp). Se un fold non completa5epoche la regola non è valutabile,
non diventa falsa. Se scatta, seguito dichiarato a15epoche per scratch e finetune,
dagli stessi pesi iniziali originali e seed0, stessa griglia/split/testa, cosine con
orizzonte15epoche; non ripresa dal checkpoint downstream selezionato a5. Preventivo
training circa3× il misurato, con costi fissi/valutazione espliciti. Prima del seguito
servono piano esecutivo, budget misurato e review; nessuna estensione sottomessa oggi.

F-M3: ogni candidato scrive subito report e stato migliore finito; nonfiniti come
esiti `stopped=nonfinite`, con scoreJSON `null` (evita Infinity non standard).
Si continua con l'altro; held solo se esiste un candidato selezionabile. Se entrambi
divergono il fold rimane fallito scientificamente, senza inventare OOF o cancellarlo
dalla matrice. OOM/errori di codice restano errori del job. Un nonfinito sul held
non permette di scegliere l'altro LR.

Preprocessing storico: filtro adattato aNyquist, notches50/60, prime10finestre
senza etichette per scala del singolo soggetto, due ritagli4s EPN mediati prima
della testa. Non si apprende una scala globale su held; la calibrazione non
supervisionata del nuovo soggetto è però una condizione esplicita, non zero-shot
senza calibrazione. LayerNorm per esempio, nessuno scaler fittato su held.

Frozen+head differisce dalla logistica storica (nuova testa/ottimizzazione), quindi
gli82,58% storici sono un riferimento separato, non il controllo sostitutivo. Il
contrasto finetune−frozen isola l'adattamento in questa ricetta; finetune−scratch
confronta inizializzazioni con pari opportunità di tuning ma LR appropriati distinti.
OOF native conservate fuoriGit; aggregazione/IC appaiati sui soggetti, come sopra.
Riportare entrambi i contrasti, distribuzione soggetto, curve e classi. Non estendere
il risultato a regressione o a «FM utile in generale»; la posa era già supervisionata.

Benchmark prima dei fit: solo soggetti train del fold0,128finestre distribuite
nell'ordine train,4update warm-up CUDA su istanza separata e poi fino32update per
regime su istanza nuova, piccoli controlli sullo stesso train,
nessun checkpoint scientifico.1GPU/8CPU/100GiB/30min: **0,5GPUh e4coreh** massimi.
Misura anche caricamento dati e preprocessing dell'intero train fold0; il tempo
di preprocessing viene proporzionato al dataset interno completo con margine1,5.
Registra RSS e memoria CUDA allocata/prenotata: proiezione RSS piena (×rapporto
finestre×1,3) entro85% di100GiB, prenotataCUDA entro85% della memoria dellaGPU.
Proiezione tempo: costo per update, tutte le validation e held addebitate come
training, ×1,5 della media steady dopo warm-up; si conserva anche la media all.
Caricamento/preprocessing misurati con margine, più600s. Ogni job deve stare entro
14100s su allocazione4h. PolicyB-R1 fissata prima della misura, per regime: se due
candidati rientrano si usa un job; altrimenti due job da un candidato se la proiezione
singola rientra. Il secondo legge la ricevuta del primo e poi seleziona/predice held.
Se neanche il singolo rientra: stop e nuovo preventivo. Nessun cambio automatico di
batch, epoche o LR. Nel benchmark OOM/nonfiniti richiedono diagnosi; nei fit solo la
divergenza numerica prevista è un esito del candidato. Pin del report e hash dei sorgenti
obbligatori prima dei fit. Il costo atteso dei fit **non è ancora misurato**.

Tetto fit congiunti:12job ×1GPU/8CPU/100GiB/4h = **48GPUh,384coreh**; nel ripiego
completo24job = **96GPUh,768coreh**, non previsione di consumo. Nessun candidato
aggiunto; combinazione mista possibile tra regimi. Namespace `FT-BENCH-20261104`,
poi `FT-EPN-<regime>-F<fold>-20261104` o suffisso`-C0/-C1`.
Report sotto `wearusfm_runs/results/consolidation_20261104/finetuning/`, output nuovi,
mai sovrascrivere una consegna incompleta. Controller e launcher implementati;
benchmark e fit bloccati dalla review. Fine-tuning49.046 non implicito: prima leggere
questa prova e il nuovo endpoint, poi fissare il confronto aggiuntivo.

## Consegna e gate ancora aperti

Test locali verificano ricetta/LR, parità del forward e dei crop, larghezze384/896,
gradiente del tokenizer, congelamento effettivo, inizializzazione appaiata della
testa, nessuna mutazione del checkpoint, split e assenza di letture held durante fit.
Non sono una misuraCUDA di velocità/memoria o una verifica dei dati remoti.

Da chiudere in ordine: critico locale e test → review Claude di questo diff e del
protocollo → correzioni convergenti → piano esecutivo con SHA, deployment e ledger
→ continuazione e benchmarkFT indipendenti → pin del benchmark e preventivoFT
misurato → fit. I controller della matrice di sonde lunga restano una consegna
tecnica da completare prima delle rispettive estrazioni, con ulteriore review se
cambiano algoritmo o criteri. Nessuna pretesa che l'intera catena sia già pronta
solo perché la continuazione del training lo è.
