# T100-LQ: proposta esecutiva per review Claude — 04/11/2026

**Review 1d6f96a recepita:** R1 (epsilon=0,001), R2, controller e disegno dei
valutatori approvati. Adattatori in implementazione per la prossima review;
nessun rilascio del training. Le prescrizioni R1-a/R1-b, N1–N4 e D1–D3 sono
in `docs/logsquare_adapters_execution_20261104.md` e nel readout aggiornato.
R1-b aggiunge solo la lettura descrittiva delle colonne di W_log ai due
endpoint. Il benchmark di inferenza avviene al 5.000; la continuazione non
lo aspetta. La release resta un commit dedicato dopo la review degli adattatori.

**Aggiornamento dopo il collaudo:** job63525730 COMPLETED0:0, audit offline
positivo. R1 segnala sensibilità della derivata; decisione motivata di mantenere
epsilon0,001 in `docs/logsquare_preflight_readout_20261104.md`, ora approvata da
Claude. R2 misurato e approvato: stima16,705GPUh, tetto training22GPUh; rho_infer ancora
da misurare. Simone ha già autorizzato il seguito operativo della linea:
non richiedere un altro consenso per il piano dopo review e controlli.
La review del training e l'adattamento dei valutatori restano da chiudere.
Il testo seguente conserva le regole fissate prima dei risultati.

**Stato dopo review Claudeb0fac8f: correzioni APPROVATE, solo collaudo
preparatorio eseguibile; training NON approvato.** Codice scientifico e launcher
invariati rispetto a d48cad5. Prima del checkout si precisano S1/S3 qui sotto;
S2 resta una lettura manuale dei report già previsti. La convergenza sul collaudo
è raggiunta; autorizzazioni operative di Simone già presenti nella conversazione. Worktree esclusivo
`wearusfm-logcontrast`, ramo `codex/tokenizer-logcontrast-20261104`.
Integrato senza riscrittura il commit `8ee6603`: controllo appaiato storico
T25-L−B e T100-G secondario. Nessuna interferenza con scaling/schedule.

## Domanda e intervento

Le risposte log-quadratiche affiancate alle risposte firmate migliorano la lettura
della rappresentazione a dose fissata? Base T100-L, finestra100ms, passo25ms,
overlap75ms e anti-aliasing esistente. Non energie di banda, non dB.

Con x = risposta del banco dopo il gain storico:

`token = W_linear x + bias + W_log log(0.001 + x²)`.

W_log ha64×384=24.576 pesi, inizialmente tutti zero. Il log e il quadrato sono
FP32 anche sotto autocast; le proiezioni seguono AMP. Due proiezioni separate
sono matematicamente equivalenti alla concatenazione128→384 e mantengono
intatta la chiamata lineare storica e l'inizializzazione di tutti gli altri pesi.
Non viene consumato RNG per W_log. Nessun nuovo bias, normalizzazione per kernel,
somma prima del log o GELU insieme al ramo. Banco e proiezioni sono apprendibili;
la trasformazione log e epsilon sono fissi. Il default disabilitato conserva
chiavi e forme dei checkpoint storici.

Si parte dall'inizializzazione NON addestrata del controllo, non dal suo checkpoint.
La funzione iniziale e i pesi condivisi devono coincidere. Il gradiente di W_log
è già non nullo al primo passo: può cambiare la norma globale e il fattore di
clipping, quindi non si promette uguaglianza del primo aggiornamento condiviso.
Non cambiare il clipping per nascondere questa conseguenza dell'intervento.

I campioni nascosti vengono azzerati prima del banco. Solo un supporto del filtro
interamente nullo produce log(epsilon); un token vicino a una maschera può usare
campioni visibili. Non chiamare questo leakage del contenuto nascosto: i test
perturbano proprio quel contenuto, anche con W_log attivo, e richiedono invarianza.

## Collaudo preparatorio implementato

`scripts/logsquare_preflight.py` riusa aggiornamento reale, optimizer, clipping,
EMA, supporti e stream del preflight storico. 64 batch globali da32, quattro rank,
sei worker/rank, stessi batch consegnati ai due bracci. Non salva checkpoint
scientifici. Piano e config in `results/infra/logsquare_review_20261104/`.

Gate tecnici: hash iniziale dei pesi condivisi uguale al T100-L storico;
RNG e funzione iniziale uguali; parità di input/query/bersagli e RNG dopo ogni
update; nessun dato mutato; parametri/gradienti/update finiti; W_log aggiornato;
input e codice immutati; presenza di dati train nativi200Hz e2000Hz.
Qualunque gate fallito ferma la fase, senza retry cieco.
Prima del confronto fra bracci si eseguono due forward del controllo con RNG
ripristinato. Se non sono identici, il report indica forward del controllo non
deterministico, senza attribuire il difetto al ramo. Si conservano esito e massimo
scarto assoluto per local/z anche quando il gate fallisce; non si introduce una
tolleranza scelta dopo aver visto lo scarto.

Per ogni rank e passo si pubblicano fattori di clipping, rapporto LQ/L,
quota della norma al quadrato dovuta a W_log e frazione dei passi clippati.
La ricostruzione segue PyTorch: min(1, grad_clip/(norma+1e-6)).
Il residuo relativo di g_LQ²−g_L²−g_log² al primo passo è descrittivo, non un
nuovo gate esatto CUDA. I rank DDP sono repliche, non osservazioni indipendenti.
Questi fattori scalano i gradienti prima di AdamW: i momenti adattivi e il
weight decay impediscono di interpretarli direttamente come rapporti degli
aggiornamenti dei pesi. Non cambiare il clipping per compensarli.
Un JSONL per rank conserva ogni coppia di update completata anche se il job
si interrompe prima del report finale.

Diagnostica separata per dataset/frequenza, inizializzazione e traiettoria:
media/deviazione delle risposte e dei log, frazione x²<epsilon per kernel,
RMS della derivata (limite analitico1/sqrt(epsilon)), mascherato e non mascherato
sullo stesso supporto visibile/QC. Nessun raw o checkpoint entra in Git.
Il collaudo include il controllo storico delle famiglie e del rango.

### R1: regola prospettica per epsilon, prima del collaudo

Epsilon resta0.001 salvo emendamento motivato e revisionato. Non è scelto sulle
metriche di sviluppo e la calibrazione non usa EPN held-out. Le soglie sotto
sono **segnalazioni operative**, non criteri di qualità della rappresentazione
né prove di instabilità. Si applicano a ciascun kernel e gruppo
dataset/fs/fase, pubblicando sempre anche il numero di token; inizializzazione e
traiettoria dei64passi restano separate. Nessuna selezione dei soli kernel favorevoli.
I trigger si applicano a LQ; le stesse statistiche di L sono pubblicate come controllo
contestuale (il suo ramo log non è attivo).

| Statistica già misurata | Trigger prospettico e direzione della revisione |
| --- | --- |
| p = frazione x²<epsilon nel modo non mascherato | p≥0,50: il pavimento domina almeno metà delle risposte; aprire una revisione verso epsilon minore, leggendo anche attenuazione del banco e contrasto fra frequenze descritto sotto, senza esenzioni automatiche |
| d = RMS[2x/(epsilon+x²)], in entrambi i modi | d·sqrt(epsilon)≥0,50: sensibilità media elevata rispetto al massimo analitico; aprire una revisione verso epsilon maggiore, leggendo anche gradienti e clipping; non basta da sola a dimostrare instabilità |
| Differenze mascherato−non mascherato sullo stesso supporto | Pubblicare delta p, delta d e delta media dei log. Se p attraversa0,50 in qualunque direzione oppure il valore assoluto di delta media log è almeno una std log non mascherata, aprire revisione dell'effetto del mascheramento prima di scegliere la direzione di epsilon |

Per l'ultimo trigger, std non mascherata zero e delta media zero non segnala
nulla; std zero e delta non zero segnala. Lo scarto è fra medie appaiate sul
supporto, non RMSE punto per punto. I cambiamenti ai bordi delle maschere possono
essere attesi e non dimostrano accesso ai campioni nascosti. La prima soglia
ha un'interpretazione semplice: la sensibilità relativa del log alla potenza,
x²/(epsilon+x²), è inferiore a metà per almeno metà delle risposte.
La soglia della derivata e quella di una std sono convenzioni operative dichiarate.

Qualunque trigger sospende il passaggio al training fino a una decisione scritta:
motivare il mantenimento di0.001 oppure proporre l'emendamento e farlo rivedere
prima di un nuovo collaudo. Segnali in direzioni opposte richiedono diagnosi,
non una ricerca automatica dell'epsilon migliore. Nonfinitezza o violazione del
limite analitico richiedono invece diagnosi tecnica, non aumento automatico di epsilon.
In assenza di trigger o di emendamento,0.001 rimane invariato; comunque il training
richiede la sua review, i pin e il preventivo misurato.

**S1, precisazione prospettica prima del collaudo.** Per lo stesso indice di
kernel, fase e modo, p≥0,50 a200Hz e p<0,50 a2000Hz è un contrasto compatibile
con attenuazione dipendente dalla frequenza di campionamento. Riportare tutte le
coppie di gruppi disponibili, con dataset e numero di token; non scegliere
soltanto il gruppo2000Hz favorevole. Il contrasto non certifica inattività:
le24uscite Fourier mescolano104armoniche nella finestra100ms, le altre24spline
e16MLP sono kernel generici; l'anti-aliasing rimuove componenti spettrali, non
necessariamente intere uscite. I gruppi200/2000Hz appartengono inoltre a dataset
diversi, con spettri e scale differenti.

**Nessun kernel è esentato dai trigger R1 in questo collaudo.** L'annotazione
non impone di modificare epsilon: la decisione documentata può motivare il
mantenimento di0.001. Una futura esenzione richiederebbe certificazione
indipendente dai dati del kernel filtrato (per esempio operatore identicamente
nullo), che i metadata attuali non forniscono. Così la clausola non apre
esclusioni scelte dopo aver letto i dati. Nessuna nuova misura o soglia del banco.

Tempi: aggiornamenti DDP separati dalla diagnostica; RAM GPU con due modelli
residenti esplicitamente distinta dalla memoria di un singolo training.
Il confronto del runtime con la base servirà al preventivo finale.

### R2: regola prospettica dei preventivi

Definire rho_update = update_mean_after8_s(LQ)/update_mean_after8_s(L),
sui56update dopo gli8iniziali, stessa procedura di aggregazione fra rank.
Il report pubblica il rapporto. Prima del training pubblicare le misure,
il preventivo calcolato e il tetto in un commit verificato su origin.

Per ogni fase (5000 e continuazione14618), la stima di durata è la durata
storica T100-L di quella fase, identificata da ricevuta/accounting, moltiplicata
per rho_update. Il tetto proposto è il walltime storico prenotato della fase
(5400s e14400s) moltiplicato per max(1,rho_update), arrotondato per eccesso a900s.
GPU-ore = quattro×walltime/3600; core-ore allocate =32×walltime/3600.
La misura è un'estrapolazione train-only breve; il margine storico resta esplicito.
S3: rho_update esclude loader e diagnostica, mentre la durata storica comprende
loader, I/O e controlli del training (non la nuova diagnostica del collaudo).
Applicare rho>1 all'intera durata include un margine conservativo su quelle
componenti. Con rho<1 la stima puntuale può sovrastimare il risparmio: il tetto
usa comunque max(1,rho), senza scontare loader/I/O.
Se un walltime supera il limite di partizione o l'allocazione proposta, fermare
la preparazione della fase e revisionare il piano, senza frazionamenti o retry impliciti.
Le17GPUh osservate e le22prenotate sono riferimenti del controllo, non il
preventivo finale del candidato.

Per le estrazioni occorre rho_infer separato, sul percorso effettivo di inferenza
dopo warmup e con sincronizzazione CUDA, stessa forma di batch/AMP/supporto.
Il rapporto degli update include backward/AdamW/EMA e non lo sostituisce.
Questo collaudo non misura rho_infer: la misura e il suo piccolo preventivo
andranno aggiunti alla consegna degli estrattori prima del loro lancio.
Per ciascuna estrazione applicare poi il rapporto alla durata storica omologa
e max(1,rho_infer) al suo walltime storico, arrotondando per eccesso a300s.
Fino alla misura,2,75GPUh resta solo riferimento storico e nessuna estrazione
scientifica è sottomettibile. Eventuali cap oltre i limiti richiedono revisione.

## Training e lettura proposti dopo il collaudo

Un solo nuovo training30M/seme0, stessa ricetta storica: ancore0.2, posa1,
JEPA0, batch32, quattroGPU, warmup1962, schedule WSD totale49046.
Checkpoint5000 e19618, continuazione completa di optimizer/EMA/RNG;
non cambiare dose o LR per migliorare il candidato. Il collaudo non è una ripresa.

Teacher congelato: posa a5000 e19618; P1/P2 a19618.
Ridge istantanea e logistica convergente con ricette, split e supporti di
`docs/tokenizer_evaluation_execution_20261103.md`, senza nuove sonde o layer.
Gli attuali orchestratori congelano tre nomi di braccio: il loro adattamento
al nuovo candidato e i pin effettivi saranno una consegna successiva alla review,
prima di qualsiasi sottomissione scientifica. Non aggirarne le whitelist.

Primario: LQ−T100-L, separatamente E_dyn della posa ed EPN612.
Secondari dichiarati: LQ−T25-L e LQ−T100-G; migliore lineare distinto per endpoint.
Non selezionare retroattivamente un unico controllo o un vincitore posa/classi.
Riusare cache/OOF storiche solo dopo parità verificata di checkpoint/dose,
configurazione condivisa, ricetta, righe, soggetti, fold e target.

Riportare stime e IC95% appaiati, stessi2000 draw utenti/seed20261102.
Per posa mantenere significato della soglia pratica2% e misure centrate,
correlazione, rapporto varianze, MAE e per-utente. Non introdurre una soglia EPN
ad hoc. Su P1/P2 mantenere margine NI1pp: limite inferiore>−1pp sostiene NI,
superiore<−1pp sostiene perdita oltre margine, altrimenti irrisolto.
Un vantaggio EPN non dimostra un effetto causale della sola frequenza200Hz.
Fit non convergenti o alpha irrisolto rendono incompleta la lettura pertinente.

Tutti i contrasti vanno pubblicati. Un seme, sviluppo riutilizzato, confronti
multipli e T100-G aggiunto dopo i suoi risultati: evidenza esplorativa.
T25−B è un controllo empirico di ripetibilità, non una stima del rumore fra
pretraining indipendenti. Nessun risultato nullo esclude in generale energia,
non linearità o approcci LEAF.

## Risorse proposte, senza prenotazioni

| Fase | GPU-ore | Core-ore | Stato |
| --- | ---: | ---: | --- |
| Checkout2CPU/10min | 0 | 0,333 | Dopo review |
| Collaudo4GPU/32CPU/30min,240GiB | 2 max | 16 allocate | Launcher pronto |
| Training candidato | circa17 osservate sul controllo;22 prenotate storiche | 176 storiche al tetto | Nuovo tetto da rho_update |
| Estrazioni posa×2,P1,P2 | 2,75 riferimento storico | 22 storiche allocate | Nuovo tetto da rho_infer, ancora da misurare |
| Sonde, ridge, benchmark e confronti | 0 | 46,333 tetto di riferimento | Benchmark prima dei fit |

Somma dei riferimenti26,75GPUh,46,667coreh soloCPU e214coreh allocate nei jobGPU.
Non è un costo già sostenuto né un tetto approvato del candidato.
Dettagli macchina in `resource_proposal.json`; nessuna estensione automatica.
Solo submitter e ledger comuni, ID prefissati LOGSQ e output propri.
Nessuna automazione di monitoraggio attivata.

## Contrastiva: seconda domanda, dopo LQ

La proposta successiva sarà un confronto sulla stessa base con/senza un solo
termine contrastivo, mantenendo gli altri termini. Non sommare ora frontend nuovo
e loss nuova: il risultato non distinguerebbe i loro contributi.
Non esiste ancora una ricetta contrastiva approvata o implementata in questa consegna.

Proposta da discutere: due viste della stessa finestra come positivi, testa
di proiezione dedicata e InfoNCE affiancata alla ricetta di base.
Prima di renderla protocollo vanno fissati trasformazioni che preservino posa
e dinamica (no invarianti di ampiezza/tempo assunte), gestione dei negativi
vicini della stessa registrazione e falsi negativi, livello del modello letto,
temperatura, peso, batch distribuito e scala dei gradienti.
La scelta della base dopo LQ va dichiarata come adattiva; il controllo senza
contrastiva deve condividere inizializzazione e tutte le altre modifiche.
Nessuna selezione del peso sulla validazione finale e nessuna griglia di training
implicita. Il prompt chiede a Claude di valutare questa sequenza, non di avviare
la contrastiva insieme al ramo logaritmico.

## Stato della consegna e ownership

Consegna iniziale:112 test CPU passati. Correzioni:129 test nella suite combinata
in25,03s; sintassi del launcher
verificata con bash -n. Non sono test CUDA né risultati scientifici.
Questa linea possiede tokenizer/log-square, config e preflight; non introduce
estrattori di livelli del backbone, che appartengono alla linea di consolidamento.
Le sole modifiche al trainer condiviso sono CLI esplicita e default di config
per compatibilità; nessun algoritmo di loss/ottimizzazione è cambiato.
Namespace previsto LOGSQ-*, output logsquare_20261104; futura contrastiva
CONTRAST-* con propria data/protocollo. Per le correzioni sono stati usati due
agenti interni: controllo critico del protocollo e test aggiuntivi, senza accesso
operativo al cluster. Nessun job o deploy avviato in questo fork.
