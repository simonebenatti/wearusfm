# T100-LQ: proposta esecutiva per review Claude — 04/11/2026

**Stato: implementazione locale, nessun deploy o job.** Simone richiede soluzione,
review Claude e convergenza prima del deploy. Worktree esclusivo
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

Diagnostica separata per dataset/frequenza, inizializzazione e traiettoria:
media/deviazione delle risposte e dei log, frazione x²<epsilon per kernel,
RMS della derivata (limite analitico1/sqrt(epsilon)), mascherato e non mascherato
sullo stesso supporto visibile/QC. Nessun raw o checkpoint entra in Git.
Il collaudo include il controllo storico delle famiglie e del rango.

Epsilon0.001 è una proposta fissa, non un iperparametro da scegliere sulle metriche
di sviluppo. Le frazioni sotto il pavimento sono diagnostiche, senza una soglia
di successo inventata dopo la lettura. Se mostrano una patologia che richiede
un altro epsilon, si documenta un emendamento e si torna alla review prima del
training scientifico. La calibrazione non usa EPN held-out.

Tempi: aggiornamenti DDP separati dalla diagnostica; RAM GPU con due modelli
residenti esplicitamente distinta dalla memoria di un singolo training.
Il confronto del runtime con la base servirà al preventivo finale.

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
| Training candidato | circa17 osservate sul controllo;22 tetto provvisorio | 176 allocate al tetto | Da ricalcolare dal collaudo |
| Estrazioni posa×2,P1,P2 | 2,75 tetto di riferimento | 22 allocate | Da adattare/pinnare |
| Sonde, ridge, benchmark e confronti | 0 | 46,333 tetto di riferimento | Benchmark prima dei fit |

Totale provvisorio26,75GPUh,46,667coreh soloCPU e214coreh allocate nei jobGPU.
Non è un costo già sostenuto né un preventivo definitivo del candidato.
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

112 test CPU passati nella suite combinata in28,81s; sintassi del launcher
verificata con bash -n. Non sono test CUDA né risultati scientifici.
Questa linea possiede tokenizer/log-square, config e preflight; non introduce
estrattori di livelli del backbone, che appartengono alla linea di consolidamento.
Le sole modifiche al trainer condiviso sono CLI esplicita e default di config
per compatibilità; nessun algoritmo di loss/ottimizzazione è cambiato.
Namespace previsto LOGSQ-*, output logsquare_20261104; futura contrastiva
CONTRAST-* con propria data/protocollo. Nessun agente o job avviato in questo fork.
