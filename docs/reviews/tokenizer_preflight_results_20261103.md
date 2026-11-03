# Collaudo tokenizer: lettura dei risultati — 03/11/2026

**Il collaudo tecnico consente di avviare i tre training con il protocollo già fissato.** Non emerge un difetto di parità, caricamento, stabilità numerica o propagazione del gradiente che richieda di cambiare gain, supporto o funzione di attivazione. La GELU agisce già in modo marcatamente nonlineare sui dati reali. Questa è una verifica della fattibilità del confronto; non prova che T100-G migliori posa o classificazione, né seleziona un braccio.

Fonti: [report del job 63370721](../../results/infra/tokenizer_execution_20261103/tokenizer_preflight_20261103/report.json), quattro `rank*.json.gz` e quattro `rank*.jsonl.gz` nello stesso archivio, [manifest degli archivi](../../results/infra/tokenizer_execution_20261103/tokenizer_preflight_20261103/archive_manifest.json), [piano](../../results/infra/tokenizer_execution_20261103/preflight_plan.json), [configurazioni](../../results/infra/tokenizer_review_20261103/comparison_configurations.json), [protocollo operativo](../tokenizer_execution_20261103.md). Lettura locale, senza nuovi job e senza modifiche al codice.

## Identità, parità e aggiornamenti

- Il report è completo, codice `ef8016a520f12a71eb9890202e836f56622d000b`, piano SHA256 `13ed48b6c88acc5fd6a9bc13e02d79287fc7d3225ec54b040bb8d5442dfa90e4`. Lo SHA256 del report è `ce0c631de18e26b0b4b6a78b3d0a1de1cd70bafd07782eed4f6b4c98b003a986`.
- Ricalcolati gli SHA256 e i byte dei quattro report rank contro il report principale; verificati anche tutti gli otto archivi gzip, prima e dopo decompressione, contro il manifest. Piano, confronto e tre configurazioni locali coincidono con i pin del run. I file scientifici remoti non sono stati riaperti in questa review: il loro controllo prima/dopo e quello degli stat dei file raw sono attestati dal runtime congelato.
- Quattro rank completi × 64 batch; 32 finestre globali per passo, **2.048 finestre presentate a ciascun braccio**. I 256 record consegnati coincidono con i rispettivi JSONL, comprese tutte le impronte di batch, segnale processato, maschere, target delle ancore, target di posa e origine.
- Nei **256 confronti fra bracci allo stesso passo/rank**, supporto delle query, target effettivi, supporto della posa, conteggi di loss, segnale mascherato e consumo RNG coincidono esattamente. Nei **768 aggiornamenti braccio × passo × rank**, il supporto usato dal decoder coincide con quello precalcolato.
- Nessun NaN/Inf nei valori numerici dei report e dei momenti. Tutte le sei famiglie diagnosticate — Fourier, spline, MLP del banco continuo, proiezione, testa di posa e primo parametro del backbone — hanno **gradiente e aggiornamento strettamente positivi in tutti i 768 aggiornamenti**: 4.608 riscontri positivi per ciascuna delle due quantità. Il runtime controlla inoltre la finitezza di tutti i parametri aggiornati. Le norme di gradiente e aggiornamento coincidono esattamente fra rank allo stesso passo/braccio.
- Ricostruita la loss totale di ogni record: `posa + 0,2 × somma delle cinque ancore`. Coincide entro l'arrotondamento FP64; **JEPA rimane a peso zero**. I valori diagnostici denominati `jepa` nel log non significano che quel termine contribuisca all'ottimizzazione. Il raggio esplicito delle query è 0.
- Tutti i bracci partono con identica impronta dei moduli comuni. T100-L e T100-G hanno anche identica impronta di tutti i pesi e del gain, su tutti i rank; i momenti delle feature prima della GELU al passo iniziale coincidono esattamente. Le differenze successive non sono una diversa inizializzazione.

| Braccio | Parametri | Gain iniziale |
|---|---:|---:|
| T25-L | 28.356.829 | 982,425067 |
| T100-L | 28.362.013 | 548,693919 |
| T100-G | 28.362.013 | 548,693919 |

I 5.184 parametri in più dei due bracci a 100 ms appartengono al frontend con supporto più lungo; i moduli comuni sono identici. La GELU non aggiunge parametri. Il gain deriva dalla stessa inizializzazione sintetica prevista dal modello, non da una calibrazione aggiunta sui dati di questa valutazione.

Il supporto, contato una sola volta per braccio, comprende **4.051.625 query** e **1.086.800 elementi di posa validi**. Sono state presentate 442 finestre emg2pose, delle quali 412 con almeno un bersaglio di posa valido; questa differenza di 30 finestre è contabilizzata ed è identica nei tre bracci. Ogni batch globale ha da 2 a 15 finestre con posa valida. Le ancore hanno 4.003.829 elementi log-RMS, 3.512.688 log-inviluppo, 19.027.705 bande, 23.591.223 multiscala veloce e 86.288.360 multiscala lenta: questi sono elementi validi, non denominatori indipendenti della loss ridotta per finestra.

Le norme globali prima del clipping vanno da 7,79 a 31,51 per T25-L, da 7,80 a 31,59 per T100-L e da 8,00 a 32,31 per T100-G. Con soglia 1 il clipping interviene sempre, secondo la ricetta comune; questo collaudo non indica un'esplosione specifica della GELU. Gli aggiornamenti effettivi restano finiti e non nulli. Il learning rate sale da `5,09684e-8` a `3,26198e-6`: i 64 passi coprono solo il 3,26% dei 1.962 passi di warmup.

## Che cosa sta facendo la GELU

Le quantità seguenti riguardano **T100-G**. `x` è la risposta del banco dopo `fe_gain`, prima della proiezione; l'uscita è `GELU(x)`. La deviazione standard riportata è la mediana delle deviazioni dei singoli filtri della famiglia: 24 Fourier, 24 spline, 16 MLP. Non è una deviazione standard normalizzata a uno per dataset.

`M` indica il segnale con campioni nascosti azzerati prima del tokenizer, valutato sulle sole posizioni visibili e valide al QC. `U` indica il segnale non mascherato valutato **sulle stesse posizioni**. Le due condizioni hanno dunque lo stesso numero di osservazioni. Il confronto misura quanto la maschera influisce sul contesto delle posizioni rimaste visibili, senza aggiungere in U i token che M esclude.

Il residuo `r = RMS(GELU(x) − 0,5x) / RMS(x)` misura lo scostamento dall'approssimazione lineare locale `0,5x`. È descrittivo, non una nuova soglia: vicino a zero indica il regime quasi lineare; vicino al limite 0,5 indica forte rettificazione. Non è un errore predittivo e non dimostra utilità per un task.

Per emg2pose, le 525.870 osservazioni canale × patch di ciascuna condizione, aggregate prima degli aggiornamenti 1–64, danno:

| Famiglia | σ(x) M | σ(x) U | σ(GELU) M | σ(GELU) U | r M | r U |
|---|---:|---:|---:|---:|---:|---:|
| Fourier | 8.789 | 8.938 | 6.055 | 6.109 | 0.499 | 0.499 |
| Spline | 2.104 | 2.149 | 1.493 | 1.503 | 0.491 | 0.491 |
| MLP | 0.996 | 0.976 | 0.651 | 0.633 | 0.480 | 0.480 |

**La GELU non è praticamente lineare sui segnali della posa.** Il ramo Fourier opera soprattutto in un regime di rettificazione; anche spline e MLP hanno un residuo elevato. Lo stesso comportamento è già presente prima del primo aggiornamento, su 8.804 osservazioni emg2pose: σ(x) M = 8,327 / 2,157 / 0,991 e r M = 0,499 / 0,490 / 0,480. Non è un effetto comparso soltanto dopo i 64 passi.

La maschera modifica moderatamente queste scale sulle posizioni appaiate; non le annulla. Il rango descrittivo delle feature **prima** della GELU, su emg2pose nella traiettoria, è 46/60 in M e 45/60 in U alle soglie relative `1e-2`/`1e-3`. Non è una misura del rango dopo la GELU né una prova che vi sia movimento decodificabile.

## Frequenza nativa e famiglie

Questa tabella aggrega i **momenti grezzi**, non le deviazioni standard dei dataset, di tutte le osservazioni della traiettoria con la stessa frequenza nativa. Pesa quindi per osservazioni canale × patch; i dataset con più canali possono pesare di più. È un riepilogo descrittivo per frequenza, **non** una prova di invarianza al ricampionamento dello stesso segnale. Frequenza, utenti, gesti e dataset sono confusi fra loro.

| Fs Hz | Finestre | Osservazioni per condizione |
|---:|---:|---:|
| 200 | 42 | 52.521 |
| 1000 | 63 | 55.725 |
| 1926 | 236 | 213.243 |
| 2000 | 1.364 | 2.159.847 |
| 2048 | 272 | 1.269.131 |
| 5120 | 71 | 133.616 |

| Fs Hz | Famiglia | σ(x) M | σ(x) U | σ(GELU) M | σ(GELU) U | r M | r U |
|---:|---|---:|---:|---:|---:|---:|---:|
| 200 | Fourier | 11.838 | 12.137 | 7.840 | 8.033 | 0.500 | 0.500 |
| 200 | Spline | 3.232 | 3.323 | 2.210 | 2.271 | 0.497 | 0.497 |
| 200 | MLP | 1.653 | 1.629 | 1.115 | 1.099 | 0.491 | 0.491 |
| 1000 | Fourier | 8.752 | 8.890 | 5.972 | 6.051 | 0.499 | 0.499 |
| 1000 | Spline | 2.024 | 2.057 | 1.372 | 1.395 | 0.490 | 0.490 |
| 1000 | MLP | 0.925 | 0.901 | 0.619 | 0.594 | 0.473 | 0.473 |
| 1926 | Fourier | 27.688 | 28.010 | 19.418 | 20.045 | 0.500 | 0.500 |
| 1926 | Spline | 7.332 | 7.314 | 4.737 | 4.805 | 0.499 | 0.499 |
| 1926 | MLP | 3.605 | 3.485 | 2.710 | 2.600 | 0.499 | 0.499 |
| 2000 | Fourier | 7.812 | 7.909 | 5.339 | 5.433 | 0.499 | 0.499 |
| 2000 | Spline | 1.824 | 1.867 | 1.243 | 1.272 | 0.487 | 0.487 |
| 2000 | MLP | 0.836 | 0.805 | 0.562 | 0.537 | 0.471 | 0.470 |
| 2048 | Fourier | 8.666 | 8.805 | 5.965 | 6.088 | 0.499 | 0.499 |
| 2048 | Spline | 1.954 | 1.965 | 1.343 | 1.354 | 0.490 | 0.490 |
| 2048 | MLP | 0.863 | 0.837 | 0.566 | 0.568 | 0.476 | 0.475 |
| 5120 | Fourier | 3.303 | 3.339 | 2.053 | 2.087 | 0.495 | 0.495 |
| 5120 | Spline | 0.767 | 0.782 | 0.489 | 0.495 | 0.433 | 0.434 |
| 5120 | MLP | 0.353 | 0.338 | 0.216 | 0.208 | 0.376 | 0.379 |

Nessuna frequenza osservata mostra un banco globalmente spento. Le scale sono però eterogenee: a 1.926 Hz, corrispondente qui a NinaPro DB10, il Fourier ha σ circa 28, ben sopra la scala di altri dataset; la tabella non attribuisce questa differenza alla frequenza né all'apprendimento. Guardando separatamente tutti i 17 dataset e le due condizioni, r varia da 0,491 a 0,500 per Fourier, da 0,360 a 0,499 per spline e da 0,242 a 0,499 per MLP. Esistono quindi famiglie meno rettificanti, ma non un'evidenza che l'intero tokenizer sia rimasto nell'approssimazione `0,5x`.

Non cambierei il gain in risposta a questi numeri: introdurrebbe un'altra variabile nel confronto L/G già definito. Gli aggiornamenti reali sono stabili; la domanda successiva è se questa trasformazione migliori le rappresentazioni ai due endpoint, non se sia abbastanza nonlineare secondo una soglia scelta dopo aver visto i dati.

I 54 gruppi di momenti per braccio sono stati ricalcolati dai quattro rank. Medie, deviazioni, residui, conteggi e ranghi coincidono esattamente col report. Nella ricostruzione locale dei soli valori singolari dalla covarianza, le code prossime a zero differiscono al massimo di `2,33e-7` per T25-L e `7,28e-8` per T100-L/G fra librerie numeriche diverse; nessun rango riportato cambia. Non sono differenze fra gli esperimenti né nuovi criteri di accettazione.

## Tempo, memoria e limiti della previsione

Dopo i primi otto passi, per 56 aggiornamenti misurati:

| Braccio | Media s/passo | Mediana s/passo | P90 s/passo | Picco allocato GPU, tre bracci residenti, GiB |
|---|---:|---:|---:|---:|
| T25-L | 0.695 | 0.698 | 0.777 | 10.20 |
| T100-L | 0.700 | 0.700 | 0.783 | 12.94 |
| T100-G | 0.706 | 0.715 | 0.783 | 12.95 |

T100-L costa circa +0,64% e T100-G circa +1,53% nella media dell'aggiornamento rispetto a T25-L in questo campione. Il tempo usa, per ogni componente, il massimo sui rank di forward/backward e di clipping/optimizer/EMA, con le barriere fuori dai timer; la ricostruzione dai rank coincide esattamente. L'intero tratto del collaudo registrato dura al massimo 174,11 secondi per rank. Questo valore non è il walltime fatturato da SLURM e non sostituisce il consuntivo del ledger.

Il massimo riservato dall'allocatore GPU è 19,33 GiB. **Questi picchi comprendono tre student, tre teacher e tre ottimizzatori residenti**: non sono i picchi di un training isolato. Il massimo RSS del processo principale è circa 1,67 GiB per rank e non comprende i worker del loader. I 64 batch non esauriscono le forme e le registrazioni possibili del corpus.

La diagnostica offre inoltre tempo aggiuntivo ai worker: l'attesa dei dati osservata qui non misura il collo di bottiglia del loader a regime. Per preventivare i run lunghi è corretto usare i rapporti fra bracci insieme al riferimento storico completo, con margine per I/O, checkpoint e overhead; non moltiplicare soltanto 0,70 secondi per il numero di passi e chiamarlo costo garantito.

## Decisione scientifica e prossima lettura

Procedere con **tutti e tre i training nuovi**, raggio 0 con JEPA 0, identità L/G e gain invariati, prima a 5.000 poi fino a 19.618 passi secondo il piano autorizzato. Il collaudo non richiede una revisione del protocollo. Non scegliere né eliminare un braccio sulla loss di questi 64 batch: cambiano i dati da un passo all'altro, siamo quasi interamente all'inizio del warmup e non esiste qui un punteggio fuori campione.

La prima risposta scientifica verrà dalla posa teacher e dalla ridge sui riferimenti appaiati a 5.000; la lettura comune a 19.618 comprende anche P1/P2 ed EPN-612. Restano da verificare capacità di inseguire il movimento, trasferimento e vantaggio effettivo della finestra/GELU. Il collaudo ha dimostrato che il confronto è tecnicamente realizzabile e che il braccio nonlineare sta davvero applicando una trasformazione diversa, non che vincerà.
