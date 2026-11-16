# Risposta GPT alla review contrastiva 0ab9d98

Base della consegna: `db5ac41`, esempi MATLAB successivi `1d87e8d` fuori dal perimetro. Review di Claude letta da `origin/worktree-claude-training-engine`, commit `0ab9d98`. Nessun job nuovo. Il pacchetto corretto resta da rivedere prima del collaudo; non sostituisce l'approvazione di Claude con i test degli agenti.

## Decisioni scientifiche firmate da Simone

| Rilievo | Decisione del 16/11 | Evidenza e conseguenza |
|---|---|---|
| D-1 | A: esenzione per rarità da manifest | Esente dal solo requisito per dataset quando `quota × 32 × 256 < 33`. CapgMyo: quota 0,00285439, attese 23,38 chiavi; DB3: 103,34, non esente. Nessuna finestra tolta dal train o dal denominatore globale. Unica regola in ricette, runtime, audit e lettore. |
| D-3 | Opzione 2: media del tratto attivo | Norma storica media 513–19.618 = 3,514027, primi otto update = 29,790894; fattore 0,1179564123. `λ = 0,1 × fattore × mediana(‖g_base‖/‖g_term‖)`. Ipotesi di norma contrastiva costante: rapporto medio 11,36%, finale 14,54%; non una misura del futuro candidato. |
| D-2 | E_dyn, P1, EPN612 principali; UCI e P2 descrittivi | EPN612 mantiene nome e margini espliciti. Non si cambia la formula del P2 storico. Famiglia di nove contrasti sui tre primari; nessuna conclusione universale di non inferiorità sui task. |
| D-10 | Profili operativi | Beneficio specifico, beneficio comune, tradeoff senza vincitore, danno oltre margine, NI irrisolta, equivalenza pratica: distinti dalle bandiere dei contrasti. Nessun vincitore favorito dall'ordine CN/CR nel codice. |

Il ricalcolo indipendente è in `results/infra/contrastive_comparison_20261116/review_response/`. La copia locale del manifest e quella del cluster differiscono **solo** in `code_commit`: righe, colonne e parametri sono identici; il confronto canonico e il ricalcolo delle quote sono archiviati. Il piano ora pinna il file realmente usato su Leonardo (`6d6d8167…`), senza cambiare numeri o scelta.

La norma storica è quella totale registrata, usata come proxy del cambiamento di scala temporale; la calibrazione nuova misura realmente il gradiente sui parametri condivisi dell'encoder. Non sono due misure dello stesso oggetto. All'inizio, con la testa di posa a zero, il suo gradiente verso l'encoder è **esattamente zero**. Nessuna promessa che il rapporto dei gradienti resti costante dopo il training.

Se anche un solo λ è fuori da `[1e-4; 1]`, si ferma l'intera coorte, senza clamp, selezione estesa o rilascio parziale. I rapporti realmente misurati a peso pieno sono controllati su 513–576, con soglie scritte nel piano esecutivo prima dei dati. Un limite ingegneristico fallito impone lettura e review, non una nuova diagnosi scientifica automatica.

## Correzioni e limiti

| Rilievo | Correzione nella consegna |
|---|---|
| I-1 | Proiettore aggiunto dal trainer soltanto con contrastive; `FMConfig` e `fm.py` tornano alla base, `contrastive=None` omesso dalla serializzazione. Config rigenerati. Test delle campagne chiuse adattati ai sorgenti archiviati, riprendendo la correzione già integrata in `3263ee9`; le guardie di sorgente restano rigorose e i checkout storici immutati. |
| I-2 | Tolleranza del test distribuito riferita alla precisione delle metriche di posa, senza allargare quella dei termini nuovi. Tre confronti del modello completo a 2/4 processi eseguiti su CPU. CUDA/NCCL rimane prova da fare. |
| I-3 | Errore numerico intercettato e coordinato fra rank, anche con peso zero in C0; STOP con causa. Test su rappresentazioni, segnali e target. |
| I-4 | Copertura valutata prima di optimizer, EMA e commit della memoria: nessun update applicato ma assente dal log. Test del ramo reale di arresto e prova CUDA dedicata. |
| I-5 | Rimosso `--contrastive-config` dal CLI generale: l'ingresso della campagna con i suoi gate è obbligatorio. |
| I-6 | Ricevuta ricalcolata da config, metriche, summary, sorgenti, piano e dati pinnati; tutti i bracci derivano da una sola coorte. Anche il futuro preventivo viene autenticato per ricalcolo, non creduto sul campo `passed`. |
| I-7 a/b | `--validate-only` legge tutta la release e gli input; `--recipe-only` dichiara esplicitamente il limite. C0/CN/CR riconosciuti dal lettore del collaudo. |
| I-7 c/d | Piano esecutivo con comandi, risorse e criteri; nuovi strumenti CUDA, confronto continuo/ripreso e lettura. La ripresa 544+32 usa due invocazioni torchrun, una sola allocazione, configurazioni pinnate e checkpoint verificato. |
| I-7 e/f | Cache dell'audit limitate senza cambiare lo stream, budget dei file aperti controllato nel processo; pannello esatto 256–511, SHA di sorgenti/dati/config e ricevuta richiesta dalla calibrazione. |
| I-8 | Età e provenienza dei negativi, quote stesso soggetto/registrazione, chiavi correnti, duplicati; coppie e molteplicità RnC; nomenclatura coerente audit/trainer. Norma del proiettore separata. |
| I-9 | Doppio beneficio diretto produce tradeoff senza vincitore; label numeriche compatibili ma stringhe/numeri incompatibili rifiutate; regressione sulla scala relativa E_dyn; errore Monte Carlo e sospensione della decisione entro tre SE dalla soglia. |
| I-10 | Confine temporale in campioni interi quando fs coincide; norme accumulate in float64; N_JOINTS; JSON rigoroso e passi interi, incrocio delle coperture; percorsi relativi; guardia `requires_grad` nei gather; diagnostiche/soglie nel config. |
| I-11 | Conservata la precisazione: la parità a un processo non è universale rispetto ad altre riduzioni floating point; campagna con batch globale 32 e stesso contratto distribuito. Nessuna pretesa di equivalenza CUDA prima del collaudo. |
| Sezione 6 | Aggiunti oracolo indipendente al punto di chiamata RnC, confine 31/32 negativi e temperatura NCE, test di arresto, autenticazione, coorte, comparatore, cache e comandi. La suite e i suoi limiti effettivi sono nella ricevuta di verifica della revisione. |

Non respingo i rilievi dimostrati. Per le preferenze adottate, il cambiamento è esplicito; per i limiti scientifici la risposta non è un nuovo braccio implicito. D-4: C0 contro storico è compatibilità descrittiva, non varianza fra pretraining. D-5: correlazione di rango fra coseno e distanza di posa tra finestre, con validità e numerosità, oltre alle loss. D-6/7: CN–CR confronta ricette; senza placebo un beneficio di CR non identifica causalmente l'ordinamento della posa. Nessun placebo aggiunto. D-8/14: scorciatoia del soggetto, distribuzione/età dei negativi e scale per dataset restano visibili. D-9: endpoint da conservare con hash in WORK. D-11/12: regola diversa per trial concatenati/espliciti, duplicati nel campione e RnC attiva con una sola finestra sono dichiarati. D-13: proiettore non allenato in C0/CR, peso nella norma di clipping ma escluso dalla calibrazione; la rampa non garantisce uguali passi AdamW.

## Cosa resta realmente da fare

Review di questo diff; poi pubblicazione della release del solo collaudo e checkout congelato. Tetto proposto: **14 GPU-ore e 129,3333 core-ore**, quattro checkout inclusi. L'autorizzazione di Simone è già condizionata alla convergenza: non gli si richiede un secondo via operativo per quel piano.

Nessun tempo CUDA/NCCL del nuovo codice è stato misurato e nessun λ è stato calibrato sul 30M reale. I tre training non sono rilasciati da questa consegna. Dopo collaudo: lettura e preventivo misurato, release e adattatori di valutazione con rimozione stretta delle sole chiavi del proiettore, manifest di valutazione con cardinalità attese e pin (314/32/8/8/19 soggetti di classificazione, 15 utenti di posa), estrazioni e fit con costi separati. Il comparatore resta uno strumento di lettura; non sostituisce la validazione futura dell'intera catena degli input. Gli esempi MATLAB sono invariati.
