# R3 dbg: diagnosi del cancello fallito e proposta di verifica

27/10/2026, GPT. Richiesta: `docs/prompt_gpt_collaudo_r3_20261027.md`, commit Claude `921cb97`.
Snapshot letto: Claude `1bfef0252498816cf36da9e56dc49da7b156a49b`; codice del run: `c5e42cb99b4dc0a105495de70946a411b0523dbc`.
Job analizzato: 62429050. Fonti: i file versionati in `results/infra/pilot_posa/collaudo_r3_dbg_62429050/`, protocolli e codice.
Nessun accesso al cluster, nuovo training o job; nessuna modifica al codice di Claude o ai checkout di campagna.
Revisione indipendente ripartita fra catena dei target, ottimizzazione/DDP e critico scientifico.

**Raccomandazione:** non aumentare il peso e non abbandonare R3 sulla base di questo collaudo. Prima controlli diretti dei target e una prova
costruita dell'ottimizzazione; poi E3 di R3c come diagnosi, ed eventualmente un solo collaudo da 5.000 passi con schedule vero e pannello
comparabile. E3 positivo non e' una condizione necessaria per autorizzare il collaudo lungo. Il vecchio dbg resta FALLITO e R3 completo non parte.
Tutto il programma sperimentale sotto e' una **proposta da firmare**, non un'autorizzazione o un cambiamento retroattivo dei criteri.

## 1. Diagnosi: correggere prima la lettura dei dati

| Voce | Riscontro indipendente | Conseguenza |
|---|---|---|
| Finestre di posa | **9.332 totali, 7,178 per aggiornamento globale**, su batch globale 32; range 0–19, un passo senza posa | Il prompt moltiplica erroneamente per quattro: non 37.300 e non 28 per passo |
| Patch / griglia | **25 ms** / 200 ms | Cinque blocchi di posa da 5 ms per patch; 200 ms e' il passo degli inizi finestra, non la risoluzione del target |
| LR | Picco 1e-4 nei passi **1.000–1.041: 42 aggiornamenti**; somma LR 0,0671 | Dose pari a 671 passi al picco, solo come integrale del LR: non equivalenza degli aggiornamenti Adam |
| Cancello firmato | 0,990860 nei passi 500–599; 1,014004 nei passi 1.201–1.300 | +2,336%; confermato il fallimento, nessuna rilettura lo trasforma in passaggio |
| Pannello fisso, 0 → 1.000 | MSE 1,074761 → 1,048031: **−2,487%**; MAE rad 0,350058 → 0,349605: **−0,129%** | Poco miglioramento misurato, non prova di apprendimento esattamente nullo |
| Copertura del pannello | 256 finestre totali, 41 emg2pose (39 coordinate uniche), 6 utenti/6 recording; solo 32 contribuiscono alla posa | Campione piccolo e correlato; non 32 osservazioni indipendenti e nessuna misura fissa al passo 1.300 |

Il conteggio globale si verifica nel codice: `training/jepa.py:535` mette `pose_windows` in `log_sum`, e `training/run.py:754–770` lo somma
fra microbatch e rank tramite all_reduce. Il massimo osservato 19, maggiore del batch locale 8, e' un controllo ulteriore. E' un errore di
interpretazione del log, **non** una perdita divisa per quattro. Le 9.332 sono esposizioni a finestre, non finestre necessariamente uniche.

Le due finestre del cancello contengono 757 e 758 esposizioni di posa: conteggi simili, ma contenuto diverso. Le deviazioni delle loss sono
0,2020 e 0,2318; errori standard sotto ipotesi iid 0,0202 e 0,0232. Non sono intervalli di generalizzazione e non annullano il cancello firmato.
La riaggregazione descrittiva pesata per numero di finestre da' 0,991639 → 1,024256: non rivela una discesa nascosta dal peso per passo.
Le correlazioni loss–LR, loss–passo, loss–conteggio sui soli passi con posa sono rispettivamente −0,0159, −0,0122, −0,0262: non discriminano
le cause. LR, rappresentazione e composizione dei batch cambiano insieme, quindi non si tratta di confronti causali.

Al passo 1.000 migliorano dieci giunti e peggiorano dieci; a 500 la MAE in radianti era scesa del 2,144%, poi risale. Non c'e' evidenza di
un miglioramento robusto e uniforme. Nel pannello nominale ci sono 30 coppie di finestre emg2pose sovrapposte, 36 finestre destre e 5 sinistre;
dal solo elenco non si identificano le nove senza contributo valido. Non inferire significativita' da questa numerosita'.

**Graduatoria delle spiegazioni:** H1 e' plausibile e il prompt ne sottostimava il peso; H2 resta possibile ma riguarda l'accessibilita' e
l'apprendimento congiunto, non una impossibilita' dimostrata; H3 sui file reali resta da verificare; H4 non trova supporto nel codice e nel
controesempio numerico. Le ipotesi possono coesistere. Non e' possibile identificare una causa unica da questi log.

## 2. Catena dei bersagli e ottimizzatore

Riferimenti relativi allo snapshot Claude; i file di target/loader/trainer/testa rilevanti sono identici al codice eseguito. Le aggiunte
successive nell'ingest riguardano profili e diagnostica, non la regola dei target a parametri fissati.

| Passaggio | Righe | Esito |
|---|---|---|
| Raw EMG e posa | `ingest/emg2pose.py:89–102,139–155`; `ingest/emg2pose_pose.py:61–78,130–139` | Stessa origine temporale; nessun taglio EMG che sposti gli indici; posa aggregata in blocchi [10j,10j+10) |
| Filtro e margini | `data/pretraining_loader.py:205–227,385–395` | Filtro avanti/indietro; il crop torna a [win.start,win.stop), non conserva il margine aggiunto |
| Blocchi → patch | `data/pose_targets.py:130–166` | Confini del front-end + win.start; blocchi assegnati tramite centro; media dei validi e successiva standardizzazione |
| Lato e unprompted | `data/pose_targets.py:37–39,72–85,114–127` | Lookup utente/stem coerente; lato da metadata EMG quando necessario; unprompted=true nel run come firmato |
| Tensori e padding | `training/run.py:532–552` | Ordine mantenuto, padding con maschera falsa, std associata alla finestra |
| Loss | `training/jepa.py:150–186,524–545` | Intersezione con patch esistenti, media interna alla finestra e poi fra finestre valide; slab inclusi come previsto |
| Riduzione | `training/jepa.py:247–252,351–361`; `training/run.py:702–774` | Somma numeratori locali / conteggio globale; fattore world_size cancellato dalla media DDP; nessuna divisione extra per accumulo |

Il controllo logico con le funzioni reali, quattro rank virtuali e due microbatch ciascuno, conteggi `[2,0,1,2,0,0,2,1]`, restituisce
gradiente **−1,024999976158142** sia per il batch unico sia per la riduzione distribuita ricostruita. Verifica dell'algebra del componente,
non una prova end-to-end GPU. Dieci test CPU esistenti su target, ingest e loss sono passati (2,48 s); non e' stato lanciato un training.

**Non abbiamo trovato un offset o una riduzione errata da far correggere immediatamente a Claude.** Restano due lacune di verifica:

- `PoseStore.entry/check_length`, `pose_targets.py:87–112`, controlla file, forme e lunghezza ma non confronta `pose_meta.json` con identita',
  lato e origine EMG. Scambiare due array di posa della stessa lunghezza supererebbe queste guardie. Questo e' un controesempio di copertura,
  **non evidenza che lo scambio sia avvenuto**. Confrontare i sidecar e ricostruire patch dal raw e' piu' discriminante di una ridge negativa.
- Una patch e' valida anche con un solo blocco su cinque (`pose_targets.py:164`); una finestra con pochi target validi pesa quanto una piena.
  E' il contratto firmato. `pose_windows` non misura quindi quanta dinamica sopravviva a M=280. Esempio: blocchi [0,1,2,3,4] producono media 2;
  se resta valido solo l'ultimo, la media e' 4 e il centro effettivo del supporto si sposta di 10 ms. Misurare supporto e dinamica senza cambiare M.

**Tre affermazioni da ridimensionare nel prompt:**

1. MSE iniziale circa uno controlla l'ordine di grandezza, non certifica allineamento o standardizzazione esatti. Le statistiche sono sui
   blocchi validi a 5 ms, la loss sulle medie a 25 ms e con pesi per finestra. Permutare tempi/registrazioni puo' conservare la varianza.
2. E4 mostra un collegamento del gradiente, non che l'encoder stia imparando la posa. `run.py:746–751,769` conserva il gradiente del microbatch
   finale locale del rank 0; non misura cancellazioni fra rank o aggiornamento Adam. Gli zeri a 800/1.100 convivono con 4/9 finestre globali.
3. L'invarianza di Adam alla scala non esclude il clipping: vale idealmente per una scala costante dell'intera storia, epsilon trascurabile.
   Qui il fattore varia ed entra nei momenti con potenze diverse. Clipping attivo in tutti i 1.300 passi; fattore medio primi/ultimi 100
   pari a 0,1104/0,2441. Controesempio scalare con beta standard, epsilon=0: gradienti [1,100] danno aggiornamento normalizzato 0,7508 al
   secondo passo, mentre clip a 1 produce [1,1] e aggiornamento 1. Questo non dimostra rallentamento nel run, ma smentisce l'esclusione.

La testa a zero ha gradiente verso encoder e LayerNorm nullo al primo backward, ma non verso i pesi/bias di uscita (`pose_head.py:28–34`).
E' un bootstrap di un passo, non una disconnessione permanente. AdamW include tutti i parametri allenabili (`run.py:1194`), freeze=false;
il solo weight decay darebbe contrazione 0,99665 sull'intero run (0,335%): una spiegazione debole della piattezza.

## 3. Conto a tavolino: che cosa si puo' stimare

Il solo rapporto segnale/rumore r non determina i passi. Mancano condizionamento e covarianza dei latenti, concentrazione del segnale nelle
coordinate, correlazioni temporali, momenti Adam e deriva del backbone. LayerNorm normalizza ciascun vettore: non rende bianca la covarianza.

Un conto **illustrativo**, non una previsione del run: target di varianza 1, r=Var(segnale lineare)/Var(rumore), q=r/(1+r); encoder congelato,
feature isotrope, segnale uniforme sulle d=384 coordinate, B esempi indipendenti, momento adattivo tenuto al suo valore iniziale. Trascuriamo
transitorio del primo momento, epsilon, weight decay, clipping variabile e correlazioni. Approssimando la varianza iniziale del gradiente per
coordinata con 1/B, il fattore adattivo e' kappa=sqrt(B)/sqrt(1+B*q/d). Il fattore 2/20 della MSE si cancella nel rapporto ideale di Adam.
Con A=somma dei LR, l'approssimazione continua da':

`L(A) ≈ 1 − q + q exp(−2 kappa A)`

`T10 ≈ −log(1 − 0,1/q) / (2 × 1e−4 × kappa)`, per q>0,1.

| Frazione spiegabile q | Passi equivalenti al picco, B=7,178 | B=28, premessa errata del prompt |
|---|---:|---:|
| 0,2 (r=0,25) | 1.296 | 660 |
| 0,5 (r=1) | 418 | 215 |
| 1 (limite senza rumore) | 198 | 103 |

Con q<0,1 una discesa del 10% dalla testa a zero e' impossibile anche all'ottimo **di quelle feature fisse**. A q=0,1 il limite e' raggiunto
solo asintoticamente nel modello esponenziale. Questi limiti non si trasferiscono al backbone addestrabile, che puo' cambiare q.

Le loss del cancello sono misurate prima dell'aggiornamento: la distanza fra le dosi medie gia' applicate alle due finestre e' **513,28 passi
equivalenti al picco**. Nel modello illustrativo con B=7,178 la discesa relativa sarebbe circa 4,50% per q=0,2, 11,49% per q=0,5. Quindi un
segnale lineare reale ma moderato potrebbe non passare il 10%. B=7,178 e' qui un'ipotesi illustrativa, non la misura del batch efficace:
ogni finestra contiene molti istanti correlati e puo' sovrapporsi ad altre. I numeri non stimano q, B_eff o il tempo di convergenza del run.

**Giudizio sul criterio:** non era stato calibrato e non si puo' considerare un controllo tecnico garantito per questa testa/schedule.
Poteva passare con segnale forte, ma il fallimento non dimostra una pipeline guasta o un obiettivo impossibile. Resta valido come esito
operativo della regola firmata; una nuova prova deve avere un nuovo criterio esplicitamente approvato.

## 4. Controlli proposti, senza lancio

| Ordine | Controllo | Tetto di risorse proposto |
|---|---|---|
| 0 | Sintesi con geometria reale + apprendibilita' della testa | Mac, 1 thread, 30 min: **0,5 core-ore locali, 0 GPU-ora** |
| 1 | Audit diretto di 20 recording reali + ridge RMS | CPU cluster, 4 core, 16 GB, 1 h: **4 core-ore, 0 GPU-ora** |
| 2 / A | E3 su R3c endpoint 2 epoche: train + val, poi rapporto | 2 × 1 A100 × 40 min: **1,333 GPU-ore**; estrazioni 10,667 core-ore + rapporto 8 core × 2 h = 16: **26,667 core-ore** |
| 3 / B, condizionale | Un collaudo R3 da 5.000 passi, peso invariato | 4 A100, 32 core, 75 min: **5 GPU-ore, 40 core-ore** |

Totale massimo dell'intera proposta: **6,333 GPU-ore e 70,667 core-ore cluster**, piu' 0,5 core-ore locali. Budget eventuale piccolo job di
deploy va contabilizzato separatamente dal launcher abituale, prima dell'approvazione. Nessun retry o rilancio automatico. Residuo GPU
dichiarato da Claude ~85 ore: resterebbero ~78,7 dopo i tetti proposti, da riconciliare col ledger; non e' stato interrogato il cluster.
Per A la stima precedente ~0,7 GPU-ore e' una previsione non misurata; per B il ritmo osservato porta a ~59 min di soli passi, circa
4–4,3 GPU-ore includendo overhead contenuto. Pannello e checkpoint devono rientrare nei 75 min.

**0 — Sintesi, due controlli distinti.** Una registrazione sintetica con EMG modulato da un segnale lento noto e posa corrispondente,
fs=2 kHz, blocchi da 10, patch=25 ms, inizi=200 ms, margini del filtro, zeri e M=280. Un oracle indipendente calcola supporto e target da
indici raw; attraversare ingest, loader diretto/a blocchi, PoseStore e _pose_tensors. Includere inizio/fine, gap, patch parziali, padding,
lato, unprompted e un abbinamento volutamente sbagliato. Poi un problema separato di testa con feature note di dimensione 384: q forte,
moderato e nullo, inizializzazione zero, loss a 20 giunti, LR esatto del dbg e vero fino a 5.000, almeno tre semi e B illustrativi 7/28.
Misurare MSE su insieme fisso, norma output/pesi e aggiornamenti. **Non** allenare il backbone casuale aspettandosi che erediti la relazione
lineare degli ingressi: quello sarebbe un terzo problema. Il tempo al 10% del sintetico calibra il test, non prevede il dataset reale.

**1 — Audit prima della ridge.** 20 recording train, idealmente 10 utenti × 2 sessioni, lati e prompted/unprompted coperti; massimo 20
finestre non sovrapposte per recording, selezione congelata. Confrontare identita'/source_member, lato, fs, lunghezze e origine temporale
EMG/posa; ricostruire dal raw un campione di patch usando regola M=280 e statistiche train. La sequenza di trasformazioni deve coincidere
entro tolleranze dichiarate prima, non solo avere correlazione alta. Riportare validi su 5 per patch, quota valida/finestra e distribuzione
del movimento prima/dopo la maschera, senza riscrivere M.

Poi RMS per canale su patch da 25 ms (eventuale contesto/lag fissato prima), ridge sulla posa. Fit/test in blocchi temporali separati,
gap che copra sia finestra sia supporto di preprocessing; nessuna random split di patch sovrapposte. Iperparametri solo dal fit, costante
stimata dal fit nello stesso utente/contesto, confronto disallineato sullo stesso supporto, metriche dinamiche e per utente/sessione.
Un successo riduce il sospetto di associazione grossolanamente errata. **Un fallimento non dimostra H3**: RMS e regressione istantanea
possono non bastare per la posa assoluta. Se l'accesso ai raw archiviati richiede piu' I/O del tetto, riportare controllo incompleto;
non trasformare la sola concordanza dei sidecar in verifica dei valori.

**A — E3 come previsto, senza usarlo come cancello nuovo.** R3c ha gia' 49.046 passi, encoder congelato senza maschera e ridge risolta
offline sui latenti grezzi. R3 dbg ha feature in evoluzione, maschere e LayerNorm nella testa. E3 positivo prova accessibilita' nelle
condizioni di E3, non che R3 debba imparare in 1.300 passi; E3 negativo non prova che il supervisionato non possa costruire quella
rappresentazione. Una lettura aggiuntiva, descrittiva e dichiarata prima, della ridge su LayerNorm(latenti) usa le stesse cache e aiuta ad
allineare testa/sonda, ma non sostituisce E3 firmata. Se si aggiunge, deve stare nel tetto CPU o avere preventivo aggiornato.

## 5. Criterio proposto per un solo collaudo lungo

**Configurazione:** run nuovo, stesso seme/pesi della definizione R3, peso posa 1 e ancore 0,2, M=280, stesso campionamento; schedule
`total_steps=49046`, `steps_at_d=98094` → warmup 1.962, **stop a 5.000**, senza ripartire dal checkpoint gia' decaduto del dbg.
La CLI attuale consente `--total-steps 49046 --steps-at-d 98094 --max-steps 5000`: `scripts/train_jepa.py:187–202` applica max_steps dopo
with_wsd, e `run.py:778` ammette max_steps minore del totale. Non usare total_steps=5000, che farebbe decadere il LR da 4.000.
In 5.000 passi la dose LR e' 0,40195, circa sei volte il dbg; non significa sei volte l'apprendimento.

Congelare prima un pannello diagnostico **di train**, distinto dalla futura val E1: ad esempio 32 utenti × 2 recording × 4 finestre
non sovrapposte = 256 finestre con posa valida. Identita', maschere e pesi invariati; quote di lato/unprompted registrate. Pannello in eval,
senza dropout del muscolo, con le medesime maschere congelate a ogni misura. Nessuna selezione del pannello in base agli errori del modello.
Valutazioni a 0, 2.000, 2.500, 3.000, 3.500, 4.000, 4.500 e **5.000**; salvare errori per finestra, non solo la media globale.

**Proposta di cancello operativo di apprendibilita':**

- mantenere i controlli tecnici/scala/finitezza e la soglia di ritmo **0,81 s/passo**; non applicare meccanicamente al run lungo gli intervalli
  e i nomi checkpoint del vecchio lettore da 1.300 passi;
- sul pannello fisso, **MSE(5.000) ≤ 0,9 × MSE(0)** e limite superiore dell'IC95 appaiato del rapporto **<1**; media per finestra e poi
  per utente, con quote fisse uguali; bootstrap di utenti, seme 20261027, 1.000 ricampionamenti, includendo tutte le loro finestre insieme;
- dati mancanti, pannello cambiato, assenza dell'endpoint o stop per tempo rendono il cancello **non valutabile**, non superato;
- confronto fisso 2.000→5.000 e loss del flusso nei blocchi 2.001–2.500 / 4.501–5.000 come diagnostiche preregistrate, insieme a MAE,
  per-giunto, supporto valido e conteggi. Registrare sempre anche il predittore costante e la varianza delle predizioni.

Scelgo il riferimento zero per non bocciare un modello che abbia gia' imparato entro 2.000. Richiedere un ulteriore 10% dopo warmup e'
un'altra condizione, piu' severa, non equivalente a verificare che la testa impari. Le finestre casuali da 100 passi del piano originario
non sono un riferimento appaiato. Questo nuovo gate e' una scelta operativa da approvare, **non una soglia statisticamente derivata dai
tre punti gia' visti**. Il suo eventuale superamento non dimostra generalizzazione, dipendenza dall'EMG o trasferimento: queste restano
domande di E1/E2/E3 e P1/P2. Un miglioramento della sola media/bias non basta a dichiarare risolta la parte scientifica.

## 6. Decisione fra A–D e consegna a Claude

1. Correggere le premesse nel registro/prompt (conteggio globale, patch, dose LR, significato di E4/clipping); non i risultati del run.
2. Proporre 0 e 1 prima di spendere GPU. Se l'oracle dei target fallisce, Claude corregge il difetto dimostrato e si ricontrolla la catena.
   Se la testa non apprende il sintetico forte, verificare prima apprendibilita', criterio e budget di ottimizzazione del controllo,
   quindi correggere l'eventuale difetto identificato; non assumere automaticamente un errore di codice e non aumentare subito il peso.
3. Eseguire A dopo il via come diagnosi gia' prevista; **non subordinare B al rapporto E3<0,9**. A orienta l'interpretazione; se negativo,
   la capacita' della posa di modificare la rappresentazione resta proprio la domanda di R3.
4. Dopo i controlli di correttezza, proporre **una sola B**, criterio firmato prima e tetto 5 GPU-ore. Se non passa, R3 completo resta
   fermo: serve una prova supervisionata piccola e discriminante, da progettare separatamente, prima delle ~51 GPU-ore di training.
5. **C no per ora:** un peso maggiore cambia la definizione e non distingue bersagli errati da ottimizzazione/accessibilita'.
   **D no come abbandono definitivo ora**, si' come mantenimento del blocco del run completo fino alla decisione sui controlli.

La raccomandazione e' quindi **controlli diretti → A diagnostica → B condizionale alla correttezza**, non «A positivo dimostra H1».
Codice, strumenti e deploy eventuali restano a Claude; GPT propone il criterio e rivede i risultati. Nessuna autorizzazione di lancio
e' implicita nella presente review.

## Riproducibilita'

Ricalcolo dei log e del modello illustrativo: `docs/reviews/analyze_r3_dbg_20261027.py`, output aggregato
`docs/reviews/r3_dbg_recalculation_20261027.json` (include SHA256 dei cinque input; nessun segnale wearable copiato).
Controllo logico di riduzione, bootstrap e Adam: `docs/reviews/repro_r3_optimizer_20261027.py` (solo autograd/algebra, nessun passo di training).

```sh
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python docs/reviews/analyze_r3_dbg_20261027.py \
  /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine \
  --out docs/reviews/r3_dbg_recalculation_20261027.json

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /Users/simonebenatti/.venvs/wearusfm-tok/bin/python docs/reviews/repro_r3_optimizer_20261027.py \
  /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine
```

Non sono disponibili qui raw, array di posa, latenti o checkpoint del run: impossibile validare da questi log l'associazione reale,
la covarianza delle feature, q, B_eff o il gradiente globale della posa. I controlli proposti restano non eseguiti.
