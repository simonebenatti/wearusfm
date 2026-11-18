# Scaling: esito a 19.618 e decisione di completare 49.046

18/11/2026. Responsabilità scientifica GPT; operazioni alla chat **1-30/300 launcher**
(`01a1e915-4dc5-7d90-bd1e-371ccd3b8560`). Base di questa consegna:
`2df4ca68119acc067977a8a588c014069b263088`, confronto concluso `64785568`.

Simone, dopo la consegna del launcher: «allora fai quello che dice, ma poi fissiamo
che si va a 49.046 o a 50000. A 19.618 i risultati sono sempre brutti seconso me».
**La continuazione è decisa: endpoint 49.046, completando lo schedule originale.**
Non recepiamo quindi la proposta del launcher di rinviarla a tempo indeterminato.
Il 30M resta il modello di lavoro economico; il 300M completa questa prova mirata.
Questo documento fissa la domanda e la lettura, non certifica un wrapper ancora
da preparare né attesta un nuovo lancio.

## 1. Esito e verifica

Confermata la lettura del launcher: entrambi i primari a 19.618 sono irrisolti.
I punti e gli IC dei contrasti naturali dei cinque dataset, P1/P2 e della posa
sono stati riprodotti da un secondo revisore dai conteggi e sufficienti per
soggetto nei report a 5.000 e 19.618: 16 contrasti, scarto entro 1e-10.
È una verifica locale delle statistiche, non una nuova lettura degli array o
dei checkpoint remoti. Integrità, pairing e convergenza rimangono attestati
dalla catena raccolta: 420 fit convergenti, 60 combinazioni cella/dataset/fold.

| Misura naturale | 30M a 5.000 | 300M a 5.000 | 30M a 19.618 | 300M a 19.618 |
|---|---:|---:|---:|---:|
| EPN612, balanced accuracy | 76,314% | 78,231% | 82,577% | 83,074% |
| Posa dinamica, E_dyn ↓ | 0,899816 | 0,899931 | 0,838933 | 0,838694 |

Da 5.000 a 19.618, E_dyn cala del **6,77% nel 30M e 6,80% nel 300M**.
I modelli imparano: sulla dinamica il guadagno osservato viene dalla dose
molto più che dalla taglia. Il vantaggio EPN del 300M si restringe da
1,917 pp a 0,497 pp; questo cambiamento del vantaggio è descrittivo, non un
test già eseguito dell'interazione dose × taglia.

A 19.618, EPN dà delta **+0,497 pp [−0,038; +1,038]** ed E_dyn
**−0,000239 [−0,002328; +0,001793]**. Il miglioramento relativo puntuale
della posa è **0,0285%**, lontano dal criterio pratico del 2%. Il delta
assoluto corrispondente al 2% del punto 30M è circa −0,01678, ben oltre
l'estremo favorevole dell'IC del delta. È un confronto di scale usando
il punto 30M come riferimento, non un IC della riduzione relativa o un test
formale di equivalenza. L'incertezza qui è condizionata ai checkpoint e
alla popolazione osservati; non copre altri semi, popolazioni o ricette.

P1/P2 sostengono la non inferiorità al margine di 1 pp: i limiti inferiori
−0,525 e −0,682 sono entrambi maggiori di −1. Non è superiorità. Il calo
descrittivo DB6, −1,002 pp [−2,070; −0,085], resta visibile; gli IC sono
marginali, senza correzione per tutti i confronti secondari.

La proiezione fissa 300M 896→384 perde 1,183 pp EPN rispetto alla vista
naturale. Non isola causalmente l'effetto della larghezza del lettore:
può eliminare informazione. La posa rimane irrisolta anche in questa vista.

Non definirei tutti i risultati «brutti»: classificare gesti e ricostruire
il movimento sono domande differenti, e le accuratezze assolute di dataset
con compiti diversi non si confrontano direttamente. Il limite evidente
è che la dinamica resta ricostruita solo in parte e la capacità aggiuntiva
non ha ancora prodotto un vantaggio convincente nei primari.

## 2. Confronto con le altre linee

| Che cosa è stato fatto | Spiegazione | Risultato disponibile | Conclusione per il seguito |
|---|---|---|---|
| Scaling 30M/300M a 19.618 | Stessi passi, ricetta e seme; encoder congelati | Primari irrisolti; E_dyn quasi identico | Non promuovere il 300M sulla sola prova breve; completare il confronto a dose lunga |
| 30M T100-G fino a 49.046 | Traiettoria già conclusa, inclusa la fase di decay | Ridge E_dyn 0,838933→0,785902: −6,32% [5,87; 6,83]; testa −6,52% [5,92; 7,15] | Esiste una ragione sperimentale per completare anche il 300M; non implica che migliorerà più del 30M |
| Sonde 30M a due livelli | Pre e post backbone, alle tre dosi | Post migliore del pre; classificazione pool49k−19k P1 +0,35 pp e P2 +1,35 pp, entrambi irrisolti; secondario post P2 favorevole | Il backbone rende accessibile informazione utile; non usare un secondario per sostituire un primario |
| Fine-tuning EPN a 15 epoche | FT da teacher19k contro scratch nella stessa pipeline | +0,83 pp [0,42; 1,26] a favore di FT | Un beneficio modesto del pretraining è già osservato; non è la prova di FT del checkpoint lungo o del 300M |
| Ramo log-quadratico a 19.618 | LQ contro T100-L, con GELU secondario | EPN +1,531 pp contro L, irrisolto contro G; nessun vantaggio dinamico contro L | Non modificare il ramo di scaling con LQ; resta una prova distinta |
| TEMPONet direttamente sull'EMG | Baseline supervisionata già conclusa | MAE EMA 14,55° su utenti nuovi; 17,67° su fasi nuove; 17,60° su entrambi nuovi | Baseline utile; split e supporti diversi impediscono di dichiararla superiore al FM da questi numeri |
| TCN/TEMPONet sui latenti FM | Lettore temporale non lineare proposto, encoder congelato | Nessun fit reale concluso nella proposta v2 letta | Domanda ancora aperta sull'accessibilità della dinamica; non è un risultato negativo né positivo |
| Contrastiva e nuovi target | Collaudo contrastivo respinto per parità/ripresa; modifiche RMS/spettro in integrazione | Nessun risultato downstream della nuova ricetta | Non sono prove contro la contrastiva o a favore dei nuovi target; non cambiare la ricetta del 300M in corsa |

Il 30M lungo ha correlazione centrata circa 0,46–0,47 e solo il **22–24%**
della varianza temporale dei bersagli, rispettivamente ridge e testa.
Il progresso nella dinamica è reale, ma incompleto. LQ, scaling e lettori
lineari non hanno finora rimosso questo limite. È ragionevole tenere aperta
la domanda sul lettore temporale e sui bersagli; non aggiungerli a questa
continuazione, che deve rispondere a una sola domanda interpretabile.

## 3. Domanda e endpoint fissati prima del risultato lungo

**La capacità maggiore trae un beneficio aggiuntivo dallo schedule completo,
oppure segue sostanzialmente il miglioramento del 30M?**

1. **Endpoint primario 49.046.** Confronto teacher300M49k contro
   teacher30M49k, stessa ricetta e stessa dose. Nessuna scelta del checkpoint
   migliore fra 5k, 19k, predecay e finale. Il controllo 30M49k esiste già;
   identità e possibilità di riusare cache/OOF vanno verificate nel piano.
2. **Ripresa completa dal 300M19.618 accettato.** Pesi, teacher EMA,
   ottimizzatore, contatori, RNG, stream dati, calibrazione, batch, clipping,
   loss e schedule invariati. Nessun nuovo warmup o restart.
3. **Ricetta storica T100-G.** Tokenizer 100 ms, GELU, passo 25 ms;
   target RMS 25 ms e spettro rapido 50 ms. Le nuove finestre dei target
   (RMS100 e rapido125, dopo il ritiro dell'intermedio100) appartengono a una
   nuova coorte. Non confondere finestra del tokenizer e finestra dei target.
   Nessun merge indiscriminato di main nel checkout della continuazione.
4. **49.046, non 50.000.** Il total_steps originale è 49.046 e il decay
   comincia al confine 39.237. Il trainer rifiuta max_steps oltre total_steps;
   cambiare anche total_steps sposterebbe il decay. Il checkpoint nativo
   predecay39.237 viene conservato, senza nuova batteria di valutazioni
   intermedie implicita. Dose aggiuntiva e decay fanno parte della stessa
   prova: non se ne isola l'effetto causale individuale.
5. **Stessa valutazione dei primari.** EPN612 e ridge E_dyn dinamiche,
   viste naturali primarie, medesimi utenti, finestre, fold, supporti e
   selezione dei lettori sul solo train. Controllo secondario con la stessa
   proiezione fissa 896→384. Non introdurre una testa temporale diversa
   o un fine-tuning dentro il confronto di scaling.
6. **Stesse definizioni della lettura a 19k.** Delta = 300M−30M;
   EPN superiore se IC95 interamente positivo, inferiore se interamente
   negativo, altrimenti irrisolto. E_dyn migliore se IC interamente negativo,
   peggiore se interamente positivo; beneficio pratico se anche la riduzione
   relativa puntuale è almeno 2%. P1/P2: non inferiorità sostenuta se il limite
   inferiore è >−1 pp, perdita oltre margine se il superiore è <−1 pp,
   altrimenti irrisolta. Bootstrap appaiato per soggetto, 2.000 repliche,
   seme 20261102, stessi pesi dei dataset. Nessuna nuova regola S8 automatica
   di prosecuzione: quella originale riguardava il passaggio 5k→19k.
7. **Lettura secondaria della dose.** Riportare entrambe le traiettorie
   19k→49k. Se gli artefatti superano la parità di supporto e lettore,
   calcolare anche `(300M49−300M19)−(30M49−30M19)`, in pp per EPN e in
   E_dyn assoluto per la posa, con le stesse estrazioni bootstrap per
   soggetto nelle quattro celle. Per la posa un valore negativo favorisce
   il guadagno del 300M. È un secondario esplicativo: non rimpiazza i primari.
   Non sottrarre IC separati né valori arrotondati; se manca la parità,
   limitarsi alla descrittiva e dichiarare il limite. Correlazione, rapporto
   di varianze e livello medio restano diagnostiche dai report già previsti;
   E2 solo se supporto e pipeline sono realmente allineati.

La lettura finale distinguerà guadagni sui gesti e sul movimento. Un vantaggio
di un endpoint non dimostra un miglioramento generale. Se i due primari
restano irrisolti, avremo completato questa verifica senza un vantaggio della
taglia dimostrato nella ricetta; niente ulteriore training automatico.
Un beneficio chiaro andrà pesato contro il costo ed eventualmente replicato:
questa resta una traiettoria a seme unico su dati di sviluppo, non un test
definitivo di scaling universale o di utilità del FM con ogni adattamento.

## 4. Costo e preparazione operativa

Il run300M riuscito `63979977` ha impiegato 42.965,633 s di trainer per
14.618 aggiornamenti: **2,93923 s/passo**. I 29.428 aggiornamenti mancanti
richiedono circa **24,03 ore su quattro GPU, ossia 96,1 GPU-ore** attese.
Il controllo 30M lungo è già disponibile: non va riaddestrato.

| Segmento proposto | Aggiornamenti | Tempo atteso | Tetto di prenotazione |
|---|---:|---:|---:|
| 19.618→39.237 | 19.619 | 16,02 h | 19 h 45 min × 4 GPU = 79 GPUh |
| 39.237→49.046 | 9.809 | 8,01 h | 10 h × 4 GPU = 40 GPUh |

Formula prudente: arrotondamento superiore a 15 min di
`passi × 2,93923 × 1,20 + 1200 s`. Sono **119 GPUh prenotate**, più 0,4
di margine scheduler se confermato il margine di tre minuti per job;
con 32 CPU, rispettivamente 952 e 955,2 core-ore. Questi sono costi del
training: estrazioni, lettori, checkout e conservazione vanno preventivati
separatamente dal launcher. Il consuntivo CPU19k **94,732778/351** chiude
la catena precedente; il residuo non autorizza un inviluppo nuovo implicito.

Occorre un wrapper nuovo: l'attuale è cablato a 5.000→19.618 e vieta di
superare il confine del decay. Preparazione, test e unica review del pacchetto
prima dell'uso; poi controlli meccanici fra i segmenti, senza nuove richieste
di consenso per il medesimo esperimento approvato. Nessun deployment o invio
da questa chat. Dettagli nella [consegna al launcher](../scaling_49046_launcher_handoff_20261118.md).

## 5. Provenienza e limiti della lettura incrociata

Fonti locali raccolte, con commit e SHA256 dei documenti, in
`results/infra/scaling_49046_20261118/source_manifest.json`.
Riferimenti principali:

- Scaling19k: `docs/reviews/gpt_scaling19618_final_results_20261118.md`
  e report `continuation19618_20261106/evidence/compare/report.json`, base2df4ca68.
- Scaling5k: `final_stages_20261106/collected_outcome_20261106_1441/report.json`.
- Lungo30M: `docs/reviews/consolidation_long_results_20261107.md`
  e `results/infra/consolidation_long_20261105/gpt_review_20261107/scientific_readout.json`.
- LQ: `docs/logsquare_scientific_readout_20261107.md`, linea3.
- TEMPONet raw: `results/temponet_emg2pose/eval_20261111/eval.json`;
  TCN latenti: `docs/reviews/pose_temponet_response_20261118.md`, linea2.
- Contrastiva: `docs/reviews/contrastive_readout_results_20261118.md`;
  target: `docs/target_windows_fix_20261118.md`, ramo d'integrazione.

I nuovi bracci TEMPONet raw con architettura residua/loss e le modifiche dei
target sono lavori in corso nei riferimenti letti, non risultati conclusi.
Nessuna lettura viva dei job delle altre linee è stata eseguita qui.

## 6. Review indipendente della consegna

Secondo revisore interno GPT: **APPROVATO, nessun rilievo bloccante**.
Verificati numeri, soglie, non inferiorità, lettura fra linee, endpoint,
interazione secondaria, aritmetica dei costi e pin documentale del controllo.
Recepite due precisazioni: il confronto fra delta assoluto e soglia relativa
non è un IC della riduzione relativa; il riuso pareggia griglia e procedura
di selezione, non obbliga a selezionare lo stesso alpha/C.
La linea2 dichiarava compatibilità numerica, non identità dei runtime:
la consegna richiede quindi una verifica esplicita prima del riuso dei lettori.
Questa review riguarda il piano scientifico; non sostituisce la verifica del
nuovo wrapper o dei pin di esecuzione che produrrà il launcher.
