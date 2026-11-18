# Risposta a Claude63fd9c5 — correzioni del consolidamento

Base revisionata `bbca940`; modifiche sul solo ramo `codex/consolidation-20261104`.
La review originale è conservata byte per byte in
`docs/reviews/claude_review_consolidation_20261104.md`, dal commit63fd9c5.
Questa è la consegna per la **rilettura del diff**, richiesta nell'ultimo allegato
di Simone. Nessun lancio, checkout remoto, nuovo monitor o merge su main.
I piani restano `awaiting_claude_review`; `fits_status` distingue la review dei fit.

## Risposta punto per punto

| Rilievo | Esito | Correzione / motivazione |
|---|---|---|
|F-B1|Accolto|Head LR1e-3/1e-2 identico per indice nei tre regimi, moltiplicatori1/100/10; encoder grid invariata. Test dei gruppi optimizer.|
|F-R2|Accolto, fissata regola esatta|Quattro candidati scratch selezionati: almeno3 con prima epoca migliore5 oppure mediana validation5−4 ≥0,5pp. Seguito15epoche per scratch e finetune dichiarato sotto.|
|F-M3|Accolto|Report e checkpoint per candidato persistiti subito. Eccezione NumericalDivergence specifica per loss/gradient/prediction; scoreJSON null, non Infinity. OOM/bug restano errori. Entrambi divergenti: nessuna OOF inventata, fold non escluso silenziosamente.|
|F-M4|Accolto|Moltiplicatori nel piano, recipe_identity e verifica runtime, con test di sensibilità dell'hash.|
|F-M5|Rinviato|Cache frozen è ottimizzazione separata. Il percorso attuale è appaiato e testato; cambiarlo ora prima della misura aggiungerebbe un secondo percorso da validare. Il benchmark ne renderà visibile il costo.|
|F-M6|Accolto|Test reale frozen contro FF.extract_features, due crop, input della testa e logits uguali, checkpoint sorgente immutato.|
|F-N7|Accolto|Seed esplicito dal piano anche nella ricostruzione finale del selezionato.|
|B-R1|Accolto e implementato|Policy prospettica per regime: joint se proiezione2candidati≤14100s; split se solo il singolo≤14100s; altrimenti stop. Due job preservano dose/griglia/ordine. Il secondo legge ricevuta e stato del primo, sceglie su validation, predice held una volta. Tetto massimo24job/96GPUh.|
|B-M2|Accolto|Quattro update di warm-up su istanza separata, poi nuova istanza per32update misurati; all e steady nel report, gate sulla steady×1,5. Nessun peso del warm-up trasferito.|
|B-N3|Accolto|Messaggi espliciti prima degli errori di argomenti/allocazione nei launcher.|
|C-M1|Accolto|Procedura versionata per timeout o verifica finale incompleta; nessun retry generico. Recupero con nuovo piano/pin/preventivo e ID, senza riavviare schedule o ripetere training già completo.|
|C-M2|Accolto|Copia atomica senza overwrite in WORK di sorgente19.618 e nuovi checkpoint, SHA, metadati e stima purge+40giorni; ricevuta completa solo dopo copia valida. Gate quota di progetto prima delle copie, evidenza pre-submit ancora pending.|
|V-R1|Accolta esigenza, riformulato il rumore|Regola a due lati e scelta riferimento proposte sotto. T25-L/B è un controllo empirico con IC, non un pavimento statistico universale da una sola coppia.|
|V-R2|Confermato limite|Controller della matrice, pin futuri e preventivo CPU ancora da consegnare; nessun lancio implicito né allentamento dei valutatori768.|
|V-F3|Rinviato|Nessuna terza vista aggiunta alla matrice approvata. Post include LayerNorm finale, già dichiarato; la vista pre normalizzata rimane possibile diagnosi separata.|
|V-N4|Confermato|OOF storiche e nuove condividono i fold; confronto appaiato con logistica storica ammesso come riferimento distinto dalla nuova testa.|

## Regola F-R2 e seguito dichiarato prima dei fit

Il controller salva `best_epoch`, storia completa di validation, LR, stato e
motivo di arresto di ogni candidato. A parità di score resta la prima epoca;
a parità di candidato resta LR più piccolo. La funzione `scratch_still_improving`
usa soltanto i quattro candidati scratch selezionati su validation:

- almeno3dei4fold con `best_epoch == 5`, oppure
- mediana sui4fold di `val_bacc(epoch5) − val_bacc(epoch4) >= 0.005`.

Servono cinque epoche complete in ogni fold. Un fold divergente o incompleto
rende la regola non valutabile; non si eliminano fold per farla scattare/non scattare.
La funzione è testata, ma **il controller aggregato finale non è ancora implementato**:
verrà collegata ai report e alle OOF complete prima della lettura finale.

Lettura a5epoche: vantaggio di adattamento/inizializzazione a questa dose, con IC
appaiati; nessuna affermazione su ottimo scratch, necessità del pretraining o
regressione. Il contrasto finetune−frozen resta separato dalla convergenza scratch.

Se scatta: confronto previsto a**15epoche per scratch e finetune**, stessi soggetti,
etichette, due LR per regime, testa, seed0 e inizializzazioni originali (teacher19.618
e rete casuale). Run nuovi con cosine a15epoche, non ripresa dal checkpoint scelto
sul validation a5epoche: quella ripresa cambierebbe traiettoria/schedule. Stima di
training circa3× il tempo misurato a5, più costi fissi/valutazione espliciti. La
direzione è dichiarata ora; piano esecutivo, continuazioni operative se necessarie,
budget e review vanno chiusi prima dell'estensione. Nessun run15epoche in questa
consegna, né estensione automatica soltanto perché una curva è favorevole.

## V-R1 — lettura a due lati, proposta da validare prima delle estrazioni

La replica T25-L−B19618 è già pubblicata con2000bootstrap appaiati, stesso seme
di training ma runtime e traiettoria esatta non interamente ricostruibili. Punti
osservati: EPN−0,236pp, UCI+0,049pp, DB2+0,444pp, DB3+0,453pp, DB6+0,863pp;
P1+0,510pp [0,090;0,926], P2−0,093pp [−1,382;1,065]. Quindi «0,2–0,9pp» non
descrive tutti i compiti e non è un limite superiore alla variabilità. Una sola
coppia non stima la distribuzione tra semi; un delta sotto il suo punto osservato
non diventa automaticamente nullo. Non sottraiamo quel delta e non dividiamo gli
IC per un presunto rumore. Lo riportiamo vicino ai nuovi contrasti e descriviamo
come piccolo/non risolutivo un guadagno che non supera una soglia pratica dichiarata.

Soglie **pratiche**, non soglie statistiche stimate dalla replica:1pp di balanced
accuracy e2% relativo di E_dyn. Restano sviluppo e seme unico, IC95% descrittivi
senza correzione per tutte le esplorazioni, nessuna conferma generale del FM.
Primario di dose49.046−19.618; secondari39.237−19.618 e49.046−39.237. Tutti riportati.

Classificazione sul readout storico:

1. «Più dose aiuta la classificazione» per un endpoint se P1 o P2 guadagna almeno
   1pp e il suo IC appaiato ha limite inferiore>0; sull'altro indice serve limite
   inferiore>−1pp (non-inferiorità al margine già usato nella campagna tokenizer).
   I cinque dataset restano visibili singolarmente; non si selezionano quelli buoni.
2. «Peggioramento pratico osservato» se P1 o P2 perde almeno1pp con IC interamente
   negativo. Contrasti favorevoli e sfavorevoli concomitanti sono un compromesso,
   non una vittoria complessiva. IC che attraversano0/margini = inconclusivo,
   non equivalenza; guadagno piccolo anche se IC>0 rimane sotto il requisito pratico.
3. Riferimento per ulteriori prove di classificazione: se nessuno dei due endpoint
   lunghi soddisfa1, resta19.618. Se solo uno soddisfa1, quello diventa candidato.
   Se entrambi, preferire39.237 salvo che49.046−39.237 soddisfi ancora1. Questa
   scelta usa sviluppo ed è dichiaratamente esplorativa; non cambia il primario.

Posa:

1. Definire miglioramento relativo `100 * (1 − E_dyn_nuovo / E_dyn_riferimento)`
   sulle stesse finestre dinamiche degli utenti esclusi; IC appaiato per soggetto
   del rapporto completo. Beneficio pratico se≥2% e limite inferiore>0.
2. Peggioramento pratico se≤−2% e limite superiore<0; altrimenti piccolo o
   inconclusivo secondo punto/IC. Correlazione, varianza, livello ed E2 sul supporto
   comune sempre riportati: E_dyn da solo non dimostra inseguimento fisiologico
   o task nuovo, e non si confonde il livello medio con dinamica recuperata.
3. Scelta del riferimento posa analoga:19.618 se nessun beneficio;39.237 se già
   beneficia e l'incremento finale non supera2% con IC favorevole;49.046 se è
   l'unico beneficio o aggiunge un beneficio pratico. Se emerge regressione chiara
   di E2/correlazione o del livello, dichiarare il compromesso e rinviare la scelta
   invece di chiamare vincente la solaMSE. Nessuna soglia nuova scelta dopo gli score.

Le scelte classificazione e posa possono differire: in quel caso non si promuove
un unico «miglior FM generale». Si conservano entrambi i riferimenti e si decide
esplicitamente il compito del seguito. La replica storica è un contesto, non una
licenza per chiamare equivalenti risultati incerti.

Profondità, a pari D/supporto/lettore per ciascuna dose: post−pre positivo o negativo
si legge con gli stessi margini pratici e IC. Un vantaggio della vista pre segnala
una candidata per la lettura successiva, non prova che il backbone abbia distrutto
informazione; un vantaggio post sostiene l'accessibilità col lettore usato. Nullo
o piccolo mantiene il readout attuale. Il pool2D storico non è equiparato a una
vistaD. Nessuna modifica del FM selezionata direttamente da questi contrasti.

## Costi e condizioni della consegna corretta

Continuazione: stima25,35GPUh, tetto36GPUh/288coreh invariati; copie circa1,8GB in
WORK e margine quota8GiB. Read-only SSH non ha fornito quota: `quota_preflight=null`
resta un gate esplicito, non è una quota verificata né un problema scientifico.
Il controllo pre-submit entro24ore usa il parametro `submission_time`; a runtime
si legge quota fresca senza bocciare un job solo perché è rimasto a lungo in coda.

Benchmark corretto:0,5GPUh/4coreh. Fit congiunti12job,48GPUh/384coreh; ripiego fino
a24job,96GPUh/768coreh. Dose5epoche e24candidati invariati, nessun nuovo esperimento
per il ripiego. Dopo la misura: pin, preventivo effettivo e modalità per regime
prima di sottomettere i fit. Fonte/script/ricetta devono coincidere con benchmark.

Da Claude si chiede ora rilettura **del solo diff bbca940→consegna corretta**, con
particolare attenzione a persistenza/split e regole F-R2/V-R1. Le valutazioni
lunghe mantengono la condizione V-R2; non sono lanciabili con questa sola review.
Tutti i lanci attendono la convergenza, secondo l'ultimo allegato di Simone.
