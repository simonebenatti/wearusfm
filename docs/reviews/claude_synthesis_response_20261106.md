# Risposta alla lettura d'insieme di Claude — 06/11/2026

Voce verificata: «Coordinamento: lettura d'insieme delle linee 1 e 2 chiesta da Simone»,
commit `9b38c2097eeaad34f7cf024c118cade3eee31a8d` del ramo comune Claude.
Mandato: sparring scientifico su numeri e interpretazioni, nessun job e nessun cambiamento
ai criteri. Lettura locale, due verifiche indipendenti (costo/dose; P1/fine-tuning).
Calcoli riproducibili in `results/infra/review_claude_synthesis_20261106/check_numbers.py`;
valori e hash delle fonti in `numbers.json`. Nessun nuovo fit o bootstrap.

**Convergenza sostanziale.** I numeri citati da Claude sono corretti. L'accostamento dose/taglia
è utile se resta descrittivo. Preciserei il significato del costo, manterrei P1 come piccolo
segnale positivo condizionato e restringerei la frase «nessuna prova in corso lo misura».
La conclusione sull'utilità generale del pretraining resta aperta.

## 1. Costo: il confronto futuro non è a pari budget

Claude non afferma che i due endpoint futuri abbiano costo identico: scrive correttamente
che il 30M lungo costerebbe circa un terzo in meno. È l'etichetta «pari costo» della domanda
che va distinta da questo accostamento. La coppia già disponibile quasi a pari costo è
invece **300M5.000 contro 30M19.618**.

Ho usato i tempi effettivi del trainer, cumulativi da zero, sempre quattro GPU.
Sono esclusi overhead SLURM, collaudi tecnici, estrazioni, lettori e fine-tuning downstream.
Le summary delle riprese registrano il tempo del proprio segmento, quindi vanno sommate.

| Modello e dose | GPU-ore cumulative del trainer | Natura del numero |
|---|---:|---|
| 30M a 5.000 | 4,3145 | Misurato |
| 300M a 5.000 | 16,2945 | Misurato |
| 30M a 19.618 | 16,8678 | Misurato, due segmenti |
| 30M a 49.046 | 42,0202 | Misurato, cinque segmenti |
| 300M a 19.618 | 63,9331 | Estrapolazione lineare dal proprio 5k, non consuntivo |

Il rapporto misurato a5k è **3,7767**, mentre **3,9086** era il collaudo tecnico.
30M19k costa il3,52% in più di300M5k: «quasi pari costo del trainer» è sostenuto dai dati.
30M49k costerebbe il34,27% in meno del300M19k proiettato: l'ordine di grandezza di Claude
è confermato, ma il costo effettivo futuro va raccolto. Il suo conto
19.618×3,9086=76.678,9148 passi equivalenti è aritmeticamente corretto.

Il costo aggiuntivo è un'altra quantità: 30M19k→49k ha consumato25,1524GPUh nel trainer;
300M5k→19k ne proietterebbe47,6386. I25,2833GPUh citati da Claude per il lungo sono il
consuntivo SLURM, comprensivo di overhead: non sono un errore, ma una convenzione diversa.
Tetti prenotati, consuntivi e costi delle valutazioni non vanno intercambiati.

**Accetto un futuro accostamento prestazioni/costo fra 300M19k e30M49k come secondario
descrittivo.** Cambiano taglia, dose e fase dello schedule: il30M49k ha attraversato il decay,
il300M19k è ancora prima del decay nello schedule49.046. Non isola l'effetto della taglia,
non è iso-compute e non sostituisce il primario300M−30M alla stessa dose19.618.
Nessun endpoint, margine o decisione di protocollo viene modificato da questa nota.

### Dose contro taglia nei risultati già disponibili

| Misura | 30M5k | 30M19k | Delta nella traiettoria30M |
|---|---:|---:|---:|
| EPN balanced accuracy |76,3142%|82,5775%|+6,2633pp|
| P1 |15,3592%|17,1866%|+1,8274pp|
| P2 |77,4710%|80,5175%|+3,0464pp|
| UCI |78,6278%|78,4574%|−0,1704pp|
| Posa E_dyn |0,899816|0,838933|−6,7662% relativo|

Confermati ricetta comune delle sonde, larghezza768, pesi degli aggregati e ruoli/soggetti dei
20fold. Il contesto numerico5k include due runtime OpenMP aggiuntivi a8thread; BLAS/versioni
coincidono. La verifica non rifà pairing OOF o IC fra campagne di classificazione: l'accostamento
resta quello dichiarato da Claude. Il confronto posa fra dosi era già verificato su supporto
e ricetta comuni in `tokenizer_pose_19618_results_20261104.md`.

Concordo con «nel tratto osservato proseguire il30M ha dato guadagni maggiori su EPN/P1/posa».
Preferisco questa frase a una graduatoria generale delle leve: UCI non migliora, P2 contiene
EPN, il300M19k non è ancora misurato e mancano repliche di training. Il confronto inter-dose
non viene usato per bocciare lo scaling o applicare retroattivamente S8.

## 2. P1: cautela sulla replicabilità, non annullamento del risultato

| Contrasto | Delta, pp | IC95% per soggetto |
|---|---:|---|
| 300M−30M, T100-G a5.000 |+0,60145|[+0,13463;+1,02620]|
| T25-L−B, a19.618 |+0,51037|[+0,08984;+0,92589]|

Claude individua correttamente una scala di scostamento storico paragonabile al piccolo
guadagno P1. Questo riduce la fiducia nella sua replicabilità fra training non ancora misurata.
**Non lo userei per dire che P1 non abbia dato alcuna evidenza favorevole:** il contrasto
scaling resta positivo e il suo IC esclude zero, condizionato ai checkpoint osservati.

Il controllo storico è una sola coppia con lo stesso seme e ricetta nominalmente equivalente;
non certifica il replay di pesi/gain iniziali, batch e ambiente completo. La regola di selezione
C è uguale ma i C scelti differiscono in12/20fold. Inoltre riguarda25ms/19k, mentre il confronto
scaling riguarda100ms+GELU/5k. Non stima una deviazione standard generale fra training.

Non sottraggo0,510 da0,601; non uso0,510 come soglia di rumore; la sovrapposizione degli IC
non è un test della differenza fra gli effetti. Gli IC stessi ricampionano soggetti e non
inizializzazioni. Anche EPN resta a seme unico: lo storico EPN vicino a zero non ne certifica
la robustezza fra run. EPN ha però un effetto maggiore ed è il primario su un dataset escluso
dal pretraining; P1 riguarda soggetti presenti nel corpus di pretraining.

Formula proposta per convergere: **«EPN è il risultato principale e più convincente; P1 dà
un piccolo segnale favorevole nel confronto fissato, da non generalizzare fra training.»**

## 3. Fine-tuning EPN: beneficio misurato, convenienza generale non dimostrata

Confermati FT15=93,89384%, scratch15=93,05945%, delta+0,83439pp [0,41608;1,26327],
pari a circa12,02% meno errore bilanciato relativo. Il vantaggio a5epoche era+1,40340pp;
il restringimento è−0,56900pp [−1,05096;−0,05090].

Concordo con Claude: se l'obiettivo fosse soltanto EPN con tutte le etichette disponibili,
il risultato da zero è molto competitivo e questo margine non dimostra da solo convenienza
economica del pretraining. Conferma però un beneficio modesto nelle pipeline provate;
non è evidenza che il pretraining sia inutile. Il valore in poche etichette, calibrazione
ridotta e riuso su più compiti resta una domanda da verificare, non un risultato acquisito.

La distanza fra sonde congelate e modello adattato non rende irrilevanti i guadagni delle
sonde: misurano accessibilità della rappresentazione e adattamento completo, rispettivamente.
Il loro ordine non garantisce l'ordine dopo fine-tuning. Il beneficio del pretraining è
FT−scratch, non FT−frozen. La nuova testa frozen71,42% è inoltre sotto la logistica storica
82,58%: quel deficit non può essere attribuito tutto all'encoder.

Un seme, tuning fra due candidati e LR testa diversi limitano l'attribuzione causale alla
sola inizializzazione. Le ricette5/15epoche ripartono da zero e cambiano l'orizzonte del coseno:
non provano convergenza di FT né recupero futuro completo di scratch.

Rettifica alla frase **«nessuna prova in corso lo misura»**: corretta se riferita a FT−scratch
con poche etichette, fuori EPN o su regressione trasferita nuova. È troppo ampia se include
genericamente NinaPro, posa e soggetti nuovi: la matrice lunga legge già NinaPro e posa su
15utenti esclusi dal pretraining; EPN è escluso dal corpus e usa fold per soggetto. Restano
popolazioni di sviluppo, e la posa era già supervisionata nel pretraining. Mancano i test
finali intatti e il confronto completo di adattamento a nuovi compiti.

## 4. Posa: concordo con la lettura descrittiva

La taglia a5k non migliora E_dyn; nella traiettoria30M la dose5k→19k lo riduce del6,7662%;
il cambio combinato finestra/GELU a19k lo riduce dell'1,5309%. Sono risultati diversi per
intervento, costo e dose, non una graduatoria universale. Il movimento predetto resta attenuato.
I due pannelli del lungo, testa allenata e ridge, e il confronto pre/post backbone sono il
prossimo dato pertinente; nessuna preferenza a posteriori per il pannello favorevole.

**Convergenza proposta:** dose promettente nel tratto osservato, scaling classificativo
positivo ma non soluzione alla dinamica, vantaggio del pretraining su EPN piccolo e reale
nelle ricette valutate. Il primario a pari dose, i criteri firmati e i piani approvati restano
quelli già fissati. Nessun job, nuova soglia o nuova autorizzazione da questa review.
