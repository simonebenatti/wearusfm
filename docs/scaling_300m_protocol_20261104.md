# Screening della taglia 300M con T100-G — 04/11/2026

Simone ha approvato la linea parallela: prima collaudo tecnico, poi300M a5.000
passi contro30M alla stessa dose. L'estensione a19.618 è una possibile seconda
tranche, non un lancio già disposto. L'operatore dei job resta GPT root tramite
submitter e ledger comuni; l'agente scaling prepara in un worktree separato.
Nessuna automazione, multinodo, nuova loss o innesto del checkpoint30M nel300M.

Stato: **Claude ha approvato il collaudo senza blocchi**, review del 04/11 nel commit
`befab4a`; ha rieseguito i 23 test, tutti passati in 13,53 s. Codice del collaudo
congelato `05d4bf0663481522a56d96c0b07847924cf5bd26`, invariato dopo la review.
Le precisazioni sotto riguardano la lettura del collaudo e il seguito a 5.000.
**Non è ancora un via al training scientifico:** occorrono risultato del
collaudo, preventivo misurato e valutatori che trattino esplicitamente la larghezza
del300M. Le sottomissioni del collaudo approvato sono coordinate da GPT root; questa
nota non afferma uno stato SLURM e non modifica il codice o la policy già approvati.

## Domanda e confronto

Quanto cambia l'informazione accessibile con più parametri, nella ricetta T100-G
attuale e alla stessa esposizione? Il confronto è condizionato a dati, dose,
obiettivo, inizializzazione convenzionale e ottimizzazione fissati. Non stima una
legge universale di scaling, né quale sia il miglior modello possibile a300M.

| Proprietà | Controllo30M già disponibile | Candidato300M da zero |
|---|---:|---:|
| Parametri student+decoder+teste, senza EMA |28.362.013|292.054.847|
| Dimensione, teste, dimensione per testa |384,6,64|896,14,64|
| Blocchi locale/backbone/decoder |2/8/2|3/18/2|
| Tokenizer / passo / slot |100ms+GELU /25ms /64|identici|
| Loss |posa1 +ancore0,2; JEPA0|identica|
| AdamW / picco LR / clip globale |WD0,05 /1e-4 /1|identici, salvo stop tecnico|
| Schedule |WSD49.046, warmup1.962, decay finale20%|identico|
| Batch globale / rank / worker per rank |32 /4 /6|identici|
| Dataset, split, preprocessing, maschere, target |pin della campagna tokenizer|identici|
| Prima dose scientifica |5.000 aggiornamenti|5.000 aggiornamenti|

Solo i cinque valori del preset LADDER cambiano. L'allarme di rango conserva la
soglia assoluta38,4 approvata per tutte le taglie; non viene moltiplicato perD.
Il seed nominale0 è comune, ma forme/depth diversi producono pesi e dropout diversi:
non si promette parità dei pesi o del consumo RNG del modello. Il banco del tokenizer
e il gain iniziali, costruiti prima della proiezione dipendente daD, sono verificati
identici. Questo mantiene il punto di lavoro della GELU già usato, senza ricalibrarlo
in risposta ai risultati downstream.

## Collaudo tecnico train-only: nessun checkpoint

Script `scripts/scaling_preflight.py`, launcher `scripts/slurm/scaling_preflight.sbatch`,
piano `results/infra/scaling_300m_20261104/preflight_plan.json`. Usa le funzioni di
aggiornamento già verificate di tokenizer_preflight e del trainer di produzione.

I due modelli sono eseguiti **uno alla volta**, ciascuno da zero sullo stesso prefisso
di128batch globali da32. Non si sommano le loro memorie GPU. Per ogni taglia:

1.64aggiornamenti con lo schedule nativo: LR da1e-4/1962 a64×1e-4/1962.
2.64aggiornamenti consecutivi sullo stesso modello/AdamW, forzando solo il LR al
  picco1e-4 (warmup temporaneamente0). Lo stress anticipa il picco a pesi ancora
  giovani; **non simula un checkpoint a1.962passi** e non è esposizione scientifica.
3.Si eliminano tutti i pesi/stati; il run scientifico ripartirà da zero con warmup
  intero. Nessun checkpoint o score downstream viene scritto.

Il controllo30M viene rieseguito solo per misurare il rapporto dei tempi nelle
medesime condizioni hardware. Non è una replica scientifica del pretraining.
Ogni rank conserva hash dei batch, segnali processati, origine, maschere e target;
si verifica parità fra taglie, inclusi supporti reali del decoder e della posa.
Manifest esclusivamente pretraining; EPN/UCI esplicitamente rifiutati dal guard.
Pin dei file e stat delle sorgenti verificati prima/dopo. Output nuovo e fuori dal
codice, nessuna sovrascrittura o ripresa implicita.

Si misurano tempi forward/backward e clip/AdamW/EMA con massimi fra rank; esclusi
i primi8passi di ciascuna fase per i riassunti. Si riportano memoria allocata e
riservata, norma del gradiente prima del clip, fattore/frequenza di clipping,
norme di aggiornamento assolute e relative per frontend/proiezione/posa e matrici
QKV temporali/MLP del primo e dell'ultimo blocco del backbone, con nomi espliciti
nei report. Le norme relative omettono una divisione per zero: la
testa iniziale a zero non è un'anomalia. Sono statistiche diagnostiche, non un
criterio di uguaglianza fra architetture di diversa larghezza/profondità.

### Gate e lettura del learning rate

Gate automatici:128batch per taglia e4rank completi, parità degli ingressi e supporti,
pesi/gradienti/aggiornamenti finiti, gradienti e aggiornamenti non nulli nei gruppi
monitorati lungo il collaudo, memoria massima allocata≤90% della memoria GPU,
pin/sorgenti immutati. Scadenza interna25min su allocazione30min. Un fallimento
ferma quella fase; niente retry cieco, riduzione batch nascosta o scelta delLR suval.

**Il collaudo non dimostra che1e-4/clip1 siano ottimali.** Si esaminano anche loss,
clipping e dimensione relativa degli update: un clipping frequente da solo non
dimostra instabilità con AdamW, e non si impone che il300M abbia le stesse norme
del30M. Se è tecnicamente stabile, la prima prova mantiene questi iperparametri,
così misura la taglia nella stessa ricetta. Se le diagnostiche mostrano un problema
concreto, si scrive una calibrazione train-only separata prima di consumare5.000
passi. Non si sceglie un LR a posteriori dalla migliore accuratezza EPN o posa val.

Memoria: il valore include le piccole copie diagnostiche dei gruppi selezionati,
quindi è conservativo, ma128batch non esauriscono tutte le forme del corpus.
Timing: non include checkpoint/pannelli, e le diagnostiche danno tempo al loader;
la stima del training include overhead storico e margine, non solo kernel timing.

**Ripiego dichiarato prima della lettura del collaudo (S4):** se il gate di memoria
fallisce, oppure la stima del percorso di produzione descritta sotto non lascia
il margine richiesto, il candidato tecnico è `accum_steps=2`, sempre batch globale
32, quattro rank e stessi batch/esposizione. Ogni rank divide le sue otto finestre
in due microbatch da quattro. Non si riduce il batch globale e non si aggiungono GPU.
È una variante da implementare, testare, revisionare e sottoporre a un **nuovo
collaudo da 128 batch**, con nuovo piano, SHA e costo pubblicati; nessun retry
automatico del job fallito. I denominatori della loss devono riferirsi allo stesso
batch globale; la loro equivalenza e quella del gradiente vanno verificate in un
test deterministico della ricetta, con tolleranze numeriche dichiarate. Non si
promette equivalenza bit a bit: ordine delle somme e consumo RNG del dropout possono
cambiare suddividendo il batch. L'eventuale confronto manterrà esplicito questo
scostamento tecnico dal controllo storico a `accum_steps=1`.

## Training scientifico5.000 e valutazioni necessarie

Il wrapper futuro ricostruirà il CLI T100-G cambiando solo preset300m e cartella
nuova; pin del codice scientifico, configurazione, input, schedule e stream.
Una ricevuta completa richiederà `summary.stopped=max_steps`, step5000, assenzaSTOP,
stato student/teacher/AdamW/RNG integro e checkpoint nominato immutabile, come nel
wrapper tokenizer. Nessuna selezione del checkpoint sullo sviluppo: si legge5000.

Per ogni taglia si usa teacherEMA, stessa popolazione/supporto di posa e stessi
split e fold P1/P2. La posa30M a5000 ha già cache e ridge; la classificazione30M a5000
richiede **nuove estrazioni**, non un nuovo training. Non si confronta300M5000 con
sonde30M19618. Gli score19618 esistenti non scelgono iperparametri della nuova prova.

| Lettura | Input30M /300M | Endpoint e controlli |
|---|---|---|
| Posa ridge istantanea |sequenza patch×384 /patch×896, media dei64slot|crop6, LNstatelessFP32, ridgeFP64, alpha solo40utenti train, stessi15utenti val; primario E_dyn, correlazione e varianza predetta come diagnostiche|
| Classificazione |pool768 /pool1792 FP32|stessa aggregazione backbone+locale, StandardScaler solo foldtrain, stesso C-grid, max_iter12.000 e audit convergenza; primario EPN-612, P1/P2/UCI secondari|
| Head posa/fixed panels |testa originale di ogni modello|train-only per dose/stabilità; non sostituiscono generalizzazione della ridge|

La diversa dimensione del readout fa parte della prova del modello più grande:
non isola «solo più profondità a uguale capacità della sonda». Nessuna PCA aggiunta
silenziosamente. Alpha eC si selezionano internamente e allo stesso modo; i benchmark
CPU precedono i fit, senza lettura degli score di sviluppo per fissare le risorse.

**Vincolo implementativo esplicito:** `tokenizer_evaluation.py`,
`tokenizer_pose_readout.py` e `tokenizer_probe.py` oggi verificano D384/pool768,
trebracci e, per la classificazione, solo19618. Le funzioni di estrazione sottostanti
usano la dimensione del checkpoint, ma le guardie di campagna non possono essere
bypassate. Si prepareranno controller/validatori della nuova campagna che registrino
D896/pool1792 e dose5000, con regressioni contro gli stessi calcoli aD384.
Il training scientifico non viene lanciato finché queste letture e i loro costi non
sono concretamente pronti e revisionati. Questo documento non afferma che lo siano.

### Requisiti prima dei 5.000 passi, dopo la review di Claude

**S1 — Identità del training.** Il piano esecutivo del 300M deve includere un
dizionario `scientific_source_sha256` verificato rispetto al commit
`afb644ea10ecc2a716160b4d8a4904e805c366fb`, che ha prodotto il controllo 30M.
Si vincolano tutti i file `src/wearusfm`, `scripts/train_jepa.py` e i file di
launcher/setup riusati che possono influire sull'esecuzione. Il diff è vuoto nel
collaudo `05d4bf0`, ma questo non basta per un futuro commit: il wrapper ricontrolla
i byte al lancio. I nuovi wrapper e valutatori hanno pin propri; non sono esentati
dalla review perché riusano il trainer. Variazioni necessarie vanno enumerate e
testate, senza dichiarare identità con il controllo se non sussiste.

**S2 — Disponibilità del controllo.** Il checkpoint nominato
`$SCRATCH/wearusfm_runs/runs/tokenizer_20261103/T100-G/checkpoint_step0005000.pt`,
compreso il teacher, e le cache di posa sono soggetti alla retention dello scratch.
La data di scadenza indicata nella review è **13 dicembre 2026**: la si tratta come
limite operativo da verificare sui metadati reali di ciascun artefatto, non come
garanzia che tutti abbiano identica data. Prima di inviare il 300M si verifica
l'esistenza e lo SHA del controllo, degli snapshot di configurazione, ricevute e
cache necessarie. Le estrazioni P1/P2 del 30M5000 vengono completate e verificate
entro quel limite; se il calendario non lo garantisce, si conserva prima una
copia in storage con retention adeguata, verificando quota e spazio e poi SHA
identico all'originale. Il piano registra il nuovo percorso e la catena di
provenienza. Non basta un log del checkpoint: senza il controllo verificato la
fase scientifica non parte.

**S3 — Memoria della produzione.** Nel summary storico T100-G30M5000 la memoria
massima allocata, ridotta con MAX fra i rank, è **15,473958969 GiB**. Il collaudo
nuovo non esercita prefetch, pannelli di posa e valutazione fissa come il trainer.
Dai massimi del collaudo, nelle medesime unità, si calcola e si pubblica:

`M300_prod_proxy = M300_preflight × max(1, 15.473958969 / M30_preflight)`.

Il fattore non riduce la stima se il collaudo30M ha copie diagnostiche che lo rendono
più oneroso della produzione. La stima deve stare entro il 90% della memoria GPU
effettivamente rilevata; non si assume esattamente 64 GiB dal nome del nodo.
Questo è un controllo prudenziale, **non una prova del picco massimo**: forme del
corpus e attivazioni dei pannelli possono scalare diversamente. Prima dei 5.000
serve anche una verifica breve del percorso reale con prefetch e diagnostiche
abilitati, comprese le fasi che producono il picco, e relativo preventivo separato.
Se la stima o la verifica superano il margine, si valuta il ripiego S4; non si
spengono i pannelli o si cambia batch per far passare silenziosamente il gate.

**S5 — Controllo secondario a dimensione pareggiata.** Lo fissiamo ora, prima dei
risultati300M, senza sostituire i primari a dimensione naturale. Si usa una
proiezione casuale fissa, indipendente da segnali, etichette, soggetti e fold:

- Generare in FP64 due matrici gaussiane standard di forma896×384, con NumPy
  `Generator(PCG64(seed))`, semi **2026110401** per il blocco backbone e
  **2026110402** per il blocco locale. QR ridotta; rendere positiva la diagonale
  diR cambiando coerentemente il segno delle colonne diQ. Le colonne diQ sono
  ortonormali; nessuna riscalatura aggiuntiva.
- Salvare le due Q in FP32, con manifest di algoritmo/versioni/seed e SHA dei byte,
  prima delle letture downstream. Tutti i dataset e fold usano gli stessi file
  congelati; non si rigenera o sceglie una matrice in base agli score.
- Per la classificazione300M, separare esplicitamente i blocchi backbone e locale
  di pool1792, applicare ciascuna Q al proprio blocco, poi concatenare a768 valori.
  Verificare ordine/identità dei blocchi nell'estrattore. Lo StandardScaler viene
  stimato **dopo** la proiezione, solo sul train del fold, come per il controllo
  30M a768. C-grid, selezione e convergenza rimangono invariati.
- Per la posa300M, applicare la stessa Q del backbone a ogni patch della cache
  media-slot, convertita FP32, passando896→384 **prima** della LayerNorm stateless
  FP32. Seguono lo stesso crop6, supporto, ridge e selezione alpha train-only del
  controllo30M. Il tempo resta conservato. Il 30M usa le viste originali a384/768,
  senza una nuova trasformazione o un nuovo training.

La proiezione non è stimata sui dati e non crea leakage; scaler, scelta diC/alpha
e fit rimangono interni al train come nelle letture principali. I confronti
secondari sono300M compresso−30M originale e300M naturale−300M compresso sugli
stessi soggetti/OOF, con bootstrap appaiato. Un beneficio che sopravvive alla
compressione sostiene che non dipende soltanto dalla maggiore larghezza della
sonda; non isola in senso assoluto «quantità di informazione» e «capacità». Un nullo
può dipendere dalla proiezione, quindi non esclude informazione nel300M naturale.
Una sola proiezione fissata, nessuna ricerca del seed o passaggio aPCA dopo gli score.
Questo aggiunge fit CPU, **nessuna estrazione GPU**. Prima dei 5.000 si congelano
implementazioni, regressioni numeriche, matrici, launcher, formule e tetti cumulativi
che comprendono le tre viste. I benchmark sulle cache reali del 300M richiedono
il checkpoint a 5.000: si eseguono dopo le estrazioni, sul solo train, prima dei fit
scientifici. Il preventivo misurato deve rientrare nei tetti già pubblicati;
altrimenti la fase CPU si ferma prima dei fit. Questa precisazione operativa,
coerente con la review Claude `a988a72` §2.3, elimina una dipendenza circolare;
non dichiara un benchmark già eseguito, non anticipa gli score e non cambia le sonde.
La readiness precedente al training attesta preparazione e limiti, non tempi
misurati su rappresentazioni che ancora non esistono.

**S6 — Ottimizzazione e significato.** LR1e-4 eclip1 restano fissati nella prima
ricetta; il risultato riguarda «maggiore taglia con questa ricetta», non l'effetto
della capacità indipendentemente dall'ottimizzatore né il miglior300M possibile.
Lo stress vede instabilità grossolane. Un problema concreto apre una calibrazione
train-only con piano separato, mai una scelta suEPN o posa val. Soglia di rango
assoluta38,4 invariata. Sono limiti di lettura e gate di stabilità, non un veto
basato sulle vecchie sonde30M.

**S7 — Appaiamento e rumore fra run.** Confermati bootstrap per soggetto,2000
estrazioni, seed20261102, supporti comuni e margini descrittivi già fissati.
La coppia storicaT25-L−B19618 è un confronto di contesto, **non una stima del
pavimento di rumore fra training**: un'unica coppia a seme nominale uguale non
stima la distribuzione della variabilità fra run. Non si sottrae il suo delta,
non si introducono soglie0,2–0,9pp e non si dichiara «indistinguibile dal rumore»
un delta300M−30M soltanto perché ha quell'ordine di grandezza. Gli IC rimangono
condizionati ai checkpoint e alle sonde di questa prova; servono repliche di
pretraining indipendenti per misurare la variabilità fra run. Questa precisazione
è coerente con l'analisi già pubblicata nel commit `8ee6603`.

**S8 — Lettura a due lati e conseguenza operativa a 5.000.** Complemento di Claude
nel commit `2f3207b`, recepito prima di qualunque estrazione 300M. Si conservano
due primari distinti; non si sceglie quello favorevole dopo i risultati.
Entrambi usano teacher e dose 5.000 contro 5.000. `E_dyn` è l'errore della ridge
istantanea sulle finestre dinamiche, con il supporto già fissato; non la loss di train.

| Misura (300M meno 30M) | Migliore | Peggiore | Irrisolto |
|---|---|---|---|
| EPN-612, delta in punti percentuali | limite inferiore IC95 > 0 | limite superiore IC95 < 0 | altrimenti, compresi i limiti uguali a zero |
| E_dyn, delta del rapporto di errore | limite superiore IC95 < 0 | limite inferiore IC95 > 0 | altrimenti, compresi i limiti uguali a zero |

Per la posa, la **rilevanza pratica** è una seconda etichetta: riduzione relativa
puntuale `(E30 − E300) / E30 ≥ 0,02` e IC del delta interamente sotto zero, come
nel criterio tokenizer. Non richiede un IC interamente oltre il 2%, né crea un
nuovo margine di danno del 2%. Un miglioramento inferiore al 2% resta misurato ma
non viene chiamato movimento migliorato in misura pratica. Nessun margine di 1 pp
viene trasferito al singolo endpoint EPN.

P1 e P2 restano controlli distinti: per ciascun IC95 `[L,U]` del delta in pp,
`L > −1` sostiene conservazione, `U < −1` sostiene perdita oltre margine;
altrimenti la non inferiorità è irrisolta. Le uguaglianze sono irrisolte.
Un punto sotto −1 con IC che attraversa −1 non è una perdita dimostrata oltre margine.

| Esito della matrice completa e valida | Raccomandazione fissata prima dei risultati |
|---|---|
| Almeno un primario migliore, nessuno peggiore; nessuna perdita P1/P2 oltre margine | Proporre 19.618 come conferma alla dose più lunga. Segnalare separatamente se il guadagno di posa è sotto il 2% e se la NI è ancora irrisolta. |
| Entrambi i primari irrisolti; nessuna perdita P1/P2 oltre margine | Proporre 19.618 per chiarire la domanda sulla dose; nessuna evidenza di beneficio a 5.000 rivendicata. |
| Qualsiasi primario peggiore, oppure perdita P1 o P2 oltre margine | Mettere in pausa questa ricetta e rivedere diagnosi e impiego delle risorse prima di proporre la continuazione. Vale anche per primari discordanti. |
| Artefatti, pairing o convergenza dei lettori non validi | Correggere la valutazione; nessuna lettura scientifica o decisione basata su score incompleti. |

La pausa riguarda questa ricetta e questo investimento; **non boccia la taglia
300M o lo scaling in generale**. Un esito irrisolto non diventa equivalenza.
Precisazione prima degli score, dopo la review Claude `5e4eb8a`: la review della
pausa deve pesare anche l'entità del peggioramento e la dose di soli 5.000 passi,
non soltanto il segno. Non cambia la riga applicata dalla tabella e non introduce
una nuova soglia. La prosecuzione a 19.618 risponde alla domanda sulla dose;
non misura la variabilità fra training indipendenti e non elimina quel limite
degli intervalli, anche quando restringe l'incertezza della lettura downstream.
NI irrisolta non equivale a perdita oltre margine e non blocca da sola una
proposta di approfondimento. I secondari a dimensione pareggiata aiutano a
interpretare i primari ma non ne sostituiscono l'esito per cambiare riga della tabella.

Bootstrap appaiato per soggetto, 2.000 estrazioni, seed 20261102; stesse popolazioni,
righe, fold e supporti entro ciascun endpoint. Gli IC sono marginali e condizionati
al singolo seme di pretraining e ai dati di sviluppo: nessuna garanzia congiunta
del 95% per questa regola su più endpoint, nessuna inferenza da un noise floor T25−B.
Non attribuiamo alla regola una probabilità numerica di falsa pausa: endpoint
correlati e margini P1/P2 diversi da zero non giustificano il semplice conto di
quattro test indipendenti sotto una stessa ipotesi nulla.

**Tetto della possibile seconda tranche.** La regola di costo viene congelata ora,
prima delle estrazioni. Sia `t5k` il tempo completo del trainer **300M** a 5.000,
e `u300` la media dei 112 update misurati del collaudo. Per i 14.618 update aggiuntivi:

`secondi = ceil_900(14618 × max(t5k / 5000, u300) × 1,20 + 1200)`;
`GPU-ore training = 4 × secondi / 3600`.

Non si usa la misura più favorevole. Prima di aprire gli score downstream a 5.000
vanno pubblicati il tetto numerico ottenuto, le eventuali segmentazioni imposte
dallo scheduler e i tetti separati per estrazioni e lettori CPU, da benchmark
tecnici senza score. Se questo passaggio manca, la lettura descrittiva può essere
pubblicata ma la prosecuzione non può essere presentata come scelta interamente
pre-registrata. Nessun preventivo viene aumentato perché uno score è promettente.

La tabella decide **quale proposta fare**, non concede una sottomissione automatica
a 19.618: quella tranche resta esterna alla prima autorizzazione, richiede piano
concreto, risorse e review, e la decisione di Simone. Il protocollo non riscrive
questa distinzione. Implementazione della lettura:
`scripts/scaling_reading.py`; policy versionata in
`results/infra/scaling_300m_20261104/scientific_reading_policy.json`.

La matrice pre/postbackbone e39.237/49.046 approvata per lo schedule lungo30M non
viene aggiunta automaticamente a questo screening. Qui resta la lettura storica.

## Risorse: collaudo fissato, seguito da misurare

Politica versionata: `results/infra/scaling_300m_20261104/resource_policy.json`.
Le cifre provvisorie tengono separati training, estrazioni e CPU. Non sono prenotazioni
né autorizzano a inviare fasi non ancora pronte.

| Fase | Risorse massime | Costo |
|---|---|---:|
| Checkout congelato |2CPU,4G,10min|0,333coreh|
| Collaudo30/300M sequenziale |4A100,32CPU,480G,30min|2GPUh +16coreh allocati|
|300M0→5000, stima provvisoria |4A100,32CPU,480G, fino6h|circa17GPUh attese da rapporto storico; tetto24GPUh da aggiornare sul benchmark|
| Estrazioni300M posa/P1/P2 +30M P1/P2 a5000 |1GPU/8CPU perjob; nessun train30M|inviluppo8,25GPUh, da verificare dopo benchmark|
| Ridge/sonde CPU e controlli |fino8CPU/30G; shard fino120min|inviluppo180coreh CPU-only, da restringere prima dei fit|

Il rapporto storico15/10 2,769/0,707≈3,92 è solo una stima iniziale: ricetta precedente.
Il T100-G30M5000 corrente registra3883,030s nel trainer; non confondere con una
misura SLURM completa. Dopo il collaudo si usa il rapporto misurato e il massimo
fra tempo perupdate300M e tempo completo30M×rapporto, +20% e1200s di margine,
arrotondato ai15min superiori. Il nuovo preventivo è pubblicato con commit/push
prima della sottomissione. CPU nei jobGPU contabilizzata separatamente.

Precisazione F7, senza cambiare la policy del collaudo: il rapporto misurato usa
la **media dei112 aggiornamenti** conservati per taglia,56nativi+56stress, dopo
aver escluso i primi8di ciascuna fase. Per ogni aggiornamento il tempo è la somma
del massimo fra rank del forward/backward e del massimo fra rank di
clip/AdamW/EMA; poi si fa la media e infine il rapporto300M/30M. Non si sceglie
la fase con il rapporto più favorevole. I due riassunti di fase restano disponibili
come diagnostica. Nel preventivo futuro questa definizione entra anche nel piano
esecutivo; nessuna modifica al JSON già approvato per il collaudo.

L'estrazione300M e la ridge possono costare più del rapporto dei parametri: per la
ridge D896/D384 implica circa12,7× nel termine cubico, mentre il classificatore
riceve2,33× feature. I benchmark reali determinano gli shard; niente invii CPU
basati soltanto sull'etichetta «inferenza» o sull'assenza diGPU.

## Cosa autorizza a concludere e seguito19.618

Contrasto appaiato300M−30M a5000 su OOF o utenti posa comuni, bootstrap2000 con seed
20261102 come nella campagna precedente. IC condizionati a questo seme, checkpoint
e sonde. Soglia pratica posa2% relativo e margineNI1pp P1/P2 sono riferimenti
descrittivi mantenuti, non prove definitive di superiorità di una famiglia.
Un solo seme e dati di sviluppo; EPN positivo non isola causalmente la frequenza
di campionamento, posa su utenti nuovi resta lo stesso task supervisionato.

Un 5.000 negativo non boccia lo scaling: dose ancora iniziale, ottimizzazione non
dimostrata ottimale, rappresentazioni sondate con lettori specifici. Il seguito
si propone **secondo la tabella S8**, senza selezionare dopo gli score una regola
più favorevole: esiti migliori o irrisolti, in assenza di peggioramenti dei primari
o perdite P1/P2 oltre margine, motivano le due diverse proposte; un peggioramento
richiede la pausa operativa della ricetta. Se deliberato, il confronto a 19.618
usa il 30M già disponibile e conserva run/schedule del 300M con altri 14.618 passi.
La vecchia stima di 49,40 GPU-ore è soltanto storica: prevalgono formula e tetti
pubblicati prima degli score come in S8. Nessuna prosecuzione implicita durante
la lettura di 5.000.
