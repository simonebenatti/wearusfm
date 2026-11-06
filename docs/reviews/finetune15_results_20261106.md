# EPN15 e valutazione lunga: lettura scientifica GPT del 06/11/2026

Consegna verificata: Claude `3ba5e6437fcda6e5149000cee3fce5374e46b7de`,
registro `440d165796e3728ffe718a3e5e6017563e275aab`. Review locale dei risultati
pubblicati e delle copie OOF raccolte da Claude. Nessun accesso operativo al
cluster, nuovo fit, inferenza, deploy, invio o monitor di GPT.

**EPN15: raccolta e conti confermati; lettura favorevole al pretraining in
questa ricetta, con vantaggio modesto. Valutazione lunga: nessun confronto
scientifico finale pubblicato nei riferimenti esaminati.**

## Risposta alle domande scientifiche

Le due dosi downstream partono dal medesimo teacher T100-G30M a **19.618 passi**.
Questa prova non misura ancora il beneficio di portare il pretraining a49.046.
EPN e' escluso dal pretraining, ma la popolazione valutata e' sviluppo gia'
utilizzato nelle sonde:314soggetti,47.100finestre,6classi, quattro fold separati
per soggetto, calibrazione non supervisionata delle prime10finestre per utente.

| Pipeline | 5 epoche | 15 epoche | Differenza15−5, pp e IC95% |
|---|---:|---:|---:|
| Encoder congelato, testa LayerNorm+Linear |70,58%|71,42%|+0,85 [ +0,64; +1,04 ]|
| Pretraining e fine-tuning dell'encoder |93,77%|93,89%|+0,13 [ −0,19; +0,45 ]|
| Stessa architettura da inizializzazione casuale |92,36%|93,06%|+0,70 [ +0,29; +1,09 ]|

Il contrasto primario **FT−scratch a15 e'+0,834pp, IC95[+0,416;+1,263]**.
E' positivo in tutti i fold: +0,928/+0,616/+0,530/+1,265pp. La riduzione relativa
dell'errore aggregato rispetto a scratch e'circa12%, descrittiva. La differenza
assoluta e' inferiore a un punto: risultato utile, non un grande salto.

Il vantaggio a5 era+1,403pp. **La differenza dei vantaggi e'−0,569pp,
IC95[−1,051;−0,051]**: secondo la regola congelata sostiene un restringimento.
L'estremo superiore e' vicino allo zero; il risultato non va trasformato in
una prova forte di una legge sulla dose. Non si sostituiscono seed, numero di
bootstrap o criterio dopo avere visto questa vicinanza.

La lettura piu' utile e' che una parte del vantaggio breve si riduce dando alla
rete casuale una ricetta piu' lunga, mentre un vantaggio resta a15. E' compatibile
con un beneficio dell'inizializzazione preaddestrata nell'adattamento supervisionato;
non dimostra gli ottimi raggiungibili, una maggiore efficienza totale di calcolo
(il pretraining ha un costo), ne' che scratch raggiungerebbe FT con altre epoche.

**Rettifica della formulazione dell'operatore:** sostituire «finetune resta
dov'era» con «non emerge un guadagno risolto del fine-tuning passando alla ricetta
a15: +0,127pp, IC95[−0,187;+0,450]». Un intervallo che comprende zero non prova
uguaglianza o convergenza. Il flag della coda di validation falso in tutti i
regimi resta coerente con il protocollo e non cambia questa conclusione.

## Cosa dice il frozen

Il contrasto FT−nuovo frozen e'+22,473pp [21,212;23,641]. Riguarda questa testa e
questa ottimizzazione: **non e' il guadagno sul miglior lettore congelato noto**.
La logistica storica e'82,577%; il frozen15 resta11,157pp sotto, mentre FT15 e'
11,317pp sopra quel riferimento. Queste due distanze sono descrittive, senza IC
appaiato derivato dal solo aggregato storico.

Allungare questa ricetta ha dato solo+0,845pp al frozen: non basta a spiegare
l'ampia distanza dal lettore storico. Non possiamo attribuirla tutta alle
rappresentazioni: il riferimento storico dimostra che un altro lettore ne
estraeva piu' informazione. Non e' pero' provato che ulteriore ottimizzazione
non aiuterebbe, ne' quale componente (normalizzazione, regolarizzazione o
ottimizzazione) sia responsabile. Questa prova non isola tali componenti.

L'architettura, adattando l'encoder, apprende bene EPN anche da zero. Quindi le
sonde congelate deboli non dimostravano che l'intero modello non fosse addestrabile.
E' ancora aperta la qualita' generale delle rappresentazioni trasferibili.

## Limiti da tenere accanto al risultato

- Le pipeline selezionano fra due candidati. FT sceglie encoder1e−4/testa1e−2,
  scratch encoder1e−4/testa1e−3 in tutti i fold. Il LR encoder coincide, quello
  della testa no; entrambi gli encoder scelti sono al bordo delle rispettive
  griglie. Non e' un confronto a identico ottimizzatore, ne' fra ottimi di tuning.
- Le15epoche ripartono dalle inizializzazioni originali e cambiano anche
  l'orizzonte del coseno. Non isolano causalmente il numero di aggiornamenti.
- Un seme; IC descrittivi appaiati per soggetto, condizionati ai modelli e ai fold,
  senza incertezza di riaddestramento. Secondo sguardo sui medesimi held di sviluppo;
  nessun test ufficiale. Nessuna soglia V-R1 importata in EPN15.
- Il guadagno non e' uniforme:157soggetti meglio,53pari,104peggio;4classi su6
  positive. Il maggior contributo e'«open» (+3,796pp);«pinch»+0,981pp,
  «fist»+0,115,«waveIn»+0,318,«noGesture»−0,038,«waveOut»−0,166.
  Descrizioni senza nuovi test per sottogruppo o selezione di endpoint.
- Non e' una dimostrazione su altri dispositivi, dataset o regressione. La posa
  usata nel pretraining non sostituisce una regressione di trasferimento nuova.

## Raccomandazione

Chiudere EPN15 alla dose concordata. Non propongo una terza dose o una nuova
griglia EPN guidata da questi held. Proseguire la valutazione lunga dopo la
risoluzione del gate delle risorse e preparare il confronto su altri dataset,
includendo una regressione, come gia' richiesto da Simone. La prossima domanda
e' quanto il vantaggio sopravviva al cambio di compito e popolazione; per un
nuovo protocollo si fissano opportunita' di tuning e popolazioni prima dei risultati.

## Valutazione lunga e tempi

Le12estrazioni sono pubblicate e verificate dall'operatore. Il registro440d165
riporta12benchmark inviati su16, con DB2/pre e DB2/post completi; e' uno snapshot
pubblicato, non una lettura corrente della coda. Sono misure train-only dei tempi,
senza punteggi validation/held. Nessun verdetto49k−19k si puo' trarre da queste.

DB2/pre:1.422s SLURM su1.800, margine378s. La ricevuta locale raccolta daClaude
contiene6fit per1.399,339s; il massimo, C=100, richiede829,720s e3.113iterazioni,
senza mancata convergenza. Questo descrive il costo del lettore, non la qualita'
della rappresentazione.

**Il problema successivo e' gia' dimostrabile, non solo un rischio di TIMEOUT:**
la formula congelata del gate dei fit usa somma dei6tempi+massimo, scala per righe
e classi di ogni fold, prende il massimo col riferimento storico e applica i
margini. Anche ignorando il riferimento storico, le coppie DB2/pre richiedono
almeno21.515,645s e20.454,240s, contro13.800s ammessi. Questi sono preventivi
prudenziali, non durate effettive future. Le6celle fisiche DB2/pre delle3dosi
non passano; il gate globale mantiene bloccati tutti i48fit di classificazione.

Serve quindi un emendamento operativo prima dei fit, non un allentamento del
gate dopo un errore. Aspettare le15ricevute di classificazione complete permette
di dimensionarlo una volta per tutte. Valutare shard per singolo fold e risorse
adeguate, senza modificare griglia, dati, solver, criteri o fattori prudenziali.
Per il solo DB2/pre, quattro shard da un fold invece di due coppie aggiungerebbero
6job:88job di calcolo piu'4checkout, tettoCPU1.903,67coreh (+192coreh allocate),
a parita' di9GPUh se si preservano le estrazioni. I limiti inferiori per singolo
fold sono10.632/11.485/10.430/10.626s. Il pavimento storico passa anch'esso solo
se il lavoro misurato nella vista pool e'almeno667,482s: ricevuta ancora da leggere.
Quindi questa e' un'opzione dimensionata parzialmente, non una promessa che tutti
i nuovi gate passeranno. La riserva aggiuntiva per benchmark30→60min sarebbe
4coreh per job; non risolve il gate dei fit. La documentazione locale indica
4h sulla partizione seriale: non assumiamo senza verifica che accetti6–8h.
Gli invii gia' fatti restano sotto la gestione diClaude; questa review non
dispone cancellazioni, rilanci o cambi ai checkout attivi.

Il cambio non si risolve editando soltanto il JSON: limiti, shard e conteggi
sono controllati anche dai sorgenti, inclusi nella firma piena. Con la regolaR-2
attuale una modifica coperta invalida le estrazioni. Un eventuale emendamento
deve dunque prevedere e far revisionare **prima** una compatibilita' esplicita e
ristretta per riusare le cache e i benchmark validi, con identita' e calcoli
scientifici invariati, oppure contabilizzare nuove estrazioni. Non basta
riscrivere un hash. Preferisco conservare gli artefatti validi con una verifica
dedicata, senza rigenerarli soltanto per cambiare la confezione dei job.

Questa e' la raccomandazione, non un emendamento eseguibile o un nuovo invio.
Il dimensionamento completo e il costo richiedono le ricevute mancanti raccolte
daClaude; un TIMEOUT resta esito incompleto da diagnosticare, senza retry cieco.

## Verifica e consuntivo

L'aggregatore sulle OOF locali e sul manifest pubblicato riproduce byte per byte
il report SHA256 `b7927ea34bfe9bb3adda1eef0338ae3b8f4dd9c35739d5352e44c1151f865688`.
Il report verifica84file e24hash di checkpoint remoti, senza caricare pesi.
Un secondo calcolo direttamente dalle OOF, senza importare gli aggregatori,
conferma selezione validation di tutte12celle, supporto e pairing dei6pannelli;
gli score e gli estremi dei contrasti differiscono al massimo3,91e−14pp.
Il bootstrap indipendente campiona direttamente i soggetti: la sua equivalenza
al macrorecall e' verificata dal supporto costante di25finestre/soggetto/classe.
Le evidenze di hash remoti, stato SLURM e assenza di riavvii sono quelle archiviate
daClaude; GPT non ha ripetuto controlli sul cluster.

Consuntivo ricalcolato dai27record archiviati: **197,272500GPUh /1.578,211667coreh**,
contro304,5GPUh/2.436,666667coreh. Residuo107,227500GPUh, non automaticamente
riassegnato. Nessuna anomalia bloccante nella raccolta EPN15.

Gli artefatti della verifica locale sono in
`results/infra/finetune15_20261105/science_review/`; dati e OOF rimangono fuoriGit.
Script di verifica: `docs/reviews/check_finetune15_results_20261106.py` e
`docs/reviews/check_long_benchmark_gate_20261106.py`. La prova numerica del gate
e le due ricevute benchmark raccolte localmente daClaude sono conservate in
`results/infra/consolidation_long_20261105/science_review/`.

La rilettura indipendente del critico GPT conferma interpretazione e dimostrazione
del gate, senza blocchi. Nessuna modifica ai sorgenti scientifici, ai piani
esecutivi o ai pin della campagna in questa review.
