# Review Claude: TCN come lettore della posa a encoder congelato, proposta di GPT `c15aac9` (17/11/2026)

Richiesta di Simone, 16/11: «ok, adesso prepapariamo unprompt per GPT in cui gli facciamo provare ad usare temponet come testa per
la regressine nel FM». Prompt di Claude: `docs/prompt_gpt_temponet_testa_fm_20261116.md` (`8917e2a`). GPT consolidamento ha
risposto con la proposta `c15aac9`, sul ramo `codex/pose-temponet-proposal-20261117`, il 17/11 alle 08:28, in stato
REVIEW_REQUIRED. Contiene il protocollo `docs/pose_temponet_frozen_protocol_20261117.md`, i materiali in
`results/infra/pose_temponet_proposal_20261117/` e le voci nel registro e in `docs/decisioni.md`. Il prompt di GPT per Claude
(`docs/prompt_claude_pose_temponet_review_20261117.md`) chiede sei verdetti separati. Nella chat del 17/11 Simone ha incollato due
risposte di GPT e ha scritto: «converge poi con te alla fine».

Ruolo di Claude: revisore. Nessun job, nessun codice, nessun file di GPT modificato. La punta del ramo di GPT è stata controllata
alle 09:37, alle 11:34 e alle 12:37, subito prima del commit di questa review: sempre
`c15aac9bcf291f4ca2e7757dd248ff61045040df`.

## Verdetti

| Verdetto chiesto da GPT | Esito | In breve |
| --- | --- | --- |
| 1. Domanda e architettura | **APPROVATO CON CORREZIONI** | Il livello congelato è giusto e l'adattamento è motivato: TEMPONet così com'è non dà un'uscita per patch. Conti, campo ricettivo e crop6 verificati. Una correzione: TCN e MLP non contengono L0 (A-1) |
| 2. Controlli e attribuzione | **APPROVATO CON CORREZIONI** | I bracci bastano per la domanda circoscritta e nulla pretende un vantaggio del pretraining. Tre correzioni: controllo di sanità della ricetta comune (C-1), regola della candidata completa (C-2), perimetro dell'esito nel §10 e in decisioni.md (C-3). Vale anche qui la cautela di A-1 sulle classi non annidate |
| 3. Selezione e supporto | **DA RIVEDERE** | Dati, supporto e fold coincidono con lo storico, la loss ha la stessa riduzione di quella di posa del pretraining e la val resta chiusa. Un rilievo bloccante: con la ricetta proposta MLP e TCN possono restare lontane dal loro ottimo, e un «no» non sarebbe leggibile (S-1). Una correzione minore (S-2) |
| 4. Criteri | **APPROVATO CON CORREZIONI** | E_dyn, media dei semi, soglia, condizione sull'IC e famiglia reggono. La copertura vera degli intervalli è circa 97% per contrasto (nominale 99%) e 85-91% per la famiglia (nominale 95%) (nota criteri-6). Tre correzioni del testo da congelare: definire «passa» (K-1), regola completa in proposal.json (K-2), esito «non a convergenza» (K-3) |
| 5. Costo e fattibilità | **APPROVATO CON CORREZIONI** | I conti tornano e, con la ricetta proposta, i tetti sono larghi. Con le correzioni di S-1 il costo dei fit va ricalcolato e può superare il tetto di 3 GPU-ore. Prima del benchmark il gate va reso meccanico (P-1). Una frase da correggere sulla dipendenza da SCRATCH (note costo-8 e costo-consegna-N1) |
| 6. Consegna e seguito | **APPROVATO CON CORREZIONI** | Consegna solo documentale, nessun merge, ruoli lasciati a Simone. Tre correzioni prima della firma: cancello che riproduce i riferimenti (R-1), voce di congelamento completa e pinnata (R-2), oggetto della prova nella tabella del §10 (R-3) |

In tutto restano 1 rilievo bloccante, 12 minori e 50 note; un rilievo è stato rifiutato dai verificatori. Il livello e
l'impostazione (TCN accanto a ridge e PoseHead rifittata) sono quelli concordati in chat (sezione 1) e non vanno rifatti; bracci e
criteri reggono con le correzioni sotto. Prima della firma vanno riviste la ricetta di allenamento (S-1) e le parti di testo delle
sezioni 4 e 5, compreso il cancello R-1. La specifica delle teste può ancora cambiare: ramo lineare di A-1, oppure
regolarizzazione o taglia se il controllo positivo di S-1 fallisce. P-1 si chiude prima del benchmark.

## 1. Impostazione condivisa e rettifiche di Claude

Nella chat del 17/11 GPT e Claude sono arrivati alla stessa impostazione. Prima si prova la TCN come lettore sull'encoder
congelato, accanto alla ridge e a una PoseHead rifittata nelle stesse condizioni. Poi, come prova separata e non automatica, si
valuta la TCN dentro il pretraining. La proposta `c15aac9` è già la prima prova (livello A).

GPT ha corretto quattro frasi di Claude scritte in quella chat. Claude le accetta:
- Se la TCN non batte la ridge, si chiude questa prova. Non si dimostra che il limite sia nell'encoder: potrebbero non andare
  quella TCN, il suo contesto, la regolarizzazione o i dati per allenarla.
- Ridge e PoseHead vicine (E_dyn 0,786 e 0,779) non dimostrano che un lettore lineare sia al massimo.
- Una testa capace che fa il lavoro al posto del backbone è un rischio, non un esito certo. Nell'addestramento congiunto potrebbe
  anche dare gradienti migliori all'encoder; per distinguerlo servono lettori comuni sui due encoder.
- Lo scarto di E_dyn fra bracci storici, 0,47-0,90% (rilievo D-4 della review `0ab9d98`), non è una stima per eccesso del rumore
  fra training. Con ricette diverse effetto e variabilità si mescolano, e con tre o quattro bracci non si ricava un limite
  affidabile.

È giusta anche la «Correzione tecnica al prompt» del §2 (riga 23). Nel prompt `8917e2a` Claude aveva scritto «Campo ricettivo 378
passi: sulla griglia di 25 ms sono 9,45 s, piu' della finestra di 4 s». I 378 passi valgono per un elemento convoluzionale finale.
A T = 160 il flatten unisce due elementi distanti 64 passi, con un'unione nominale di 442. Misurata dai revisori, l'uscita della
TEMPONet trapiantata dipende da tutti i 160 passi con `causal=False` e dai soli passi 0-123 con il default `causal=True`. La
conclusione del prompt («piu' della finestra di 4 s») resta vera per il campo nominale: 378 e 442 superano i 160 passi, e con
`causal=False` l'uscita dipende da tutta la finestra. E, come scrive GPT alla riga 23, il trapianto diretto non ha il significato temporale dei 128 ms di TEMPONet raw.

Il rilievo S-1 rende verificabile la cautela del primo punto. Se la ricetta non fa convergere le teste, il «no» va riconosciuto come
difetto di allenamento, e non come esito scientifico, prima di chiudere la prova.

## 2. Materiali e metodo

- Oggetto della review: l'albero di `c15aac9`, esportato con `git archive` in una copia esterna al repository, e il codice
  TEMPONet sul ramo `claude/temponet-db8-20261109` (`8917e2a`).
- Sei revisori, uno per verdetto, e tre verificatori avversari, uno per coppia di dimensioni. Ogni verificatore ha provato a
  confutare i rilievi con metodi e script propri e ha riattaccato una parte delle affermazioni che i revisori davano per buone: 60
  delle 126.
- Il primo giro (09:01-09:26) si è fermato per il limite d'uso del modello, senza consegne. Il secondo (09:39-11:34) è ripartito
  dagli script del primo e li ha rieseguiti per i numeri che decidono i verdetti.
- Esiti della verifica: 56 rilievi giudicati (34 confermati, 21 confermati in parte, 1 rifiutato) e 8 rilievi nuovi. Delle 60
  affermazioni riattaccate, 57 reggono e 3 cadono (sezione 3).
- Una rilettura indipendente della bozza contro le fonti, con tre lenti (numeri, riferimenti, fedeltà), ha fatto 72 segnalazioni
  (8 errori, 57 imprecisioni e 7 problemi di forma), in parte doppie fra le lenti: gli errori distinti sono 6. Un controllo delle
  correzioni le ha trovate tutte risolte e ha segnalato altri 15 problemi, in parte nati dalle correzioni. Sono corretti in questa
  versione.
- Tutto è stato eseguito sul Mac, senza job: su CPU e, per alcuni micro-benchmark del costo, sulla GPU del Mac (MPS). L'unica
  lettura su Leonardo è un `ls -l` dei cinque file di input, fatto da Claude prima della review.
- Livelli di prova come nei rapporti: ESEGUITO, CALCOLATO, LETTO. Gli esperimenti su dati sintetici mostrano meccanismi, non i
  numeri veri dei latenti.
- Rapporti e script degli agenti: `docs/reviews/claude_review_pose_temponet_proposal_20261117_evidence/` (vedi il `LEGGIMI.md`).

## 3. Che cosa regge

Provenienza e input:
- `src/wearusfm/model/temponet.py` e `scripts/temponet_emg2pose.py` hanno byte e sha256 identici a `0c19c4b` e `8917e2a`, e il
  modello anche a `60b35bd`, come in `raw_source_identity.json`.
- I cinque file di input esistono su `$WORK` con i byte dichiarati (`ls -l` di Claude del 17/11). I pin di `proposal.json`
  coincidono con la ricevuta storica del fit (sha256 `b08e6e32…5037c14`). Gli sha256 dei file su `$WORK` non sono stati
  ricalcolati oggi.

Architettura:
- TEMPONet così com'è accetta solo (B, C, T) a lunghezza fissa e dà una posa per finestra. BatchNorm e pooling sono fissati nei
  blocchi privati.
- Conteggi: L0 8.468, MLP 65.300, L13 100.628 e TCN 65.204 parametri. MLP e TCN hanno 64.128 pesi nelle trasformazioni e
  10.260.480 MAC per finestra.
- TCN e L13 vedono 13 patch contigue (offset da -6 a +6); L0 e MLP leggono solo la patch centrale. I centri 6-153 non dipendono dal
  padding e coincidono con il crop6 storico. Tutte le finestre in cache hanno 160 patch reali.
- L0 è letteralmente la PoseHead (LayerNorm affine, eps 1e-5, uscita a zero) e ha la stessa classe di funzioni della ridge
  storica. Con l'uscita a zero i gradienti interni sono nulli al primo backward e non lo sono più dopo un passo di AdamW.
- Non c'è una ricerca nascosta di architetture.

Dati e supporto:
- Cache: 1.907 finestre e 40 utenti train, 18.548 finestre e 15 utenti val. Il supporto proposto è identico a quello storico della
  ridge e di E_dyn, con le stesse 8.327 finestre dinamiche.
- La loss ha la stessa riduzione di quella di posa del pretraining: media per finestra, poi media fra finestre. Le statistiche
  della posa vengono dai 158 utenti train e non contengono i val. I fold alternano gli ID ordinati, come nella ridge storica.
- Nella proposta attuale nessun dato dei 15 utenti val entra in selezione, normalizzazione o arresto, e il benchmark usa solo il
  train.
- Le predizioni storiche della ridge e della testa sono salvate per finestra e patch sul crop6 dei val: i contrasti 4 e 5 si fanno
  senza nuovi fit.

Criteri:
- Con codice proprio, partendo dalle somme per utente della ricevuta pinnata, i revisori hanno riprodotto i due E_dyn storici:
  0,7859020004638595 (ridge post) e 0,7786289843513977 (testa allenata). Il ricalcolo dai file delle predizioni resta da fare: è il
  cancello R-1.
- La media degli accumulatori per utente fra i tre semi è esattamente la media aritmetica dei tre E_dyn, anche in ogni replica
  bootstrap. Non è un ensemble di predizioni.
- La soglia del 2% e la condizione «limite inferiore aggiustato >0» (riga 95) coincidono con il criterio del 02/11, che la scrive
  come limite superiore del delta assoluto < 0. Qui l'IC è al 99% per contrasto (Bonferroni); nel criterio del 02/11 era al 95%.
  La regola è a due lati, è fissata prima e non deduce equivalenze da un esito non significativo.
- In simulazione l'intervallo percentile al 99% copre davvero circa il 96-97% per contrasto e l'85-91% per la famiglia (nota
  criteri-6).

Costo:
- Le cifre tornano: 4 GPU-ore come somma delle fasi, 12 core-ore solo CPU, 44 core-ore allocate e 44 ore locali col billing
  registrato.
- Con la ricetta proposta i tetti dei fit (3 GPU-ore) e dell'inferenza (0,5) sono larghi. Anche con l'analogo peggiore del
  progetto, cioè TEMPONet raw con cuDNN deterministico in eager (18,5 ms per passo), i 75.000 passi della matrice dei fit,
  raddoppiati, fanno circa 0,77 GPU-ore. Con le valutazioni interne restano sotto un'ora. I micro-benchmark sintetici del revisore
  danno 0,27-0,69 ore. Con le correzioni di S-1 il costo va ricalcolato (vedi S-1).
- Gli NPZ sono compressi e la RAM proposta basta. Le 15 predizioni val in FP32 occupano 3,32 GiB.

Consegna:
- La consegna è solo documentale: nessun merge, nessun file di Claude o della linea 3 toccato. Ruoli e budget restano a Simone.
- Il piano del codice del livello A non tocca `src/wearusfm`, `FMConfig`, `PoseHead` o i config: è compatibile con le guardie I-1
  della linea 3.

Le tre affermazioni che cadono:
- «Sui dati finti [...] il disegno sa vedere una struttura temporale non lineare quando c'è» (revisore dell'architettura). Dipende
  dal mondo sintetico: vedi S-1.
- Due revisori avevano accettato la frase del §9, riga 169: «I payload A sono su WORK e non dipendono dal rinfresco SCRATCH del
  03/12». Il revisore del costo la dava per buona con la sola eccezione possibile del file delle statistiche; quello della consegna
  senza eccezioni (la sua parte sul fermo del 23-25/11 regge). La frase non regge per intero. Il file delle statistiche della posa
  e i riferimenti di posa del pilot B19618 (`train_subset.npz` e `val.npz`), che il codice della valutazione lunga apre per
  validare le cache, stanno solo su SCRATCH (note costo-8 e costo-consegna-N1). Le predizioni storiche di ridge e testa dei
  contrasti 4 e 5 sono invece su `$WORK`.

## 4. Da rivedere prima della firma

### S-1. La ricetta può lasciare MLP e TCN lontane dal loro ottimo (bloccante; selezione-1)

Che cosa dice la proposta. La ricetta del §4 (righe 63-75) usa AdamW con weight decay 0,01, LR 1e-4 o 1e-3 costante, al più 50
epoche e batch da 32 finestre: circa 30 passi per epoca nei fold e 60 nel rifit. Gli strati interni hanno l'inizializzazione
standard di Torch e l'uscita parte da zero (riga 35). La riga 75 dice: «Se il migliore è all'ultima epoca o al bordo LR si riporta;
non si estende automaticamente la ricerca». Alla riga 98 l'esito inconclusivo chiude il pilot senza promozione. Per la riga 100 la
TCN è candidata solo se passa, e per la riga 118 il livello B non è il seguito automatico. Ogni esito diverso dal beneficio chiude
quindi la prova.

Che cosa hanno trovato revisori e verificatori:
- In due mondi sintetici costruiti in modo indipendente, con la ricetta proposta la TCN perde contro L0 anche quando c'è un
  guadagno da prendere:
  - nel mondo temporale del revisore la TCN fa -5,2% di E_dyn rispetto a L0, mentre il tetto teorico era +8,7%;
  - nel mondo V del verificatore il primario vale -3,1%. Il mondo è tarato per somigliare ai numeri veri: ridge CV 0,7555 contro
    0,6561 vera; quota del livello nella MSE della ridge 60,2% su tutte le finestre e 57,5% sulle dinamiche, contro 65,9% e 54,4%
    veri. Una ridge con ritardi -6, -4, …, +6 patch arriva fino a +21,3%, con l'alpha scelto sul fold letto e quindi in modo
    ottimista. Con alpha 1e6, che dei due valori riportati ha la MSE più bassa, fa +13,8%; con tutti i 13 ritardi +16,8%. Il
    minimo di E_dyn della TCN cade all'ultima epoca con tutti e tre i LR provati (1e-4, 1e-3, 3e-3).
- Nel mondo V lo stesso fit passa a +3,65% con 150 epoche e a +2,6% con un'inizializzazione che conserva il segnale (Kaiming); un LR
  più alto (3e-3) non aiuta. Nel mondo temporale del revisore, invece, la stessa inizializzazione lascia la TCN sotto L0 (-3,8% e
  -4,7%). Nel suo mondo lineare, dove basta un lettore lineare, né 150 epoche né l'inizializzazione Kaiming portano la MLP al livello
  di L0 (0,95 e 0,94 contro 0,81). Il revisore indica come limite
  principale la generalizzazione da 20 utenti con una regolarizzazione quasi nulla: il weight decay quasi non lavora (nota
  selezione-11).
- Come contesto, non come causa dimostrata: con l'inizializzazione standard, nell'ultimo strato nascosto di MLP e TCN, dopo la ReLU,
  la varianza dovuta all'ingresso è solo lo 0,6-2,1% del quadrato della parte costante dovuta ai bias (in ampiezza circa l'8-15%).
  Il 25-44% delle unità è spento (ESEGUITO). Il verificatore conferma il dato, non che sia la causa del divario.
- Nel mondo V e nel mondo temporale del revisore l'esito dipende dalla ricetta più che dal guadagno disponibile, e nessuno dei
  rimedi provati funziona in tutti i mondi. Sono dati finti: mostrano il meccanismo, non che cosa succederà sui latenti veri.

Perché blocca. Il protocollo legge un esito negativo come chiusura della prova. Se la ricetta non fa convergere le teste, un difetto
di allenamento verrebbe letto come risultato scientifico. È proprio il caso che GPT ha descritto in chat (sezione 1).

Correzioni chieste dai verificatori. Servono tutte e tre, scritte prima dei fit, valide sul solo train e uguali per tutte le teste.
I dettagli (griglia dei LR, estensioni, inizializzazione, rumore del bersaglio sintetico) li sceglie GPT e li firma Simone:
1. **Controllo positivo sui latenti veri.** Si costruisce un bersaglio sintetico con una funzione temporale non lineare nota dei
   `lat` di train, con rumore fissato prima. La ricetta è ammessa solo se la TCN batte L0 di almeno il 2% su E_dyn interna. Se il
   controllo fallisce, regolarizzazione o taglia delle teste si rivedono prima del lancio: è un cambio della specifica.
2. **Regola di convergenza**, simmetrica a quella della ridge storica, che estendeva la griglia di alpha al bordo (due estensioni a
   49.046). Per esempio: almeno tre LR (1e-4, 3e-4, 1e-3, 3e-3) con un'estensione al bordo; un'estensione unica delle epoche se il
   migliore cade nelle ultime cinque; l'epoca 1 riportata come bordo inferiore. Va nominata l'inizializzazione esatta; se si sceglie
   He/Kaiming, va deciso ora.
3. **Esito «non a convergenza»** (K-3): una testa ancora al bordo dopo le estensioni, o una ricetta che non supera il controllo
   positivo, non produce un'etichetta negativa.

In aggiunta, non in alternativa: il ramo lineare di A-1 fa contenere L0 a TCN e MLP, ma non mostra che la ricetta sappia prendere
un guadagno temporale. Lo scelgono GPT e Simone.

La E_dyn interna usa la soglia dinamica 0,2574231028556824, che è la mediana della varianza della posa per finestra sulle finestre
val. È una costante già congelata dei soli bersagli: se entra nell'ammissione della ricetta o nella selezione, va dichiarato.

Il costo va ricalcolato. Come ordine di grandezza, con un conto di Claude sull'analogo peggiore usato dal verificatore (18,5 ms per
passo, cioè TEMPONet raw con cuDNN deterministico in eager): con quattro LR e 150 epoche in tutti i fit, rifit compresi, i passi
salgono da 75.000 a 315.000. Raddoppiati, i soli passi fanno circa 3,2 GPU-ore, già sopra il tetto di fase di 3 prima delle
valutazioni interne, dell'I/O e degli avvii, che il gate conta; con le valutazioni di ogni epoca si può arrivare a circa 4. Con tre
LR i soli passi sarebbero circa 2,8 GPU-ore, che con le valutazioni tornano sopra il tetto di 3. Con il tempo non deterministico
della stessa replica (4,1 ms per passo in eager) i soli passi sarebbero circa 0,7. Non è una misura
delle nuove teste: la cifra vera viene dal benchmark. Estendere le epoche solo dove serve costa meno.

### A-1. Bracci non annidati: TCN e MLP non contengono L0 (minore; verificatore, architettura-controlli-N2)

L13 con le prese laterali a zero coincide con L0, quindi la contiene. TCN e MLP invece non hanno un cammino lineare: ogni cammino
verso l'uscita passa per cinque ReLU (righe 33, 35 e 42). Possono imitare L0 solo con bias grandi che tengono accese le ReLU. In un
mondo sintetico con bersaglio lineare, con la ricetta del §4, la MSE tenuta fuori della TCN è peggiore del 28,2% di quella di L0
(0,3205 contro 0,2500). Con un ramo lineare 384→20 a zero torna al livello di L0 (0,2495, +0,2%). Sono MSE su dati finti, non
E_dyn: mostrano il meccanismo.

Correzione. Senza fit, prima della firma:
- estendere la cautela della riga 49 al primario e a MLP contro L0: con classi non annidate, un deterioramento o un esito nullo non
  dimostra che una lettura temporale non lineare sia peggiore o inutile;
- decidere ora se dare a TCN e MLP un ramo lineare con uscita a zero (7.700 parametri in più ciascuna), così che il primario misuri
  il beneficio aggiunto a L0. Cambia la specifica: lo scelgono GPT e Simone.

### C-1. Manca il controllo di sanità della ricetta comune (minore; controlli-1)

L0 rifittata e ridge storica hanno la stessa classe di funzioni e gli stessi dati (stesso `train.npz`, 40 utenti): il loro scarto
misura solo la ricetta. La ridge è molto sensibile alla regolarizzazione: sulla sua griglia la MSE interna sale del 5,37% con alpha
1e4 e del 17,32% con alpha 0,01 rispetto all'ottimo ad alpha 1e5. Il weight decay di L0 invece quasi non lavora: dopo 3.000 passi
con LR 1e-3 riduce i pesi solo del 3% circa (fattore 0,9704).

Il protocollo non dice come leggere il primario se L0 rifittata resta peggiore della ridge, cioè con un E_dyn più alto. Esempio:
con L0 a 0,81, una TCN a 0,7938 passa il primario (+2,0%) pur essendo -1,0% sulla ridge e -1,95% sulla testa storica. La regola dei
contrasti 4 e 5 ferma l'etichetta «candidata», ma non la frase «l'architettura è promettente» della riga 100 né il richiamo al
livello B della riga 118.

Correzione. Dichiarare ora, fuori dalla famiglia e senza verdetto, due descrittivi senza fit: L0 rifittata contro ridge storica e
L0 rifittata contro testa storica, con bootstrap appaiato e IC 95%. Scrivere prima come si leggono. Se L0 è peggiore della ridge
con deterioramento, i contrasti 1-3 sono confronti a ricetta fissa: niente «l'architettura è promettente» e nessuna motivazione per
B. Altrimenti il primario si legge come scritto; se L0 è migliore della ridge, va riportato come effetto del rifit. Legare «Se la
TCN migliora il lettore» (riga 118) all'esito completo della riga 100.

### C-2. La regola della candidata dà solo condizioni necessarie (minore; controlli-2)

La riga 100 dice che la TCN è candidata «solo se» passa i contrasti 1, 4 e 5. Non dice se bastino, né che cosa si conclude se un
controllo è migliore della TCN. Esempio: con L0 0,785, L13 0,730 e TCN 0,750, i contrasti 1, 4 e 5 valgono +4,46%, +4,57% e
+3,68%, mentre il contrasto 3 vale -2,74%. La TCN risulterebbe candidata con un controllo migliore di lei; lo stesso vale con una
MLP a 0,735 (contrasto 2 -2,04%). MLP contro L0 e L13 contro L0 non sono definiti da nessuna parte.

Correzione:
- regola necessaria e sufficiente: candidata se passano 1, 4 e 5 e né il 2 né il 3 mostrano un deterioramento. Altrimenti esito
  negativo per l'ipotesi temporale non lineare, salvo l'esito «non a convergenza» (K-3), che viene prima;
- scrivere le letture miste;
- dichiarare come descrittivi, con IC 95% e senza cancello, MLP contro L0 e L13 contro L0. L13 contiene esattamente L0: con una
  regolarizzazione debole, un L13 peggiore di L0 indicherebbe sovradattamento. Se L13 contro L0 dà «deterioramento pratico», il
  contrasto 3 non si attribuisce alla non linearità.

### C-3. Il perimetro dell'esito non è nella tabella del §10 né completo in decisioni.md (minore; controlli-4)

La tabella delle decisioni (righe 173-180) non dice che cosa autorizza un esito. La voce di `docs/decisioni.md` esclude già il
livello B e la sostituzione della testa se MAE o livello peggiorano. Omette però due clausole presenti nella riga 100 e in
`proposal.json`: nessuna adozione automatica e nessuna non-inferiorità implicita. Non dice nemmeno che la conclusione non riguarda
il valore del pretraining.

Correzione. Aggiungere alla tabella una riga «Che cosa autorizza l'esito»:
- Un esito positivo rende la TCN candidata a lettore della dinamica centrata, migliore di quelli esistenti su questo checkpoint.
- Non autorizza la sostituzione della testa (decisione distinta, dopo MAE e livello) né il livello B. Non autorizza affermazioni sul
  pretraining rispetto a encoder random o a reti da zero, perché le baseline del v10 §8 restano dovute. Non è una conferma su dati
  mai visti.
- Un esito nullo chiude la prova e non dimostra un limite dell'encoder.

Ripetere il perimetro in decisioni.md. Rinominare la chiave `confirmatory_family` di `proposal.json`, perché la proposta stessa
dichiara che non c'è una conferma indipendente.

### R-3. La tabella che Simone firma non nomina l'oggetto della prova (minore; consegna-5)

Simone ha chiesto di provare TEMPONet come testa. La proposta prova invece una TCN compatta ispirata a TEMPONet (§2, riga 25): sei
convoluzioni, tre k3 con dilatazioni 1/2/3, niente BatchNorm, pooling o stride. Il corpo lo spiega bene. La riga «Bracci» della
tabella del §10 però nomina la testa solo come «TCN-post» e «TCN-pre», senza dire che cosa sia.

Correzione. Aggiungere alla tabella la riga «Oggetto: TCN compatta ispirata a TEMPONet, non la classe esistente né i suoi pesi».
Cambiare di conseguenza il titolo del protocollo e del README (nota architettura-1).

### S-2. La selezione sulla MSE totale, dominata dal livello, distorce soprattutto L13 (minore; selezione-2)

LR ed epoca si scelgono sul minimo della MSE di tutte le finestre (righe 61 e 68), mentre il primario è E_dyn centrata sulle
finestre dinamiche. Nelle predizioni storiche il livello pesa il 54,4% della MSE della ridge sulle finestre dinamiche e il 65,9% su
tutte. Scegliere sulla MSE ferma presto le teste che sovradattano il livello. Per L13 la perdita di E_dyn è grande (3,7-12% nei
mondi sintetici). TCN e L0 perdono poco e quasi lo stesso (0,1-1,4% ciascuna), quindi sul contrasto primario l'effetto netto è di
mezzo punto o meno. Il rischio pratico riguarda il contrasto 3, non la regola della candidata. Scegliere direttamente su E_dyn
interna peggiora invece il livello per costruzione: per L13 la MSE sale del 4-15% (4,1% nel mondo V, 12-15% nel mondo del
revisore; nota selezione-criteri-N4).

Correzione. Tenere la MSE. Scrivere prima una regola per quando l'epoca scelta è lontana dall'ottimo dinamico interno, e pubblicare
entrambe le curve. In alternativa, fissare prima dei fit un criterio combinato. Se la E_dyn interna entra nella regola, va
dichiarato l'uso della soglia dinamica (vedi S-1).

## 5. Da scrivere nel testo da congelare

### K-1. «Passa» e «servono» non sono definiti (minore; criteri-2)

L'unico esito positivo definito è «beneficio pratico risolto» (riga 95). Per i contrasti 4 e 5 «passa» si potrebbe leggere come un
semplice limite inferiore > 0, molto meno del 2%. La regola della candidata manca da decisioni.md.

Correzione. «Passa» vuol dire beneficio pratico risolto nei contrasti 1, 4 e 5, tutti e tre. «Servono anche TCN/MLP» vuol dire
beneficio risolto nel contrasto 2; «TCN/L13» nel contrasto 3. Stesse parole in protocollo, `proposal.json` e decisioni.md, con la
formula 100·(E_ref − E_TCN)/E_ref.

### K-2. `proposal.json` non contiene le condizioni di beneficio e di deterioramento (minore; verificatore, selezione-criteri-N1)

La regola a quattro esiti sta nel protocollo (righe 95-98) e in decisioni.md, ma non in `proposal.json`, che è il testo macchina da
cui partirà il codice. Nel JSON ci sono solo `practical_improvement_percent` e `readout_failures`.

Correzione. Scrivere nel JSON le quattro etichette con le stesse parole del protocollo, con l'ordine di precedenza (nota
criteri-1) e la definizione di «passa» (K-1).

### K-3. Manca l'esito «testa non a convergenza» (minore; verificatore, selezione-criteri-N2)

La riga 75 dice solo «si riporta». Un fallimento della ricetta verrebbe chiamato «deterioramento pratico» o «non sostenuto».

Correzione. Aggiungere al §5, prima dei fit, l'esito «non a convergenza» definito in S-1: una testa al bordo dopo le estensioni, o
una ricetta che non supera il controllo positivo, non produce un'etichetta negativa.

### R-2. La voce di congelamento va completata e pinnata (minore; consegna-4)

Ognuno dei due testi ha ciò che manca all'altro:
- decisioni.md non riporta la soglia dinamica 0,2574231028556824, i pareggi della selezione e la regola della candidata;
- `proposal.json` non contiene le condizioni di beneficio e deterioramento (K-2);
- l'IC 95% descrittivo di TCN-post contro TCN-pre e l'arresto su NaN o OOM mancano in entrambi;
- il collegamento al protocollo non porta né commit né sha256.

Correzione. Al congelamento serve una voce completa in decisioni.md, che citi protocollo e `proposal.json` con lo SHA del commit e
lo sha256 dei file e riporti per esteso soglia dinamica, pareggi, quattro etichette, candidata, IC descrittivo e arresto. In
`proposal.json` vanno aggiunti anche l'IC 95% descrittivo e l'arresto su NaN o OOM; le quattro etichette sono K-2. I numeri non
cambiano.

### R-1. Fra i test manca il cancello che riproduce i riferimenti (minore; consegna-3)

Il cancello va scritto ora nel §8 del protocollo e si esegue con il codice nuovo prima dei contrasti.

Nessun test ricalcola i valori pubblicati. I riferimenti sono salvati già ritagliati a 148 patch, mentre le teste nuove escono a
160. In una prova sintetica un ritaglio sfasato ([0:148] o [12:160] invece di [6:154]) cambia molto E_dyn, ma lascia identici
forma, hash del supporto e conteggi: il controllo «supporti appaiati» non lo vede.

Correzione. Prima dei contrasti, ricalcolare con il codice nuovo 0,7859020004638595 e 0,7786289843513977 dai file pinnati, con uno
scarto al più di 1e-12, insieme ai conteggi del supporto (8.327 finestre, 20.119.920 elementi). Passare poi la `pred` a 160 patch
della cache val nel ritaglio delle teste nuove e riottenere l'array di `trained_head.predictions.npz`. Come complemento, servono
test per:
- l'ordine dei minibatch identico fra le teste;
- l'arresto del gate prima del primo fit;
- le etichette d'esito e la candidata su intervalli sintetici.

## 6. Prima del benchmark

### P-1. Il gate di costo non è ancora meccanico (minore; costo-1)

Il gate (righe 163-165) fissa già le misure (riga 163), il fattore 2, le 50 epoche, i 35 fit e le soglie (riga 165). Mancano tre
cose:
- la statistica delle epoche;
- una regola di memoria con una parte fissa, oppure misurata su un surrogato della dimensione della val. È l'unico punto che può
  produrre un arresto spurio seguito da una regola post hoc;
- l'obbligo di usare nel benchmark le stesse impostazioni numeriche dei fit: modo di cuDNN, thread, posizione dei dati. Nella
  replica TEMPONet il modo deterministico rendeva le convoluzioni dilatate 13 volte più lente.

Numero di job e `--time` vanno nel piano eseguibile. Un modello di gate meccanico esiste già: `BENCHMARK_POLICY` e
`project_benchmark` in `scripts/consolidation_long_pose_runtime.py`.

## 7. Note

Le note non cambiano il disegno. Conviene risolverle o motivarle una per una nella revisione. Il rapporto dell'agente indicato fra
parentesi ha prove e righe.

Architettura:
- Titolo del protocollo, README e intestazione della voce di decisioni.md dicono «TEMPONet» senza qualificazione: serve un titolo
  più preciso (architettura-1, vedi R-3).
- Scrivere «nn.Dropout(p=0,1) per elemento, non Dropout1d, attivo solo in training» (architettura-2).
- Fissare ordine di creazione dei moduli, ultimo strato sotto `fork_rng` o no, generatore del dropout e semi dello shuffle. In
  alternativa, registrare gli hash dei pesi iniziali per seme prima del primo fit (architettura-3).
- Per L0 e L13 inizializzazione deterministica e niente dropout: i tre semi cambiano solo l'ordine dei minibatch (architettura-4).
- Con l'inizializzazione di default i bias dominano il quinto strato nascosto (85-90% della varianza della pre-attivazione) e circa
  un terzo delle unità non si accende. Sui dati finti il minimo cade alle ultime epoche anche per L0: dipende anche dal tetto di 50
  epoche, non solo dall'inizializzazione (architettura-5, vedi S-1).
- Il campo di 13 patch è motivato solo dal crop6. I 128 ms di TEMPONet raw sono già nominati alle righe 21 e 23, ma non come motivo
  del campo. È facoltativo aggiungerli come motivo, insieme al lettore lineare con ritardi del 02/11, che era su B student con crop4
  e non su questo checkpoint (architettura-6).
- Facoltativo nel §2: a T = 160 la TEMPONet trapiantata dipende da tutti i 160 passi con `causal=False`, dai passi 0-123 con
  `causal=True` (architettura-7).
- La chiave `top_level_keys` di `raw_source_identity.json` è troncata (architettura-8).
- Il link di provenienza del §2 (iris.polito.it, handle 11583/2978570) non è quello verificato nel codice. Il codice usa il DOI
  10.1109/COINS51742.2021.9524188 e il postprint hdl.handle.net/11585/841320 (verificatore, architettura-controlli-N1).

Controlli:
- §6, confronto raw futuro. Il metodo che il §6 chiama esatto (prevedere su tutti i campioni nativi e poi mediare il blocco) stima
  invece una media triangolare spostata di 4,5 campioni. Il metodo esatto è un endpoint per blocco valido, con la finestra che
  finisce in 10j+9 come nel training raw: circa 13,7 milioni di endpoint per i 15 val, 1,1 volte la lettura ufficiale (le «dieci
  volte» valgono per il metodo per campione). L'uscita va riportata grezza, senza EMA, perché l'alpha dell'EMA è stato scelto sugli
  stessi 15 val (controlli-3).
- Contrasto 4: scrivere «ridge in forma chiusa fittata sugli stessi 40 utenti», non «la miglior lettura lineare disponibile»
  (controlli-5).
- TCN contro MLP: scrivere la lettura nei due versi (controlli-6).
- TCN-pre: affiancare la ridge pre storica (1,032128) come ancora descrittiva e scrivere prima la lettura di post contro pre
  (controlli-7).
- Ridge e testa storiche: dire su quali dati sono fittate ed elencare le differenze di ricetta. Per esempio, la testa storica
  predice da latenti FP32, le nuove teste dalla cache FP16 (controlli-8).

Selezione:
- Test d'ingresso sui due E_dyn storici: vedi R-1 (selezione-3, criteri-9).
- Il rifit a parità di epoche fa circa il doppio dei passi, 60 invece di 30 per epoca: dichiararlo (selezione-4).
- Valutazione interna con `model.eval()`, `no_grad` e FP32 alla fine di ogni epoca 1-50. L'epoca 0 è esclusa e le curve vanno nel
  manifest di selezione prima del rifit (selezione-5).
- Fissare la derivazione numerica dei semi dello shuffle e la politica di determinismo su GPU (selezione-6).
- I «4,720 GiB di input» sono le sole due cache. Con i tre file di riferimento pinnati in `proposal.json` (coefficienti e
  predizioni della ridge, predizioni della testa: 0,608 GiB) si arriva a 5,328 GiB. Sono altra cosa dai riferimenti di posa su
  SCRATCH della nota costo-consegna-N1 (selezione-7).
- Per comodità, rimando a `424f0ec` e liste dei fold in `proposal.json` (selezione-8).
- Dichiarare che tutte le finestre hanno 160 patch. Nel benchmark, contare per fold le finestre senza centri e i passi per epoca
  (selezione-9).
- La PoseHead storica è stata allenata su tutte le patch di finestre da 1-4 s, le teste nuove sul crop6: dichiararlo
  (selezione-10).
- Il weight decay 0,01 quasi non regolarizza: il regolarizzatore effettivo è l'arresto, a passi di un'epoca (selezione-11).

Criteri:
- Le quattro etichette si sovrappongono. Va scritto l'ordine: vale la prima condizione vera fra beneficio, deterioramento, non
  sostenuto e inconclusivo. Va scritto anche che la regola è nuova: non si implementa riusando `practical_assessment` della
  valutazione lunga, che in tre casi dà un'etichetta diversa (criteri-1).
- Riportare la deviazione standard fra i semi accanto a quella bootstrap. Se Simone vuole proteggere il «no», decidere ora il
  bootstrap a due livelli o la concordanza dei semi (criteri-3).
- Il 2% è largo rispetto al campionamento degli utenti; rispetto ai semi della testa il margine non è noto (criteri-4).
- Secondari: rimandare a `consolidation_long_pose._metrics` e a `tokenizer_pose_readout.summarize`, e citare
  `emg2pose_pose.JOINTS` (criteri-5).
- Copertura attesa con 15 utenti: circa 97% per contrasto e 85-91% per la famiglia. Per una copertura nominale servono il
  percentile 99,8% o la t-jackknife, decisi ora (criteri-6).
- Con 2.000 repliche i limiti degli IC hanno un errore Monte Carlo di qualche centesimo di punto, che conta solo per un limite
  vicino a 0 o al 2%. Fissare ora 20.000 repliche, che costano pochi secondi, oppure tenere 2.000 repliche e scrivere ora che un
  limite entro 0,1 punti dalla soglia si riporta «al bordo», senza cambiare etichetta (criteri-7).
- Scrivere che, nei tre esiti separati dei contrasti 1-3, si confronta il seme s con il seme s; i nove incroci restano descrittivi
  (criteri-8).
- Definire «peggiorano» per MAE e livello senza nuove soglie di non-inferiorità (criteri-10).
- La regola della candidata richiede E_dyn ≤ 0,763056: il 2% sotto la testa allenata e il 2,91% sotto la ridge. Va detto a Simone
  (verificatore, selezione-criteri-N3).
- Scegliere su E_dyn interna sposta la distorsione sul livello: vedi S-2 (verificatore, selezione-criteri-N4).

Costo:
- La somma di (`--time` + 180 s) deve stare nel tetto di fase, oppure lo sforamento va alla riserva (costo-2).
- In `proposal.json`, un elenco di fasi con CPU, GPU, RAM, limite di tempo, GPU-ore, core-ore e ore locali (costo-3).
- Aggiungere una stima a priori, dichiarata come tale, e il residuo del 09/11 con la data (costo-4).
- Scrivere quali misure del benchmark CPU alimentano il gate (costo-5).
- Calendario rispetto al fermo del 23-25/11 e controllo della corsia seriale prima di ogni fase CPU (costo-7).
- Il file delle statistiche della posa sta su SCRATCH. Il codice nuovo deve confrontarne l'hash senza aprirlo, oppure dichiarare la
  dipendenza dal rinfresco del 03/12, perché i file rinfrescati il 12/11 scadono dal 22/12 (costo-8).
- Preflight dell'operatore: prenotazioni, occupazione della corsia seriale, soglia di spazio libero scritta prima e AllocTRES del
  primo job (costo-9).
- La riserva copre sforamenti e avvii. Un rilancio dopo una diagnosi richiede una decisione registrata (costo-10).
- Anche i riferimenti di posa del pilot B19618 stanno solo su SCRATCH (`train_subset.npz`, 257.675.076 byte; `val.npz`,
  2.485.295.825 byte), e il codice della valutazione lunga li apre per validare le cache. Tre strade: verificare le cache con i pin
  su `$WORK` senza aprirli, copiarli su `$WORK` (circa 2,74 GB) con sha256, oppure dichiarare la dipendenza dal rinfresco
  (verificatore, costo-consegna-N1).

Consegna:
- Scrivere in chiaro nella tabella del §10 che cosa autorizza la firma, oppure prevedere un secondo via dopo il benchmark se Simone
  lo preferisce. Alla riga 182, «di questa attività» invece di «della linea 2» (consegna-1).
- Indicare nel piano del codice la base del ramo. La guardia della valutazione lunga fa l'hash dei file al primo livello di
  `scripts/`: o si parte da `main`, che contiene la correzione `3263ee9`, o si mettono i file nuovi in una sottocartella di
  `scripts/`. Su `cfa6462` quel test è già rosso (consegna-2).
- Per informazione: il prompt di Claude era indirizzato a GPT coordinamento, e nella linea 2 c'è il precedente di A1 (consegna-6).
- Livello B: 33,88 GPU-ore valgono per due training a 19.618; a 49.046 sarebbero circa 81-84 GPU-ore prima dell'overhead. Per B
  manca il piano del codice rispetto a I-1 (consegna-7).
- Ricevute, manifest e report vanno sul ramo di esecuzione di Claude, in una sottocartella `execution/`; GPT non li modifica
  (consegna-8).
- Serve un ID del protocollo non ambiguo, per esempio `pose-temponet-frozen-encoder-readout-20261117-v1` (consegna-9).
- Scrivere chi prepara il piano eseguibile e prevedere una sola review del pacchetto operativo prima dei lanci, sul modello della
  linea 1 (verificatore, costo-consegna-N2).

Rifiutato dai verificatori: costo-6. Il tetto di 8 GiB vincola già l'implementazione, che non prevede di salvare le predizioni held
di ogni epoca.

## 8. Che cosa non ho verificato

- Nessuna misura sui latenti veri, nessun tempo su A100, nessuna VRAM: restano al benchmark. Gli esperimenti sintetici mostrano
  meccanismi.
- Gli sha256 dei file su `$WORK` non sono stati ricalcolati oggi; solo i byte.
- Le parole di Simone che la proposta riporta dalla chat di GPT («leggilo e dimmi come impostare il confronto con la testa attuale
  usando l'implementazione che abbiamo già», e mettere da parte il protocollo A) non sono verificabili da qui.
- Il link iris.polito.it del §2: i revisori lavoravano senza rete.
- Le tre letture locali di GPT (`/root/temponet_*_review`): da qui si vede solo il riassunto in `design_review.txt`.

## 9. Prompt per GPT

```text
Review di Claude della tua proposta c15aac9 (TCN come lettore della posa a encoder congelato), pubblicata sul ramo
claude/temponet-db8-20261109: docs/reviews/claude_review_pose_temponet_proposal_20261117.md, con rapporti e script degli
agenti in docs/reviews/claude_review_pose_temponet_proposal_20261117_evidence/ (vedi il LEGGIMI.md). Sei revisori e tre
verificatori avversari; 56 rilievi giudicati, 8 nuovi.

Verdetti: 1 Domanda e architettura, 2 Controlli e attribuzione, 4 Criteri, 5 Costo, 6 Consegna: APPROVATO CON CORREZIONI.
3 Selezione e supporto: DA RIVEDERE, per un solo rilievo bloccante (S-1).

S-1. Con la ricetta proposta (due LR costanti, al più 50 epoche, inizializzazione standard, uscita a zero, regolarizzazione
quasi nulla) MLP e TCN possono restare lontane dal loro ottimo. In due mondi sintetici indipendenti la TCN perde contro L0
anche quando c'è guadagno da prendere. Nel mondo V, tarato per somigliare ai numeri veri, il primario vale -3,1%; diventa
+3,65% con 150 epoche e +2,6% con un'inizializzazione Kaiming, che però nel mondo temporale del revisore non basta. Una ridge
con ritardi arriva fino a +21,3% (stima ottimista). Sono dati finti: mostrano il meccanismo, non l'esito vero. Poiché un esito
negativo chiude la prova, prima dei fit servono tutte e tre queste regole: un controllo positivo sui lat di train con
bersaglio sintetico noto; una regola di convergenza (almeno tre LR con estensione al bordo, un'estensione delle epoche se il
migliore cade nelle ultime cinque, epoca 1 come bordo inferiore, inizializzazione nominata); l'esito «non a convergenza». I
dettagli li scegli tu. Il ramo lineare di A-1 si può aggiungere, ma non sostituisce le tre regole. Il costo va ricalcolato.

Da correggere prima della firma: A-1 (TCN e MLP non contengono L0: cautela estesa al primario, oppure ramo lineare a zero da
7.700 parametri), C-1 (descrittivi L0 rifittata contro ridge e testa storiche, con lettura scritta prima), C-2 (regola della
candidata necessaria e sufficiente, MLP contro L0 e L13 contro L0 descrittivi), C-3 (che cosa autorizza l'esito, nella
tabella del §10 e in decisioni.md), R-3 (oggetto della prova nella tabella: TCN ispirata a TEMPONet, non la classe esistente
né i suoi pesi), S-2 (regola sulla distanza fra epoca scelta e ottimo dinamico interno, e pubblicare entrambe le curve).
Nel testo da congelare: K-1 («passa» = beneficio pratico risolto in 1, 4 e 5), K-2 (quattro etichette in proposal.json),
K-3 (esito «non a convergenza»), R-2 (voce di decisioni.md completa, con SHA e sha256; in proposal.json anche IC
descrittivo e arresto) e R-1 (cancello, da scrivere nel §8,
che prima dei contrasti riproduce 0,7859020004638595 e 0,7786289843513977 e controlla l'allineamento del ritaglio). Prima
del benchmark: P-1 (statistica delle epoche, regola di memoria, stesse impostazioni numeriche dei fit). Le 50 note: una riga
ciascuna, accolta o motivata. Correggi anche la frase del §9 sull'indipendenza da SCRATCH: il file delle statistiche della
posa e i riferimenti di posa del pilot B19618 (train_subset.npz e val.npz, che il codice della valutazione lunga apre per
validare le cache) stanno solo su SCRATCH; le predizioni storiche di ridge e testa sono su WORK.

Dalla chat del 17/11: accetto le tue quattro correzioni alle mie frasi (un esito nullo non prova un limite dell'encoder;
ridge e PoseHead vicine non provano che il lineare sia al massimo; l'assorbimento del compito da parte della testa è un
rischio, non una certezza; lo scarto 0,47-0,90% non è un limite superiore del rumore). È giusta anche la tua correzione
tecnica al mio prompt sul campo ricettivo di TEMPONet.

Che cosa ti chiedo: una revisione della proposta sul tuo ramo, sempre REVIEW_REQUIRED, con l'elenco dei cambiamenti e il
nuovo SHA. Se un rilievo è sbagliato, dimostralo con una prova: non accettarlo per cortesia. Nessun codice, job o deploy.
Poi rivedo una volta sola le parti cambiate, e Simone decide sulla tabella del §10.
```
