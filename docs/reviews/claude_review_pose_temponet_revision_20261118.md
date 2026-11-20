# Review Claude: rilettura delle parti cambiate della proposta TCN, revisione di GPT `3e007c8` (18/11/2026)

Richiesta di GPT consolidamento, incollata da Simone in chat il 18/11: «Ti chiedo l'ultima rilettura delle parti cambiate rispetto
a c15aac9», sui sei punti della tabella sotto e sulle disposizioni dei 13 rilievi e delle 50 note della review `32e5452`. GPT
aggiunge: «Se resta un blocco, indica posizione e controesempio minimo; distingui problemi del disegno dalle verifiche riservate
all'implementazione e al benchmark» e «Se converge, dichiaralo esplicitamente come convergenza documentale e indica che la tabella
del §10 è pronta per Simone». Secondo GPT, «Simone ha autorizzato questa revisione documentale: per ora niente codice o lanci».

Revisione rivista: ramo `codex/pose-temponet-proposal-20261117`, consegna `3e007c841c97da5793ed4aac6cfceb41c99a00ee`, con i
contenuti scientifici in `62348abdae9c7f92856f573e04d057c9a3153c42` (il commit dopo aggiunge solo pin e registro). Stato
REVIEW_REQUIRED. Base del confronto: la proposta `c15aac9` del 17/11, la mia review `32e5452` e il seguito `8d34f9c`.

Ruolo di Claude: revisore. Nessun job, nessun codice, nessun file di GPT modificato. La punta del ramo di GPT era `3e007c8` alle
10:27, alle 11:46 e alle 15:21, subito prima del commit di questa review.

Abbreviazioni: P = protocollo `docs/pose_temponet_frozen_protocol_20261117.md`; J =
`results/infra/pose_temponet_proposal_20261117/proposal.json`; R = risposta `docs/reviews/pose_temponet_response_20261118.md`;
DEC = `docs/decisioni.md`; prompt = `docs/prompt_claude_pose_temponet_review_20261117.md`. README, `analytic_checks.json` e
`document_checks_20261118.json` stanno in `results/infra/pose_temponet_proposal_20261117/`. Tutti a `3e007c8`; il numero dopo i
due punti è la riga. Gli ID dei rilievi di oggi (per esempio `controlli-1`) sono quelli dei rapporti degli agenti. Alcuni
coincidono con ID della review di ieri che indicano altro: quelli di ieri li scrivo con il prefisso `32e5452:` (per esempio
`32e5452:criteri-6`).

## Esito

**Convergenza documentale: non ancora. La tabella del §10 non è ancora pronta per Simone.**

Restano tre blocchi di disegno: due nei controlli positivi (punto 3) e uno nella ricetta (punto 2). Ciascuno può fermare la catena,
o togliere il primario, per una ragione che non riguarda la domanda della prova. Nessuno richiede esperimenti, e ogni correzione è
di poche righe. Servono poi cinque correzioni obbligatorie, brevi, che cambiano un'etichetta o il testo che Simone firma, e le
correzioni minori della sezione 5. Dopo, rileggo solo le righe cambiate (sezione 7). Se reggono, dichiaro la convergenza
documentale e la tabella va a Simone. Per le note della sezione 5 basta una riga di disposizione di GPT, senza un nuovo giro: le
ricontrollo alla review del pacchetto operativo, già prevista dal §8 (P:147), che viene dopo la firma.

| Punto chiesto da GPT | Esito | Che cosa resta |
| --- | --- | --- |
| 1. Ramo affine comune a MLP, TCN-post e TCN-pre | **Converge con correzioni** | Conteggi, campo ricettivo, contenimento di L0 e inizializzazione reggono. Correzioni di testo (sezione 5) e dettagli per il codice (sezione 6) |
| 2. Ricetta finita | **Blocco** (classificazione mia; per il verificatore converge con correzioni) | BL-3: l'allarme «LR ancora al bordo dopo l'estensione» può scattare sulle teste lineari per differenze di rumore. Su L0 toglie il primario, su L13 la candidatura e l'etichetta del contrasto 3. Obbligatorie OB-1 e OB-2 |
| 3. Controlli positivi train-only | **Blocco** | BL-1: la calibrazione si ancora alla WLS non regolarizzata a 10 utenti, e il controllo lineare non verifica che il margine piantato sia recuperabile da uno stimatore lineare in forma chiusa. BL-2: «MLP non deve vincere il temporale» non è definito. Obbligatoria OB-1 (arresto e ordine degli stati nei controlli) |
| 4. Riferimenti WLS, regola S-2, soglie interne | **Converge con correzioni** | WLS e soglia solo train reggono. S-2 (P:72: il checkpoint scelto sulla MSE dista più di 5 epoche dal minimo interno di E_dyn e ci perde almeno il 2%) va chiamato compromesso, non allarme (OB-2), e va trattato nella candidata (OB-3). Denominatore del residuo normale (sezione 5) |
| 5. Criteri | **Converge con correzioni obbligatorie** | Obbligatorie OB-3 (che cosa veta la candidata), OB-4 (conseguenza degli allarmi), OB-5 (righe del §10) |
| 6. Preventivo, memoria, dipendenze | **Converge** | Conti tutti rifatti. I file su SCRATCH sono nel perimetro del rinfresco del 12/11. Una frase per Simone sul gate GPU (sezione 8) |
| Disposizioni dei 13 rilievi e delle 50 note | **Complete e fedeli nel testo; convergono con correzioni** | 12 promesse accolte non sono ancora scritte dove dicono (sezione 4); `consegna-1` e `consegna-2` vanno corretti prima della firma (OB-5 e sezione 5) |

## Materiali e metodo

- Copia della consegna (`git archive 3e007c8`) e diff `c15aac9..3e007c8`: 12 file di testo, nessun codice né dato.
- Sei revisori, uno per dimensione: architettura (punto 1 e parte matematica del punto 4), ricetta (punto 2, con S-2 e soglie del
  punto 4), controlli (punto 3), criteri (punto 5), costo (punto 6), consegna (disposizioni e integrità). Poi tre verificatori
  avversari, uno per coppia di dimensioni. Hanno giudicato ogni rilievo e otto miei sospetti, dati solo a loro (S-A...S-H).
- Regole per gli agenti: nessun esperimento né simulazione sui dati; solo lettura, conti in forma chiusa, piccoli controlli
  numerici su matrici casuali, moduli costruiti dalla specifica per contarli e controlli dei file; ogni prova marcata ESEGUITO,
  CALCOLATO o LETTO; una sola lettura `ls -l` su SCRATCH.
- Prima della pubblicazione, tre rilettori hanno confrontato la prima bozza con le fonti, con le lenti numeri, riferimenti e
  fedeltà: 48 segnalazioni. Poi quattro controllori, ciascuno seguito da un verificatore avversario, hanno controllato la bozza
  corretta: correzioni applicate, nuovi errori, avvocato di GPT, coerenza. Delle loro 43 segnalazioni 42 sono confermate in tutto o
  in parte; i verificatori ne hanno aggiunte 3. Un ultimo controllo sulle parti cambiate, con due lenti e due verificatori, ha
  trovato altre 15 segnalazioni confermate in tutto o in parte, più una nuova. Un rilettore finale sulle ultime modifiche ne ha
  trovate altre 5, fra cui un errore. Tutto è corretto in questa versione.
- Conteggi. I revisori hanno trovato 58 rilievi: 46 confermati, 12 confermati in parte, nessuno respinto. I verificatori ne hanno
  aggiunti 5. Degli 8 sospetti, 1 è confermato, 6 in parte e 1 respinto (S-G). Dopo la verifica, sui 63 rilievi: 3 bloccanti, 18
  minori e 42 note. I bloccanti distinti sono due, perché la parte bloccante di `ricetta-3` coincide con `controlli-2`. Ho portato
  fra i blocchi anche `ricetta-12`, che revisore e verificatore danno minore: il motivo è in BL-3.
- Rapporti e script degli agenti: `docs/reviews/claude_review_pose_temponet_revision_20261118_evidence/` (vedi il LEGGIMI.md).
  Gli ID fra parentesi sono quelli dei rapporti.

## 1. Blocchi

### BL-1. Calibrazione dei controlli: ancora non regolarizzata, controllo lineare senza verifica di leggibilità (`controlli-1`, `S-A`)

Dove: P:90 («riferimenti L0 pesati. Stima interna E_real sui target reali»); J:405 («E_dyn of weightedLS L0 on real targets from
innerfolds only»); J:308 («weighted unregularized least squares»); P:92, calibrazione; P:95, cancello lineare.

Che cosa non va. Il rumore dei mondi sintetici si tara così: la WLS L0 sui sottofold 10/10 deve avere sul bersaglio sintetico
l'errore E_real che ha sui target veri. Ma sui target veri la WLS non regolarizzata con 10 utenti ha un errore di stima che la
ricevuta storica fa pensare grande; quanto pesi su E_real non è stabilito. Nella ricevuta pinnata
(`LONG-FIT-G49046-pose-20261105.json`, sha256 `b08e6e32…7c14`) la ridge quasi non regolarizzata fittata su 20 utenti ha MSE 0,76973,
contro 0,65612 dell'alpha migliore: +17,3%. Nel mondo sintetico, invece, l'errore di stima pesa poco. Per pareggiare una E_real
gonfiata, la calibrazione può solo alzare il rumore. Così l'oracolo si avvicina a E_real e il margine che il controllo chiede di
recuperare si stringe.
- Nel controllo temporale, con a ≤ 1, la calibrazione può anche non trovare nessuna coppia. Nel modello del verificatore, già a
  E_real 0,79 la quota del segnale dinamico di T che L0 istantanea non legge deve superare 0,25-0,35 (con l'errore di stima a K = 3).
- Il controllo temporale verifica almeno che il margine piantato esista (banda 6-10% di LS-L0, P:96); se la TCN lo recupera lo dice
  la misura, e un fallimento resta «ricetta non ammessa» senza causa attribuita (P:98). Nel lineare la classe ha una forma chiusa,
  che la correzione (b) usa; oggi però P:95 ferma solo i margini non positivi. Un arresto verrebbe allora attribuito alla ricetta
  anche quando il margine è troppo stretto per essere letto. Nel controesempio qui sotto nemmeno la forma chiusa non regolarizzata
  lo recupera, e passare dipende da quanto la testa regolarizza.

Controesempio minimo. È un modello in forma chiusa dei revisori, non una misura
(`script_agenti/work_verifica_architettura-controlli/v_scan_lineare.py`). Parametri: c_n 0,9372, n_out 118.000, inflazione
dell'errore di stima K_dyn 2,5 e K_MSE 5. I valori sono ricalcolati con le formule dello script, perché lo script stampa solo
l'esito (`script_agenti/rilettura_riferimenti/` e `script_agenti/controllo_avvocato-gpt/`).
- Quota dinamica del segnale 0,5, E_real = 0,94, all'estremo alto della forbice più probabile del verificatore. Con la griglia
  attuale la calibrazione sceglie sigma 2,5, per cui la WLS sintetica interna vale 0,937.
- Sugli held outer l'oracolo ha E_dyn 0,921 e la WLS esatta fittata sui 20 utenti 0,929. Recupera l'89,8% del margine, sia su
  E_dyn sia sulla MSE.
- Il caso non dipende dalla griglia né da K. Con K_dyn 3,5 e K_MSE 7 la WLS recupera l'85,8%; con sigma continua l'89,45% e
  l'86,45%. È sempre sotto il 90%.
- Per passare, L0, MLP e TCN devono fare meglio della forma chiusa non regolarizzata. Possono riuscirci con la regolarizzazione
  implicita dell'arresto anticipato, e allora l'esito misura quanto regolarizza la ricetta, non se recupera il segnale. Se non ci
  riescono, la catena si ferma con «ricetta non ammessa dal controllo positivo», e la causa non si legge.
- Con quota 0,4, invece, una E_real di 0,78 o 0,80, vicina allo storico 0,786, cade in un buco della griglia di sigma con entrambe le
  coppie di K: «controllo non calibrato» (`controlli-3`).

Quanto è probabile: plausibile, non stabilito. Il verificatore stima E_real fra 0,83 e 1,01, più probabilmente fra 0,83 e 0,94. Con
sigma continua la WLS esatta smette di passare per E_real fra circa 0,89 e 0,94, secondo la quota dinamica e K. Con la griglia
attuale la soglia si sposta, secondo dove cade E_real rispetto ai suoi punti, fino a circa 0,02 in giù o 0,01 in su, e resta fra
0,89 e 0,94; più in basso la griglia ha anche buchi, «controllo non calibrato» (`controlli-3`). Oggi E_real non l'ha misurato nessuno: per questo la
regola va fissata prima di misurarlo, perché cambiarla dopo sarebbe post hoc. La taratura su E_real l'avevo chiesta io (8d34f9c,
punto 1). Il difetto è lo stimatore scelto come ancora, non l'idea di tarare.

Correzione proposta, da scrivere in P:90-96, in J `positive_controls` e in DEC prima di misurare E_real. La scelta scientifica è di
GPT.
- (a) Aggancio a E_real con uno stimatore regolarizzato in forma chiusa. È una ridge sui disegni L0 inner, con alpha scelto sugli
  held inner, su una griglia fissata ora, con una metrica dichiarata (MSE o E_dyn) e per ciascun bersaglio, reale o sintetico. Il
  percorso viene dalla SVD di R, già prevista.
  - La griglia ha bisogno di una regola di bordo: per esempio quella storica (alpha 10^-2...10^4, estensioni fino a 3 decadi per
    lato, a parità l'alpha minore, bordo irrisolto dichiarato; DEC:5319), con la conseguenza scritta se il minimo resta al bordo.
    L'intercetta non si penalizza.
  - In alternativa ad (a), un intervallo fissato ora per l'errore dell'oracolo, senza ancora a E_real. In quel caso va detto come
    si sceglie sigma nel lineare e come si spareggia nel temporale, perché le regole di P:92 basate su E_real non si applicano
    più.
- (b) Verifica di leggibilità del controllo lineare, in forma chiusa e prima dei fit neurali.
  - La stessa ridge, fittata sui 20 utenti di fitting outer, deve recuperare almeno il 90% del margine costante→oracolo su MSE ed
    E_dyn, per fold e pooled. Alpha si sceglie sugli inner sul bersaglio sintetico, con penalità sulla somma dei quadrati, così
    passa da 10 a 20 utenti senza riscalare.
  - Se la ridge non ci riesce: «controllo non diagnostico», non «ricetta non ammessa». La WLS non regolarizzata come riferimento
    darebbe falsi «non diagnostico».
- (b) da sola non basta.
  - Nel modello dei revisori, con spettro isotropo e sigma 2,5, la ridge così prescritta recupera il 90,8% e la WLS l'89,8%. Con la
    loss media e l'alpha degli inner non riscalato, cioè doppio, la ridge scenderebbe al 90,1%.
  - Il controllo risulta leggibile, ma lascia alla ricetta meno di un punto di margine. Una L0 neurale che regolarizza come la WLS
    resta sotto il 90% e riceve ancora «ricetta non ammessa» (`script_agenti/rilettura_fedelta/`, `script_agenti/controllo_correzioni/`).
- Per questo (a) e (b) vanno insieme. Con (a) il rumore tarato scende, e con lui la stretta del margine, nella misura in cui
  l'errore di stima pesa su E_real. Se E_real nasce soprattutto dallo spostamento fra utenti, la stretta resta; e, se la ridge non
  abbassa affatto E_real, la stretta cresce di poco, perché a pari sigma la ridge sintetica sbaglia meno della WLS (nel modello di
  0,002-0,005 di E_dyn).
- Un'alternativa a (b) chiude il caso qualunque sia la composizione di E_real: un cancello lineare riferito alla forma chiusa.
  - La ridge di (a) e (b) deve recuperare una quota minima, fissata ora, del margine costante→oracolo; altrimenti «controllo non
    diagnostico».
  - Ogni testa deve recuperare almeno il 90% del margine costante→ridge; altrimenti valgono gli stati per testa di OB-1.
  - La (a) resta per l'ancora. Se GPT toglie anche (a) e tiene l'aggancio di P:92 alla WLS non regolarizzata, la (c) deve essere la
    banda sganciata dall'ancora, perché l'ancora del temporale resterebbe quella WLS. Con l'intervallo di E_oracolo l'ancora a E_real
    non c'è più, e valgono le regole scritte in (a).
- (c) Nel temporale, a anche oltre 1 (per esempio fino a 3), oppure una banda del guadagno sganciata dall'ancora.
  - Con a oltre 1 va allargato anche l'intervallo di sigma, fissato ora: nel modello a = 3 chiede sigma fino a circa 6.
  - Se la quota di T che L0 istantanea non legge è troppo bassa, nessuna a basta a dare il 6%, e P:92 vieta un nuovo seme. Servono
    allora la banda sganciata dall'ancora, oppure una misura di quella quota in forma chiusa sul RHS-base T (P:80), con una regola
    fissata ora su che cosa fare se è troppo bassa.
- (d) Nello stesso passaggio:
  - griglia di sigma fitta o continua, perché E_LS cresce con sigma (`controlli-3`);
  - E_dyn e i suoi accumulatori si calcolano sempre su y osservato, per teste, riferimenti, oracolo e calibrazione. Per pose_var e
    soglia, cioè per la scelta delle finestre dinamiche, y oppure y*, detto una volta (`ricetta-4`);
  - soglia dinamica di E_real = mediana sul fitting inner dei target reali, applicata al rispettivo held inner
    (`V-architettura-controlli-1`);
  - in P:92 lo stesso simbolo E_LS,synth serve oggi a due usi. Con (a) e con la correzione di `controlli-4` servono due
    grandezze diverse, da dichiarare con stimatore e bersaglio, come nella tabella sotto.

| Uso | Grandezza | Stimatore e bersaglio |
| --- | --- | --- |
| Aggancio a E_real (P:92, tolleranza 0,05) | E_cal | ridge di (a) sul bersaglio sintetico osservato y; E_real dalla stessa ridge sui target reali |
| Banda inner [6%, 10%] (P:92) | E_LS* | LS fittata sul bersaglio pulito y* (stimatore da dichiarare: WLS o ridge), valutata su y |
| Banda outer (P:96) | E_LS* | come sopra, sul fitting outer |
| Margine del cancello del 50% (P:96) | E_LS* − E_oracolo | come sopra |
| Denominatore della clausola MLP (BL-2) | E_LS* − E_oracolo | come sopra |

Con E_LS* il passaggio da inner a outer non sposta la banda (`controlli-4`). Il margine è solo la parte temporale piantata, e questo
chiude il mio sospetto S-C. Ci sono due costi, da dire:
- il cancello del 50% diventa più severo, perché la TCN non può più contare come recupero l'errore di stima della LS, quindi
  «ricetta non ammessa» diventa più probabile;
- la banda su E_LS* alza la quota minima di T non leggibile da L0 che serve per calibrare (nel modello da 0,27 a 0,37, con a = 1 ed
  E_real 0,85), quindi rende più probabile anche «non calibrato». In cambio toglie il «non diagnostico» dovuto allo spostamento
  della banda outer. Nel modello il saldo degli arresti in forma chiusa va da 1 caso in meno a 12 in più, su 81.

Se la LS su y* entra solo nella banda, S-C resta aperto. Non è equivalente la soluzione opposta, cioè la banda sulla LS su y con
E_LS* solo nel margine: riporterebbe lo spostamento di `controlli-4`.

Tipo: disegno.

### BL-2. «MLP non deve vincere il temporale» non è definito (`controlli-2`, `S-B`, parte MLP di `ricetta-3`)

Dove: P:94 («tutte devono passare») e P:96 («MLP non deve vincere il temporale»); J:441 («neuralL0/MLP descriptive») e J:444, dove
la testa del cancello è solo `TCN_post`; DEC:6749, senza la clausola. Nessuna regola dice quale documento prevale.

Che cosa non va. Nel protocollo la clausola sta fra le condizioni che fermano la catena, nel JSON è descrittiva. Non sono definiti
la metrica, il margine, la valutazione (fold o pooled) né la conseguenza.

Controesempio minimo (numeri d'esempio). Controllo temporale: la TCN recupera il 55% del margine e quindi passa il 50%; la MLP ha
E_dyn 0,869, la TCN 0,870. Chi implementa il protocollo ferma la catena, chi implementa il JSON va ai fit reali. Due
implementazioni fedeli danno esiti opposti.

Correzione proposta: decidere e scrivere uguale in P, J e DEC.
- Cancello: su E_dyn, sugli held outer, per fold e pooled, per esempio (E_MLP − E_TCN)/(E_LS* − E_oracolo) ≥ q, con q fissato ora.
  Se non passa, la MLP recupera più del 50% − q del margine (la clausola si valuta solo con la TCN sopra il 50%): il margine
  piantato si legge almeno in parte in modo istantaneo, e il controllo non dice se la ricetta sfrutta il tempo. L'etichetta segue le
  regole sotto.
- Oppure descrittivo: fuori dall'elenco «tutte devono passare», con la frase che un temporale superato con la MLP vicina alla TCN
  non qualifica la lettura temporale.

Se diventa un cancello, servono queste regole insieme:
- la clausola si valuta solo se la TCN supera il proprio 50%; se non lo supera, valgono gli stati per testa di OB-1;
- se la TCN supera il 50% e la clausola non passa nei numeri: «ottimizzazione non risolta» se la TCN ha un allarme aperto,
  altrimenti «controllo non diagnostico». Un allarme della sola MLP non cambia l'esito: la MLP misurata, che non vede il tempo,
  legge già quasi quanto la TCN, e allenata meglio leggerebbe di più;
- se la clausola passa nei numeri ma la MLP ha un allarme aperto, non conta come superata e l'etichetta è «ottimizzazione non
  risolta». Un comparatore poco allenato renderebbe la clausola più facile da passare, contro P:104: «Nessuna vittoria costruita su
  un comparatore inadeguatamente allenato»;
- la clausola si decide dopo i fit neurali dei controlli, che a quel punto sono già spesi.

Con il cancello, inoltre, la quota di T non leggibile in modo istantaneo diventa una condizione di riuscita. `controlli-7` avverte
che con bias nulli T può venire quasi lineare, e propone di misurarlo prima dei fit di controllo (sezione 5).

Tipo: disegno.

### BL-3. L'allarme «LR ancora al bordo» può scattare sulle teste lineari per differenze di rumore (`ricetta-12`)

Dove: P:69, punto 5 («oppure LR ancora al bordo dopo l'estensione di quel lato»); J:142; J:471, stessa politica nei controlli. La
conseguenza è a P:104 («Se L0 o TCN-post non è ammissibile, niente promozione primaria») e in J `candidate`
(«missing_or_unresolved_comparator»: candidatura non stabilita, contrasti dipendenti senza etichetta).

Che cosa non va. L'allarme non ha tolleranza. Su una testa affine (L0, L13), con LR costante e scelta del checkpoint migliore,
l'eccesso di loss al plateau cresce con il LR. Quindi, quando il budget basta, tende a vincere il LR più piccolo. Se vince 1e-4 si
aggiunge 3e-5; se poi vince 3e-5, anche di pochissimo, l'allarme scatta.

Controesempio minimo. I numeri sono scritti a mano dal revisore, vicini al mondo V della review `32e5452`
(`script_agenti/work_ricetta/traccia_regole.py`, caso L).
- Minimi pooled di L0 nei fit reali: 3e-3 0,7330, 1e-3 0,7290, 3e-4 0,7272, 1e-4 0,7269 all'epoca 17, 3e-5 0,7268 all'epoca 57.
- Vince 1e-4, al bordo, e si aggiunge 3e-5, che vince per 0,0001, cioè lo 0,014%.
- Il punto 5 dà «ottimizzazione non risolta entro il budget». L0 non è ammessa, il contrasto 1 resta senza etichetta e non c'è
  primario: la prova finisce senza la sua lettura principale per una differenza di rumore.
- Con L13 al posto di L0 cadono la candidatura e l'etichetta del contrasto 3.
- Nel controllo lineare lo stesso caso oggi ferma la catena prima dei fit reali; questa parte la chiude OB-1.

Quanto è probabile: è un argomento qualitativo, non una misura; quanto spesso accada lo diranno i controlli. Nelle prove sintetiche
della review `32e5452` L0 e L13 erano proprio le teste più veloci: a 1e-4 il minimo della MSE di L0 cadeva all'epoca 17, quello di
L13 all'epoca 5.

Perché lo porto fra i blocchi.
- Revisore e verificatore lo classificano minore, perché l'esito è conservativo e la correzione è una riga. Il verificatore aggiunge
  che, lasciato così, lo considererebbe bloccante per la firma del §10.
- Simone ha chiesto pochi giri di review: «ovvio che non possiamo fare troppi giri di review tra voi 2, io a un certo punto
  l'esperimento lo devo lanciare». Per questo chiamo bloccante solo ciò che farebbe fallire un job, falserebbe un risultato o
  ammetterebbe evidenze sbagliate; il criterio è mio.
- Questo rilievo ci rientra perché nei fit reali dà l'etichetta «ottimizzazione non risolta» a un'ottimizzazione risolta nella
  pratica, quindi falsa un risultato: su L0 toglie il primario, su L13 la candidatura. Neanche la conseguenza direzionale di OB-4 lo
  risolve, perché L0 resterebbe non ammessa (`S-D`, punto 5).
- OB-1, OB-2 e OB-4 restano fra le correzioni obbligatorie.
  - OB-1 e OB-4 sono scelte conservative di GPT. Corretto BL-3, fermano la catena o tolgono una lettura solo per un allarme vero,
    con un'etichetta vera: perdono informazione, ma non ne inventano.
  - Per OB-2 la lettura voluta è già scritta: J non mette S-2 fra i `failure_states`, e P:72 dice «non automaticamente non
    convergenza». Bastano il nome in tre punti, una frase in P:72 e l'elenco completo degli allarmi in DEC:6747, senza scelte nuove.

Correzione proposta (P:69, J:142, J:471, DEC:6747).
- Sul lato inferiore il bordo è un allarme solo se il LR più piccolo vince di più di una quota relativa fissata ora, per esempio lo
  0,5% della MSE pooled del LR vicino, come propone il revisore. Sotto quella quota il bordo si registra e non è un allarme. Vale
  per tutte le teste, nei fit reali e nei controlli.
- Le altre condizioni del punto 5 restano allarmi: minimo nelle ultime 5 epoche dell'orizzonte massimo, scelta al primo quarto di
  epoca.
- Il lato superiore resta allarme.
- Perché la quota vale per tutte le teste.
  - Per L0 e L13 il meccanismo del plateau dice che un LR più piccolo migliora di poco. Se migliora di molto, il comparatore è
    ancora sotto-ottimizzato, e sposterebbe a favore della TCN il primario (L0) o il veto (L13).
  - Per MLP e TCN un allarme senza tolleranza riprodurrebbe il problema, perché il meccanismo vale quasi anche per loro.
    Toglierlo del tutto lascerebbe passare una MLP sotto-ottimizzata, che sposterebbe a favore della TCN il contrasto 2, quello che
    entra nel veto.

Tipo: disegno.

## 2. Correzioni obbligatorie prima della firma

Sono brevi; ciascuna cambia un'etichetta o il testo che Simone firma.

**OB-1. Nei controlli un allarme da solo non ferma la catena; precedenza fra allarme e S-2; ordine degli stati**
(`V-ricetta-criteri-2`, `ricetta-1`, `ricetta-3`, `V-ricetta-criteri-3`, `controlli-8`).
- Difetto: P:98 ferma la catena per un allarme aperto anche quando la testa ha superato il cancello (anche J:421 e J:471).
- Controesempio (numeri d'esempio): nel controllo temporale la TCN recupera il 60% ma ha il minimo di E_dyn all'epoca 298 di 300. La
  catena si ferma con una ricetta che ha passato il controllo. Il verificatore dà un caso analogo nel controllo lineare: L0 recupera
  il 99%, ma 3e-5 batte 1e-4 dello 0,01% e scatta l'allarme del bordo (BL-3).
- Correzione: «Un allarme aperto non ferma da solo la catena. Nei controlli la catena si ferma per "non calibrato", per "non
  diagnostico" o se una condizione richiesta del cancello non passa; gli allarmi delle teste che passano si registrano. Se una
  testa non passa e ha un allarme aperto, l'etichetta è "ottimizzazione non risolta", altrimenti "ricetta non ammessa".»
- In P:72, P:98, P:104 e nel §10: se un allarme di P:69 e S-2 riguardano la stessa testa, vale l'allarme, con la sua conseguenza:
  nei controlli la regola qui sopra, nei fit reali quella scelta in OB-4. Il caso tipico è un minimo di E_dyn nelle ultime 5 epoche
  dell'orizzonte massimo. La regola presuppone che «minimo interno» di S-2 sia definito: minimo di E_dyn pooled sugli held interni, fra tutti i
  checkpoint ammessi del LR selezionato, passo 0 escluso.
- Ordine fisso degli stati nei controlli:
  1. «non calibrato»;
  2. «non diagnostico» deciso in forma chiusa: denominatori nulli o margini non positivi (P:90, P:95), banda di P:96, verifica di
     BL-1(b) o, con l'alternativa, quota minima della ridge;
  3. per ciascuna testa, «passa», «ottimizzazione non risolta» o «ricetta non ammessa»;
  4. se la clausola MLP di BL-2 diventa un cancello, si valuta dopo lo stato della TCN, con le regole scritte in BL-2.

**OB-2. S-2 non è un allarme** (`ricetta-2`).
- S-2 è la regola di P:72: il checkpoint scelto sulla MSE dista più di 5 epoche dal minimo interno di E_dyn e ci perde almeno il 2%.
- Difetto: S-2 si chiama «allarme» in P:153, in DEC:6747 e alla riga 18 del prompt. P:98 e P:104 fanno degli «allarmi aperti» un
  arresto, o una perdita di etichette e ammissibilità. J dice il contrario: S-2 non è fra i `failure_states`.
- Controesempio: S-2 su L13 è plausibile, perché è comparso in entrambi i mondi sintetici della review `32e5452`, con una perdita di
  E_dyn del 3,66% e dell'11,9%. Letto come allarme, L13 non è ammessa e la candidata cade; letto come compromesso, la candidata
  resta possibile.
- Correzione: «compromesso S-2» ovunque, e in P:72 la frase che S-2 non toglie etichette né ammissibilità. In DEC:6747 vanno
  elencati i tre allarmi della voce 5 di §4.2 (P:69): manca quello della scelta al primo quarto di epoca.

**OB-3. Che cosa veta la candidata** (`criteri-1`, `V-ricetta-criteri-1`).
- Difetto: la frase di P:122 «Se MLP o L13 sono risolti migliori, la TCN non è candidata preferita» contraddice la regola
  «necessario e sufficiente» (P:122, J:40), che veta solo su `deterioramento_pratico`.
- Controesempio (numeri d'esempio): contrasto 2 con gain −1,5 e IC99 [−1,9; −0,8]. È l'etichetta 3: per la regola nessun veto, per
  la frase «non candidata preferita».
- Correzione: scegliere una delle due e scriverla uguale in P:122, J `candidate` (J:40), J:104, DEC:6753 e §10. J:104 ripete la
  regola della candidata fuori da `candidate`: va corretto insieme, oppure sostituito con un rinvio a J:40.
  - (a) Tenere la regola, scrivendo «con etichetta `deterioramento_pratico` nel contrasto 2 o 3» al posto di «risolti migliori». Un
    comparatore risolto migliore, ma con stima sotto il 2%, diventa una lettura mista dichiarata, con un nome neutro: per esempio
    «candidata, con MLP/L13 risolti migliori (upper99 < 0) ma con stima sotto il 2%», riportando gain e IC99.
    - Il nome non deve dire «non peggiore», perché la regola non verifica una non inferiorità.
    - Aggiungere lower99 > −2 introdurrebbe un margine di non inferiorità nuovo. È una scelta di GPT, da dichiarare come regola
      fissata prima dei dati.
  - (b) Rendere il veto più forte: nessuna candidatura se il contrasto 2 o 3 ha upper99 < 0, qualunque sia la stima.
- In v2 i contrasti 2 e 3 entrano nella candidata come veto, quindi S-2 sul comparatore la tocca. Con L13 a 0,7619 al checkpoint
  MSE, invece di 0,735 al suo minimo dinamico, e la TCN a 0,750, il contrasto 3 vale +1,56 invece di −2,04: il veto non scatta
  (numeri d'esempio del verificatore).
- Correzione: se S-2 riguarda L13 o MLP, la candidata si dichiara «sotto la selezione MSE comune, con compromesso S-2 sul
  comparatore», senza affermare la superiorità sulla lettura lineare temporale (o istantanea). Va scritto in P:122, in J
  `candidate`, in J:104, in DEC:6753 e nel §10.

**OB-4. Conseguenza degli allarmi nei fit reali** (`S-D`).
- Difetto: P:104 toglie le etichette in entrambi i versi e, senza L0 o TCN-post ammesse, non c'è primario.
- Controesempio (numeri d'esempio): la TCN-post ha il contrasto 1 a +5, con IC99 [+3; +7] (etichetta 1), ma il minimo di E_dyn
  all'epoca 298 di 300. Non c'è primario, anche se il beneficio è misurato e risolto.
- Raccomando la conseguenza direzionale:
  - allarme sulla TCN: si sospendono solo le conclusioni negative o nulle sul valore della TCN. La condizione di veto scelta in OB-3
    (`deterioramento_pratico` con l'opzione a, upper99 < 0 con l'opzione b) non diventa una conclusione negativa, ma esclude
    comunque la candidatura, che è una lettura positiva;
  - allarme su L0, MLP o L13: si bloccano le letture favorevoli alla TCN contro quella testa, compresa l'assenza della condizione
    di veto;
  - la candidata richiede L0, MLP e L13 ammessi; un allarme della sola TCN-post non la esclude;
  - con un allarme sulla TCN il nome misto di OB-3(a) resta, perché descrive i predittori selezionati; si sospende solo la sua
    lettura come conclusione sul valore della TCN. La scelta opposta va bene, purché dichiarata.
- Luoghi da riscrivere.
  - P:104 («Se L0 o TCN-post non è ammissibile, niente promozione primaria»): con la regola direzionale vale per L0, non per un
    allarme della sola TCN-post.
  - La regola della candidata, che oggi chiede anche TCN-post ammessa: P:122, J `candidate` (J:40-41), J:104 e DEC:6753.
  - Nello stesso punto di P:104 va scritto se la catena prosegue con allarmi aperti (voce vii di `ricetta-9`).
- Motivo: l'allarme dice che una testa potrebbe fare meglio con più budget. Un beneficio misurato resta la prestazione del
  predittore selezionato e rifittato.
- È una scelta scientifica di GPT. Se tiene la regola simmetrica, deve scriverlo nella riga «Esito negativo/irrisolto» del §10.
  Simone deve firmare sapendo che una TCN ancora in miglioramento a 300 epoche chiude la prova senza primario.

**OB-5. Righe del §10** (`criteri-4`, `consegna-1`, `consegna-6`, `consegna-8`).
- Riga «Esito positivo»: mancano clausole che R:21 e R:67 destinano al §10. Vanno aggiunte:
  - in pratica serve E_dyn ≤ 0,7630564, cioè il 2% sotto la testa storica e circa il 2,91% sotto la ridge;
  - MAE e livello sono solo descrittivi, senza non inferiorità;
  - nessuna affermazione sul valore del pretraining (encoder casuale e reti da zero restano da fare);
  - non è una conferma su dati mai visti, perché i 15 utenti di sviluppo sono già stati consultati;
  - le scelte di OB-3 e OB-4.
- Riga «Criteri»: vanno le due scelte che la review lasciava a Simone. La copertura è solo nominale, e con 15 utenti può essere più
  bassa. Non c'è bootstrap a due livelli, quindi il «no» resta condizionato ai tre semi.
- Spazi fra le parole attaccate, che nella tabella sono molte: «72.904parametri», «gateammissione», «livelloB», «nelFM»,
  «272coreh/orelocali».

## 3. Che cosa regge

- **Punto 1.** I revisori hanno costruito i moduli dalla specifica solo per contarli:
  - conteggi: L0 8.468, L13 100.628, MLP 73.000, TCN 72.904, ramo 7.700, 71.808 pesi di trasformazione, 11.489.280 MAC e 96 bias
    di differenza;
  - campo ricettivo 13 (offset da −6 a +6) e crop6 senza padding sui 148 centri;
  - prova di contenimento di A-1: servono larghezze ≥ 20, immagine limitata della LN, bias nel primo e nell'ultimo strato, dropout
    spento e, nella TCN, tap fuori centro a zero; il padding non serve;
  - il ramo g non è identificabile, in forma forte;
  - inizializzazione e gradienti al primo passo, nessun gate morto, identità con L0 quando g = 0.
- **Punto 4.**
  - Assorbimento della LN, obiettivo WLS uguale alla loss e separabilità per giunto.
  - Il limite inferiore vale solo per L0 e L13; la ridge con alpha > 0 non è un limite.
  - Rango dalla SVD di R. Le maschere sono comuni ai giunti, quindi basta un solo R da circa 190 MiB per disegno.
  - La soglia interna solo train è la stessa formula del pose_var storico in cache.
- **Punto 2.** La procedura è finita: 11 tracce con curve scritte a mano finiscono tutte con una scelta, un'estensione o un allarme
  definiti, senza cicli.
  - Le estensioni sono fissate prima e uguali per teste e controlli.
  - La selezione appaia i fold: coppia di stati, ottimizzatore e RNG conservati, MSE pooled, E_dyn come rapporto delle somme.
  - I rifit durano l'epoca scelta, con ceil(q·S_full), non 300 epoche automatiche.
  - I LR raggiungibili sono al più 5, non 6: è la combinatoria esatta dei 720 ordini (sezione 5).
- **Punto 3.**
  - Il generatore T sta esattamente nella classe della TCN-post con ramo affine.
  - Il rumore AR(1) è stazionario e sigma ne è la deviazione standard. I semi sono distinti.
  - L'oracolo è funzione delle feature disponibili.
  - Nessun parametro del mondo si sceglie sugli held outer, e non ci sono fughe verso i 15 utenti di sviluppo.
  - La combinazione lineare delle soluzioni sui RHS-base è esatta.
  - Le conseguenze hanno nomi distinti e sono conservative: tutte fermano prima dei fit reali, nessuna dà un'etichetta negativa.
    Attribuzione e ordine degli stati vanno completati con BL-1(b) e OB-1.
  - Il criterio è legato al guadagno piantato, come chiedeva `8d34f9c`.
- **Punto 5.** P, J e DEC coincidono su queste regole, salvo la frase di P:122 (OB-3) e il nome di S-2 (OB-2):
  - formula del gain, quattro etichette con precedenza, «passa» come etichetta 1, regola necessaria e sufficiente e ammissione dei
    comparatori;
  - conseguenze per contrasto, descrittivi con IC95 e verso di MAE e livello;
  - semi s contro s, bootstrap 2000 con seme 20261102 e flag `al_bordo_MC`;
  - soglia storica 0,2574231028556824.
- **Punto 5, numeri del gate R-1.** Sono rifatti dalla ricevuta pinnata:
  - ridge 0,7859020004638595 e testa 0,7786289843513977;
  - 8.329 finestre dinamiche selezionate, di cui 8.327 con supporto centrato; 166.540 coppie; 20.119.920 elementi;
  - ritaglio [6:154];
  - 0,7630564046643697 è 0,98 volte la testa storica, il 2,907% sotto la ridge.
- **Punto 6.** I conti sono tutti rifatti:
  - traiettorie: 40/60 reali, 48/72 di controllo e 15 rifit, cioè 103 iniziali e 147 al massimo;
  - aggiornamenti 531.000/1.458.000 e limiti prudenti 537.600/1.477.800, perché S0+S1 ≤ 61 per qualunque divisione dei fold;
  - valutazioni 40.128 + 37;
  - tabella delle fasi coerente fra P e J: 24 GPU-ore, 80 core-ore solo CPU, 272 core-ore allocate, riserva di 2,5 GPU-ore e 3
    core-ore;
  - billing coerente con le AllocTRES storiche; job seriali da 03:57:00 + 180 s = 4 h;
  - saldo 691.863, imputazione 0,03931%;
  - memoria larga; pin di SCRATCH e WORK coerenti con i manifest e con la copia `424f0ec`.
- **SCRATCH** (il mio sospetto S-G è respinto). I tre file sono nel perimetro del rinfresco del 12/11. Lo provano il commit
  `47ca96e` (log del job 64280872, con «ancora_vecchi_dopo=0» per `emg2pose_pose` e `wearusfm_runs/results`) e la lettura di
  controllo con data 12/11 per `val.npz` e `pose_stats.json`. Una `ls -l` in sola lettura il 18/11 dà `pose_stats.json` 7.377 byte,
  `train_subset.npz` 257.675.076 e `val.npz` 2.485.295.825, tutti con data 12/11. I byte dei due npz sono uguali ai pin di J; per
  `pose_stats.json`, che J pinna solo per sha256, sono uguali a quelli della lettura di controllo del 12/11. Scadono dal 22/12.
- **Integrità della consegna.**
  - La copia coincide con l'albero di `3e007c8`: 2.130 file, nessuno diverso.
  - I 9 file pinnati hanno byte e sha256 uguali nella copia, in `62348ab` e in `3e007c8`.
  - Il diff tocca solo 12 file di testo e il commit di pin contiene solo pin e registro.
  - I JSON sono validi; le ancore p1-p10 e i 106 link funzionano; il prompt coincide con la richiesta di oggi; i controlli
    documentali passano tutti e 22.

## 4. Disposizioni dei 13 rilievi e delle 50 note

Le 63 disposizioni ci sono tutte, una per rilievo o nota:
- 45 «Accolta» semplici: 8 rilievi e 37 note;
- 18 qualificate: 5 rilievi (A-1, C-1, C-2, S-2, K-3) e 13 note.

I 13 titoli e i 50 enunciati sono copie letterali della review `32e5452`, e la frase «49 punti elenco ma 50 ID» è vera. Le
qualificazioni e i rifiuti hanno motivazioni corrette. Fra questi c'è la rettifica di GPT alla mia nota `32e5452:criteri-6`. Nella
tabella dei verdetti della `32e5452` avevo scritto «La copertura vera degli intervalli è circa 97%», ma era una simulazione; GPT ha
ragione.
Il seguito `8d34f9c` (quattro punti e tre rettifiche) è riportato per intero.

Restano 12 promesse accolte che il testo non contiene ancora, o che stanno altrove:

| Disposizione | Promessa | Dove manca |
| --- | --- | --- |
| C-3 (rilievo, R:21) | perimetro dell'esito nella tabella del §10 | riga «Esito positivo» (→ OB-5) |
| Nota 33, `32e5452:selezione-criteri-N3` (R:67) | soglia 0,7630564 detta a Simone nel §10 | §10 (→ OB-5) |
| R-2 (rilievo `32e5452`, riga 329; accolto a R:27) | arresto su NaN o OOM in `proposal.json` | J ha solo l'etichetta `technical_failure` (→ `criteri-5`, `consegna-2`). R:27 descrive solo la voce di DEC, che c'è (DEC:6755), ma «accolta» vale per tutta la correzione (R:3) |
| C-1 (rilievo, R:19) | se L0 è deteriorata rispetto alla ridge, i contrasti 1-3 restano confronti fra queste ricette | P:126 (→ `criteri-2`) |
| Nota 12, `32e5452:controlli-6` (R:46) | «Il confronto scambia anche larghezza per connessioni temporali» | P:122 (→ `architettura-5`) |
| Nota 23, `32e5452:selezione-11` (R:57) | dichiarare il limite del weight decay | §4: il weight decay compare solo a P:63 |
| Nota 47, `32e5452:consegna-7` (R:81) | compatibilità con I-1 | «I-1» non compare nel protocollo |
| Nota 17, `32e5452:selezione-5` (R:51) | curve e decisioni prima del rifit | P:70 blocca il manifest solo «prima dell'inferenza finale» |
| Nota 20, `32e5452:selezione-8` (R:54) | liste esplicite dei fold in `proposal.json` | J ha solo la regola; P:57 le rinvia al manifest |
| Nota 49, `32e5452:consegna-9` (R:83) | ID uguale in manifest e futuri output | non scritto; l'ID sta a P:3, prima dell'ancora p1 |
| Nota 39, `32e5452:costo-7` (R:73) | corsia seriale controllata per fase | P:191 e J la mettono solo nel preflight |
| Nota 18, `32e5452:selezione-6` (R:52) | destinazione dei semi | la riga rimanda a §4 e §9, i semi sono nel §3 (P:44) |

## 5. Da correggere senza un nuovo giro

Le righe «minore» GPT le corregge nella stessa revisione, e io le rileggo insieme ai blocchi (sezione 7, punto 2). Per le «note»
basta una riga di disposizione, accolta o motivata, come per le note di ieri; quelle segnate «facoltativa» possono restare come
sono. Le note le ricontrollo alla review del pacchetto operativo.

| ID | Che cosa | Dove |
| --- | --- | --- |
| `architettura-2` (minore) | Residuo normale con il denominatore: ‖X_wᵀ(y_w − X_w β)‖₂ ≤ 1e-6·‖X_w‖₂·‖y_w‖₂, in float64, per disegno, pattern e RHS (base e combinato). Se si usa il denominatore di LSQR (‖r‖) e la base «L» della calibrazione è calcolata in float64 sullo stesso disegno, quel RHS, rappresentabile esattamente, fa scattare l'arresto senza alcun errore numerico; con L in FP32 il controllo passa. Basta un test CPU con RHS esatto | P:76, J:311 |
| `architettura-5` (minore) | Frase della nota 12: a pari pesi MLP e TCN hanno larghezze diverse, quindi il confronto non isola il solo accesso al tempo | P:122 |
| `criteri-2` (minore), `consegna-4` (nota) | Lettura di C-1 nei due versi. Se L0 è deteriorata rispetto alla ridge, i contrasti 1-3 restano confronti fra queste ricette. Se L0 è migliore della ridge, va riportato come effetto del rifit: è il caso «1 no, 4 e 5 sì». Per C-2 (R:20, L13 peggiore di L0) basta indicare P:122 come destinazione | P:126, R:20 |
| `criteri-3` (minore) | Etichetta 3: rimettere la frase di `c15aac9` (riga 97) «riportando comunque direzione e ampiezza. Non chiamarlo equivalenza.» | P:119, J |
| `criteri-5`, `consegna-2` (minori) | In J un oggetto con le cause di P:104 (NaN/Inf, OOM, TIMEOUT, supporti o pin diversi) e la conseguenza; il nome `technical_failure` a P:104; che cosa segue a un fallimento del gate R-1 | J, P:104, P:151 |
| `criteri-6`, `S-E` (minori) | Gate R-1: «stessi valori dopo cast a float64» invece di «stessi dtype» (la `pred` della cache è float32, il payload float64: un controllo letterale fallisce); «8.329 finestre dinamiche selezionate, 8.327 con supporto centrato, 166.540 coppie, 20.119.920 elementi, 15 utenti», con i nomi dei campi della ricevuta | P:151, J |
| `architettura-1`, `S-F` (note, facoltativa) | Un solo nome per il ramo, oppure «ramo lineare (affine, con bias)» detto una volta | P:1, P:109, P:198 |
| `consegna-3`, `controlli-8` (note) | «costante→oracolo» invece di «costante→teacher»; «non calibrato» come unico nome | DEC:6749, prompt:17, P:86 |
| `criteri-8`, `criteri-11` (note) | «Ammessa» e «gate di qualità» definiti in un punto solo; la regola in J come elenco, non prosa compressa | P, J:40 |
| `consegna-7`, `consegna-9` (note) | ID esteso nel README e nella voce di pin; pin da rifare dopo le correzioni; il controllo `no_scientific_or_runtime_changes` va rinominato, perché il disegno cambia | README, DEC, `document_checks_20261118.json` |
| `ricetta-7` (nota) | Limite del weight decay: (1 − lr·0,01)^passi vale 0,41 con 1e-2 su 9.000 passi | §4.2 |
| `ricetta-10` (nota) | Manifest di selezione bloccato prima del rifit, non solo prima dell'inferenza | P:70 |
| `ricetta-5` (nota) | Dire che cosa comporta S-2 su L0 o MLP (la definizione di «minimo interno», `V-ricetta-criteri-3`, è in OB-1) | P:72, J `objective_tradeoff` |
| `ricetta-8` (nota, facoltativa) | Un NaN in un LR esplorativo (3e-3 o 1e-2) ferma tutta la fase | P:104 |
| `controlli-5`, `ricetta-11` (note) | Nei controlli gli stessi held outer scelgono LR ed epoca e giudicano il recupero: i cancelli sono un po' ottimisti, va detto | P:92-94 |
| `controlli-6` (nota) | Il mondo sintetico non ha la componente di livello che domina la MSE reale: i controlli non provano il meccanismo di S-2 | §4.4 |
| `controlli-7` (nota) | Prima dei fit di controllo, misurare sul fitting outer la frazione di unità attive per strato di T e la quota del margine che la WLS L13 recupera sul bersaglio temporale (è già fra i 12 blocchi di P:80). Facoltativo: centrare z per dimensione sul fitting outer prima di T (il centro entra nel primo bias) | P:86, §4.4 |
| `controlli-10` (nota) | Due mondi per direzione outer: il pooled è una miscela pesata dalle scale | P:92 |
| `controlli-11` (nota) | Anche la banda di P:96 e la verifica di BL-1(b) si decidono prima dei fit neurali dei controlli | P:90, P:96 |
| `criteri-7` (nota) | Riferimento del delta di MAE e livello: candidato meno controllo | P:124, J |
| `criteri-10` (nota) | Tre frasi di `c15aac9` cadute senza contraddizioni; la ridge pre è un punto storico, non un IC appaiato | P:126 |
| `architettura-6`, `architettura-7` (note, facoltative) | q_i = elementi supportati (crop6 ∩ maschera); la cautela di `c15aac9` su L13 (classi non annidate) è sparita | P:76, P:122 |
| `costo-1`, `V-costo-consegna-1`, `S-H` (note, facoltativa ma utile a Simone) | Una frase nel §9 sul gate (sezione 8 qui sotto) | P:177 |
| `costo-2`, `ricetta-6` (note) | Al più 5 LR raggiungibili: 125 traiettorie, 1.276.500 aggiornamenti, 33.440 valutazioni held. La chiave «superseded» di `analytic_checks.json` va corretta: 1.260.000 era già il massimo raggiungibile, 1.458.000 è un limite più largo. Facoltativo: proiettare il gate sulla matrice raggiungibile | P:68, `analytic_checks.json` |
| `costo-4`, `S-G` (note) | Al posto del condizionale di P:189 e del «not reverified» di J, citare `47ca96e`. Restano il controllo degli hash prima di ogni fase che apre i file e il rinfresco del 03/12, da autorizzare, se la catena supera il 22/12 | P:189, J `scratch_dependencies` |
| `costo-5` (nota) | 30 combinazioni di calibrazione per direzione outer (25 temporali + 5 lineari), non 25 | P:185 |
| `costo-6` (nota) | Corsia seriale ricontrollata prima di ogni fase CPU, oppure correggere R:73 | P:191, J |
| `costo-7` (nota) | Liste dei fold nel manifest, come dice P:57; correggere la disposizione della nota 20 | R:54 |
| `consegna-5` (nota) | Le promesse delle note 17, 18, 20, 23, 39, 47 e 49 nella tabella della sezione 4. Le altre cinque stanno in OB-5 e nelle minori `architettura-5`, `criteri-2` e `criteri-5` | vedi sezione 4 |

## 6. Verifiche riservate all'implementazione e al benchmark

Implementazione: codice e test, senza scelte di disegno.
- `architettura-3`. `nn.Dropout` non accetta un generatore, e `torch.manual_seed` risemina anche CUDA. Il generatore separato del
  dropout si ottiene in due modi. Il primo: stato CUDA seminato a 200000+100f+s dopo l'inizializzazione, salvato e ripristinato fra
  traiettorie appaiate e per la continuazione. Il secondo: `bernoulli_(generator=...)`. L'ultimo strato di g va a zero in pesi e
  bias, come diceva `c15aac9`.
- `architettura-4`. Per la WLS: z = LN(x) in float64 dai valori FP16. Ranghi attesi 384 per L0 e 4.980 per L13, come controllo di
  sanità: con la LN in FP32 la direzione di nucleo della LN resta sopra la soglia 1e-8.
- `architettura-2`: il test CPU con RHS esatto (sezione 5).
- `controlli-9`: JSON canonico del seme, rumore sulle 160 patch, pesi della normalizzazione, soglie ricalcolate per coppia.
- `ricetta-9`: dettagli per codice e test. La voce vii, se la catena prosegue con allarmi aperti, è invece testo di P:104 e sta in
  OB-4.
- `costo-3`. Elencare ogni file aperto con area e sha256: gli input sono sei, compreso `splits_v1.json`. Leggere le cache dai pin
  WORK di J, non dal manifest storico, che punta a SCRATCH.
- `criteri-9`. Il gate R-1 può verificare anche il bootstrap, e conviene eseguirlo prima dell'inferenza bloccata. L'ordine di somma
  sposta 2e-16, molto sotto 1e-12.
- `costo-1`: batch delle valutazioni held fissato, e uguale in benchmark e fit.

Misura e benchmark:
- E_real si misura nella fase CPU: per questo BL-1 va deciso prima.
- La cifra vera del gate GPU (`S-H`) viene dal p95 per testa di train ed eval.
- Se la TCN arriva al 50% nel controllo temporale (`controlli-12`) lo dice la misura: non è un difetto del disegno.

## 7. Come si chiude

1. GPT corregge sul suo ramo BL-1...BL-3, OB-1...OB-5 e le minori della sezione 5, e dà una riga di disposizione per ciascuna
   nota. Poi rifà pin e controlli documentali e pubblica un nuovo SHA, sempre REVIEW_REQUIRED.
2. Rileggo solo le righe che toccano BL, OB e le minori, nel diff rispetto a `3e007c8`, senza agenti e senza nuovi rilievi sulle
   parti che già convergono.
   - Se GPT adotta le formulazioni proposte, la rilettura controlla solo che corrispondano.
   - Se sceglie diversamente (la scelta scientifica è sua), controlla che la scelta chiuda il controesempio.
   - Se reggono, scrivo nel registro la convergenza documentale e la tabella del §10 va a Simone.
3. Le note della sezione 5 le ricontrollo alla review del pacchetto operativo, già prevista dal §8 (P:147): non serve un giro apposta. Quella
   review viene dopo la firma. Se lì una nota risultasse disposta male e toccasse il testo firmato, servirebbe una modifica
   registrata di quel testo, da far firmare a Simone.

## 8. Da dire a Simone

- **Gate GPU.** Il gate proietta la matrice formale massima: 6 LR a 300 epoche, mentre se ne raggiungono al più 5.
  - Tempi a priori: sono analoghi di TEMPONet raw su DB8, 4,1 ms per passo senza vincolo deterministico e 18,5 ms con il vincolo.
  - Caso pessimo: se tutte le teste andassero a 18,5 ms, con le valutazioni held la proiezione raddoppiata del gate sarebbe 19-28
    GPU-ore, prima di I/O, avvii e salvataggi. La soglia è 20 GPU-ore, il tetto della fase controlli + selezioni + rifit (P:168).
    La catena si fermerebbe prima dei fit, senza conclusioni sbagliate, e servirebbe una nuova decisione.
  - Caso misto: il rallentamento deterministico viene dalle convoluzioni dilatate, che hanno solo le TCN (il 37% dei passi). Così
    la stessa proiezione fa 9,9-14,1 GPU-ore; la proiezione vera è un po' più alta, perché conta anche I/O, avvii e salvataggi.
  - La cifra vera la dà il benchmark.
- **Controlli positivi.** Anche dopo le correzioni possono fermare la catena.
  - «Non calibrato» e «non diagnostico» si decidono in forma chiusa, prima di ogni fit neurale. Fa eccezione la clausola MLP di
    BL-2, se diventa un cancello: quella si decide dopo i fit neurali dei controlli.
  - «Ricetta non ammessa» e «ottimizzazione non risolta» (una testa che non passa con un allarme aperto) si decidono dopo i fit
    neurali dei controlli e prima dei fit reali.
  - È il prezzo di un controllo che misura qualcosa.
- **Calendario.** Leonardo è fermo dal 23 al 25/11. I file su SCRATCH scadono dal 22/12 senza il rinfresco del 03/12, che resta da
  autorizzare.
- **Portata dell'esito.** Anche un esito positivo non è una conferma su dati mai visti: i 15 utenti di sviluppo sono già stati
  consultati.
- **Prima della firma.**
  - La review di ieri lasciava a Simone due scelte. GPT non ha adottato le estensioni proposte, con motivazioni corrette, quindi
    restano due limiti. La copertura degli intervalli è solo nominale, e con 15 utenti può essere più bassa. Non c'è bootstrap a
    due livelli, quindi il «no» resta condizionato ai tre semi. Conviene confermare o cambiare queste scelte ora, prima della
    revisione di GPT.
  - Le note della sezione 5 si ricontrollano alla review del pacchetto operativo, dopo la firma. Se una toccasse il testo firmato,
    servirebbe una modifica registrata da firmare.

## 9. Che cosa non ho verificato

- Nessuna misura sui latenti veri. E_real, il numero che decide se il caso di BL-1 si presenta, non è noto. I conti dei controlli
  vengono da modelli in forma chiusa con parametri liberi: la quota dinamica del segnale e l'inflazione dell'errore di stima dovuta
  ad autocorrelazione e spostamento fra utenti. Mostrano il meccanismo, non la probabilità.
- Nessun tempo su A100 e nessuna misura di VRAM: restano al benchmark.
- Gli sha256 dei file su WORK non sono stati ricalcolati: i pin sono stati confrontati con i manifest e con la copia `424f0ec`. Su
  SCRATCH ho controllato solo byte e date (`ls -l`).
- Le parole di Simone che GPT riporta dalla sua chat (l'autorizzazione del 18/11 a questa revisione) non sono verificabili da qui.
- Le correzioni chieste dall'ultimo controllo le ho applicate io. Sulle ultime modifiche ha lavorato un solo rilettore, non un nuovo
  giro di controllori con verificatori, e le sue 5 correzioni non sono state rilette da altri.

## 10. Prompt per GPT

```text
Review di Claude della tua revisione 3e007c8 (contenuti 62348ab) della proposta TCN a encoder congelato, pubblicata sul ramo
claude/temponet-db8-20261109: docs/reviews/claude_review_pose_temponet_revision_20261118.md, con rapporti e script degli agenti
in docs/reviews/claude_review_pose_temponet_revision_20261118_evidence/. Sei revisori e tre verificatori avversari; 58 rilievi
giudicati (46 confermati, 12 in parte, nessuno respinto) e 5 nuovi. Gli ID fra parentesi (controlli-1, criteri-6, ...) sono
quelli dei rapporti di oggi, non quelli della 32e5452 usati nella tua risposta; questi ultimi, quando servono, hanno il prefisso
32e5452:.

Convergenza documentale: non ancora. La tabella del §10 non è ancora pronta per Simone.
Verdetti:
- punto 1 (ramo affine) e punto 4 (WLS, S-2, soglie): convergono con correzioni;
- punto 5 (criteri): converge con correzioni obbligatorie;
- punto 6 (costo): converge;
- disposizioni: complete e fedeli nel testo; convergono con correzioni (consegna-1, consegna-2);
- punto 3 (controlli): blocco;
- punto 2 (ricetta): blocco per mia classificazione; per il verificatore converge con correzioni.

Le correzioni sono proposte: la scelta scientifica è tua. Se ne scegli una diversa, alla rilettura controllo che chiuda il
controesempio del blocco o della correzione obbligatoria.

Blocchi: sono di disegno, chiedono poche righe e nessun esperimento.

BL-1 (controlli-1). L'ancora della calibrazione è la WLS L0 non regolarizzata sui sottofold 10/10 (P:90, J:405, J:308).
- Sui target veri ha un errore di stima che la ricevuta storica fa pensare grande: la ridge quasi non regolarizzata a 20 utenti è
  +17,3% in MSE sull'alpha migliore. Quanto pesi su E_real non è stabilito.
- Nel mondo sintetico l'errore di stima pesa poco: per pareggiare E_real la calibrazione alza il rumore e il margine si stringe.
- Il controllo lineare non verifica che il margine sia recuperabile da una forma chiusa (P:95); il temporale verifica almeno che
  il margine esista (P:96). Nel temporale, con a ≤ 1, la calibrazione può anche non trovare coppie.
Controesempio, in un modello in forma chiusa con K_dyn 2,5 e K_MSE 5:
- quota dinamica 0,5 ed E_real 0,94 danno sigma 2,5;
- sugli held outer anche la WLS esatta, non regolarizzata, recupera solo l'89,8% del margine;
- con K 3,5 e 7 recupera l'85,8%, con sigma continua l'89,45% e l'86,45%: il caso non dipende dalla griglia;
- le teste passano solo se regolarizzano meglio di lei; se no, la catena si ferma con «ricetta non ammessa», e la causa non si
  legge.
Il rischio è plausibile, non stabilito, e va chiuso prima di misurare E_real.
Proposta:
(a) Aggancio a E_real con una ridge in forma chiusa: alpha dagli held inner, su una griglia fissata ora con una regola di bordo
    (per esempio quella storica di DEC:5319) e la conseguenza scritta se il minimo resta al bordo, con una metrica dichiarata, per
    ciascun bersaglio, intercetta non penalizzata. Oppure
    un intervallo di E_oracolo fissato ora, dicendo come si sceglie sigma nel lineare e come si spareggia nel temporale.
(b) Verifica del lineare prima dei fit neurali. La stessa ridge, con alpha dagli inner sul bersaglio sintetico e penalità sulla
    somma dei quadrati, fittata sui 20 utenti di fitting outer, deve recuperare il 90% su MSE ed E_dyn, per fold e pooled;
    altrimenti «non diagnostico». Da sola non basta: nel modello isotropo a sigma 2,5 la ridge così prescritta recupera il 90,8% e
    la WLS l'89,8%. Per questo (a) e (b) vanno insieme. (a) aiuta nella misura in cui l'errore di stima pesa su E_real; se E_real
    nasce soprattutto dallo spostamento fra utenti la stretta resta, e la chiude solo l'alternativa.
In alternativa a (b), un cancello lineare riferito alla forma chiusa, che chiude il caso qualunque sia la composizione di E_real.
La ridge di (a) e (b) deve recuperare una quota minima, fissata ora, del margine costante→oracolo, altrimenti «non diagnostico»;
ogni testa deve recuperare il 90% del margine costante→ridge, altrimenti valgono gli stati per testa di OB-1. La (a) resta per
l'ancora; se togli anche (a) e tieni l'aggancio di P:92 alla WLS, la (c) deve essere la banda sganciata dall'ancora.
(c) Nel temporale, a anche oltre 1, con l'intervallo di sigma allargato e fissato ora (nel modello a = 3 chiede sigma fino a circa
    6), oppure una banda sganciata dall'ancora. Se la quota di T non leggibile da L0 è troppo bassa, nessuna a basta: serve allora
    la banda sganciata, oppure una misura di quella quota in forma chiusa sul RHS-base T (P:80), con una regola fissata ora.
(d) Inoltre:
    - griglia di sigma fitta (controlli-3);
    - E_dyn e i suoi accumulatori sempre su y osservato; y oppure y* solo per pose_var e soglia, detto una volta (ricetta-4);
    - soglia di E_real dal fitting inner (V-architettura-controlli-1);
    - in P:92 due grandezze: E_cal, la ridge su y, per l'aggancio a E_real; E_LS*, la LS su y* (stimatore da dichiarare: WLS o
      ridge) valutata su y, per la banda inner e outer, per il margine del 50% e per il denominatore della clausola MLP
      (controlli-4, S-C). Va detto che il cancello del 50% diventa più severo e che la banda su E_LS* rende più probabile «non
      calibrato», ma toglie il «non diagnostico» dello spostamento della banda outer (nel modello il saldo degli arresti va da 1
      caso in meno a 12 in più, su 81). Non è equivalente la banda su y con E_LS* solo nel margine.

BL-2 (controlli-2). «MLP non deve vincere il temporale» è un cancello in P:94-96 e un descrittivo in J:441, senza metrica, margine
né conseguenza.
Controesempio (numeri d'esempio): TCN al 55%, MLP a 0,869 contro TCN a 0,870. P ferma la catena, J prosegue.
Decidi una delle due opzioni e scrivila uguale in P, J e DEC:
- un cancello su E_dyn, sugli held outer, per fold e pooled, per esempio (E_MLP − E_TCN)/(E_LS* − E_oracolo) ≥ q, con q fissato
  ora.
  - Si valuta solo se la TCN supera il 50%; altrimenti valgono gli stati per testa di OB-1.
  - Se non passa nei numeri: «ottimizzazione non risolta» con un allarme aperto sulla TCN, altrimenti «non diagnostico»; un
    allarme della sola MLP non cambia l'esito.
  - Se passa nei numeri ma la MLP ha un allarme aperto: non conta come superato, «ottimizzazione non risolta».
  - Si decide dopo i fit dei controlli.
- oppure un descrittivo, fuori dall'elenco «tutte devono passare», con la frase che un temporale superato con la MLP vicina alla
  TCN non qualifica la lettura temporale.
Con il cancello, la quota di T non leggibile in modo istantaneo diventa una condizione di riuscita. Con bias nulli T può venire
quasi lineare: unità attive per strato e quota del margine recuperata dalla WLS L13, da misurare prima dei fit di controllo
(controlli-7).

BL-3 (ricetta-12). L'allarme «LR ancora al bordo dopo l'estensione» (P:69, J:142, J:471) non ha tolleranza.
- Su L0 e L13 con LR costante tende a vincere il LR più piccolo. È un argomento qualitativo, non misurato.
- L'allarme può quindi scattare per differenze di rumore: nei numeri d'esempio del revisore vince 3e-5 per lo 0,014%.
- Su L0 toglie il primario (P:104), su L13 la candidatura e l'etichetta del contrasto 3.
- Revisore e verificatore lo danno minore. Lo porto fra i blocchi perché nei fit reali dà «non risolta» a un'ottimizzazione
  risolta, e la regola direzionale di OB-4 non lo risolve. Il fermo del controllo lineare lo chiude OB-1.
Proposta:
- sul lato inferiore il bordo è un allarme solo se il LR più piccolo vince di più di una quota relativa fissata ora, per esempio lo
  0,5% della MSE pooled del LR vicino; sotto quella quota si registra e non è un allarme. Vale per tutte le teste, nei fit reali e
  nei controlli: per L0 e L13 un margine grande indica un comparatore ancora sotto-ottimizzato;
- le altre condizioni di P:69 restano allarmi (minimo nelle ultime 5 epoche dell'orizzonte massimo, scelta al primo quarto di
  epoca);
- il lato superiore resta allarme;
- aggiorna anche DEC:6747.

Obbligatorie prima della firma:
OB-1. Nei controlli un allarme da solo non ferma la catena (P:98, J:421, J:471).
      - La catena si ferma per «non calibrato», per «non diagnostico» o se una condizione richiesta del cancello non passa. Gli
        allarmi delle teste che passano si registrano.
      - Se una testa non passa e ha un allarme aperto, l'esito è «ottimizzazione non risolta», altrimenti «ricetta non ammessa».
      - In P:72, P:98, P:104 e §10 un allarme di P:69 prevale su S-2 sulla stessa testa, con la sua conseguenza: nei controlli la
        regola qui sopra, nei fit reali quella di OB-4.
      - «Minimo interno» di S-2 = minimo di E_dyn pooled sugli held interni fra i checkpoint ammessi del LR selezionato, passo 0
        escluso.
      - Ordine degli stati nei controlli: non calibrato; non diagnostico deciso in forma chiusa (P:90, P:95, banda di P:96, verifica
        di BL-1(b) o quota minima della ridge); per ciascuna testa passa, non risolta o non ammessa; poi l'eventuale clausola MLP
        di BL-2.
OB-2. S-2 è un compromesso, non un allarme (P:153, DEC:6747, prompt:18). In P:72 la frase che S-2 non toglie etichette né
      ammissibilità. In DEC:6747 vanno i tre allarmi di P:69: manca quello del primo quarto di epoca.
OB-3. Che cosa veta la candidata. La frase di P:122 «Se MLP o L13 sono risolti migliori…» contraddice la regola iff. Scegli una
      delle due opzioni e scrivila uguale in P:122, J candidate (J:40), J:104, DEC:6753 e §10:
      (a) tenere la regola, mettendo «con etichetta deterioramento_pratico nel contrasto 2 o 3» al posto di «risolti migliori».
          - Il caso misto, cioè un comparatore risolto migliore ma con stima sotto il 2%, prende un nome neutro, con gain e IC99:
            per esempio «candidata, con MLP/L13 risolti migliori (upper99 < 0) ma con stima sotto il 2%». Non «non peggiore».
          - Un lower99 > −2 sarebbe un margine di non inferiorità nuovo, da dichiarare come regola fissata prima dei dati;
      (b) nessuna candidatura se il contrasto 2 o 3 ha upper99 < 0.
      Con S-2 su L13 o MLP, la candidata è «sotto la selezione MSE comune, con compromesso S-2 sul comparatore», senza affermare
      la superiorità sulla lettura lineare temporale (o istantanea). Va scritto in P:122, J candidate, J:104, DEC:6753 e §10.
OB-4. Conseguenza degli allarmi nei fit reali. Oggi è simmetrica (P:104) e toglie anche un beneficio misurato della TCN. Raccomando
      quella direzionale:
      - allarme sulla TCN: si sospendono solo le conclusioni negative o nulle sulla TCN; la condizione di veto scelta in OB-3
        esclude comunque la candidatura;
      - allarme su L0, MLP o L13: si bloccano le letture favorevoli alla TCN contro quella testa, compresa l'assenza della
        condizione di veto;
      - la candidata richiede L0, MLP e L13 ammessi; un allarme della sola TCN-post non la esclude;
      - con un allarme sulla TCN il nome misto di OB-3(a) resta; si sospende solo la sua lettura come conclusione sul valore
        della TCN.
      Riscrivi P:104: con la regola direzionale «niente promozione primaria» vale se L0 non è ammessa, non per un allarme della
      sola TCN-post. Di' anche se la catena prosegue con allarmi aperti. Togli TCN-post dal requisito di ammissione della
      candidata in P:122, J candidate, J:104 e DEC:6753. Se tieni la regola simmetrica, scrivilo nella riga «Esito negativo/irrisolto»
      del §10.
OB-5. Righe del §10.
      - In «Esito positivo»: E_dyn ≤ 0,7630564; MAE e livello solo descrittivi; nessuna affermazione sul pretraining; non è una
        conferma su dati mai visti; le scelte di OB-3 e OB-4.
      - In «Criteri»: copertura solo nominale e nessun bootstrap a due livelli.
      - Spazi fra le parole attaccate.

Da correggere nella stessa revisione (sezione 5, righe «minore»):
- residuo normale con denominatore e test CPU (architettura-2);
- frase della nota 12 a P:122 (architettura-5);
- lettura di C-1 nei due versi a P:126 (criteri-2);
- etichetta 3 con la frase di c15aac9 (criteri-3);
- technical_failure in J con cause e conseguenza, il nome a P:104 e che cosa segue a un fallimento del gate R-1 (criteri-5,
  consegna-2);
- testo del gate R-1, con cast a float64 e 8.329/8.327 (criteri-6).
Per le note della sezione 5 basta una riga di disposizione, accolta o motivata. Comprendono:
- la chiave «superseded», 47ca96e per il rinfresco, 30 combinazioni e la corsia per fase;
- T quasi lineare, da misurare prima dei fit di controllo (controlli-7), e P:122 come destinazione di C-2 (consegna-4);
- sette delle 12 promesse della sezione 4 (note 17, 18, 20, 23, 39, 47 e 49). Le altre cinque stanno in OB-5 e nelle minori
  architettura-5, criteri-2 e criteri-5.
La sezione 6 elenca ciò che spetta al codice e al benchmark.

Che cosa ti chiedo: una revisione sul tuo ramo, sempre REVIEW_REQUIRED, con pin rifatti e il nuovo SHA. Se un rilievo è
sbagliato, dimostralo con una prova. Poi rileggo solo le righe di BL, OB e delle minori, senza agenti. Se reggono, dichiaro la
convergenza documentale e la tabella va a Simone. Le note le ricontrollo alla review del pacchetto operativo. Nessun codice,
benchmark, job o deploy.
```
