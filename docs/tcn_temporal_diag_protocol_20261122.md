# TCN: diagnostici del controllo temporale fallito (D0, D1, D2) — protocollo del 22/11/2026

Stato: **regole congelate prima di qualsiasi job**. Ramo `claude/tcn-temporal-diag-20261122`, base
`e8c9824461b5e18494ff462f1be8abda71b76d5e` (punta di `codex/pose-temponet-launcher-20261120`).

## 1. Incarico, basi dell'autorizzazione e perimetro

Messaggio di Simone del 22/11/2026. Dopo il testo di GPT incollato, il messaggio continua con un riepilogo in tre punti che
comincia con «Procederei così:» e con queste due frasi: «Non abbasserei la soglia ma allungherei subito i training.» e
«Diagnosi e prompt completo pubblicati nel commit `e8c98244`, push verificato.». Chiude con le parole di Simone:

> «GPT ha finito i token, quindi qua, quando hai fatto la tua review e proposto una soluzione puoi lanciare tu direttamente e
> impostarti un timer per guardare ogni 30 minuti»

- **Lettura di Claude, da confermare:** il riepilogo e le due frasi hanno la voce di GPT («push verificato»), ma stanno fuori dal
  blocco incollato. Non so dire se «allungherei subito i training» sia di Simone. La tratto come una richiesta possibile: D2 la
  esegue in forma descrittiva, e D1 segue la regola congelata che prolunga a 300 epoche (sezioni 5 e 6).
- **Basi del lancio.** (a) La frase di Simone qui sopra: review, proposta, poi lancio diretto. La review è
  `docs/reviews/claude_review_pose_temponet_temporal_diagnosis_launcher_20261122.md` sul ramo
  `claude/temponet-db8-20261109`, pubblicata prima del primo invio. (b) `CLAUDE.md`, Simone, 28/10/2026: «se devi sottomettere
  cose che durano meno di 5h-GPU fallo pure in autonomia». L'intera catena prenota 2,40 GPU-ore.
- **Differenza dichiarata da ciò che avevo chiesto io.** La mia review `f6669115` e la proposta di GPT `8fac3a82` chiedevano
  una decisione di Simone registrata sul pacchetto, con ID e costi, prima del primo invio. Simone ha dato il via prima che il
  pacchetto esistesse. ID, costi e sotto-budget sono registrati qui e in `docs/decisioni.md` prima del primo invio; la decisione
  sul pacchetto come tale non c'è.
- Per questa sola catena l'operatore è Claude, in questa chat. Il launcher GPT della linea TCN non invia: nessun secondo
  submitter. Prima di ogni invio si legge il ledger comune.
- La campagna A1 resta `recipe_not_admitted` e ferma. Nessuna soglia cambia. Nessun utente di sviluppo viene letto.
- Qui non si lancia nessuna variante della testa. Il candidato residuale di GPT e ogni altra ricetta restano una decisione
  successiva di Simone, con i risultati in mano.
- Il disegno di D0 e D1 è quello già concordato fra Claude e GPT: proposta `8fac3a8239089d64dbefdf3ad70bd365094d49ff`
  (`codex/pose-temponet-temporal-diagnosis-20261121`), nota di convergenza `4a48b814a32452a22c3525afe375635b251ecc68`.
  Differenze, tutte in questo documento: le letture a 1, 10 e 75 epoche e le letture sui due rami chieste dalla seconda
  diagnosi di GPT (`e8c98244`); il riferimento `cleanLS` di fitting (sezione 4); l'orizzonte di D1 (sezione 5); D2 (sezione 6).

## 2. Codice e identità

- Codice nuovo fuori dal manifest dei sorgenti congelati: `scripts/tcn_temporal_diag/` e
  `scripts/slurm/tcn_temporal_diag{,_dispatch}.sbatch`. I 247 sorgenti congelati non cambiano:
  `source_sha256 = 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea`, codice rivisto
  `a1ae9607681c1a181034d7f72770537054af419d`.
- Il driver importa i moduli congelati senza modificarli e riusa le loro funzioni: lettura verificata degli input, catena delle
  ricevute, costruzione del bundle, traiettorie, calendario dei checkpoint, selezione, valutatore.
- All'avvio ogni job verifica: piano dei controlli `plan_controls.json` (sha256 `b334ece8…04e3`) con le review; sorgenti congelati;
  checkout discendente del codice rivisto; contesto numerico uguale al pin `gpu8` `12cb2a0b…7eb3f`; ricevute di A1 uguali alle
  copie archiviate nel repository; bundle ricostruito uguale, lock per lock, a quello della selezione archiviata; identità della
  valutazione interna uguale.
- Uscite sotto una radice nuova: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_diag_20261122`. Niente si
  scrive sotto la radice di A1. Un job che trova già la sua cartella o la sua ricevuta si rifiuta.

## 3. Job, risorse e costo

| ID logico | Che cosa fa | Risorse | Limite | GPU-ore prenotate |
| --- | --- | --- | --- | ---: |
| `TCND-checkout-20261122` | checkout di campagna `pose_temponet_diag_20261122` | `lrd_all_serial`, 2 CPU, 24 GB | 12 min | 0 |
| `TCND-d0-replay-20261122` | D0: rilettura degli stati archiviati | `boost_usr_prod`, 1 GPU, 8 CPU, 32 GB, QoS `normal` | 20 min | 0,38 |
| `TCND-d1-cleanfit-20261122` | D1: parità del driver e due fit sul bersaglio pulito; parte solo dopo D0 riuscito | come D0 | 75 min | 1,30 |
| `TCND-d2-extend-20261122` | D2: la corsa archiviata sul rumoroso portata da 150 a 300 epoche; parte solo dopo D0 riuscito | come D0 | 40 min | 0,72 |

- GPU-ore prenotate: limite più 180 secondi, come nel piano A1. **Totale 2,40 GPU-ore**, cioè 19,2 core-ore sui nodi GPU e
  19,2 ore locali (un'ora di 1 GPU con 8 CPU conta 8 ore locali), più 0,4 core-ore del checkout (2 CPU per 12 minuti).
- Consumo atteso 0,9-1,3 GPU-ore: D0 circa 4 minuti, D1 circa 25 minuti (45 se prolunga a 300 epoche), D2 circa 25 minuti.
  Durate dal benchmark 65016447 e dai sei controlli: 2,33 s per un'epoca dei due fold, 2,72 s per una valutazione degli esclusi.
- **Addebito e residuo.** Account `IscrB_WearUsFM`, voce nuova «diagnostici del controllo temporale TCN». I tetti di A1 (3,681
  GPU-ore usate su 56,10) non la coprono e non vengono toccati. Residuo dell'account letto il 22/11 alle 11:47: 652.656 ore
  locali su 2.000.000; nel mese 116.468 su 164.383.
- **Sotto-budget di questa catena: 4 GPU-ore prenotate in tutto.** Copre i tre job e al più un rilancio di uno stadio dopo un
  errore tecnico diagnosticato e corretto, con ID e checkout nuovi. Un fermo di parità o un esito scientifico non si rilanciano:
  tornano a Simone. Oltre il sotto-budget serve una decisione nuova.
- Manutenzione di tutto Leonardo dal 23/11 alle 08:00 al 25/11 alle 08:00: un job che non può finire prima resta in coda.

## 4. D0 — rilettura degli stati archiviati

Si rileggono i 20 stati della TCN del controllo temporale (cinque learning rate, stato migliore e stato finale, due fold) e i
quattro stati scelti di L0 e MLP. Ogni stato è ricaricato con `Trajectory.restore`, che ne verifica il checksum, la ricetta, il
contesto numerico, l'inizializzazione e il set di training bit per bit. Il punteggio è quello congelato, a blocchi di 32
finestre, contro `y` e contro `y*`, sugli esclusi interni e sul fitting.

**Parità a due livelli, fissati qui prima dei job.** A1 non ha mai mostrato che due job su nodi diversi diano gli stessi bit:
ogni job ha verificato la parità solo al proprio interno. Per questo i livelli sono due.
- **Esatta:** `|nuovo − archiviato| <= 1e-12 + 1e-12·|archiviato|`, la tolleranza della selezione congelata. È l'esito atteso.
- **Numerica:** differenza relativa fino a 1e-4 sui punteggi, con tutte le identità discrete uguali (update, numero di finestre,
  hash dei pesi iniziali, del set di training e del mescolamento). È l'arrotondamento FP32 fra due nodi dello stesso modello: uno
  stato, un bersaglio o un flusso casuale diversi spostano i punteggi di molto più. Le letture restano valide e portano la
  marca `numerical`.
- **Fallita:** oltre 1e-4, oppure un'identità discreta diversa. Niente viene allentato dopo.

È una differenza dalla proposta `8fac3a82`, che prevedeva il solo livello esatto.

**Cancelli di validità.** Se uno fallisce il job chiude con stato `gate_failed` e D1 e D2 non partono:
1. **Parità**, esatta o numerica. Per ogni coppia di stati della TCN (10) e per la coppia scelta di L0 e di MLP: MSE ed `E_dyn`
   pooled contro il punto archiviato della curva, stessi update. Per lo stato scelto di ogni testa anche gli accumulatori per
   fold: esclusi contro `y`, esclusi contro `y*`, fitting contro `y`.
2. **Aritmetica del cancello.** I recuperi dello stato scelto della TCN, riletto, coincidono con quelli del cancello archiviato
   (0,24173052888786836 / 0,1887545039561508 / 0,21710109906955868): entro `1e-12·(1+|R|)` se la parità è esatta; se è
   numerica, entro `1e-12·(1+|R|) + 2,2·1e-4·E_dyn/(E_cleanLS − E_oracolo)` per pannello, cioè circa 0,0027. È lo spostamento
   del recupero che corrisponde a una differenza relativa di 1e-4 sui due accumulatori dinamici. La funzione congelata
   `temporal_head_gate` deve ridare lo stato `recipe_not_admitted`.
3. **Riferimenti.** L'oracolo sugli esclusi, ricalcolato dagli array, coincide con quello archiviato entro 1e-9 relativo.

**Riferimenti di fitting.** `R_fit` usa `cleanLS` stimato su `y*` e valutato su `y` sul pannello di fitting, preso dalla
diagnostica lineare archiviata e autenticata (`temporal_margin_clean_fit_statistics`), e l'oracolo calcolato dagli array. È una
differenza dalla proposta `8fac3a82`, che chiedeva di ricalcolare entrambi. `cleanLS` viene anche ricalcolato dagli array con la
funzione congelata, come controllo (entro 1e-9) e per avere il suo errore contro `y*`: è una lettura descrittiva, non un cancello.

**Letture, tutte descrittive.** Per ogni stato, per fold e pooled, su fitting ed esclusi:
- MSE ed `E_dyn` contro `y` e contro `y*`, con numeratori e denominatori;
- recupero sulla scala del cancello contro `y`;
- scomposizione esatta dell'errore contro `y` sullo stesso supporto: `E_dyn = errore pulito + livello del rumore + allineamento`,
  dove `allineamento = −2<e,n>/TSS`, `e = predizione − y*`, `n = y − y*`. Deriva da
  `SSE_y = SSE_y* + SSE_oracolo − 2<e,n>`. Si riporta anche `<e,n>/<n,n>`: 0 se la predizione ignora il rumore realizzato, 1 se
  lo riproduce;
- differenza esclusi meno fitting dei tre termini, con segno, per lo stato scelto (epoca 31) e per quello finale (epoca 150);
- quota dell'errore pulito di `cleanLS` che la testa toglie, contro `y*`;
- i due rami della testa: per ogni ReLU del ramo profondo la frazione di attivazioni positive e il numero di unità mai attive
  sulle finestre valutate; energia del ramo affine e del ramo profondo, anche centrata nel tempo; scarto della LayerNorm
  dall'identità. Si leggono con ganci di sola lettura durante il punteggio: nessun passaggio in più, nessuna modifica;
- loss di training per epoca, in modalità train;
- statistiche del generatore conservate in `calibration.pt` (unità attive per strato, medie e scale);
- il generatore scritto dentro la testa (L nel ramo affine, T nei primi otto canali) e valutato dal valutatore congelato sui
  latenti veri: dice che cosa dà il percorso di punteggio per la funzione vera.

**Nessuna etichetta.** D0 riporta i numeri e non assegna cause: lo scarto fra fitting ed esclusi si legge nei tre termini, che
sommano esattamente. Che cosa misura ciascuno:
- l'**allineamento** dice quanto la predizione riproduce il rumore realizzato di quel pannello;
- l'**errore pulito** dice quanto la predizione dista dalla funzione vera su quel pannello;
- il **livello del rumore** dice solo quanto i due pannelli differiscono per costruzione.

**Previsioni di Claude, scritte prima dei risultati.** Sono previsioni, non regole:
1. nel pannello pooled, allo stato scelto, la differenza di allineamento è più di metà dello scarto esclusi meno fitting di
   `E_dyn`. Se è la differenza di errore pulito a superare la metà, la previsione è sbagliata;
2. nello stato finale del learning rate 0,01 l'ultimo strato nascosto del ramo profondo ha almeno 28 unità su 32 mai attive sul
   fitting, in entrambi i fold; sbagliata altrimenti. Con meno di 16 in entrambi i fold il sospetto del ramo spento cade; fra
   16 e 27 l'esito è intermedio e si riporta così;
3. nello stato scelto, a 0,003, quelle unità mai attive sono meno di 16 su 32 in entrambi i fold; sbagliata se sono almeno 16
   in un fold.

## 5. D1 — parità del driver e due fit sul bersaglio pulito

**Unico intervento:** il bersaglio di training è `y*` invece di `y`. Restano quelli di A1: testa `TCN_post`, seme 0,
inizializzazione, ordine dei batch, flusso del dropout, dropout 0,1, weight decay 0,01, clip 1,0, batch 32, loss, learning rate
0,003, calendario dei checkpoint, valutatore. Niente griglia, niente altri semi.
Il learning rate viene dal mondo già osservato: è un diagnostico esplorativo, non una conferma indipendente.

**Orizzonte: la regola congelata della ricerca.** 150 epoche; si prosegue a 300 solo se il minimo pooled della MSE oppure di
`E_dyn`, contro `y`, cade nelle epoche 146-150 (`selection.needs_epoch_extension`, come in `train._run_search`). È una
differenza dalla proposta `8fac3a82`, che fissava 150 epoche senza estensione: così D1 è la ricerca congelata a un learning
rate, con il solo bersaglio di training cambiato.

**Cancello di parità, prima dei fit.** Lo stesso driver, con bersaglio di training `y`, deve ripercorrere la prima epoca
archiviata del learning rate 0,003:
- pesi iniziali e set di training identici a quelli archiviati (hash);
- cinque punti della curva, esposizioni 0, 0,25, 0,5, 0,75 e 1, con update `[0,0] [8,8] [15,15] [22,23] [29,30]`: MSE ed
  `E_dyn` pooled con i due livelli di D0;
- hash del primo mescolamento uguale a quello archiviato e loss di training della prima epoca con i due livelli, per fold.

Con parità solo numerica i fit puliti partono, ma la traiettoria non è più identica bit per bit a quella di A1: il confronto
fra A1 e D1 alla stessa esposizione porta quella marca.

Durante la parità si eseguono a ogni punto le stesse valutazioni in più dei fit puliti: se spostassero la traiettoria, la
parità fallirebbe. Il ciclo dei fit puliti è quello congelato (`PairedCandidate.advance`); un test su CPU verifica che con le
valutazioni in più la curva e gli stati restano identici a quelli del ciclo senza.
Se la parità fallisce il job chiude con `gate_failed` e i fit puliti non partono.

**Selezione, invariata.** Punto scelto: minimo della MSE pooled contro `y` sugli esclusi interni, esposizione positiva, a parità
la prima. Le curve contro `y*` sono descrittive e non scelgono niente. Lo stato scelto, ricaricato, deve ridare il suo punto
della curva entro la tolleranza.

**Valutazioni in più.** A ogni checkpoint: esclusi contro `y*`, fitting contro `y` e contro `y*`, rami e LayerNorm, in modalità
di valutazione. Non estraggono numeri casuali. Si salvano gli stati alle esposizioni 1, 10, 31, 75, 150 (e 200, 250, 300 se si
prosegue), quelli scelti e quelli finali.

### Regola di lettura, in ordine e senza sovrapposizioni

Per ogni pannello `p` fra fold 0, fold 1 e pooled:
- `R_held,p` è la formula del cancello `(E_cleanLS − E_TCN)/(E_cleanLS − E_oracolo)` su `E_dyn` contro `y`, sugli esclusi, al
  punto scelto, con i riferimenti archiviati del cancello;
- `R_fit,p` è la stessa formula sul fitting, contro `y`, allo stesso punto scelto, con i riferimenti di fitting della sezione 4;
- il pooled somma gli accumulatori dei due fold prima dei rapporti.

`bordo` è vero solo se la corsa è arrivata a 300 epoche e il minimo pooled della MSE oppure di `E_dyn`, contro `y`, cade nelle
epoche 296-300 (la regola di `selection.assessment`). A 150 epoche il bordo non resta mai aperto: fa proseguire.
Soglia: 0,50, la costante congelata `TEMPORAL_HEAD_RECOVERY`.

| Priorità | Condizione | Esito |
| --- | --- | --- |
| 0 | un cancello di validità fallito, lo stato scelto che non ridà il suo punto, oppure un recupero non calcolabile (riferimenti non finiti o margine non positivo) | **non valutabile** |
| 1 | `R_held,p >= 0,50` nei tre pannelli | **sensibilità recuperata allenando sul pulito**; il bordo si annota soltanto |
| 2 | almeno un `R_held,p < 0,50` e `bordo` vero | **non risolto entro 300 epoche**; nessuna causa assegnata |
| 3a | almeno un `R_held,p < 0,50`, `bordo` falso, `R_fit,p >= 0,50` nei tre pannelli | **recupero raggiunto sul fitting, non in tutti i pannelli esclusi** |
| 3b | come sopra, `R_fit,p < 0,50` nei tre pannelli | **recupero insufficiente anche sul fitting** |
| 3c | come sopra, fitting misto | **esito eterogeneo** |

- I confronti usano i numeri non arrotondati. Nessun esito dichiara un ottimo, un'impossibilità statistica o una causa unica.
- **Un esito 1 non ammette la ricetta:** il cancello vero resta quello sul bersaglio rumoroso, fallito il 21/11. Dice che la
  testa, la ricetta e i 20 utenti per fold bastano a leggere la funzione quando le etichette sono pulite.
- Letture descrittive dichiarate prima: tutte le grandezze della sezione 4 alle esposizioni 1, 10, 31, 75, 150 e al punto scelto.
  Il confronto fra A1 e D1 all'esposizione 31, sullo stesso `y`, misura l'effetto del bersaglio di training a parità di tutto il
  resto, per questi dati e questi semi.

**Previsione scritta prima dei risultati** (review `f6669115`): D1 passa, esito 1. Se non passa, la diagnosi cambia direzione.

## 6. D2 — la corsa archiviata sul rumoroso, portata a 300 epoche

Risponde alla frase «allungherei subito i training» con una misura. Le due traiettorie del learning rate scelto, 0,003, sono
ricaricate dagli stati finali archiviati a 150 epoche, con ottimizzatore, mescolamento e dropout, e proseguono fino a 300 epoche
sul bersaglio rumoroso `y`, con il calendario e il valutatore congelati. 300 è il tetto della ricetta congelata.

- **Cancello di validità:** la coppia ricaricata deve ridare il punto archiviato a 150 epoche, con i due livelli di D0.
  Se la parità è fallita il job chiude con `gate_failed` senza allenare.
- **Solo descrittivo.** Non si seleziona niente e non si valuta nessun cancello. Qualunque valore esca, la selezione archiviata
  (epoca 31) e lo stato `recipe_not_admitted` di A1 non cambiano. Un punto oltre le 150 epoche che superasse la soglia sarebbe
  un'osservazione fatta dopo, sullo stesso mondo: andrebbe confermata su un mondo nuovo, come ogni variante.
- A ogni epoca si leggono le stesse grandezze della sezione 4. Si salvano gli stati a 200, 250 e 300 epoche.
- Non è la ricerca congelata: quella prolunga solo quando il minimo cade nelle ultime cinque epoche, e qui non accadeva.

**Previsioni scritte prima dei risultati:**
1. in nessuna epoca fra 151 e 300 l'`E_dyn` pooled sugli esclusi scende sotto quello del punto scelto all'epoca 31 (0,715926).
   Se un'epoca scende sotto, la previsione è sbagliata e allungare aiuta;
2. `<e,n>/<n,n>` pooled sul fitting all'epoca 300 è maggiore che all'epoca 150; sbagliata altrimenti.

## 7. Dopo i job

- Ricevute, JSON dei risultati, log e `sacct` si raccolgono sul Mac e si committano su questo ramo. Gli stati dei fit restano
  su `$WORK`.
- Nessun passo successivo parte da solo. Con i numeri in mano Simone decide se e quale variante provare. Restano valide le
  condizioni già concordate: variante fissata prima, conferma su un mondo nuovo con semi `20261122 + 1000·fold`, un solo
  tentativo.

## 8. Fase 2 — esplorazione della dose di weight decay (D3), 23/11/2026

Stato: **regole congelate prima dei job**. Esiti di D0, D1 e D2 in `docs/decisioni.md`, voce del 22/11 «esiti dei diagnostici».

Incarico. Alla domanda «Che cosa vuoi provare come variante, da fissare prima e confermare su un mondo nuovo?» Simone ha
risposto, il 23/11: «niente gpt per ora, vai avanti tu». Lettura di Claude: scegliere la variante, fissarne le regole e lanciare
la parte che rientra nell'autonomia del 28/10 («se devi sottomettere cose che durano meno di 5h-GPU fallo pure in autonomia»).
La conferma su un mondo nuovo è un passo separato (sezione 8.6).
La griglia delle quattro dosi è la selezione interna che la proposta `8fac3a82` rimandava al protocollo operativo; l'unico
tentativo di variante («un solo tentativo di variante, poi a Simone», condizione 2 della convergenza `4a48b814`) è la conferma
della sezione 9. Operatore unico di D3 e della conferma: Claude.

### 8.1 Perché il weight decay

- Con etichette pulite la stessa testa e la stessa ricetta arrivano al 94,6% (D1): non mancano capacità, rappresentazione o
  ottimizzazione.
- Con il rumore la TCN impara il rumore del fitting mentre impara la funzione: allo stato scelto il 79% dello scarto fra fitting
  ed esclusi è allineamento al rumore, e l'errore pulito è tolto per un quinto sugli esclusi (D0). Allungare peggiora (D2).
- La regolarizzazione esplicita oggi è quasi assente: AdamW con weight decay 0,01 a LR 0,003 contrae i pesi del 2,7% in 899
  passi. Restano l'arresto anticipato e il dropout 0,1 su due strati del ramo profondo.
- Nella simulazione giocattolo del 22/11 un weight decay di 0,3 portava il recupero da 0,31 a 0,42 (seme 0) e da 0,29 a 0,37
  (seme 2): è l'unica modifica singola provata che alzasse il livello. Evidenza: ramo `claude/temponet-db8-20261109`, commit
  `c6a530d5`, `docs/reviews/claude_review_pose_temponet_temporal_diagnosis_launcher_20261122_evidence/residual_candidate/notes.md`,
  §3d. Limiti: latenti sintetici, un solo fold, uno o due semi, al più 40 epoche con tetti di tempo; il weight decay 0,3 al
  seme 2 e il weight decay 1 erano ancora in salita a 22 epoche; nessuna variante arrivava a 0,50.
- La scelta è quella scritta prima di D0 e D1 nella review del 22/11 (`c6a530d5`, sezione 6): se D0 trova il ramo sano e D1
  passa, la leva è la regolarizzazione. D0 ha trovato il ramo profondo vivo a LR 0,003 (al più un'unità mai attiva per strato)
  e spento soltanto a 0,01; D1 è passato.
- È un solo parametro della ricetta comune: cambia nello stesso modo L0, MLP e TCN e lascia intatto il confronto fra le teste.

### 8.2 Disegno

- Dosi: 0,1, 0,3, 1 e 3. Il valore congelato è 0,01. Unica modifica: il weight decay di AdamW, applicato come nella ricetta
  congelata ai soli pesi di matrici e kernel.
- Celle: la TCN nel mondo temporale; TCN, L0 e MLP nel mondo lineare. Mondo originale: calibrazioni, bersagli, utenti e fold di A1.
- Per ogni dose e cella: learning rate 0,003, seme 0, inizializzazione, ordine dei batch e flusso del dropout di A1; calendario
  dei checkpoint, estensione a 300 epoche e selezione congelati (minimo della MSE pooled contro `y` sugli esclusi interni, a parità
  la prima esposizione).
- Nel test su CPU del mondo sintetico, con la ricetta privata di poche epoche, il driver al weight decay congelato ridà
  esattamente curve e stati archiviati per ogni learning rate
  (`test_variant_fit_at_the_frozen_weight_decay_is_the_archived_curve_of_that_learning_rate`). Con dati di forma di produzione il
  passaggio dalla ricetta congelata alla variante al weight decay congelato non cambia nessun numero
  (`test_frozen_weight_decay_variant_trains_bit_for_bit_like_the_frozen_recipe`).
- Sui dati veri ogni cella confronta il proprio punto a esposizione 0 con quello della curva archiviata allo stesso learning
  rate, con i due livelli di D0: esatto a 1e-12, oppure numerico a 1e-4 con esposizione e passi uguali. Se fallisce, il job
  chiude `gate_failed` (8.3). All'inizio la testa dà un'uscita costante: il punto 0 controlla bersagli, utenti e valutatore;
  l'uguaglianza dell'addestramento con A1 resta mostrata dal solo test su CPU.
- Per la TCN la differenza fra A1 e D3 è quindi la sola dose. Per L0 e MLP anche il learning rate: nel mondo lineare A1 li ha
  scelti a 1e-4, D3 li allena a 0,003, e la loro ammissibilità è il cancello congelato a quel learning rate. Sulle curve
  archiviate a 0,003, al minimo della MSE e con il weight decay congelato, il recupero pooled del cancello lineare sta fra 0,98 e
  0,99 per le tre teste (calcolo della review del 23/11; i valori per fold non sono archiviati).
- La ricetta variante è marcata `test_only`: nessun punto d'ingresso congelato la accetta.
- È un'esplorazione sul mondo in cui A1 ha già scelto learning rate ed epoca: serve a scegliere, non a confermare.
- Letture: recupero del cancello temporale congelato (fold 0, fold 1, pooled) per la TCN; cancello lineare congelato (MSE ed
  `E_dyn` nei tre pannelli) per le tre teste. Descrittivi come in D1: ramo profondo, rumore riprodotto, letture alle esposizioni
  1, 10, 31, 75, 150 (e 200, 250, 300 se si prosegue).

### 8.3 Regola di scelta, fissata prima

- Una dose è **ammissibile** se ha i risultati di tutte e quattro le celle, il punto scelto e il punto a esposizione 0 di ogni
  cella si riproducono (8.2) e le tre teste passano il cancello lineare congelato (recupero almeno 0,90 in MSE ed `E_dyn` nei tre
  pannelli, `controls.linear_head_gate`). Una dose senza tutte le celle, per esempio per un job interrotto, non è ammissibile.
- Contano solo le sintesi dei job con ricevuta `completed`. Un punto scelto o a esposizione 0 non riprodotto, a una dose
  qualsiasi, chiude il suo job `gate_failed`: come per un TIMEOUT, le sue celle mancano a tutte le dosi, nessuna dose è
  ammissibile e la linea torna a Simone. Il controllo riga per riga di `core.d3_results` resta come seconda difesa.
- Si **sceglie** la dose ammissibile con il recupero temporale pooled più alto; a parità, la dose più piccola.
- La **conferma** su un mondo nuovo si propone solo se la dose scelta raggiunge 0,50 nei tre pannelli del mondo temporale
  originale, la soglia congelata `TEMPORAL_HEAD_RECOVERY`. Altrimenti la linea torna a Simone con la curva dose-risposta.
- È voluto che la conferma riguardi la sola dose scelta: se questa manca 0,50 in un pannello, la conferma non si propone anche
  quando un'altra dose ammissibile passasse i tre pannelli. Si conferma la dose che l'esplorazione stessa mette in cima; una
  dose prima nel pooled ma sotto soglia in un fold è un risultato instabile.
- La regola è `core.choose_dose` su `core.d3_results` e richiede le quattro dosi registrate. La applicano
  `results/infra/tcn_temporal_diag_20261122/summarize.py` dopo la raccolta e il driver all'inizio di ogni ricerca della fase 3.
- Limiti dichiarati: un solo learning rate, un solo seme, il mondo già usato da A1. Il cancello di ogni cella non riceve gli
  allarmi d'ottimizzazione: cambia solo l'etichetta dei casi non passati (`recipe_not_admitted` invece di
  `optimization_unresolved`), non `passed` né i recuperi, che sono ciò che la regola usa.

### 8.4 Previsioni di Claude, scritte prima

Le previsioni 1 e 2 sono state precisate il 23/11 dopo la review, prima dei job, perché ogni esito le renda giuste o sbagliate.

1. Il recupero temporale pooled della TCN ha il massimo a una dose interna. Giusta se, fra le quattro dosi (ammissibili o no),
   il massimo cade a 0,3 o a 1; sbagliata se cade a 0,1 o a 3. A parità conta la dose più piccola, come nella regola.
2. La regola non propone la conferma: la dose scelta non raggiunge 0,50 nei tre pannelli. Sbagliata se la regola la propone.
3. Il massimo pooled fra le quattro dosi (ammissibili o no) sta fra 0,25 e 0,40. Sbagliata fuori da questo intervallo.

Se manca il recupero temporale di una dose (`d3t` non concluso), le previsioni 1 e 3 non sono valutabili. Se `d3t` chiude
`gate_failed`, i recuperi si leggono dai JSON di cella come descrittivi.

### 8.5 Job, risorse e costo

| ID logico | Celle | Limite | GPU-ore prenotate |
| --- | --- | --- | ---: |
| `TCND-checkout2-20261123` | checkout di campagna `pose_temponet_diag_20261123` | 12 min, `lrd_all_serial` | 0 |
| `TCND-d3t-wd-20261123` | TCN, mondo temporale, quattro dosi | 111 min | 1,90 |
| `TCND-d3l-wd-20261123` | TCN, mondo lineare, quattro dosi | 110 min | 1,88 |
| `TCND-d3s-wd-20261123` | L0 e MLP, mondo lineare, quattro dosi | 69 min | 1,20 |

- Risorse dei tre job GPU come D0-D2: `boost_usr_prod`, 1 GPU, 8 CPU, 32 GB, QoS `normal`; tutti `afterok` sul checkout.
  Ognuno richiede anche la ricevuta completata di D0.
- **Totale prenotato 4,98 GPU-ore** (limite più 180 secondi), cioè 39,9 ore locali, più 0,5 core-ore del checkout: sotto le 5
  GPU-ore dell'autonomia del 28/10. Consumo atteso circa 2,3 GPU-ore: 13,4 minuti per una TCN a 150 epoche, 4 per L0 e MLP.
  Caso peggiore stimato dalle review del 23/11, con tutte e quattro le dosi prolungate a 300 epoche: `d3t` 102-109 minuti,
  `d3l` 101-108, `d3s` 65-68. I limiti lo coprono, con un margine di pochi minuti; dopo la verifica delle correzioni sono stati
  ribilanciati (111 / 110 / 69 al posto di 112 / 110 / 68) a parità di totale, per non lasciare `d3s` al margine più stretto.
  I risultati di ogni dose si scrivono appena finisce.
- Nessun rilancio dentro questa fase: con un rilancio si supererebbero le 5 GPU-ore, quindi serve una decisione nuova. Un job
  in TIMEOUT non scrive la sintesi: tutte le dosi della sua cella mancano, nessuna dose è ammissibile e la linea torna a Simone.
- Il checkout nasce dal commit che chiude le review delle fasi 2 e 3; il suo SHA completo è registrato nel registro condiviso e
  nella sequenza operativa (`results/infra/tcn_temporal_diag_20261122/sequenza_operativa_fasi_2_3.md`) prima dell'invio.
- Leonardo è in manutenzione dal 23/11 alle 08:00 al 25/11 alle 08:00, e il certificato ssh va rinnovato: l'invio avviene dopo.
- **Uscite su `$SCRATCH`, 27/11.** Con `$WORK` a 991 GB su 1 TB di quota, Simone: «le prossime cose salvale su scratch». D3 e la
  fase 3 scrivono sotto `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pose_temponet_diag_20261123`, i log
  sotto `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/logs`. D0-D2 restano sotto `$WORK`; da lì gli stadi nuovi
  leggono solo la ricevuta di D0. Ogni stadio nuovo scrive per prima cosa un piccolo indicatore nella radice: se il percorso non
  è scrivibile, il job si ferma in pochi secondi e non dopo il primo fit. Il checkout del codice (circa 119 MB) resta su
  `$WORK`, dove lo crea la procedura approvata. Su `$SCRATCH` i file non usati per 40 giorni vengono cancellati: JSON, ricevute e
  log si raccolgono nel repository subito dopo i job; gli stati `.pt` restano lì e sono rigenerabili. Cambia solo dove si
  scrive: regole, soglie, dosi, limiti e costi sono invariati.

### 8.6 Passo successivo, non incluso: conferma su un mondo nuovo

Solo se la regola 8.3 lo propone. Semi `20261122 + 1000·fold` (L = base, T = base + 1, rumore = base + 2); nuova calibrazione
con la regola congelata, sui soli inner-train; sei controlli (tre teste, due mondi) con la dose scelta e la ricerca congelata
completa; cancello congelato sul mondo nuovo; un solo tentativo. Un fallimento torna a Simone. Costo stimato: circa 2 core-ore
CPU per le calibrazioni e 3,6 GPU-ore misurate per i controlli, con una prenotazione maggiore: serve l'approvazione di Simone.
Approvata da Simone il 23/11: regole nella sezione 9.

## 9. Fase 3 — conferma su un mondo nuovo (B), regole fissate il 23/11/2026 prima di ogni risultato di D3

Approvazione. Alla domanda «Approvi già ora la conferma su un mondo nuovo, se l'esplorazione rispetta la regola fissata?»
Simone ha risposto, il 23/11: «Sì, approvo». La fase 3 parte quindi se, e solo se, la regola 8.3 la propone. La regola la
applica il driver stesso, sulle tre sintesi autenticate di D3: nessuna scelta a mano.
La domanda non riportava il costo della fase 3, come chiede CLAUDE.md per ogni richiesta di conferma: prima dell'invio Simone
conferma la prenotazione della sezione 9.3. Senza quella conferma la fase 3 non si invia.
Conferma del costo, 23/11: alla domanda «Confermi per la fase B 9,30 GPU-ore prenotate (74,4 ore locali)?», accompagnata nel
messaggio dalle altre cifre (12,1 core-ore CPU, consumo atteso circa 4-5 GPU-ore, residuo dell'account letto il 22/11: 652.656
ore locali su 2.000.000), Simone ha risposto «Sì, confermo». La fase 3 resta subordinata alla regola 8.3 e a un solo tentativo.

### 9.1 Il mondo nuovo

- Semi dichiarati prima di D1 (proposta `8fac3a82`): `base = 20261122 + 1000·fold`; L con `base`, T con `base + 1`, rumore con
  `base + 2`. Fold 0: 20261122 / 20261123 / 20261124; fold 1: 20262122 / 20262123 / 20262124.
- Generatore: `controls.generate_components` congelato, con il solo seme come parametro (`core.generate_components_seeded`).
  Al seme congelato ridà il generatore di A1 bit per bit (test).
- Calibrazione: `controls.build_positive_control_fold` congelato, sui soli inner-train, con la stessa regola per `a` e `sigma`, sugli
  stessi utenti, fold e latenti. Contesto numerico `cpu8` uguale a quello delle calibrazioni di A1: modello di CPU, versioni delle
  librerie, thread e flag. Se il nodo non lo riproduce, la calibrazione si ferma prima di ogni lavoro e la linea torna a Simone.
- Se la calibrazione non è `calibrated`, o i riferimenti in forma chiusa non sono diagnostici, la fase si ferma: non si estrae un
  altro seme e la linea torna a Simone.
- Non sono utenti nuovi: è un'altra funzione e un'altra realizzazione del rumore sugli stessi latenti.

### 9.2 I controlli e il cancello

- Sei ricerche: L0, MLP e TCN, nei mondi lineare e temporale del mondo nuovo, con la ricetta congelata e il solo weight decay
  della dose scelta da 8.3. Ricerca congelata completa: griglia dei learning rate con le estensioni ai bordi, 150 epoche e 300
  secondo la regola, selezione MSE (`workflow._run_selection`).
- Cancello: `workflow.qualify_controls` congelato sulle due calibrazioni nuove e sulle sei selezioni; l'esito è il suo campo
  `status`. È `passed` se i tre cancelli lineari (recupero almeno 0,90 in MSE ed `E_dyn` nei tre pannelli) e quello temporale della
  TCN (almeno 0,50 nei tre pannelli) sono `passed`. Gli allarmi d'ottimizzazione si registrano e cambiano solo l'etichetta di un
  mancato superamento: `optimization_unresolved` invece di `recipe_not_admitted`. Un recupero sopra soglia con un allarme resta
  `passed`, come nel cancello archiviato di A1, dove L0 lineare passa con l'allarme `minimum_in_last_five_maximum_epochs`.
- **Un solo tentativo.** Un esito diverso da `passed`, un errore o un fermo tornano a Simone, senza un'altra dose, un altro seme o
  un'altra ricetta. Anche con `passed` la variante non entra da sola nella catena congelata: serve una ricetta congelata nuova,
  con identità e review sue, e la decisione di Simone.
- Il mondo originale non si rifà con la ricerca completa: per il confronto con A1 restano le letture di D3 a LR 0,003.
- Limite dichiarato: la ricerca completa può scegliere un learning rate diverso da 0,003, e il weight decay effettivo per passo
  (prodotto di learning rate e dose) cambia con esso; a 1e-3 è tre volte più debole. La conferma prova quindi la ricetta con la
  dose scelta e la ricerca congelata, cioè ciò che entrerebbe nella catena, non la coppia di D3 (LR 0,003, dose).

### 9.3 Job, risorse e costo

| ID logico | Che cosa fa | Risorse | Limite |
| --- | --- | --- | --- |
| `TCND-b-calib-0-20261123`, `TCND-b-calib-1-20261123` | calibrazione del mondo nuovo, un fold ciascuno | `lrd_all_serial`, 8 CPU, 24 GB | 40 min |
| `TCND-b-controls-{linear,temporal}-{L0,MLP}-20261123` | quattro ricerche di L0 e MLP | `boost_usr_prod`, 1 GPU, 8 CPU, 32 GB | 60 min |
| `TCND-b-controls-{linear,temporal}-TCN_post-20261123` | due ricerche della TCN | come sopra | 150 min |
| `TCND-b-gate-20261123` | cancello congelato sul mondo nuovo | `lrd_all_serial`, 2 CPU, 16 GB | 15 min |

- Le uscite della fase 3 stanno sotto la stessa radice di D3 su `$SCRATCH` (8.5).
- La fase 3 si invia solo dopo la raccolta di D3, con le tre ricevute di D3 `completed`, se la regola propone la conferma e Simone
  ha confermato il costo. Ogni stadio della fase 3, calibrazioni comprese, rilegge la regola dalle sintesi autenticate di D3 prima
  di qualunque setup: se non la propone, si ferma `gate_failed` senza lavoro.
- Le ricerche partono `afterok` sulle due calibrazioni, il cancello `afterok` sulle sei ricerche. Ogni stadio verifica anche le
  ricevute completate dei predecessori. Un cancello congelato non superato dentro uno stadio (un generatore degenere, una
  calibrazione non `calibrated`) si registra `gate_failed`, con `failed_gate.json`, come nel runner congelato.
- **Prenotato: 9,30 GPU-ore** (limite più 180 secondi), cioè 74,4 ore locali sui nodi GPU, più 11,5 core-ore delle calibrazioni
  e 0,6 del cancello. Consumo atteso circa 4-5 GPU-ore: le sei ricerche di A1 hanno usato 3,57 GPU-ore, e con un weight decay più
  forte i minimi possono cadere più tardi e allungare le ricerche.
- Caso peggiore della ricerca congelata: cinque learning rate (i quattro iniziali e una sola estensione a un bordo), tutti a 300
  epoche. Con i costi unitari misurati in A1 sono circa 129 minuti per una TCN, 43 per l'MLP e 39 per L0, setup compreso: i limiti
  di 150 e 60 minuti li coprono. Le calibrazioni di A1 sono durate 6,5 minuti l'una, contro un limite di 40.
- Addebito: account `IscrB_WearUsFM`, voce «diagnostici del controllo temporale TCN», fuori dai tetti di A1. È sopra le 5 GPU-ore:
  lo copre l'approvazione di Simone qui sopra, con la conferma del costo chiesta nella sezione 9. Nessun rilancio senza decisione
  nuova.
- Stato della voce: la catena D0-D2 aveva un sotto-budget di 4 GPU-ore e ne ha usate 0,745; D3 ha la sua prenotazione di 4,98 e
  la fase 3 di 9,30, senza un tetto comune della voce. Residuo dell'account letto il 22/11 alle 11:47: 652.656 ore locali su
  2.000.000 (nel mese 116.468 su 164.383); si rilegge prima dell'invio.
