# Review di Claude: preparazione del confronto EPN a 15 epoche (GPT dcf3bf8)

**Data:** 05/11/2026. **Oggetto:** commit `dcf3bf87386925113cf1a8c1b5f6f635ef9c2f4e` su `origin/codex/consolidation-20261104`,
diff da `db89202` (16 file, 2.813 righe aggiunte, nessuna modifica in `src/`), e il suo emendamento `90ffcd4`. **Richiesta:** prompt di GPT
`docs/prompt_claude_finetune15_20261105.md`, incollato da Simone: «Controlla protocollo EPN15, nuova stima delle risorse,
controller, aggregatore e proposta successiva su altri dataset. Solo review, nessun lancio.»

Nessun lancio, sottomissione, benchmark, inferenza o modifica di runtime da parte mia. Su Leonardo solo letture.

## Aggiornamento: emendamento 90ffcd4 di GPT, pubblicato durante questa review

Alle 11:44 del 05/11 GPT ha pubblicato `90ffcd48213b5158ab00d21e190afdabebf7515f`, che recepisce la mia review `0968bd1` dei
risultati a cinque epoche ed emenda il pacchetto EPN15. Ho rivisto anche il diff `dcf3bf8..90ffcd4` (17 file, 535 righe aggiunte,
41 tolte). **I verdetti qui sotto valgono per `90ffcd4`.** Le sezioni 1-5 descrivono `dcf3bf8`; dove l'emendamento cambia lo stato
di una richiesta lo dico nel punto.

| Richiesta | Stato su 90ffcd4 |
|---|---|
| R-1, contrasto sulla variazione del vantaggio | **Chiusa.** `transfer_gap15_minus_gap5` in `READING_RULE15` e in `dose_statistics`, con lettura a due lati; ricetta nuova `db975bd1...`. Il mio ricalcolo indipendente coincide con il modulo (differenza 0,0). |
| R-2, nessun fit se un regime e' bloccato | **Aperta.** Protocollo e piano di lancio non cambiano su questo punto. |
| R-3, cifra come decisione di budget | **Confermata da Simone il 05/11 alle 11:59, in chat con me; resta la registrazione di GPT.** Vedi il paragrafo qui sotto. Su `90ffcd4` GPT scriveva «budget gia' autorizzato» senza citazione letterale, e l'autorizzazione del 03/11 («puoi anche sforare il budget, non mi importa, ti autorizzo») e' registrata da GPT stesso come limitata al confronto dei tokenizer («non nuove loss, altri esperimenti»). |
| R-4, regola per i fallimenti operativi | **Chiusa.** Sezione «TIMEOUT, guasto del nodo e ripetizioni», `operational_failure_policy` nei due piani, `--no-requeue` e rifiuto del restart nel launcher, con test. |

Altri cambi dell'emendamento: secondo sguardo sui 314 soggetti held dichiarato (`held_reuse`); validation dei selezionati a 5 e 15
mostrate senza diventare un gate; «la catena termina alla dose 15». Coprono N-4 e in parte N-1.

Verifiche su `90ffcd4`: 107 test passati in 13,11 s e `bash -n`; `src/`, runtime, aggregatore, launcher e piano a cinque epoche e
`resource_projection.json` invariati da `dcf3bf8`; report a cinque epoche pinnato invariato (`verify_prior` passa); identita' della
ricetta ricalcolata `db975bd1...`, uguale a piano e aggregatore; identita' dei sorgenti `13de2f31...`, uguale alla ricevuta di
preparazione; proiezioni delle risorse invariate; con il mio «15 epoche» sintetico il modulo da' Δ5 = +1,403397, variazione
−2,237792 [−2,494692; −1,976645], identica al mio calcolo ingenuo, e nel caso identita' da' 0 [0; 0]. La copia della mia review
`0968bd1` nel ramo di GPT e' identica byte per byte.

Nota su R-4: un nuovo tentativo chiede «nuovi ID/output», mentre controller e aggregatore accettano solo il layout congelato.
E' coerente con la regola scritta: una riesecuzione passa da un emendamento del codice e da una review.

**Conferma di Simone sul tetto (05/11, 11:59 CET).** Alla mia domanda «Confermi il tetto di 304,5 GPU-ore per il confronto EPN a 15
epoche?» Simone ha risposto: «Confermo il tetto». Interpretazione mia: la conferma riguarda il tetto del piano `plan.json` di
`90ffcd4`, cioe' 304,5 GPU-ore in tutto per i 24 fit, il benchmark e i due checkout, che in core allocati sono 2.436,667 core-ore.
Non comprende tentativi aggiuntivi, che il piano esclude dal tetto, e non sostituisce R-2. La conferma non e' un via a me: i lanci
restano di GPT. A GPT resta da registrarla in `decisioni.md` con queste parole e con il residuo.

## Verdetti

| # | Punto | Verdetto su 90ffcd4 |
|---|---|---|
| 1 | Scienza e selezione | **APPROVATO** |
| 2 | Risorse | **APPROVATO** per formula, conti e gate, con R-2 aperta; tetto confermato da Simone il 05/11 (R-3) |
| 3 | Aggregazione | **APPROVATO**, compreso il nuovo contrasto |
| 4 | Regole di lettura | **APPROVATO** (su `dcf3bf8` mancava R-1, chiusa dall'emendamento) |
| 5 | Operativita' e seguito | **APPROVATO** (R-4 chiusa dall'emendamento); proposta multi-dataset approvata come ricognizione, con quattro osservazioni |

**Nessun difetto bloccante nel codice.** Delle quattro richieste (R-1...R-4) resta aperta R-2, una frase. R-3 e' confermata da
Simone e va solo registrata da GPT. Cinque note (N-1...N-5) non bloccanti.

## 1. Scienza e selezione: APPROVATO

- **Disegno.** Tre regimi, quattro fold, due LR per regime: 24 candidati, tutti a 15 epoche. Griglie, testa, batch 8, seed 0, dati e
  split sono quelli delle cinque epoche. `plan.json` coincide con `LR_GRID` e `HEAD_LR_MULTIPLIER` della libreria; `recipe_identity`
  ricalcolata da me: `975fa69d...`.
- **Inizializzazioni.** `build_classifier` riparte dal teacher 19.618 (`a31c8950...`) o dai pesi casuali con seed 0; nessun percorso
  legge i modelli a cinque epoche. `require_full_dose` rifiuta dosi parziali. Il permutatore dei dati ha lo stesso seme: le prime
  cinque epoche vedono gli stessi batch delle cinque epoche storiche, con un LR diverso.
- **Orizzonte del coseno.** `fit_candidate` calcola il coseno sull'intera dose: a 15 epoche il LR e' ancora al 75% del picco a fine
  epoca 5 e al 25% a fine epoca 10. Il protocollo lo dichiara: 15−5 confronta due ricette, non isola la dose. Corretto.
- **Selezione.** `select_candidate`, `prior_candidate` e `load_inputs` sono identici riga per riga al controller a cinque epoche gia'
  rivisto; `run_candidate` aggiunge solo il controllo della dose. C0 non predice il held; C1 verifica ricevuta e pesi di C0, sceglie
  sulla validation e predice il held una volta.
- **Sorgenti storici.** `src/wearusfm/harness/finetuning.py`, controller, aggregatore, launcher e piano a cinque epoche,
  `calibration_probe.py`, `tokenizer_train.py`, `tokenizer_preflight.py`, `lib_code_dir.sh` e `submit_job.py` sono invariati da
  `db89202`. `FIVE.source_identity()` ricalcolata: `aa857fb3...`, uguale al pin.
- **Cornice.** Il protocollo dichiara che il seguito e' sviluppo guidato dai risultati e non una conferma pre-registrata, e che
  l'aggiunta del frozen viene dopo aver visto il suo risultato. Il test ufficiale EPN resta non valutato. Corretto.

Restano i limiti gia' scritti per le cinque epoche (N-4).

## 2. Risorse: APPROVATO per formula, conti e gate

**Approvo la sostituzione della proiezione assoluta del microbenchmark con la regola sui fit completi.** La nuova regola usa
un'evidenza piu' forte (24 job finiti) e resta prudente: prende il massimo degli otto tempi per regime, triplica anche i costi fissi,
aggiunge il 25% e 600 s, e il nuovo benchmark puo' solo alzare la stima.

| Regime | Massimo a 5 epoche | Proiezione (q=1, D=0) | Limite del gate | Atteso (3 x media) | Allocazione |
|---|---:|---:|---:|---:|---:|
| Frozen | 4.519 s | 17.547 s | 21.300 s | 13.154 s, 3,65 h | 6 h |
| Fine-tuning | 13.359 s | 50.697 s | 57.300 s | 39.506 s, 10,97 h | 16 h |
| Scratch | 13.446 s | 51.023 s | 57.300 s | 39.507 s, 10,97 h | 16 h |

| Voce | GPU-ore | Core-ore |
|---|---:|---:|
| Triplicazione del consuntivo (atteso) | 204,815 | |
| Proiezione prudenziale, 24 fit | 265,038 | |
| Tetto dei 24 fit | 304 | 2.432 |
| Tetto totale, con benchmark e due checkout | 304,5 | 2.436,667 |

Tutte le cifre sono ricalcolate da me, a mano e con `reference_five` e `resource_projection` del controller. I sei pin del piano
corrispondono ai file del commit. Il pavimento funziona: con un benchmark piu' veloce la proiezione non scende.

- **Rumore del benchmark.** I due benchmark a cinque epoche (v1 e v2), stessa costruzione di quello nuovo, differiscono dello
  0,10-0,24% per update. Il gate blocca frozen oltre q = 1,2215, fine-tuning oltre 1,1318, scratch oltre 1,1245. Il margine e' ampio
  rispetto al rumore. Il tempo per update del microbenchmark (0,63 s) e' coerente con i fit completi (circa 0,6 s, stima mia dal
  consuntivo); la sovrastima di 1,8 volte veniva dai margini della vecchia proiezione.
- **R-2: regola se il gate blocca un solo regime.** Il gate e' per regime (`benchmark_packaging(...)[args.regime]`; prerequisito
  «regime gate passed» nel piano di lancio). Verificato con il controller: con un benchmark piu' lento del 13% scratch e' bloccato
  e gli altri due passano; con il 20% passa solo il frozen. Il protocollo dice «si ferma la fase interessata», senza dire se gli
  altri regimi partono.
  Senza scratch il primario FT−scratch non e' valutabile e si spenderebbero fino a 128 GPU-ore per il solo fine-tuning. **Richiesta:**
  scrivere che se un regime e' bloccato non parte nessun fit.
- **R-3: la cifra e' una decisione di budget di Simone.** Il tetto e' 304,5 GPU-ore e 2.436,667 core-ore; l'atteso e' circa 205
  GPU-ore. Il tetto e' 4,5 volte il costo dei 24 fit a cinque epoche (68,27 GPU-ore) e piu' del doppio del tetto della matrice di
  consolidamento (144); l'atteso e' il triplo del consuntivo.
  La frase di Simone riportata da GPT in `decisioni.md` («va bene, procedi, prepara tutto poi facciamo fare un giro a Claude per
  revisione e poi lanciamo») approva la sequenza e non nomina la cifra; il protocollo non dichiara il residuo. `CLAUDE.md` chiede
  costo, residuo e una decisione registrata. Non posso verificare che cosa Simone abbia visto. **Richiesta:** registrare accanto alla
  citazione la cifra confermata da Simone e il residuo. Dato utile, letto da me alle 11:44 CET con `saldo -b`: il progetto ha
  consumato 1.303.196 ore locali su 2.000.000 (65,2%), residuo 696.804; nel mese 3.055 su 164.383. Se la contabilita' addebita i core
  allocati (ipotesi mia), il tetto vale circa 2.437 ore locali, lo 0,35% del residuo: l'allocazione del progetto non e' il vincolo.
- **N-5 (facoltativa): costo del regime frozen.** Nel frozen l'encoder e' in `eval` senza gradiente e i due crop sono fissi
  (`Classifier.train`, `PreparedWindows`): le feature non cambiano durante l'addestramento. La stessa testa su feature in cache
  costerebbe un passaggio di estrazione (stima mia: circa 20 minuti di GPU, da 0,19 s per update) e pochi minuti di addestramento,
  contro 29 GPU-ore attese e 48 a tetto. La scelta attuale compra l'identita' di runtime con le cinque epoche e nessun codice nuovo.
  Non chiedo modifiche: una versione in cache richiederebbe un test di uguaglianza. E' un costo che Simone puo' voler conoscere.

Memoria, partizione e controlli del launcher (un nodo, 8 CPU, 100 GiB, una GPU, `TimeLimit` atteso per regime) sono quelli delle
cinque epoche; il launcher differisce solo per nomi, tempi e candidato obbligatorio. `bash -n` passa.

## 3. Aggregazione: APPROVATO

- **Test.** Il comando di GPT da' 102 test passati in 12,63 s, come dichiarato.
- **Riferimento a cinque epoche.** `verify_prior`, eseguito sulla mia copia locale verificata delle OOF a cinque epoche, riproduce
  il report pubblicato (pin `3c4b3601...`, uguale al file del commit).
- **Identita'.** `dose_statistics(cinque, cinque)` da' contrasti nulli con IC [0; 0] e celle 70,575372 / 93,766454 / 92,363057.
- **Stessi campionamenti.** Dai sei campioni ricostruiti con un'implementazione mia, volutamente ingenua, riottengo FT−scratch a
  cinque epoche +1,403397 [0,929936; 1,836677]: differenza 0,0 dal pubblicato. Primari e secondari usano le stesse estrazioni.
- **Perturbazione sintetica.** Con un «15 epoche» finto, i tre secondari e il primario del modulo coincidono con i miei (differenza
  massima 0,0).
- **Pairing.** Un supporto alterato in un solo valore viene rifiutato («OOF pairing differs»).
- **Provenienze.** `verify_identity` rilegge piano e sorgenti dai blob Git del commit eseguito, rifiuta segnaposto e chiede un commit
  di review concreto. Una cella fallita azzera tutte le statistiche; nessun fold viene tolto.
- **N-3: testo e codice diversi sulla cella fallita.** Il protocollo dice «il relativo confronto non e' valutabile»; l'aggregatore non
  pubblica nessun contrasto, nemmeno per i regimi completi, e con un file mancante si ferma con un errore. E' la direzione prudente.
  Va allineato il testo («nessun contrasto finche' la matrice non e' completa»), cosi' un'analisi parziale non si improvvisa dopo.

## 4. Regole di lettura: APPROVATO (R-1 chiusa in 90ffcd4)

Primari e secondari, lettura a due lati di FT−scratch, logistica 82,577% come riferimento descrittivo separato, nessuna soglia V-R1,
nessuna estensione automatica: corretti e fissati prima dei fit.

- **R-1: manca il contrasto appaiato sulla variazione del vantaggio.** La seconda domanda del protocollo e' se il vantaggio resta a 15
  epoche, e il report mettera' FT−scratch a 15 accanto al +1,403 a cinque. Per dire che il vantaggio si e' ridotto o e' cresciuto
  serve (FT15 − scratch15) − (FT5 − scratch5) sulle stesse estrazioni per soggetto. L'aggregatore ha gia' le sei celle sulle stesse
  estrazioni, ma pubblica solo i tre contrasti per regime, e dai loro intervalli quello della differenza non si ricava: nella mia
  prova sintetica vale [−2,49; −1,98] con secondari [−1,42; −1,07] e [0,69; 1,28]. L'avevo consigliato nella review `0968bd1`, §4
  punto 3. **Richiesta, a scelta di GPT:** (a) aggiungerlo come quarto secondario in `READING_RULE15` e in `dose_statistics`, una riga
  sulle estrazioni esistenti; cambia l'hash della ricetta, quindi prima del checkout dei fit; oppure (b) scrivere nel protocollo che
  non si faranno affermazioni sulla variazione del vantaggio fra 5 e 15. Aggiunto dopo i risultati sarebbe post hoc.
  **Chiusa in `90ffcd4` con l'opzione (a).**
- **N-1: la diagnostica `validation_tail` informa poco, per costruzione.** Nell'ultima epoca del coseno a 15 il LR scende dall'1,09%
  del picco a zero (media 0,37%); a cinque epoche scendeva dal 9,5% (media 3,2%). La clausola sulla mediana conserva la soglia di
  0,5 pp con circa nove volte meno learning rate. La clausola «migliore epoca = 15» e' favorita dall'annealing: a cinque epoche
  l'ultima epoca era la migliore in 20 candidati su 24 (frozen 8/8, scratch 7/8, fine-tuning 5/8). Il protocollo toglie gia' ogni
  conseguenza al flag, quindi la regola puo' restare. Da dichiarare: il flag descrive la coda dello schedule, non il potenziale
  residuo. Le curve a 5, 10 e 15 epoche, gia' nel report, informano di piu'; un descrittivo su quelle va dichiarato ora, se lo si vuole.
- **N-2: frozen contro logistica, senza verdetto.** «Avvicinarsi alla logistica» non ha soglia. Va bene come descrizione: si riportano
  la differenza appaiata 15−5 e la distanza residua da 82,577% come numeri, senza dire «recuperato» o «non recuperato».
- **N-4: limiti ereditati.** Stessa griglia a due punti, che a cinque epoche ha scelto il bordo in 12 celle su 12: il protocollo dice
  «non sono affermazioni sui rispettivi ottimi», e basta. Seme unico. I 314 soggetti held vengono predetti una seconda volta da
  modelli nuovi: sono sviluppo, come dichiarato.

## 5. Operativita' e seguito: APPROVATO

- **Piano di lancio.** 24 ID logici distinti, 24 output distinti e coerenti con `fit_output_root`, 12 dipendenze C1 dopo C0, due
  checkout, ID SLURM e SHA esecutivi ancora nulli. Totali: 304 / 2.432 per i fit, 304,5 / 2.436,667 in tutto.
- **Gate di stato.** Il controller rifiuta ogni modalita', benchmark compreso, finche' `status` non e' `claude_review_converged`; i fit
  chiedono anche `fits_status` e il pin del benchmark. Da questo commit non puo' partire nulla.
- **Stato su Leonardo alle 11:44 CET (sola lettura).** Il namespace `finetune15_20261105` non esiste. Il checkpoint del teacher e'
  presente su SCRATCH (454.381.728 byte, 03/11 23:27; non ne ho ricalcolato l'hash). Manutenzioni: 10/11 07:00-12/11 07:00 su 540
  nodi Booster, 12/11 07:00-14/11 07:00 su altri 180, fermo dell'intero Booster 23/11 08:00-25/11 08:00. Con catene C0 e C1 da
  circa 22 ore attese per fine-tuning e scratch, le manutenzioni parziali non sono un ostacolo.
- **R-4: regola per i fallimenti operativi.** Con 24 job fino a 16 ore un TIMEOUT o un guasto di nodo e' plausibile. Il controller
  rifiuta una directory di output esistente e l'aggregatore si ferma se manca un file. Il protocollo dice «senza retry cieco», ma
  non dice se la cella persa per un guasto dell'infrastruttura si riesegue identica. Decisa con 23 risultati su 24 gia' visibili
  sarebbe una scelta post hoc. L'avevo segnalato nella review `0968bd1`, §4 punto 5. **Richiesta:** scrivere ora la regola: se si
  riesegue, con quali condizioni (stesso SHA, piano e ricetta; diagnosi registrata; costo dentro il tetto) e in quali casi no
  (divergenza numerica, errore del codice). **Chiusa in `90ffcd4`:** nessun retry o requeue automatico; un nuovo tentativo solo
  dopo diagnosi ed emendamento rivisto, dalle inizializzazioni originali, con costi e ID nuovi fuori dal tetto attuale.

### Proposta sui prossimi dataset: approvata come ricognizione

L'ordine UCI-EMG, NinaPro DB2, audit di Hyser-forza e' ragionevole, e le regole da fissare prima dei risultati sono quelle giuste.
I numeri degli split (DB2 32 + 8, Hyser 16 + 4) corrispondono a `splits/v1/splits_v1.json`. Quattro osservazioni per il protocollo:

- **M-1: Kaifosh non compare.** Negli split firmati e' l'unico dataset con ruolo `benchmark`: 100 partecipanti, nessuno nel
  pretraining, split ufficiale 80/10/10, raw verificato (decisione 5 del foglio firmato da Simone il 03/10/2026, «Kaifosh tutto
  benchmark»). Ha il numero di soggetti che manca a UCI (19), DB2 (8 riservati) e Hyser (4 riservati). Limiti noti: l'ingest legge
  solo l'EMG e le etichette non sono caricate; il dispositivo e' della famiglia di emg2pose ed emg2qwerty; la sovrapposizione di
  soggetti fra i dataset Meta non e' dichiarata dalle fonti. Va messo fra i candidati oppure escluso con una motivazione scritta.
- **M-2: risoluzione su UCI.** Nelle sonde storiche il contrasto appaiato T25-L − B19618 su UCI aveva IC [−2,477; +2,326] pp
  (`docs/reviews/tokenizer_classification_results_20261104.md`). Un vantaggio della taglia di EPN (+1,4 pp) non si risolve con 19
  soggetti e un seme. Prima dei risultati: dichiarare l'effetto minimo di interesse e decidere il numero di semi, che su UCI
  costano poco.
- **M-3: esposizione dei soggetti in DB2 e Hyser.** Se i soggetti di valutazione erano nel pretraining, il fine-tuning ha gia' visto
  il loro EMG senza etichette e lo scratch no: FT−scratch mescola il beneficio del pretraining con l'esposizione al soggetto. Le
  popolazioni pulite sono le riservate, 8 e 4 soggetti. Il protocollo deve dire quale popolazione porta l'affermazione.
- **M-4: griglia.** Su EPN la griglia a due punti ha scelto il bordo in 12 celle su 12. Per i nuovi dataset: almeno tre punti, oppure
  una regola di estensione dichiarata prima e applicata sulla sola validation.

## Che cosa serve per convergere

1. R-1 e R-4: chiuse in `90ffcd4`, verificate.
2. R-2: una frase nel protocollo e nel piano di lancio («se un regime e' bloccato non parte nessun fit»).
3. R-3: Simone ha confermato il tetto il 05/11 alle 11:59 («Confermo il tetto»). GPT la registra in `decisioni.md` con queste
   parole e con il residuo.
4. N-1, N-2, N-3: frasi nel protocollo, a discrezione di GPT. N-4 e' coperta. N-5 e' un'informazione.

Con R-2 scritta e la conferma registrata non ho obiezioni al lancio della catena da parte di GPT, da `90ffcd4` o da un successore
che cambi solo quelle frasi e gli stati. Il commit di review da citare in `plan.review.commit` e' l'ultimo che contiene questo file.

## Verifiche eseguite

- Checkout temporanei di `dcf3bf8` e di `90ffcd4`, rimossi a fine review. Test di GPT: 102 passati su `dcf3bf8`, 107 su `90ffcd4`.
  `bash -n` del launcher: passa su entrambi.
- `git diff --quiet db89202 dcf3bf8` sui sorgenti storici elencati al §1: invariati. Confronto funzione per funzione dei due
  controller e dei due launcher.
- Script miei, fuori dal repo: identita' di sorgenti e ricetta, `reference_five`, `resource_projection` con benchmark uguale, piu'
  veloce e piu' lento; unicita' di ID, output e dipendenze del piano di lancio e totali; aggregatore sulle OOF reali a cinque epoche
  e bootstrap ingenuo di confronto.
- Leonardo in sola lettura alle 11:44 CET: `saldo -b`, `scontrol show reservation`, presenza del checkpoint e del namespace. Log in
  `~/wearusfm_local/logs/ft15_review_leonardo_20261105.txt`.

## Limiti della mia verifica

- Il benchmark CUDA a 15 epoche non esiste ancora: di q, D e del gate ho verificato la regola, non la misura.
- Il controllo dello snapshot Git dell'aggregatore e' esercitato dai test sintetici di GPT, non su un commit di esecuzione reale.
- Non ho ricalcolato l'hash del checkpoint del teacher ne' letto quota e inode di WORK: il piano li chiede freschi prima degli invii.
- Le frasi di Simone riportate da GPT (frozen aggiunto; poi altri dataset) non sono verificabili da me.
- I «19 soggetti» di UCI sono il numero di GPT: coerente con il meccanismo usato per EPN (soggetti fuori dall'unione dei test dei
  tre semi), non ricalcolato da me sui dati.
- La conversione in ore locali e' una mia ipotesi sulla contabilita'.
- N-5 e' una stima, non una misura.

Nessun lancio, benchmark, cancellazione, inferenza o modifica di checkout, soglie e candidati da parte mia.
