# Review Claude: prototipo Rank-N-Contrast sulla posa (linea 3), 16/11/2026

Richiesta di Simone, 16/11: «eccoci di nuovo, unclaude simile a te ha fatto questa implementaizone della contrastive loss :», con
incollata la chat dell'altro Claude (modulo `rank_contrast.py` in due versioni dell'11 e del 12/11, 12 test, schizzi d'aggancio per
`training/jepa.py` e `training/run.py`, limiti dichiarati). **Interpretazione mia:** mi chiede una review tecnica e scientifica di
quel prototipo, nel mio ruolo di revisore della linea 3. Nessun codice della linea 3 scritto nel repository, nessun job, nessun ssh:
tutto è stato eseguito in copie esterne al repository.

**Verdetto: nucleo numerico corretto e riutilizzabile; come termine da allenare il prototipo è DA RIVEDERE.** I difetti sono di
definizione e di disegno, non di calcolo. Converge con la review di GPT `924e19d` (R1, R2, R3, N1) e aggiunge misure sui batch
veri e rilievi di disegno. Lo stato è già cambiato durante questa review: vedi il paragrafo qui sotto e la sezione 6.

**Stato alla consegna (riletto con `git ls-remote` alle 11:34 del 16/11).** Mentre i miei controlli giravano, sul ramo
`origin/codex/tokenizer-logcontrast-20261104` sono arrivati tre commit: `924e19d` (09:53, review di GPT dello stesso prototipo),
`db5ac41` (11:09, RnC corretta e InfoNCE implementate, tre bracci C0/CN/CR) e `77b7003` (11:11, prompt di review per me). Sul ramo
del coordinamento c'è `8eba5e4` (09:49, sintesi sulla regressione). Questa review riguarda **il prototipo**. La review del
pacchetto `db5ac41` chiesta dal prompt di GPT **non è fatta qui**: la sezione 6 dice solo quali rilievi del prototipo risultano
chiusi nel modulo corretto, con un controllo limitato fatto di persona.

## 1. Materiali, metodo, livelli di prova

- Prototipo: modulo v2 (157 righe, sha256 `8883c510…2b8cc4`) e test v2 (112 righe, 12 test, sha256 `8abac7f0…379dad`), trascritti
  alla lettera dalla chat. Sono byte per byte uguali alle copie che GPT ha pubblicato in `results/infra/rnc_review_20261116/` a
  `924e19d`, quindi i miei controlli si rifanno da git (ricetta in `claude_review_rnc_prototype_20261116_numbers.txt`).
- Base: `origin/main` `e114a78`. Tutti i nomi e le firme usati dagli schizzi esistono lì (`pose_time_valid`, `latent_mean`,
  `masked_sum_count`, `LossTerms`, `loss_counts`, `_accumulate_terms`, `_shared_grad_stats`).
- Regole di riferimento: `docs/contratto_loss_distribuite.md` (righe 5-6: «1 GPU e N GPU (con o senza accumulo) diano **lo stesso
  aggiornamento** sullo stesso batch globale»; §5: tolleranza relativa 1e-10 in float64), `docs/pilot_posa_v1.md` (R3),
  CLAUDE.md (ruoli della linea 3, soglie prima dei dati, `critico` prima dei lanci).
- Due workflow con verifica avversaria: ogni dimensione analizzata da un agente e poi attaccata da un secondo agente con prove
  proprie. Valutazione scientifica: 8 agenti su 8 conclusi (disegno, conteggi sui batch veri, confronto con la proposta InfoNCE,
  letteratura). Verifica tecnica: 9 agenti conclusi su 11 (formula, numerica, contratto e forza dei test, ciascuna con
  verifica; applicazione degli schizzi senza verifica dedicata; la parte «trainer» interrotta, vedi sezione 7).

| Sigla | Significato |
|---|---|
| **[io]** | controllo fatto di persona; script e uscite in `claude_review_rnc_prototype_20261116_*.py` e `_numbers.txt` |
| **[2 agenti]** | trovato da un agente e confermato da un secondo con prove indipendenti |
| **[1 agente]** | un solo agente, senza verifica avversaria: da prendere come indicazione |
| **[GPT]** | dalla review `924e19d`; dove scrivo «riprodotto» l'ho rifatto io |

Le prove degli agenti (script e uscite) stanno fuori dal repository, in `~/wearusfm_local/reviews/rnc_review_20261116/` sul Mac.
Dati usati: solo tensori sintetici, log di metriche e manifest (indici e conteggi). Nessun segnale da wearable.

## 2. Che cosa regge

| Punto | Esito | Prova |
|---|---|---|
| 12 test dell'autore | 12 passati su torch 2.14.0 | [io] numbers [A] |
| Gradiente del calcolo veloce contro la definizione coppia per coppia | uguale in 24 combinazioni, errore relativo massimo 1,6e-15 | [io] numbers [B] |
| Fedeltà alla formula dell'articolo (S_ij con «maggiore o uguale», k diverso da i) | nucleo uguale a una riscrittura dell'algoritmo ufficiale in 225 casi, scarto 1,8e-14; uguale anche alla definizione riscritta da zero dal verificatore | [2 agenti] |
| Numerica nel campo d'uso (temperatura 0,07-0,2) | float32 contro float64: perdita entro 1,5e-07, gradiente entro 4,1e-06; CPU deterministica | [2 agenti] |
| Conteggio dalle sole maschere uguale al conteggio del forward | 36.000 confronti su maschere avverse, 0 differenze | [2 agenti] |
| Nessun numero casuale, nessun parametro | generatori fermi; `_ddp_wrap` invariato | [2 agenti] |
| Peso 0 dopo aver applicato gli schizzi | `jepa_losses`, termini del contratto, conteggi e stato dell'RNG bit per bit uguali alla base | [1 agente], ripetuto su E4 da un secondo |
| Costo alle taglie vere | al più 160 ancore per micro-batch a stride 8: trascurabile | [1 agente] |

Il lavoro dell'altro Claude è accurato nell'ingegneria: conteggi precalcolati, ramo vuoto derivabile, default invariato, limiti
dichiarati con onestà («su EMG non è una pratica collaudata», «proposta, `critico` e firma restano da fare»).

## 3. Rilievi principali

### P-1. `cross` respinge pose uguali della stessa finestra, e nel flusso vero il caso si presenta

Con `positives="cross"` gli istanti della stessa finestra restano fra i candidati ma non sono mai coppia positiva: compaiono solo
nei denominatori. È il rilievo R1 di GPT. **Riprodotto [io]:** quattro pose identiche, due per finestra, logit uniformi: la
discesa abbassa la similarità con la posa identica della stessa finestra (derivata +0,303413) e alza quella con le due dell'altra
finestra (-0,151707 ciascuna). Con `all` le derivate sono nulle (numbers [C]).

In più rispetto a GPT: non è solo un caso costruito. Gli inizi di finestra stanno su una griglia di 200 ms (`align_ms = 200.0`,
`pretraining_loader.py` righe 64 e 336 a `e114a78`, letto da me) e lo stride proposto è 8 patch da 25 ms, cioè 200 ms. Le ancore
di due finestre sovrapposte della stessa registrazione cadono quindi sugli stessi istanti fisici, con distanza di posa esattamente
zero. Sui 19.618 passi veri del controllo 30M il 7,0% delle ancore dei micro-batch attivi ha un gemello, nel 17,1% dei micro-batch
attivi [2 agenti]. Con uno stride che non divide la griglia i gemelli esatti spariscono, quelli quasi esatti no [1 agente].

### P-2. Candidati del micro-batch: il termine dipende da come si divide il batch

È il rilievo R2 di GPT. L'avevo trovato in modo indipendente: il mio controllo è delle 09:55, la review di GPT (pubblicata
alle 09:53) l'ho letta alle 11:15 (numbers [B]). Stesse 8 finestre, tre con posa:

| Divisione | `cross`: valore | finestre contate | `all`: valore |
|---|---:|---:|---:|
| batch intero | 1,889877 | 3 | 1,726708 |
| due metà | 2,322421 | 2 | 1,744976 |
| quattro quarti | 0 | 0 | 1,719271 |

- Il contratto chiede lo stesso aggiornamento a 1 e a N processi, con o senza accumulo. Per questo termine non vale: gradiente
  diverso in 52 casi su 52 (scarto relativo da 0,55 a 3,93), uguale solo se tutte le finestre di posa stanno in un processo
  [2 agenti].
- Il test «batch intero = micro-batch» che l'autore elenca fra quelli da scrivere «ricalcando quelli della posa» non può passare
  su un batch che metta finestre di posa in micro-batch diversi. **Attenzione [2 agenti]:** ricalcato alla lettera dal test della
  posa passa a vuoto, perché quel batch di prova ha una sola finestra di posa su 4. Il rischio vero è un test verde che non prova
  l'invariante.
- I controlli meccanici del trainer non se ne accorgono: il conteggio precalcolato usa la stessa regola per micro-batch, il run
  arriva a `max_steps` e il summary continua a scrivere che i conteggi sono globali [2 agenti].
- La frase dell'autore «`LossTerms` vieta le operazioni collettive, quindi i candidati sono solo quelli del processo» ha premessa
  vera e conclusione imprecisa: i candidati sono quelli del micro-batch (con accumulo si dividono anche a un processo solo), e il
  vincolo che conta è l'additività dei numeratori, non il divieto di collettive.
- Dentro un run la partizione non cambia in silenzio (ripresa con altro accumulo rifiutata, provato; con altro numero di
  processi rifiutata, letto nel codice) [1 agente]. Un confronto fra due bracci a partizione uguale resta quindi interno e
  valido. Serve una deroga firmata al contratto oppure un ridisegno: non è un errore di calcolo.

### P-3. Nei batch veri il termine `cross` per processo è povero di varietà

Riconteggio mio sui 19.618 passi veri del controllo 30M (78.472 micro-batch da 8, numbers [D]); coincide cifra per cifra con i
conteggi dei due agenti.

| Finestre di emg2pose nel micro-batch | Quota |
|---|---:|
| 0 | 16,59% |
| 1 | 26,16% |
| 2 | 25,45% |
| 3 o più | 31,80% |

| Misura | Valore | Prova |
|---|---:|---|
| `cross` nullo (zero o una finestra) | 42,75% dei micro-batch | [io] |
| idem contando la validità della posa | 45,2% (limiti 43,1-48,5%) | [2 agenti] |
| almeno 2 registrazioni di posa distinte nel micro-batch | 49,92% | [io] |
| almeno 3 registrazioni distinte | 20,30% | [io] |
| finestre di posa sole nel loro micro-batch | 13,55% | [io] |
| passi con `cross` nullo su tutti e 4 i processi | 3,57% (700 passi) | [io] |
| passi con `cross` attivo su un solo processo | 17,60% | [io] |
| coppie di finestre di posa dalla stessa registrazione | 19,06% | [2 agenti] |
| coppie sovrapposte nel tempo | 5,78% | [2 agenti] |
| peso del termine su finestre con al più un'altra registrazione | 59,1% | [1 agente] |

- L'autore cita «un micro-batch su cinque o sei senza emg2pose»: è giusto (16,6%), ma è il numero del caso «zero». `cross` si
  annulla con «zero o una»: 42,75%, due volte e mezza tanto. Lo stride non cambia questa quota.
- Non manca la dose, manca la varietà: il termine è non nullo nel 96,4% dei passi e ci entra l'86,5% delle finestre di posa, ma
  quasi sempre contro una o due altre finestre. Un cancello copiato da R3 («conteggio > 0 nel 90% dei passi») passerebbe al 97%
  senza dire nulla.
- Le finestre di emg2pose arrivano a gruppi (pool di 16 blocchi per worker): varianza 2,02 contro 1,46 binomiale, serie fino a
  40 micro-batch dello stesso worker con meno di due finestre [2 agenti].
- GPT nel suo R3 portava il solo esempio IID (41,89%) e chiedeva la misura sui batch veri: è questa.

### P-4. Peso e temperatura senza regola; «parte da circa 1» non ne dà una

- Il valore di partenza non è circa 1 in generale. Architettura vera 30M non allenata, segnali sintetici, encode mascherato:
  1,01-1,10 con finestre di utenti diversi dello stesso lato, 1,03-1,30 con mano destra e sinistra nello stesso micro-batch
  [1 agente]; con finestre quasi coincidenti della stessa sessione un altro agente aveva misurato 1,006-1,038.
- Il minimo raggiungibile non è 0 e non è noto: con embedding liberi 0,30-0,81 secondo temperatura e numero di ancore. Una
  discesa da 1,0 a 0,8 non ha una lettura assoluta, e il valore non è confrontabile fra batch di taglia diversa [2 agenti].
- La forza del gradiente, a peso fisso, cambia di 19-27 volte con la temperatura e con la componente comune dei readout, e fino
  a 66 volte quando i readout sono quasi collineari [2 agenti]. Col coseno conta la componente costante del readout, mai misurata.
- La testa di posa nasce a zero: al passo 0 il suo gradiente sull'encoder è esattamente nullo, mentre RnC agisce subito
  [2 agenti]. Sui dati veri di T100-G il pannello della posa resta alla predizione costante almeno fino al passo 500 [1 agente]:
  acceso dal passo 0, RnC sarebbe per centinaia di passi l'unica pressione legata alla posa sull'encoder. Il prototipo non ha
  rampa né regola di calibrazione; l'autore lo dichiara («Valori da fissare tu»).

### P-5. Due bracci non attribuiscono un effetto all'ordinamento per posa

Il termine porta insieme l'ordinamento per posa e una pressione che non dipende dalla posa. Nei giocattoli degli agenti (encoder
piccolo, dati sintetici: dicono meccanismo e segno, non grandezze) [2 agenti]:

- con `cross` l'informazione costante nella finestra e non di posa esce dal readout nella stessa misura col termine vero e con un
  placebo a pose scambiate (R2 lineare -0,063 e -0,062); con `all` resta (+0,020);
- nel regime in cui la posa è difficile da leggere, dove sta WearUsFM (E_dyn 0,84), l'effetto su E_dyn è nullo; dove è facile il
  placebo riproduce circa metà del guadagno;
- con `all` e candidati del micro-batch il 42% dei positivi di un'ancora sta nella sua finestra [io], e alla varianza mediana
  vera dentro la finestra (0,257) una similarità cieca alla posa ottiene il 16-34% della perdita riducibile (sintetico, 2-3
  finestre di posa nel micro-batch; la quota cala al crescere delle finestre).

Quindi «R3 + RnC contro R3» risponde a «la ricetta col termine contro la ricetta senza», non a «ordinare per posa aiuta». La
scelta fra le due domande è di Simone. Il vettore su cui agisce il termine è lo stesso che, mediato nel tempo, leggono le sonde
P1 e P2: la scelta fra `cross` e `all` ha effetti collaterali opposti proprio lì, e va scritta come ipotesi con guardia.

## 4. Rilievi minori e note

| # | Rilievo | Gravità | Prova |
|---|---|---|---|
| m-1 | Sentinella `-1e4` sui logit mascherati: risultato sbagliato in silenzio per temperature fino a 1e-4 (nessun effetto sopra 1,01e-4; margine di 700 volte sotto 0,07). È l'N1 di GPT. | minore | [2 agenti] |
| m-2 | I 12 test non fissano il gradiente né la precisione: un mutante col logsumexp staccato dal grafo e uno che calcola in bf16 passano 12 su 12. | minore | [2 agenti] |
| m-3 | Punteggio di mutazione: 33 mutanti uccisi su 90 (40,2% tolti gli 8 equivalenti); 27 uccisioni da un test solo. Un solo valore di temperatura (0,2) e di stride (2) in tutte le chiamate; i valori proposti (0,1 e 8) mai eseguiti. | minore | [2 agenti] |
| m-4 | Schizzi: i 5 campi nuovi entrano in `config_to_dict` anche a peso 0; due test esistenti della linea 3 falliscono e `scripts/logsquare_preflight.py` rifiuta il piano registrato. | minore | [1 agente] |
| m-5 | Schizzi: chiavi di metrica diverse nei due percorsi del trainer (`rnc_pairs` e `rnc_chance` solo col contratto; `rnc_raw` vale 0,0 oppure manca senza ancore). | minore | [2 agenti] |
| n-1 | «Due pose quasi uguali non diventano mai negativi» è falso alla lettera: un vicino entra nel denominatore di uno ancora più vicino. Sta solo nella chat, il docstring è corretto. Lo rileva anche GPT. | nota | [2 agenti] |
| n-2 | Scostamenti dall'articolo non dichiarati nel docstring: vista singola, RMS al posto di L1, divisione per ancora (è un ripesamento fra ancore, non sola scala, appena cambiano candidati o parità). | nota | [2 agenti] |
| n-3 | Target di posa non finito: `ValueError` dentro il forward invece dell'arresto coordinato del contratto. Caso fuori disegno (l'ingest invalida i campioni non finiti). | nota | [2 agenti] |
| n-4 | Modello con testa di posa presente e `pose_weight` 0: DDP si ferma al secondo passo. Preesistente, non della proposta; conta per un braccio «solo RnC». | nota | [2 agenti] |
| n-5 | Il ramo «per giunto» (candidati per giunti in comune) non viene mai esercitato dal loader vero, dove la maschera è uguale sui 20 giunti, e i test non lo fissano. | nota | [2 agenti] |
| n-6 | Pannello fisso: 41 finestre di emg2pose da 6 registrazioni; il 40% delle coppie è della stessa registrazione. Il valore `rnc` del pannello misurerebbe altro rispetto al training. | nota | [2 agenti] |
| n-7 | CUDA e torch 2.2 di Leonardo non eseguiti: `logcumsumexp`, `sort(stable=True)`, `searchsorted` 2-D e `gather` non sono mai stati usati dal progetto sul cluster. | nota | [2 agenti] |

## 5. Lettura scientifica (mia; la scienza della linea 3 è di GPT, le decisioni di Simone)

1. **L'idea è pertinente alla regressione, ma non porta informazione nuova.** Usa gli stessi 20 angoli, lo stesso readout e lo
   stesso supporto del termine di posa R3. Aggiunge un vincolo geometrico: tutta la distanza fra readout standardizzati deve
   seguire la distanza fra pose, non solo 20 direzioni. Su questo concordo con GPT.
2. **Ordina pose istantanee, non il movimento.** Il difetto noto è l'inseguimento temporale attenuato (sintesi del coordinamento
   `8eba5e4`: varianza predetta 17-24% del bersaglio, E_dyn 0,839 a 19.618). Fra finestre diverse la distanza di posa è dominata
   dal livello medio: per giunto il quadrato della media di finestra vale 0,687 e la varianza dentro la finestra 0,313
   (`results/infra/pilot_posa/b_dynamics_20261028/final/report.json` su `origin/main`, riletto da me). Gran parte dei vincoli
   riguarda quindi la postura, che il modello legge già meglio. E_dyn misura le traiettorie centrate. **Ipotesi mia, da
   scrivere prima dei dati:** effetto piccolo o nullo su E_dyn a 19.618, forse un guadagno sul livello. I giocattoli vanno
   nello stesso verso.
3. **Il regime è lontano da quello in cui il metodo è stato misurato.** Nell'articolo: 256 campioni indipendenti in due viste,
   similarità -L2, temperatura 2, encoder allenato con la sola RnC e poi congelato. Le sue ablation danno peggiori il termine
   congiunto, il coseno, la vista singola e i pochi positivi. Qui valgono tutte e quattro le condizioni, più candidati
   autocorrelati presi da 2-3 traiettorie. Sono fatti di letteratura letti tramite strumenti automatici: **da verificare**,
   non premesse. La ricerca degli agenti non ha trovato usi di RnC su sEMG verso la posa; non è una ricerca sistematica.
4. **Il metro per una soglia su E_dyn esiste già:** la sola dose da 19.618 a 49.046 passi vale -6,3% (stessa sintesi). Una
   soglia del 2% con un seme solo sta vicino agli scarti fra bracci già visti (fino a 1,6%).
5. **Che cosa renderebbe leggibile un esperimento:** una configurazione sola, congelata prima dei dati; un controllo di
   manipolazione (la correlazione di rango fra coseno e distanza di posa sugli utenti di validazione deve salire, altrimenti un
   nullo non si legge); guardia su P1 e P2; se si vuole l'attribuzione, un terzo braccio placebo (un training in più).

## 6. Stato dei rilievi nel modulo corretto da GPT (`db5ac41`): controllo limitato, non la review del pacchetto

Ho letto solo `src/wearusfm/model/rank_contrast.py` (179 righe) e il protocollo `docs/contrastive_comparison_protocol_20261116.md`,
ed eseguito i 23 test del modulo e un mio controllo (numbers [E] e [F]).

| Rilievo del prototipo | Nel modulo di `db5ac41` | Prova |
|---|---|---|
| P-1 `cross` asimmetrico | `cross` rifiutato, solo `all`; derivate nulle nel controesempio | [io] |
| P-1 gemelli sullo stesso istante fisico | il protocollo li esclude in modo simmetrico (`pair_allowed`) | dichiarato, **non verificato** |
| P-2 partizione | il modulo dichiara che dividere la popolazione non riproduce la perdita globale; la raccolta globale con gradiente sta nel trainer | **non verificato**: i sei test distribuiti nuovi non sono stati eseguiti da GPT |
| P-3 varietà | candidati dal batch globale: nei passi veri almeno 3 finestre di emg2pose nel 97,7% dei passi, al più una con posa valida nello 0,85% | [io] sui conteggi; non sul codice |
| P-5 positivi dentro la finestra con `all` | coi candidati globali la quota media scende dal 42% al 15% | [io], validità ideale |
| m-1 sentinella | tolta: uguale alla mia definizione a temperature 1e-5, 5e-5, 1e-4, 1e-3 | [io] |
| valore e gradiente | uguali alla mia definizione coppia per coppia in 96 casi (5,1e-16 e 3,0e-15), anche con `pair_allowed` | [io] |
| P-4 peso | calibrazione comune a pesi fermi, rampa fra 256 e 512 update | dichiarato, **non verificato** |

Punti che questa review porta alla review del pacchetto, senza anticiparne i verdetti:

1. I sei test distribuiti non eseguiti sono il punto tecnico centrale di P-2. Sul Mac un agente ha fatto girare DDP `gloo` a 2
   processi: vanno provati qui prima di dichiararli non eseguibili.
2. La calibrazione del peso a pesi fermi allo stato iniziale misura una base fatta di sole ancore (testa di posa a zero). Sui
   dati veri di T100-G la norma del gradiente di base cala di circa 3 volte nei primi 64 passi [1 agente]: il 10% all'avvio non
   dice quale rapporto ci sarà a fine rampa. Il protocollo lo dichiara; va deciso che cosa si legge dai gradienti registrati.
3. Con mano destra e sinistra nello stesso batch la struttura di similarità più grande all'avvio è il lato [1 agente, segnali
   sintetici]. Il termine chiede anche l'invarianza al lato del readout per istante: va deciso se è voluta.
4. Con `all` resta una parte di perdita riducibile senza posa (vicinanza nel tempo). Coi candidati globali è più piccola, ma
   senza placebo o controllo di manipolazione un effetto di CR non si attribuisce all'ordinamento per posa.
5. Il cancello RnC «almeno una finestra contata nel 90% dei batch globali» passa sempre e non misura la varietà: una soglia in
   registrazioni distinte, pesata per passo, direbbe di più.

## 7. Che cosa non ho verificato

- La parte «trainer» della verifica tecnica (comportamento degli schizzi dentro `train()`, E4, ripresa, pannello) non era
  conclusa alle 11:32. Riguarda gli schizzi del prototipo, ormai sostituiti dall'integrazione vera di GPT: l'ho interrotta.
  Il rilievo m-4 resta perciò di un solo agente.
- Nessuna esecuzione su GPU, con NCCL o con la build torch 2.2 di Leonardo. Nessun DDP a 4 processi.
- Le prove sul modello vero 30M usano segnali sintetici; i giocattoli hanno 2-3 semi. Grandezze non trasferibili.
- Valori veri della posa nei batch (quanto spesso un istante della stessa finestra entra in S_ij): servono i target su Leonardo.
- La provenienza del log dei 19.618 passi (fuori dal repository) è provata solo per coerenza: i primi 500 passi coincidono con il
  log committato `results/infra/scaling_300m_20261104/production_memory_63527446` [1 agente].
- Le fonti di letteratura sono state lette quasi tutte tramite WebFetch; di prima mano solo il PDF della v1 dell'articolo.
- Non ho fatto la review della proposta InfoNCE del 07/11 (`df36272`), il cui prompt non mi era arrivato, né quella di `db5ac41`.

## 8. Rettifica a una mia nota precedente

Nella review `975f6a3` (lettura finale di T100-LQ) la nota N-3 trattava T25-L contro T100-L come una stima della ripetibilità.
GPT ha corretto il 07/11 (`docs/reviews/risposta_claude_lq_final_20261107.md`): quel contrasto cambia la finestra, quindi non
stima il rumore fra repliche; e «non aggiunge nulla a GELU» va detto «nessuna superiorità dimostrata». Accolgo entrambe.

## 9. Prompt per GPT

Testo da incollare nella chat GPT logcontrast (linea 3), se Simone vuole:

> Claude ha rivisto il prototipo Rank-N-Contrast dell'altro Claude su richiesta di Simone: review
> `docs/reviews/claude_review_rnc_prototype_20261116.md` sul ramo `worktree-claude-training-engine`, con script e uscite accanto.
> Converge con la tua `924e19d`: R1 riprodotto (+0,303413 e -0,151707), R2 trovato in modo indipendente (stesso batch: 1,889877
> intero, 2,322421 a metà, 0 a quarti con `cross`), N1 confermato. Aggiunge: (a) R1 non è solo un caso costruito, perché con
> stride 8 e inizi su griglia di 200 ms le ancore di finestre sovrapposte coincidono: 7,0% delle ancore dei micro-batch attivi sui
> 19.618 passi veri del controllo 30M; (b) la misura che chiedevi in R3: `cross` per processo è nullo nel 42,75% dei micro-batch
> veri (45,2% con la validità), almeno 3 registrazioni distinte solo nel 20,30%, 59,1% del peso su finestre con al più un'altra
> registrazione; (c) il test «batch intero = micro-batch» ricalcato dal test della posa passa a vuoto, perché quel batch ha una
> sola finestra di posa; (d) la scala: partenza 1,01-1,30 secondo il lato della mano, minimo 0,30-0,81, forza del gradiente che
> varia di 19-66 volte con temperatura e componente comune del readout; (e) nei giocattoli un placebo a pose scambiate riproduce
> l'uscita dell'informazione di finestra con `cross` e, dove la posa è facile da leggere, circa metà del guadagno su E_dyn. Controllo limitato del tuo modulo a
> `db5ac41`: 23 test passati, valore e gradiente uguali alla definizione coppia per coppia di Claude in 96 casi, R1 annullato,
> temperature estreme esatte. Non è la review del pacchetto, che Claude farà sul tuo prompt `77b7003` appena Simone la conferma.
> Punti che porterà lì: i sei test distribuiti da eseguire; calibrazione del peso all'avvio con testa di posa a zero e base che
> cala di circa 3 volte in 64 passi; lato della mano nel batch; parte di perdita riducibile senza posa con `all` e assenza di un
> controllo di manipolazione o di un placebo; cancello RnC che non misura la varietà. Claude accoglie anche la tua rettifica del
> 07/11 alla sua nota N-3 su T100-LQ. Nessun job è autorizzato da questa review.
