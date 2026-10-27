# Review del seguito R3 di Claude, e198a87

27/10/2026, GPT. Commit esaminato: `e198a8707b72a6f0433134eaf80921358fe16d72`, ramo Claude `worktree-claude-training-engine`.
Richiesta di Simone: verificare oracolo, split/ritardi, guardia e preventivi prima del via. Codice Claude e checkout di campagna non modificati;
nessun accesso al cluster e nessun job sottomesso. Tre letture indipendenti: target/guardia, critica scientifica dell'audit, sintetico/risorse.

**Aggiornamento operativo, d037a3e:** Claude ha registrato che entrambe le estrazioni E3 hanno letto `val`: la cache train manca.
Prima di J1 occorrono recupero della val, verifica delle identita' e nuova estrazione del vero train. Questo aggiorna la raccomandazione
e il preventivo iniziali; il codice esaminato resta e198a87, poiche' d037a3e modifica soltanto documenti/comandi. Dettagli e tetti sotto.

**Raccomandazione operativa:** J1 (rapporto E3 R3c) puo' procedere dopo il recupero corretto delle cache, con il via di Simone. **J2 va corretto prima
di presentarlo come audit completo dell'associazione EMG-posa**: i controesempi sotto mostrano false rassicurazioni e falsi allarmi possibili.
Non sono prove di un difetto nei dati o nel training R3. Nessun nuovo motivo per aumentare il peso della posa o abbandonare R3.

## Esito delle verifiche richieste

| Domanda | Esito |
|---|---|
| Oracolo indipendente? | Si' per aggregazione, centri e dilatazione degli zeri nel test; incompleto per NaN/Inf e per la catena EMG. L'audit riusa `runs_of` dell'ingest, ma non le sue funzioni di aggregazione dei target nell'oracolo |
| Split senza perdite? | Nessun mescolamento casuale di patch fra fit/test esterno; finestre distinte e gap. Validazione interna alpha da correggere; ordine per sessione/stem non garantisce cronologia globale o isolamento di acquisizioni simultanee |
| Lag valida l'allineamento? | **No come verdetto**. Utile come esplorazione, con supporto controllato e incertezza. Minimo nonzero compatibile con dati allineati; il test di shift non verifica il ritardo dichiarato |
| Guardia compatibile? | Nessuna regressione trovata sui sidecar conformi; mantiene lettura senza sidecar. Controlla metadati, non identita' del contenuto degli array |
| Preventivi corretti? | J1 16 core-ore, J2 4, checkout 1/6. Dopo d037a3e aggiungere nuova estrazione train: 1 A100/8 core per 40 min. Nuovo totale **25,5 core-ore e 0,6667 GPU-ore** |

## Correzioni necessarie alla diagnosi J2

### R1 — Verificare anche l'EMG raw/processato; rendere esplicita la copertura

`scripts/pose_audit.py:232–246` legge dal raw solo gli angoli e controlla lato/lunghezza dell'EMG. Non confronta `source_member` dei due
sidecar, campioni EMG, origine temporale o quantizzazione. L'oracolo valida le trasformazioni della posa, non la sua associazione con
l'EMG effettivamente usato dal modello.

**Controesempio confermato nella fixture consegnata:** `tests/cpu/test_pose_audit.py:48–56` scrive EMG modulato nel processato, mentre
`_hdf5_bytes(pose)` genera nel tar un altro EMG casuale (`test_ingest_emg2pose_pose.py:44`). Audit completo: identita', ricostruzione e
oracolo tutti `True`, pur con differenza massima raw/processato **10,3390** nei primi 100 campioni. Questo test, oggi verde, dimostra
esattamente la lacuna. Anche il test chain costruisce separatamente l'albero EMG e non attraversa l'ingest EMG o i margini di `read_window`.

Correzione: un unico HDF5 sintetico passato a entrambi gli ingest; confronto raw EMG → quantizzazione/dequantizzazione → campioni
processati alle coordinate giuste, con tolleranza derivata dalla quantizzazione; poi almeno finestre dirette e da blocco con bordi/gap.
Sui reali confrontare anche source_member/identita', frequenza e origine temporale; il test deve fallire per EMG spostato o sostituito,
anche con sidecar apparentemente coerente. Non usare l'argmin della ridge per sostituire questo controllo.

Inoltre `pose_audit.py:300–301` tratta una ricostruzione assente come `True` e un oracolo assente come accettabile. **Repro con budget
indice tar zero:** found=0, missing=4, n_ricostruite=0, n_finestre_oracolo=0, ma entrambe le bandiere `*_ok_tutte` sono `True`.
Il budget di indicizzazione da 20 minuti rende questa condizione operativamente rilevante. Usare stati **verificato / fallito / non
verificato**, copertura attesa/effettiva e nessun successo complessivo con controlli mancanti. L'audit puo' produrre un rapporto parziale
utile senza estendere il budget; deve chiamarlo parziale.

### R2 — Correggere l'oracolo per NaN/Inf; rispettare le statistiche della finestra

`pose_audit.py:45–48` e `test_pose_chain_oracle.py:72–75` dilatano solo gli zeri; l'ingest esclude anche righe non finite
(`ingest/emg2pose_pose.py:121–126`). Repro: angoli=1 con NaN nel primo giunto della prima riga. L'ingest esclude quel blocco e produce
target validi finiti; l'oracolo conta cinque blocchi validi nella prima patch e produce NaN. Risulterebbe erroneamente un problema nei dati.
Implementare indipendentemente la regola completa, senza dilatare i non-finiti se il contratto dilata soltanto gli zeri; fixture NaN e Inf.

`pose_audit.py:202,263` passa all'oracolo mean_all/std_all, mentre `PoseStore.window` usa `norm(view)`. Con per_side=true si produce un
falso mismatch. Usare le statistiche del lato della finestra e testarlo. **Non attivo sui dati congelati attuali**, per_side=false nella
voce ingest di `decisioni.md:4788–4789`; non e' da presentare come spiegazione del collaudo fallito.

### R3 — Togliere il verdetto di allineamento dal minimo della lag scan

`pose_audit.py:14` afferma che il miglior k deve essere zero; `163–172,307` selezionano l'argmin sul test e producono `lag_zero_per_tutti`.
Non e' un criterio valido. Con contesto ±2 patch, piu' lag possono essere predetti altrettanto bene dallo stesso input; rumore e supporti
diversi cambiano il minimo. Inoltre 17 lag sono confrontati sul test: `ratio_at_best` e' selezionato, non una prestazione indipendente.

**Repro senza modello fisiologico o training neurale:** target esattamente uguale alla feature centrale all'istante t; feature includono
i vicini ±2. Con seed5 il codice restituisce best_lag=+2 pur con associazione esatta. Ratio zero=0,0000910413, ratio +2=0,0000910397:
curva praticamente piatta, ma il verdetto binario sarebbe negativo. Il ritardo predittivo e l'errore degli indici sono domande diverse.

`stack(...,k)` cambia anche righe/maschere e `yk.mean` cambia la costante (`106–118,169`). Per una curva quantitativamente confrontabile,
definire supporto comune, medesima distribuzione dei target e baseline comparabile, riportando anche MAE assolute e numerosita'. Se si
vuole scegliere un lag operativo, sceglierlo dentro il fit e valutarlo una volta sul test. Altrimenti etichettare tutta la scansione come
esplorativa, senza “zero=allineato / nonzero=errore”, e separarla dai controlli deterministici raw/processato.

**Il test dello shift contiene un errore di unita' e di copertura:** `test_pose_audit.py:95` sposta 40 blocchi ×10 campioni /2.000 Hz =
**200 ms = 8 patch**, non 400 ms. L'invocazione a riga72 usa lags6, cioe' ±150 ms: il ritardo vero e' fuori scansione. L'asserzione
abs(best_lag)>=4 non verifica recupero di segno o magnitudine. Correggere test e documento: segnale artificiale identificabile, range
che includa il ritardo con margine, controllo del recupero atteso; disabilitare o esplicitare l'ambiguita' introdotta dal contesto.

### R4 — Non comprimere il tempo nelle misure di movimento

`pose_audit.py:275–276,283–284` applica `diff(t[v])` e `diff(bw[vw])`: salta i buchi della maschera e tratta come adiacenti due punti
lontani. Il salto attraverso il buco puo' dominare la misura di movimento “valido” e la selezione delle patch dinamiche.
Calcolare differenze sugli indici originali e tenere soltanto coppie adiacenti entrambe valide. Se si vogliono differenze a tempi
variabili, normalizzare per il tempo reale e denominarle diversamente. Una fixture con plateau prima/dopo un buco deve dimostrare
che il salto non viene contato come movimento osservato durante il tratto valido.

## Split, controllo disallineato e guardia

**Split esterno:** `pose_audit.py:98–101` assegna finestre intere, separate da almeno una finestra; `fixed_windows` evita sovrapposizioni
nello stesso recording. Non ho trovato una fuga di etichette del test esterno nel fit della ridge a lag zero. L'ordine
`(session,stem,start)` (`286`) e' pero' lessicografico fra recording: non equivale a ordinare tutti gli intervalli per tempo assoluto,
specialmente per i due lati della stessa acquisizione. Salvare recording/sessione/lato/fase/intervallo e verificare esplicitamente
che acquisizioni simultanee o duplicate non attraversino lo split se l'obiettivo e' una separazione temporale indipendente.

**Validazione interna alpha:** `124–129` normalizza usando tutto il fit e lo taglia poi a meta' delle patch valide concatenate. Puo'
tagliare una finestra e mettere vicini correlati nei due lati senza gap; la normalizzazione vede l'inner holdout. Non e' leakage dal
test esterno, ma la selezione alpha puo' essere ottimistica. Fare lo split interno per finestre/recording con gap e fit-only normalization.

**Disallineato:** `145–160` permuta finestre solo entro utente, senza vincoli di fase/lato/distanza. Il dizionario `286` non conserva
questi campi. Puo' cambiare il compito o la mano oltre al movimento, oppure confrontare finestre troppo vicine. Portare i metadati,
applicare vincoli espliciti e contabilizzare esclusioni. Il supporto delle due MAE e' gia' intersecato, ma i due rapporti hanno denominatori
diversi: riportare MAE allineata, disallineata e gap sul supporto comune. In alternativa dichiarare il controllo generico/descrittivo
senza usarlo per certificare dipendenza temporale; non chiamarlo equivalente al controllo E2 corretto.

**Guardia PoseStore.entry:** `pose_targets.py:99–111` sanitizza utente/stem come l'ingest e confronta i lati in minuscolo. Alberi senza
sidecar restano compatibili. Nessuna regressione trovata sui metadati conformi; gli input reali non sono stati letti da GPT. Rileva lo
scambio insieme al sidecar, non lo scambio dei soli .npy con sidecar lasciato al suo posto. La cache e i metadati parziali mantengono
limiti noti: e' una guardia di coerenza, non un checksum del contenuto.

## Sintetico: utile, due dettagli da correggere e una conclusione da limitare

- `pose_head_synthetic.py:37` anticipa il decay di un aggiornamento: al passo1.041 applica9,961538e-5 anziche'1e-4, al1.300 zero anziche'
  3,846154e-7. Dose **670, non671**. La tolleranza<2 del test nasconde il difetto. Usare la funzione del trainer o verificarla per tutti
  i1.300 passi con tolleranza stringente. Non invalida da solo il messaggio qualitativo del sintetico e non blocca J1/J2.
- Teacher `y=xW+rumore`, testa su LayerNorm(x): `optimum_ratio_at_fixed_features=1-q` non e' il floor garantito della testa, che perde
  media/norma di x. Generare il teacher dalle feature normalizzate oppure dichiarare q riferito al raw e1-q come limite inferiore.
- “H1 regge solo se B_eff basso o q piccolo” (`decisioni.md:4912`) non segue dalla prova: mancano encoder mobile, ancore e feature
  anisotrope. Anche B_eff fra7e70 e' un'ipotesi, non un intervallo misurato. Il test usa seme0; CLI predefinita ha tre semi ma soloP=10,
  mentre P=1 e' nel test. Registrare gli scenari effettivamente eseguiti e non dedurre una diagnosi esclusiva del run reale.

## Via ai job e budget

**J1, rapporto E3: favorevole dopo recupero train/val.** Le correzioni precedenti a E2 non toccano E3; R3c ha has_head=false e non entra
in E1/E2. Il confronto del checkout41f6ac1 con e198a87 conferma che il delta del report riguarda E2, non la ridge E3.

**Novita' d037a3e, fonte: registro e comandi versionati di Claude, senza verifica live del cluster:** 62437148 e 62437151 risultano
COMPLETED, ma entrambi hanno estratto val. In zsh `$tail_args` non quotato e' arrivato come unico argomento nel nome del file; lo split
e' rimasto al default val. Una val e' recuperabile (18.548 finestre/1.950 recording dichiarati); la seconda non e' un train. Il costo
gia' sostenuto e' **0,84 GPU-ore**, distinto da quello futuro. Non basta che un job sia COMPLETED o che il nome contenga “train”.

Usare `results/infra/pilot_posa/comandi_e3_r3c.txt` aggiornato: recupero val senza sovrascrivere un file esistente, nuova estrazione
train di 40 utenti/200 recording, poi rapporto. I comandi [E3b]/[E3c] sono stati verificati con parsing locale degli argomenti, senza
eseguirli: `--split train`, `--n-users 40`, `--n-recordings 200` sono token separati. Verificare prima del rapporto le identita' effettive
nelle cache contro il CSV/split, numerosita', disgiunzione train/val prevista dal protocollo, checkpoint e statistiche congelate;
non dedurre lo split dal nome del file. Se la cache non contiene tutti i metadati, ricostruire la provenienza dai recording/manifest/log
e registrare le verifiche mancanti. Nessun rapporto E3 ammissibile con due cache val.

Nuova estrazione: **1 A100, 8 core, 60 GB ×40 min = 0,6667 GPU-ore e 5,3333 core-ore**; i 4 minuti attesi indicati da Claude sono una stima,
non una misura. Rapporto: **8 core ×2h,60GB =16 core-ore**, senza GPU. E3 puo' avanzare indipendentemente dalle correzioni di J2.

**J2: correggere prima R1–R4 e chiarire split/disallineato**, poi un nuovo SHA e checkout. La parte di sola ricostruzione della posa
sarebbe gia' utile come audit parziale, ma il job presentato promette una diagnosi piu' ampia e potrebbe generare conclusioni sbagliate.
Le correzioni sono locali agli strumenti; non richiedono di cambiare R3, M=280 o il peso. Tetto esistente **4 core ×1h,16GB =4 core-ore**,
checkout **1 core ×10min,2GB =0,1667 core-ore** (0,2 arrotondato). Il confronto raw EMG deve essere campionato e restare nel tetto;
se serve ampliare tempi/memoria, aggiornare il preventivo prima del via. J1+J2+checkout restano **20,1667 core-ore,0 GPU-ora**;
**includendo il recupero train necessario dopo d037a3e, totale futuro 25,5 core-ore e 0,6667 GPU-ore** (tetto GPU arrotondato 0,67).
Somma delle durate massime dei job 3h50min, esclusi attese in coda e lavoro locale; non e' una previsione del tempo a calendario.

Entrambi i launcher passano bash -n, verificano CODE_DIR/SHA e scrivono fuori dal repo. **B da5.000 passi resta una proposta separata**:
nuovo pannello, criterio e compatibilita' della ripresa da verificare e firmare. Le tre obiezioni tecniche di Claude sono ragionevoli;
il pannello diagnostico deve preservare RNG e traiettoria del training. Nessun via implicito a B in questa review.

## Verifica e riproduzione

Rieseguiti da GPT **18 test**: test_pose_chain_oracle.py, test_pose_audit.py, test_pose_targets.py e test_pose_objective.py: tutti passati
in19,82s. Un warning preesistente di conversione tensore/scalare; suite completa non rieseguita. I test verdi non coprono i controesempi.

`docs/reviews/repro_pose_audit_review_20261027.py` riproduce: argmin nonzero con target perfettamente allineato; NaN dell'oracolo;
dose670/671; EMG raw/processato diverso con controlli verdi; nessuna lettura raw con bandiere True. Solo dati sintetici temporanei,
autograd/analisi e regressione ridge, nessun training neurale nel repro. Le asserzioni dimostrano difetti attuali, non comportamento desiderato.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /Users/simonebenatti/.venvs/wearusfm-tok/bin/python docs/reviews/repro_pose_audit_review_20261027.py \
  /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine
```
