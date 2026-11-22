# TCN: diagnostici del controllo temporale fallito (D0 e D1) — protocollo del 22/11/2026

Stato: **regole congelate prima di qualsiasi job**. Ramo `claude/tcn-temporal-diag-20261122`, base
`e8c9824461b5e18494ff462f1be8abda71b76d5e` (punta di `codex/pose-temponet-launcher-20261120`).

## 1. Incarico e perimetro

Simone, 22/11/2026, in chat: «GPT ha finito i token, quindi qua, quando hai fatto la tua review e proposto una soluzione puoi
lanciare tu direttamente e impostarti un timer per guardare ogni 30 minuti».

- Per questa sola catena l'operatore è Claude, in questa chat. Il launcher GPT della linea TCN non invia: nessun secondo
  submitter. Prima di ogni invio si legge il ledger comune.
- La campagna A1 resta `recipe_not_admitted` e ferma. Nessuna soglia cambia. Nessun utente di sviluppo viene letto.
- Qui non si lancia nessuna variante della testa. Il candidato residuale di GPT e ogni altra ricetta restano una decisione
  successiva di Simone, con i risultati in mano.
- Il disegno è quello su cui Claude e GPT avevano già converguto: proposta `8fac3a8239089d64dbefdf3ad70bd365094d49ff`
  (`codex/pose-temponet-temporal-diagnosis-20261121`), nota di convergenza `4a48b814a32452a22c3525afe375635b251ecc68`.
  Si aggiungono le letture alle esposizioni 1, 10 e 75 chieste dalla seconda diagnosi di GPT (`e8c98244`).

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
| `TCND-d1-cleanfit-20261122` | D1: parità del driver e due fit sul bersaglio pulito; parte solo dopo D0 riuscito | come D0 | 50 min | 0,88 |

- GPU-ore prenotate: limite più 180 secondi, come nel piano A1. Totale 1,27 GPU-ore, 10,1 core-ore sui nodi GPU più 0,4 del
  checkout. Consumo atteso 0,4-0,5 GPU-ore (durate misurate nel benchmark 65016447 e nei sei controlli).
- È una voce nuova: i tetti di A1 (3,681 GPU-ore usate su 56,10) non la coprono e non vengono toccati.
- **Sotto-budget di questa catena: 3 GPU-ore prenotate in tutto.** Copre i tre job e al più un rilancio per stadio dopo un errore
  tecnico diagnosticato e corretto, con ID e checkout nuovi. Un fermo di parità o un esito scientifico non si rilanciano: tornano a
  Simone. Oltre il sotto-budget serve una decisione nuova.
- Manutenzione di tutto Leonardo dal 23/11 alle 08:00 al 25/11 alle 08:00: un job che non può finire prima resta in coda.

## 4. D0 — rilettura degli stati archiviati

Si rileggono i 20 stati della TCN del controllo temporale (cinque learning rate, stato migliore e stato finale, due fold) e i
quattro stati scelti di L0 e MLP. Ogni stato è ricaricato con `Trajectory.restore`, che ne verifica il checksum, la ricetta, il
contesto numerico, l'inizializzazione e il set di training bit per bit. Il punteggio è quello congelato, a blocchi di 32
finestre, contro `y` e contro `y*`, sugli esclusi interni e sul fitting.

**Cancelli di validità.** Se uno fallisce il job chiude con stato `gate_failed`, D1 non parte e niente viene allentato:
1. **Parità.** Per ogni coppia di stati della TCN (10) e per la coppia scelta di L0 e di MLP: MSE ed `E_dyn` pooled uguali al
   punto archiviato della curva, `|nuovo − archiviato| <= 1e-12 + 1e-12·|archiviato|`, stessi update. Per lo stato scelto di ogni
   testa anche gli accumulatori per fold: esclusi contro `y`, esclusi contro `y*`, fitting contro `y`.
2. **Aritmetica del cancello.** Lo stato scelto della TCN, riletto, ridà i recuperi archiviati (24,17% / 18,88% / 21,71%) e lo
   stato `recipe_not_admitted` con la funzione congelata `temporal_head_gate`.
3. **Riferimenti.** L'oracolo sugli esclusi, ricalcolato dagli array, coincide con quello archiviato entro 1e-9 relativo.

**Letture, descrittive.** Per ogni stato, per fold e pooled, su fitting ed esclusi:
- MSE ed `E_dyn` contro `y` e contro `y*`, con numeratori e denominatori;
- recupero sulla scala del cancello contro `y`. Sul fitting i riferimenti sono `cleanLS` (stimato su `y*` e valutato su `y`,
  dalla diagnostica lineare archiviata) e l'oracolo calcolato dagli array;
- scomposizione esatta dell'errore contro `y` sullo stesso supporto: `E_dyn = errore pulito + livello del rumore + allineamento`,
  dove `allineamento = −2<e,n>/TSS`, `e = predizione − y*`, `n = y − y*`. Deriva da
  `SSE_y = SSE_y* + SSE_oracolo − 2<e,n>`. Si riporta anche `<e,n>/<n,n>`: 0 se la predizione ignora il rumore realizzato, 1 se
  lo riproduce;
- differenza esclusi meno fitting, termine per termine, per lo stato scelto (epoca 31) e per quello finale (epoca 150);
- loss di training per epoca, in modalità train.

**Lettura a due lati, scritta prima del job.** Lo scarto fra fitting ed esclusi si divide in tre termini che sommano esattamente:
- se domina la differenza di **allineamento**, la rete riproduce sul fitting il rumore di quegli utenti: rumore memorizzato;
- se domina la differenza di **errore pulito**, la funzione imparata vale meno sugli utenti esclusi: trasferimento fra utenti;
- la differenza di **livello del rumore** dice solo quanto i due pannelli differiscono per costruzione.

D0 riporta i tre termini e non assegna etichette con soglie: non decide niente da solo.

## 5. D1 — parità del driver e due fit sul bersaglio pulito

**Unico intervento:** il bersaglio di training è `y*` invece di `y`. Restano quelli di A1: testa `TCN_post`, seme 0,
inizializzazione, ordine dei batch, flusso del dropout, dropout 0,1, weight decay 0,01, clip 1,0, batch 32, loss, learning rate
0,003, 150 epoche fisse, calendario dei checkpoint, valutatore. Niente griglia, niente altri semi, niente estensione a 300.
Il learning rate viene dal mondo già osservato: è un diagnostico esplorativo, non una conferma indipendente.

**Cancello di parità, prima dei fit.** Lo stesso driver, con bersaglio di training `y`, deve ripercorrere la prima epoca
archiviata del learning rate 0,003:
- pesi iniziali e set di training identici a quelli archiviati (hash);
- cinque punti della curva, esposizioni 0, 0,25, 0,5, 0,75 e 1, con update `[0,0] [8,8] [15,15] [22,23] [29,30]`: MSE ed
  `E_dyn` pooled entro la tolleranza di D0;
- hash del primo mescolamento e loss di training della prima epoca, per fold, uguali a quelli archiviati.

Se la parità fallisce il job chiude con `gate_failed` e i fit puliti non partono.

**Selezione, invariata.** Punto scelto: minimo della MSE pooled contro `y` sugli esclusi interni, esposizione positiva, a parità
la prima. Le curve contro `y*` sono descrittive e non scelgono niente. Lo stato scelto, ricaricato, deve ridare il suo punto
della curva entro la tolleranza.

**Valutazioni in più.** A ogni checkpoint: esclusi contro `y*`, fitting contro `y` e contro `y*`, in modalità di valutazione. Non
estraggono numeri casuali: la traiettoria resta quella. Si salvano gli stati alle esposizioni 1, 10, 31, 75 e 150 e quelli scelti.

### Regola di lettura, in ordine e senza sovrapposizioni

Per ogni pannello `p` fra fold 0, fold 1 e pooled:
- `R_held,p` è la formula del cancello `(E_cleanLS − E_TCN)/(E_cleanLS − E_oracolo)` su `E_dyn` contro `y`, sugli esclusi, al
  punto scelto, con i riferimenti archiviati del cancello;
- `R_fit,p` è la stessa formula sul fitting, contro `y`, allo stesso punto scelto, con i riferimenti di fitting della sezione 4;
- il pooled somma gli accumulatori dei due fold prima dei rapporti.

`bordo` è vero se il minimo pooled della MSE oppure quello di `E_dyn`, contro `y`, cade nelle epoche 146-150
(`selection.needs_epoch_extension`, la regola congelata). Soglia: 0,50, la costante congelata `TEMPORAL_HEAD_RECOVERY`.

| Priorità | Condizione | Esito |
| --- | --- | --- |
| 0 | un cancello di validità fallito, lo stato scelto che non ridà il suo punto, oppure un recupero non calcolabile (riferimenti non finiti o margine non positivo) | **non valutabile** |
| 1 | `R_held,p >= 0,50` nei tre pannelli | **sensibilità recuperata allenando sul pulito**; il bordo si annota soltanto |
| 2 | almeno un `R_held,p < 0,50` e `bordo` vero | **non risolto entro 150 epoche**; nessuna estensione automatica, nessuna causa assegnata |
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

## 6. Dopo i job

- Ricevute, JSON dei risultati, log e `sacct` si raccolgono sul Mac e si committano su questo ramo. Gli stati dei fit puliti
  restano su `$WORK`.
- Nessun passo successivo parte da solo. Con i numeri in mano Simone decide se e quale variante provare. Restano valide le
  condizioni già concordate: variante fissata prima, conferma su un mondo nuovo con semi `20261122 + 1000·fold`, un solo
  tentativo.
