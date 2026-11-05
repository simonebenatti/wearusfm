# Dopo EPN15: valutare trasferimento su più dataset

Simone, 05/11: dopo il controllo a 15 epoche iniziare a usare altri dataset,
altrimenti si rischia di valutare tutto soltanto su EPN. È la priorità
scientifica successiva, non una nuova campagna inclusa nei 24 fit EPN15.

Le sonde storiche hanno già misurato UCI e NinaPro, ma il confronto
frozen/fine-tuning/scratch completo è finora solo EPN. Si distingue quindi
copertura delle sonde da verifica del contributo del pretraining con adattamento.

Ricognizione locale di codice, inventario e split, verificata indipendentemente;
**presenza attuale dei file remoti e costi non ancora verificati**.

| Priorità proposta | Dataset e domanda | Rapporto col pretraining | Pronto e mancante |
|---|---|---|---|
| 1. Secondo dataset escluso | **UCI-EMG, classificazione:** il vantaggio FT−scratch si ripete? | Escluso dal pretraining; popolazione interna P2 già usata per le sonde, 19 soggetti. È sviluppo, non test intatto. | Loader, split e sonde disponibili. Servono adattatore FT, fingerprint e benchmark specifici; pochi soggetti impongono lettura prudente degli IC. |
| 2. Candidato benchmark escluso, prima audit delle etichette | **Kaifosh Discrete Gestures, classificazione:** trasferimento su un altro dataset escluso dal corpus? | Ruolo firmato `benchmark`: 100 partecipanti, nessuno nel pretraining tramite Kaifosh; split pubblico 80 train / 10 val / 10 test. La sovrapposizione di persone con emg2pose/emg2qwerty non è ricostruita. | Raw storicamente verificato e EMG ingerito; `prompts`/`stages` non letti dall'ingest. Servono audit/lettura delle etichette, allineamento al tempo e ai buchi registrati, definizione del compito, adapter e benchmark. Non è pronto quanto UCI. |
| 3. Montaggio/compito diversi | **NinaPro DB2, classificazione:** le rappresentazioni si adattano a un'altra acquisizione e tassonomia? | Nel corpus, 32 soggetti pretraining e 8 test riservati. Le sonde interne hanno usato la popolazione pretraining. | Loader P1 ed etichette disponibili. Servono adattatore FT per metadati/montaggio e supporto trial; non ereditare l'assunzione Myo8 di EPN. Sviluppo sui 32 già esposti, conferma sugli 8 riservati soltanto dopo congelamento. |
| 4. Nuovo bersaglio di regressione, prima audit | **Hyser, forza:** il pretraining aiuta a stimare un bersaglio diverso dalla posa emg2pose? | Nel corpus, 16 soggetti pretraining e 4 test riservati. La ricetta corrente supervisiona posa emg2pose, non forza Hyser. | Raw forza documentato, 5 canali/100 Hz; l'ingest attuale conserva soltanto EMG. Mancano sidecar/loader forza e audit di unità, identità, sincronizzazione, durate e supporti; poi protocollo/baseline/costi per 256 canali. Sviluppo sui 16 già esposti, conferma sui 4 riservati soltanto dopo congelamento. Non è una pipeline pronta. |

Riferimenti interni: [inventario](dataset_access.md),
[risultati delle sonde](reviews/tokenizer_classification_results_20261104.md),
[loader UCI](../src/wearusfm/harness/benchmarks.py),
[loader DB2](../src/wearusfm/harness/p1_cross_subject.py),
[ingest Hyser](../src/wearusfm/ingest/hyser.py),
[split congelati](../splits/v1/splits_v1.json).
I punteggi storici T100-G sono 78,457% UCI e 18,224% DB2; non sono stime
di ciò che il fine-tuning otterrà e non scelgono automaticamente il prossimo LR.

## Precisazioni dopo la review Claude 9338c04

**Kaifosh va incluso fra i candidati.** Il ruolo non è più da decidere, nonostante
alcune frasi storiche del riferimento: la decisione 5 del foglio firmato è
«Kaifosh tutto benchmark» ([decisioni](decisioni.md), voce del 03/10, punto 5;
`splits/v1/splits_v1.json`, voce `kaifosh`). Il conteggio locale degli
[split ufficiali](../splits/official/kaifosh_users.json) conferma 100 ID:
80 train, 10 val, 10 test; la lista `pretraining` è vuota. La lettura delle
etichette resta da implementare: l'[ingest](../src/wearusfm/ingest/kaifosh.py)
documenta i gruppi `prompts` e `stages`, ma non li deserializza; il nome dei gesti
è in blocchi pandas pickled. L'audit dovrà verificare formato e percorso di
lettura, senza dare per disponibile un loader supervisionato.

Kaifosh amplia il campione di sviluppo, ma **100 partecipanti non significano
100 soggetti di test**: il test ufficiale ne ha 10 e va preservato. Il dispositivo
è della famiglia Meta; la disgiunzione fra persone dei dataset Meta non è
dichiarata dalle fonti (`fm_emg_reference_v10.md`, §8, righe 1115–1117).
L'affermazione sostenibile è quindi «dataset escluso dal pretraining tramite
questa sorgente», con limite esplicito sulla sovrapposizione fra sorgenti;
non «persone certamente mai viste» né sensore completamente nuovo.

**UCI è una verifica utile, con risoluzione da motivare.** I 19 soggetti di
sviluppo sono elencati in
`results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json`, voce
`uci_emg`. Il contrasto storico appaiato T25-L−B19618 ha IC95%
**[−2,477; +2,326] pp**, riportato nella
[lettura delle sonde](reviews/tokenizer_classification_results_20261104.md).
Questo precedente non garantisce potenza sufficiente per un vantaggio di circa
1,4 pp e non va trasformato in una prova di impossibilità: riguarda altri modelli
e una specifica variabilità appaiata. Prima dei nuovi risultati fissare effetto
minimo di interesse, numero di semi e precisione attesa sulla base dei dati di
sviluppo. Aumentare i semi misura la variabilità di training, ma non sostituisce
nuovi soggetti né rende più grande la popolazione indipendente.

**DB2 e Hyser richiedono due affermazioni distinte.** Nei rispettivi gruppi di
sviluppo (32 e 16 persone), FT ha già visto l'EMG dei soggetti durante il
pretraining e scratch no: il contrasto comprende beneficio del pretraining
ed esposizione non supervisionata alle persone. Può sostenere adattamento a
etichette/compiti sul corpus esposto; non generalizzazione a persone escluse
dal pretraining. I test riservati degli split firmati sono **8 DB2 e 4 Hyser**:
solo una valutazione finale dopo congelamento di modello, target, griglia e
regole di lettura può sostenere l'affermazione su tali soggetti riservati,
con incertezza dovuta al campione piccolo. Non usarli per scegliere LR, dose,
lag della forza o preprocessing, né per iterare il protocollo. La forza Hyser
è un bersaglio diverso dalla posa supervisionata, ma questo non rende nuovo
il corpus EMG usato in sviluppo.

## Regole da fissare prima dei prossimi risultati

- Selezionare candidati e checkpoint prima di leggere i nuovi confronti.
  Confrontare frozen/FT/scratch con opportunità di tuning e budget supervisionato
  motivati per ciascun dataset, senza assumere la ricetta EPN ottimale ovunque.
- Per i **nuovi dataset**, predisporre almeno tre punti di griglia oppure una
  regola di estensione predefinita quando l'ottimo cade sul bordo, applicata
  soltanto alla validation e con budget comparabile fra regimi. Non ampliare la
  griglia dopo aver letto held/test. Questa regola riguarda la fase successiva:
  **la griglia EPN15 già rivista resta invariata**.
- Separare dataset esclusi dal pretraining, nuovi soggetti e nuovi compiti su
  dati già esposti. Sono evidenze diverse, non un'unica categoria di trasferimento.
- Mantenere gruppi soggetto/sessione/trial disgiunti secondo il compito; evitare
  contaminazione da finestre sovrapposte. Per i target nuovi, fit delle
  normalizzazioni e selezione degli iperparametri soltanto sui dati ammessi.
- Predefinire metriche e confronti, includendo baseline pertinenti. Per la
  regressione distinguere livello e dinamica, e confrontare anche con baseline
  semplici: il solo miglioramento della posa già supervisionata non basta.
- Riservare una verifica indipendente dopo lo sviluppo. Le popolazioni già
  osservate nelle sonde non diventano test nuovi cambiando il lettore.

Non presentare alternative come già pronte: NinaPro DB8 documenta target glove,
ma l'ingest li esclude e i 12 soggetti sono tutti nel pretraining; putEMG-force
non è incluso nel download locale documentato (`emg_gestures` soltanto).
L'audit di disponibilità e validità dei bersagli precede la scelta finale
della regressione. Nessun nuovo job, download, addestramento o budget da questa nota.
