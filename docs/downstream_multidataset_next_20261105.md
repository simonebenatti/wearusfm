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
| 2. Montaggio/compito diversi | **NinaPro DB2, classificazione:** le rappresentazioni si adattano a un'altra acquisizione e tassonomia? | Nel corpus, 32 soggetti pretraining e 8 test riservati. Le sonde interne hanno usato la popolazione pretraining. | Loader P1 ed etichette disponibili. Servono adattatore FT per metadati/montaggio e supporto trial; non ereditare l'assunzione Myo8 di EPN. Esplicitare esposizione precedente e preservare i test. |
| 3. Nuovo bersaglio di regressione, prima audit | **Hyser, forza:** il pretraining aiuta a stimare un bersaglio diverso dalla posa emg2pose? | Nel corpus, 16 soggetti pretraining e 4 test riservati. La ricetta corrente supervisiona posa emg2pose, non forza Hyser. | Raw forza documentato, 5 canali/100 Hz; l'ingest attuale conserva soltanto EMG. Mancano sidecar/loader forza e audit di unità, identità, sincronizzazione, durate e supporti; poi protocollo/baseline/costi per 256 canali. Non è una pipeline pronta. |

Riferimenti interni: [inventario](dataset_access.md),
[risultati delle sonde](reviews/tokenizer_classification_results_20261104.md),
[loader UCI](../src/wearusfm/harness/benchmarks.py),
[loader DB2](../src/wearusfm/harness/p1_cross_subject.py),
[ingest Hyser](../src/wearusfm/ingest/hyser.py),
[split congelati](../splits/v1/splits_v1.json).
I punteggi storici T100-G sono 78,457% UCI e 18,224% DB2; non sono stime
di ciò che il fine-tuning otterrà e non scelgono automaticamente il prossimo LR.

## Regole da fissare prima dei prossimi risultati

- Selezionare candidati e checkpoint prima di leggere i nuovi confronti.
  Confrontare frozen/FT/scratch con opportunità di tuning e budget supervisionato
  motivati per ciascun dataset, senza assumere la ricetta EPN ottimale ovunque.
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
