# Consegna al launcher: 300M 19.618→49.046

18/11/2026. Per **1-30/300 launcher**, thread
`01a1e915-4dc5-7d90-bd1e-371ccd3b8560`. Operatore unico della linea 1 dal
passaggio di Simone del 17/11. Questa chat scientifica non avvia runner,
checkout o job. La precedente catena 19k, confronto 64785568, è conclusa.

## Aggiornamento operativo del 21/11: un solo via per la catena

Letta la sezione «Un solo via per le tre chat e i loro launcher» del riferimento
Mac `/Users/simonebenatti/dev/wearusfm/CLAUDE.md`, pubblicata in
`92d1fc0a16bfda9c2d0309c5f1d8769fa2880074`. Il via già dato da Simone copre
l'intera catena 49.046: controlli, riprese previste, attese di slot e dipendenze,
raccolta, archiviazione e commit/push. Il launcher prosegue appena i requisiti
del piano sono soddisfatti, senza altri consensi o review fra gli stadi.

Il pacchetto operativo definitivo è `87da975bc0fd4824684e166659bb996e118ff107`,
approvato dalla review unica `f6ba1c925e9619d096eea8d68371b1f8811a36a3` in
`docs/reviews/gpt_review_scaling49046_release_20261119.md`: prevale sui dettagli
preparatori qui sotto. In particolare il secondo segmento viene inviato dopo
raccolta e verifica del primo, e l'interazione dose×taglia resta descrittiva.
Nessun cambio ai pin scientifici, ai criteri o ai tetti approvati.

Unico operatore resta 1-30/300 launcher, che riconcilia il ledger prima degli
invii; questa chat non crea job o duplicati. Per il submitter segue l'invocazione
diretta già consentita descritta nel CLAUDE.md operativo, solo dopo verifica che
i byte coincidano con quelli previsti dal piano. Una risposta ambigua richiede
riconciliazione dell'ID, non risottomissione.

Un permesso tecnico realmente imposto dall'app va rispettato: un solo operatore
segnala una volta l'operazione precisa e il motivo se serve intervento dell'utente.
Non si deduce una richiesta di consenso dal solo indicatore dell'app. Restano da
segnalare decisioni scientifiche nuove, risorse fuori perimetro e arresti previsti
dai criteri; nessuno di questi è introdotto da questo aggiornamento procedurale.

## Decisione e perimetro

Simone ha chiesto di procedere al termine lungo dopo la lettura19k:
«fissiamo che si va a 49.046 o a 50000». **Fissato 49.046**, termine dello
schedule originale e del controllo30M. L'autorizzazione alla prova persiste;
non chiedere un nuovo via per ogni job. La lettura scientifica è fissata in
`docs/reviews/scaling_19618_crossline_and_49046_20261118.md`.

Prepara il pacchetto operativo completo, con piano/costi, sorgenti e test
della ripresa, pin effettivi, preservazione e percorso della valutazione.
Il wrapper precedente non è un'autorizzazione tecnica al nuovo endpoint:
una sola review del nuovo pacchetto prima dell'uso, poi la catena approvata
prosegue con i suoi controlli meccanici, senza review per ogni invio.
Non attendere un altro messaggio di Simone per cominciare la preparazione.
Nessun nuovo seme, loss, target, fine-tuning o training del 30M.

## Ripresa e due segmenti

- Checkpoint 300M a 19.618 accettato, SHA256
  `d7ebe9d87a08882ae0c1c836b62988ea21c7f1246998a5f82613878a0499387e`.
  Riverifica il pin e la ricevuta completi prima dell'uso. Origine immutabile;
  nuove directory con copia e verifica dei file necessari.
- Endpoint 19.618→39.237→49.046, un nodo/quattro GPU, batch globale 32,
  accumulo 1, stesso trainer e configurazione scientifica. Ripristino di
  student, teacher, AdamW/contatori, gain, RNG, stream/worker e allarmi.
  Nessun nuovo warmup o inizializzazione dai soli pesi.
- Conserva l'ordine di import corretto (torch prima di NumPy), ambiente e
  thread, e i controlli d'identità del campione fisso. Non reintrodurre il
  difetto già diagnosticato di BLAS/import order.
- Non cambiare `total_steps=49046`, warmup 1962, decay_fraction 0,2 o
  `keep_checkpoints`. `checkpoint_predecay.pt` a 39.237 e
  `checkpoint_postdecay.pt` a 49.046 sono salvataggi nativi automatici.
  Il vecchio wrapper 5000→19618 contiene endpoint e guardie cablati:
  serve una versione esplicita, non un cambio isolato di max_steps.
- Testa la ripresa in directory nuova anche attraverso il confine di decay:
  LR originale per ciascun update, stato/contatori/RNG/stream, rollover,
  avanzamento esatto e checkpoint nativi. Il checkpoint 39.237 precede
  l'update registrato 39.238, che usa ancora il picco per l'indicizzazione
  originale. Non dichiarare bit-identità GPU se misurata solo su CPU.
- Il secondo segmento può essere accodato con afterok; deve leggere e
  verificare la ricevuta e il checkpoint effettivi del primo, con termine
  `max_steps`, nessun STOP, supporto/log completi, identità e dose corrette.
  Dipendenza scheduler da sola insufficiente. Nessun retry automatico.
- Gate memoria sull'allocato entro il 90% della GPU, riservato registrato per
  rank. Riferimento prudente 37,7518 GiB; ultimo run allocato 36,8984 GiB,
  riservato 61,2266 GiB: non scambiare le due misure.
- Conserva pannelli e log periodici già previsti; non inventare un pannello
  all'endpoint non multiplo di 500. Nessuna valutazione downstream a 39.237
  implicita: salvataggio previsto per la ripresa e la conservazione.

**Non incorporare i nuovi target di main.** Questa è una ripresa storica
T100-G: tokenizer 100 ms/GELU/stride 25 ms, ma target RMS 25 ms e spettro rapido 50 ms.
RMS 100 ms/spettro 125 ms sono una nuova ricetta, non un aggiornamento trasparente.
Usa i sorgenti congelati; se un wrapper serializza i nuovi campi, deve
esplicitare la compatibilità storica e provarla, senza cambiare il training.

## Controllo, valutazione e conservazione

Il 30M a 49.046 è già documentato come COMPLETED 0:0, job 63525809:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/continuation/checkpoints/checkpoint_postdecay.pt`.
SHA256 `bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490`,
454.381.728 byte. È un pin documentale: riverifica presenza e byte prima di
rilasciare il confronto. Non riaddestrare il controllo.

Inventaria le cache/OOF 49k già prodotte dalla linea 2. Riutilizzale solo se
teacher, gain, input, utenti, finestre, viste, fold, supporto della posa,
lettori, griglie/procedure di selezione e contesto numerico sono compatibili con
la valutazione scaling. Il nome del checkpoint o la metrica uguale non
bastano. I valori selezionati di alpha o C possono legittimamente differire
fra dosi/taglie: si pareggia la procedura, non il suo esito.
Nessun riuso silenzioso di aggregati di un'altra ricetta di lettura.
Se serve estrarre dal controllo esistente, documenta ciò che manca e il costo
nel pacchetto, prima dei lanci. L'endpoint è teacher 30M contro teacher 300M a 49k,
vista naturale primaria e proiezione 896→384 secondaria già congelata.

Conserva checkpoint 19k, predecay e finale con hash e ricevute, compresi i
materiali della precedente catena. Ogni 300M pesa 4.673.783.740 byte (~4,35 GiB):
tre copie più una temporanea sono già ~17,4 GiB; prevedi almeno l'ordine di 20 GiB
di spazio effettivo, oltre a cache e altri materiali. Non riusare il vecchio
gate 8 GiB del 30M. Verifica quota/retention reale prima della scelta WORK/SCRATCH.

Preparare la raccolta e la lettura finale insieme alla catena: primari e
secondari restano quelli del documento scientifico, inclusa la lettura
secondaria dell'interazione dose × taglia solo su artefatti confrontabili.
Nessuna nuova ricerca di iperparametri sul held, scelta del checkpoint
migliore o modifica delle soglie dopo aver visto gli score.

## Risorse, rilascio e tracciabilità

Previsione dai tempi del run 63979977: 2,93923 s/update; training residuo atteso
~96,1 GPUh. Proposta di prenotazione: 19 h 45 min + 10 h su 4 GPU, ossia 119 GPUh;
più 0,4 GPUh di margine scheduler se resta valido il margine documentato.
Con 32 CPU sono 952 coreh, 955,2 con margine. Time-limit del trainer proposti
69.300 s e 34.200 s, lasciando 30 min al wrapper. Riverifica partizione/QoS e
limiti attuali; questi sono preventivi, non nuove letture del cluster.

Pubblica separatamente tetti di training, estrazioni, lettori CPU, checkout
e storage; usa i benchmark disponibili e il piano di riuso verificato.
Le 351 coreh erano l'inviluppo della catena 19k ormai conclusa: il suo residuo
non è un budget implicito per49k. La decisione di Simone autorizza questa
continuazione e la sua normale valutazione, non esperimenti ulteriori.

Pubblicazione e SHA origin verificato prima del deploy, checkout congelati,
submitter/lock/ledger comuni, ID nuovi ed elenco chiuso senza duplicati.
Usa la guardia di finalizzazione P1-r2 già approvata per le nuove estrazioni,
mantenendo gli hash di contenuto e i controlli sugli input: non reintrodurre
la vecchia guardia instabile né allentare l'integrità.
Primo errore reale, TIMEOUT, costo fuori dal piano o evidenza incompleta:
ferma la fase e diagnostica, senza rilanci ciechi. Registro in entrambe le
sezioni, sacct grezzo accanto all'accounting e ricevute/pin per ogni fase.

Restituisci il commit del pacchetto per la review tecnica unica. Nessun nuovo
consenso di Simone richiesto per eseguire il medesimo piano una volta pronto.
