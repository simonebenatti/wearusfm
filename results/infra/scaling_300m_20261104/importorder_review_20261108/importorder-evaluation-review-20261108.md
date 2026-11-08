# Import order: revisione indipendente dello stadio B scaling19618

Verificato in sola lettura `/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec`, HEAD `913b4ae4bdfd8de709641ef39383a9fa7e0020f0`; letta la proposta `0c5b974af2483faf767a2e171183ae1af0dcdb19`. Nessun SSH, job, deploy, modifica del checkout o score nuovo. Evidenza riassunta in `/private/tmp/importorder-evaluation-evidence-20261108.json`, SHA256 `eacb1f7e4404fd3f4d3fe0655c9175fcea5067583598a36b4c604e7ceea148bd`.

## Conclusione

La correzione minima resta **torch-first nel solo wrapper di continuazione**, per riprendere il percorso numerico del training originale. **Non convertire per riflesso gli estrattori o i lettori a torch-first.** La valutazione storica30M usava già i percorsi NumPy-first; lo stadio B attuale li conserva. Non esiste un guard B che imponga l'uguaglianza bit a bit fra i segnali/ancore rigenerati dal training e quelli del percorso di valutazione.

Un dettaglio decisivo: `scaling_evaluation19618.py` è un supervisore. Invoca gli estrattori storici in **nuovi processi Python mediante subprocess**, quindi l'ordine degli import del supervisore o del precedente job di training non si trasmette ai figli. Il caso che ha rotto la ripresa avviene invece all'interno dello stesso worker di training.

La frase «i lettori importano solo numpy, quindi percorso OpenBLAS» è troppo forte: importano Torch transitivamente, la normalizzazione LayerNorm usa Torch direttamente, SciPy può avere un'altra libreria e l'interposizione può interessare operazioni diverse. La descrizione corretta è **entry point NumPy-first, con l'insieme delle dipendenze comune e congelato**.

## Mappa dei percorsi e del controllo storico

| Percorso | Primo ingresso rilevante | Verifica e conseguenza |
|---|---|---|
| Training originale30M/300M `train_jepa.py` |Torch prima di `training.run`/NumPy |Il resume deve mantenere questo ordine per ritrovare gli hash del fixed set e del pose panel. Il fix0c5b974 ripristina questa condizione.|
| `scaling_preflight.py:21–22` |NumPy, poi Torch |Il preflight accoppia30M/300M **fra loro** nello stesso processo, confrontando batch/segnali/ancore/supporti e stato iniziale del front-end (`:143–148`,`:216–224`). Non impone uguaglianza con il fixed set storico del training. Non cambiarlo né invalidare retroattivamente il confronto tecnico: è un benchmark a ricetta comune, non prova di identità bit a bit con la CLI di produzione.|
| `scaling_evaluation19618.py:27–28` |NumPy, poi Torch |Carica e verifica il teacher, poi costruisce comandi e avvia processi nuovi (`:274`,`:344` e seguito). Il cambio dell'ordine nel trainer non cambia questo processo.|
| Figlio posa `scripts/pose_eval.py` |`pose_targets` e `harness.pose_eval` importano NumPy prima del successivo Torch |È il percorso effettivo dell'estrazione. Il supervisore non precarica Torch nel figlio.|
| Figlio classificazione `scripts/channel_readout.py:35` |NumPy prima di `harness.channel_readout`/Torch |È il percorso effettivo P1/P2; identico a quello del controllo storico.|
| Lettori CPU e confronto |`scaling_compare19618.py:17`, `scaling_probe19618.py:23`, `scaling_pose_readout19618.py:20`: NumPy prima delle dipendenze transitive |Il dispatcher precarica lo stesso `scaling_compare19618` nel processo padre e in ogni figlio, prima di runpy (`scaling_readers19618.sbatch:174,267`). Nessun confronto obbligatorio del loro runtime con quello del training.|

Il codice di estrazione registrato nei piani del **controllo30M a19.618** è `453ce6c496745f2d73aa3007406721586e6dd767`, checkpoint `a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056`. Ho confrontato tramite Git i byte di nove file fra quella revisione e HEAD913b4ae: sono tutti identici. Comprendono i tre script `pose_eval`, `channel_readout`, `tokenizer_evaluation` e i moduli `pose_targets`, `harness.pose_eval`, `p1_cross_subject`, `benchmarks`, `fm_features`, `harness.channel_readout`. Gli hash completi stanno nell'evidenza JSON.

Il commento degli estrattori scaling che cita f2061b8/b315dad si riferisce alle primitive identiche di precedenti campagne: per identificare il **controllo attuale** va usato il pin reale453ce6c. `tokenizer_evaluation.extraction_commands:279` mostra esplicitamente i comandi `sys.executable scripts/pose_eval.py` e `sys.executable scripts/channel_readout.py`; non sono chiamate numeriche nello stesso processo del trainer. La verifica locale `python3 scripts/scaling_derive19618.py --check` è passata, senza scrittura.

## Quali identità dello stadio B sono sensibili, e a cosa

| Guard/identità | Contenuto esatto | Effetto atteso del solo fix al trainer |
|---|---|---|
| Checkpoint, config, receipt, source contract; `torch.equal` teacher/gain (`scaling_evaluation19618:96–191`) |Byte dei file, configurazione, firma manifest/scale/finestre, pesi e buffer salvati caricati esattamente |Non dipende dal BLAS scelto per filtrare una finestra. Il nuovo checkpoint avrà il suo pin reale dopo il training; non si ricalibrano gain o front-end sui dati di valutazione.|
| `raw_inputs`/`source_states` |Identità dei file sorgente prima/dopo l'estrazione; inventario e registrazioni selezionate |Non confronta i segnali già filtrati del fixed set. Cambiare una libreria numerica non modifica i byte dei file.|
| Posa: `verify_pose_cache:194–225` e `validate_artifact:100–145` |`target`, `mask`, `std`, `pose_var`, metadata e maschera centrale, contro il riferimento storico |Non comprende `signals`, `anchor_targets`, latenti o predizioni. La posa deriva dai blocchi di angoli già salvati, da `np.add.at`, somme/divisioni/standardizzazione; `pose_var` usa varianza/riduzione NumPy. Non passa da `sosfiltfilt`, `np.linalg.solve` o `power @ bands`. La diagnosi ha infatti trovato pose e finestre uguali. Nessuna rottura attesa da quel solo fix.|
| Classificazione: `input_sha256`, `verify_classification_cache:228–272` |P1: finestre dequantizzate/segmentate, label, fs, montaggio e chiave sessione; P2: finestre parse/padded o convertite nei codici Myo, label, soggetti, fs e montaggio |L'hash è calcolato **prima** di `extract_channel_readouts`, del filtro FM e delle scale di normalizzazione. Non è l'hash di un tensore già filtrato. Le operazioni preparatorie non contengono il solve/matmul diagnosticato.|
| Classificazione: y/s/subjects/all_subjects/ordered rows/fold roles |Tipi e uguaglianza esatta degli array di etichette e soggetti; fold deterministici |Non dipende dal filtro; EPN mantiene label testuali. Il guard resta esatto e non va allentato.|
| Hash integrale delle cache, provenance e ricevute |Tutti i byte di ciascun output reale |Può cambiare se cambia una feature per ordine numerico, ma gli output300M sono nuovi e ricevono nuovi pin. **Non** è richiesta uguaglianza della cache300M con la30M, né con un'altra estrazione rigenerata. Una riscrittura dopo il pin viene correttamente rifiutata.|
| Forma/precisione/finitudine delle feature |PosaD896/384,FP16; pool1792/768,FP32; predFP32 |Non è un confronto numerico esatto con le feature storiche. Perturbazioni di un ulp non fanno fallire una guardia di finitezza/forma.|
| Matrici S5 |Byte delle due NPY già pubblicate, hash degli array e manifest; ortogonalità con atol2e-6 |Il loader non rigenera QR: import order non cambia i byte salvati. Cambiare BLAS può cambiare il risultato di `x @ projection`, perciò tutti i fit devono usare lo stesso bootstrap. Non rigenerare le matrici per «uniformarle».|
| Benchmark→fit CPU e report→comparatore |Piano/codice/ricetta, tempi, versioni, threadpool e ricevute coerenti |Un cambio di bootstrap fra fasi può rompere questi confronti e invalidare tempi/fit. Il fix al trainer, lasciando i lettori invariati, non introduce quel cambio.|
| Comparatore finale (`scaling_compare19618:145–188`,`:194–228`) |Hash coefficienti/predizioni; ricostruzione esatta delle statistiche dagli stessi pred/target, OOF ricomposte dagli shard; pairing row/y/s/fold |Non rifitta il modello da zero per confrontare coefficienti ottenuti con due BLAS. Verifica gli artefatti di quella catena. Rimane necessario tenere uniforme l'ambiente fra produttori e comparatore.|

Per P1, `session_windows` dequantizza i campioniint16 con divisioni per scala e segmenta (`p1_cross_subject:47–75`); l'hash viene aggiornato a `channel_readout:119`, prima del filtro a `harness.channel_readout:120`. P2 fa altrettanto a `channel_readout:151`. La posa legge/filtra EMG per il **forward**, ma costruisce separatamente i target tramite `pose_store.window` e la loro varianza (`harness.pose_eval:129–203`). Gli hash dei bersagli non certificano che due percorsi di preprocessingEMG siano identici: sono guardie del target/supporto, ed è corretto non attribuire loro un'altra funzione.

Non posso garantire da questi dati che una perturbazione sia irrilevante per tutte le future feature o sonde: Claude ha misurato gli ultimi bit su un sottoinsieme di batch. Non serve tale assunzione per la raccomandazione, perché il percorso di valutazione rimane quello già usato dal controllo.

## Limite dimostrato dei fingerprint runtime

I fingerprint attuali sono utili, ma **non certificano il provider BLAS effettivamente invocato da NumPy**:

- La receipt di estrazione (`scaling_evaluation19618:400–403`) registra versioni, device, numero thread Torch e variabili dei thread. Non registra il binding BLAS del figlio; il runtime riportato è quello del supervisore.
- `tokenizer_probe.numeric_context:183` usa la lista ordinata dei pool di threadpoolctl, con sei campi; `scaling_compare19618._blas_identity:89` ordina cinque campi e ignora num_threads nel confronto fra posa e sonde. Questi sono inventari di librerie visibili, non la risoluzione delle chiamate all'interno del modulo NumPy.
- `tokenizer_pose_readout.runtime_identity:118` conserva l'ordine originario dei pool e i thread; il launcher confronta queste liste esattamente. Può quindi rilevare un diverso ordine di caricamento, ma soltanto indirettamente.

Ho riletto i **rapporti reali** `diag4_63864244/numpy_first_report.json` e `torch_first_report.json`, ricavando i quattro pool dal loro `numpy_runtime`. Proiettando esattamente i sei campi della sonda e ordinandoli: **liste identiche e stesso hash**. Nell'ordine originale sono differenti: la posa potrebbe rifiutare quel confronto. Le versioniPython/NumPy/Torch e i thread sono uguali. Il numero di librerie mappate e i kernelMKL caricati danno invece l'evidenza aggiuntiva presente nella diagnosi; MKL non compare in quell'inventario threadpoolctl pur essendo attivato dalle operazioni nella variante torch-first.

Anche il `dladdr` di un simbolo **globale** non basta da solo: il file `probe_login/probe_mkl_order_output.txt` restituisce MKL per gli stessi simboli in entrambe le varianti. La differenza è il binding già risolto all'interno dell'estensione NumPy; nella prova raccolta emerge quando `np.linalg.solve`/matmul fanno caricare i kernelMKL in torch-first. Non trasformare perciò `numeric_sha256` in una promessa di identità numerica più forte di quella che misura.

## Raccomandazione minima per la review

1. Accettare in principio il fix locale del wrapper di continuazione; rinnovare i pin operativi del wrapper/piano e verificare con il worker corretto le identità fixed set e pose panel. Questo riguarda la ripresaA; i dataset e i guard esatti non cambiano.
2. Lasciare invariati estrattori, valutatori, matrici e bootstrap dello stadioB. Conservare la stessa ricetta NumPy-first del controllo storico e la medesima pipeline per entrambe le taglie. Cambiare solo l'ordine del supervisore di estrazione non cambierebbe i figli e non sarebbe una soluzione all'allineamento.
3. Nella consegna dichiarare separatamente training torch-first, estrazione NumPy-first storica e CPU NumPy-first con preload comune. I guard B devono restare esatti su target/supporti/row/fold e coerenti sul runtime; non sostituire equality con allclose per far passare la catena.
4. Come miglioramento circoscritto della futura attestazione, un trace dell'entry point/import order e un piccolo canary numerico solve/matmul, eseguito nel **processo effettivo** e confrontato con un riferimento esplicitamente congelato, distinguerebbero meglio il backend. È una proposta di diagnostica, non un motivo per alterare adesso tutti i moduli o reinterpretare come falliti gli score storici. Un semplice elenco di librerie o `dladdr` globale non risolve la lacuna. Nessun canary remoto è stato eseguito in questa review.

Conclusione operativa: **non emerge un blocco dello stadioB causato dal passaggio del solo training a torch-first**. L'esecuzione reale futura dovrà ancora superare i suoi guard. Il rischio aggiuntivo evitabile sarebbe modificare adesso la valutazione300M, conservando invece cache30M prodotte con il percorso storico, oppure mescolare bootstrap diversi fra benchmark, fit e confronto.
