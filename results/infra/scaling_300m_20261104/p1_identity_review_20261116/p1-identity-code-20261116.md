# Review tecnica indipendente: p1 identity, 16/11/2026

Base esatta `c2ae7ee02b3df5bdc3310b90d6608e211753dd07` in checkout locale separato. Nessuna operazione remota, nuova feature, fit, inferenza, modifica di codice scientifico o dei worktree Claude.

## Verdetto

B con modifiche, subordinata alla diagnostica. La causa Lustre è plausibile, non dimostrata; il messaggio non conserva la coppia di stat e non identifica il campo cambiato. Non approvare ora né un retry cieco né una modifica globale di file_identity/check_unchanged. Il criterio verbale «sync, secondo hash uguale e seconda stat stabile» è incompleto: manca il vincolo esplicito all'identità dell'oggetto fra le letture e manca la distinzione nuovo output / input già pinato.

## Codice effettivo e ipotesi

- `tokenizer_preflight.py:43–52`: `Path.stat()`, `R.sha256_file(path)` (apertura autonoma del percorso), `Path.stat()`; uguaglianza di inode, size, mtime_ns, ctime_ns. Il ritorno comprende SHA e questi campi, ma non st_dev. Non usa il medesimo descriptor per stat/hash, non registra before/after in caso di rifiuto.
- **atime non viene confrontato**. Un solo aggiornamento atime non basta a produrre questo errore. La prova sintetica eseguendo le due funzioni originali via AST lo conferma.
- Non vi è conversione float dei timestamp nella guardia Python: usa gli interi `st_mtime_ns` e `st_ctime_ns`. Il `stat` testuale raccolto dopo il job non ricostruisce la coppia che ha fallito.
- `scaling_evaluation19618.py:344–408`: attende `subprocess.run(check=True)` in modo sincrono (`:371`), poi ricontrolla gli input grezzi e inventario (`:373–382`), poi valida le cache (`:384–395`). `channel_readout._save:79–87` usa `np.savez` normale con percorso, senza thread di salvataggio dichiarato. Nessun writer asincrono evidente.
- Primo controllo su output precede il contenuto (`:385`); poi verifica formato, dati, supporti/provenienza, quindi intera identità ancora uguale (`:390`). Non basta sostituire la sola prima file_identity se il fallback cambiando metadata produce risultati diversi al secondo confronto.
- Alternative da distinguere: attributi riportati in modo incoerente dal client/filesystem; scrittura o modifica metadata esterna (touch/chmod/manutenzione); sostituzione del percorso/inode/symlink; effettivo cambiamento del contenuto; errore del client/durabilità. Stabilità del semplice SHA dopo il fatto non identifica la causa.

## Contratto di correzione minimo proposto

Prima misurare, poi consentire soltanto l'assestamento effettivamente diagnosticato. Non accettare qualsiasi eccezione di file_identity.

1. Output nuovo e directory a singolo autore, subprocess terminato0, nessun writer ancora attivo. Non sovrascrivere il tentativo fallito.
2. Sincronizzazione mirata dei file completati, se la diagnostica la sostiene. `os.sync()` è una barriera da misurare, non prova di contenuto, durata garantita o metadata stabili; preferire la barriera mirata se sufficiente. Non cambiare gli estrattori numerici per questo.
3. Osservazioni complete: tempo monotono, hostname, percorso canonicale, lstat/stat e fstat del descriptor effettivamente letto, dev/ino, size, mtime_ns, ctime_ns, atime_ns (solo diagnostico), numero di byte letti, SHA completo, esito e durata barriera. Registrare anche la prima lettura rifiutata.
4. Dev/ino e legame percorso→file devono restare identici lungo entrambe le letture; la dimensione e il numero di byte letti devono concordare col file completo. Inode/device/path cambiati, file sparito, symlink/replacement, short read o hash diversi: STOP, nessuna seconda opportunità per trasformare una modifica reale in successo.
5. Se il solo campo temporale diagnosticato cambia, al massimo una nuova lettura completa dopo una pausa fissata prima, sullo stesso oggetto. Hash1 = Hash2, stessa dimensione, seconda coppia stat/fstat stabile e nessuna deriva del legame del percorso. La nuova lettura non sceglie un nuovo hash da accettare. Entrambe vanno nell'audit separato dall'identità canonica.
6. Validazione completa della cache invariata e verifica dopo la lettura ancora contro la stessa identità di contenuto. Due hash uguali non provano che nessun processo abbia riscritto byte identici; per questo servono proprietà e controllo del writer. Non promettere una garanzia assoluta di mutazione mai avvenuta.
7. Per gli input pinati il digest di riferimento deve restare quello già congelato. Non chiamare «nuova baseline» una seconda lettura che ha hash diverso dal pin. Nessun `allclose`, rimozione generale di ctime, catch-all o retry illimitato.

La diagnostica sintetica deve avere hash del payload deterministico noti PRIMA della scrittura; così distingue contenuto atteso da sola stabilità successiva. Le cache reali non pinabili storicamente vanno dichiarate tali, non trattate come prova retrospettiva del contenuto originale.

## Dove intervenire e vincoli dei sorgenti

**Non modificare `tokenizer_preflight.py`.** È un file congelato nel contratto scientifico del training:

- `scaling_train.py:42–43`: SCRIPT_PATHS lo include esplicitamente;
- `training_source_contract.json:9`: SHA `3bdd90d0a30c1089953345c0b63415f8c40974939af6dd05da63f11d59affe69`;
- `scaling_train.check_source_contract:111–120` ne impone il contenuto esatto;
- l'estrattore richiama quel contratto a inizio e fine, e lo richiamano i lettori.

Un cambio globale farebbe fallire il contratto del run accettato. Non riscrivere tale contratto storico né eludere la verifica per un bug di finalizzazione output.

Intervento minimo: helper nuovo di sola libreria standard per finalizzazione/attestazione uscite; richiamo circoscritto nel wrapper di estrazione19618. La modifica del file derivato richiede una sostituzione dichiarata aggiuntiva in `scaling_derive19618.py`, senza editare la sorgente congelata a5000. Nuovo helper esplicitamente incluso nei pin operativi e nell'audit; vecchi checkout immutati. Evitare monkeypatch globali di P o import numerici nuovi che cambino il bootstrap.

## Lettori: due famiglie, non una sola guardia

Classificazione:
- `scaling_probe19618.load_cache:121–155`: ST.resolve_pin sulle cache/receipt/provenance, poi verifica cache, poi E.P.check_unchanged.
- `:330`: verifica esatta della resource_plan dopo il fit.
- comparatore: stesso P per manifest e ingressi.

Posa:
- `tokenizer_pose_readout.verify_file:129–135` usa `pose_temporal_readout.file_state:120`, con **dev**, ino, size, mtime_ns, ctime_ns.
- `scaling_pose_readout19618:143`: stat dopo la validazione;
- `:349`: stat complessivo a fine lettura/fit.

Cambiare solo P non risolve la posa. **Prima dei fit lunghi esercitare entrambe le famiglie reali nella diagnostica su cache pinabili esistenti**, conservando file senza scritture; un esito sul solo helper estrazione non copre i lettori. Se entrambe restano stabili, conservare i lettori congelati e lo stop già prescritto. Se un percorso read-only fallisce, tenere fermo il rilascio CPU e trattarlo separatamente con evidenza e costo. Non ampliare oggi tutta la libreria preventiva senza dato.

## Commit misti e alternativa di sola attestazione

I lettori già verificano extraction_code_sha **per cache**:
- `scaling_probe19618:104,127–145`;
- `scaling_pose_readout19618:86,108–136`.

Il controller `scaling_plans19618` impone invece un solo oggetto `extraction` per `candidate_probe_caches`, `_extraction_identity:198–206`, loop`:241–258`; il costruttore latoMac deve accettare riferimenti per task, non togliere il controllo SHA. Pose/p2 già valide non hanno motivo scientifico di essere riestratte se il nuovo codice cambia soltanto attestazione e i sorgenti numerici restano byte-identici. Ogni cache mantiene il proprio SHA/receipt/codeSHA; nessuna uniformazione retroattiva.

Alternativa: verificatore separato legge le tre cache già scritte e scrive attestazione in una NUOVA directory, senza mutare quella FAILED. Il divieto del README di «adottare» assumeva di scrivere nella directory fallita; questa scrittura non è necessaria in astratto. Richiede però un esplicito emendamento di protocollo, completa validazione e catena di provenienza, senza dichiarare retroattivamente COMPLETED il job originale. Le prove attuali non bastano a garantire che tutte le condizioni siano recuperabili: mancano ricevuta finale, alcuni hash e controlli a fine job. È una possibilità da valutare, non un recupero già approvato.

## Prove locali

`/private/tmp/p1-identity-code-20261116.py` e `.json`:9 controlli su AST esatto delle guardie, scenari sintetici di metadata, controesempio del fallback verbale, pin della funzione condivisa. Nessuna riproduzione Lustre pretesa, nessun benchmark filesystem. I test mostrano l'ambito delle guardie, non la causa dei due fallimenti reali.

## Aggiunta: oracolo NPZ e non perturbare il braccio originale

Prova piccola su NumPy locale2.3.5: NPZ non compresso dello stesso dizionario/array, prodotto due volte in BytesIO a distanza di oltre2secondi e una volta per percorso, byte-identico (19.908byte, SHA1010d569d425844e5aa4b8df226bb4339107608c2ae4ee122e87af10553d5fe8). Membri ZIP con timestamp1980-01-01 e compress_type0. Prova in `/private/tmp/p1-identity-npz-determinism-20261116.json`.

Questo sostiene un oracolo di byte preparato in RAM prima dell'IO su Lustre, con una prova piccola iniziale nella versione NumPy effettiva del job. Non usare un semplice sink hasher non seekable come oracolo: ZIP può differire in header/data-descriptor rispetto al percorso seekable. BytesIO seekable, un file alla volta, evita l'ambiguità; dichiarare la RAM per array e bufferNPZ.

Nel braccio diagnostico originale preservare la sequenza stat→sha256_file→stat e registrare i valori già letti, senza open/fstat/lstat aggiuntivi prima: potrebbero rinfrescare la cache attributi e modificare il fenomeno. Le sonde descriptor/percorso aggiuntive vengono dopo la coppia originale o nei bracci correttivi esplicitamente distinti.
