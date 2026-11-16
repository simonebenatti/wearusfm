# Protocollo proposto da congelare: SCALING-P1-IDENTITY-DIAG-v1

16/11/2026. Parte della review `gpt_review_scaling_p1_identity_20261116.md`.
**Proposta, non autorizzazione a un job.** Dopo la decisione di Simone,
Claude pubblica codice, test, manifest, risorse e commit prima dell'invio.
Un solo job diagnostico; nessun retry o rilancio P1 automatico.

## Domanda e perimetro

Distinguere un cambiamento dei soli timestamp a contenuto e oggetto
invariati da una modifica dei byte/oggetto o da un errore I/O; verificare
se lo stesso fenomeno compare su input chiusi dei lettori. Non misurare
qualità del modello o prestazioni delle sonde e non attribuire a priori
la causa a Lustre. Nessuna inferenza, fit o modifica del training.

## Risorse e arresto

- Un nodo Booster, 1 GPU allocata ma non usata, 8 CPU, 32 GiB RAM.
- SLURM richiesto **00:27:00**, più il margine di 180 s già assunto nella
  campagna: tetto **30 min, 0,5 GPUh, 4 coreh allocate**. Confermare tale
  margine nel piano; se non vale, il preventivo va corretto prima dell'invio.
- Deadline applicativa a **1.500 s** dall'inizio: controllata tra chunk,
  file e fasi. Dopo la deadline non iniziare nuovi cicli o hash completi;
  un controllo interrotto non è valido. Chiudere il report come INCOMPLETE.
  Scrivere incrementalmente eventi e risultato prima/dopo operazioni
  potenzialmente bloccanti. TIMEOUT, report troncato o errore non atteso:
  nessun via a P1, nessun retry automatico.
- Directory nuova sotto SCRATCH/`wearusfm_runs/diag`, massimo 32 GB di
  fixture scritte. Nessuna scrittura nelle cartelle delle cache reali.
  Un solo writer per fixture, concluso prima delle letture. Non modificare
  mount, stripe di file esistenti, tunable Lustre, cache del kernel o permessi
  delle cache reali. Nessun accesso amministrativo.

## Identità dell'ambiente e del programma

Prima delle prove registrare: commit e hash del diagnostico, Python/NumPy,
hostname, JobID, kernel e client Lustre se leggibili, mount e filesystem,
device e layout delle directory/fixture, numero stripe e stripe size,
thread e allocazione, processi figli propri e loro exit. Campi non accessibili
sono `unavailable`, non zero. Nessuna scansione del corpus.

Manifest dei sei input reali con percorsi canonici e pin disponibili, fissato
prima del job. Separare pin storici accettati e impronte soltanto diagnostiche.
Usare l'ordine degli import e l'ambiente dei lettori congelati per le loro
funzioni; non cambiare dipendenze o impostazioni numeriche.

## Fixture e ordine fisso

Sei triplette distinte, ognuna con file analoghi a DB2/DB3/DB6, rispettivamente
entro l'1% di **1.934.763.898 / 493.636.450 / 2.191.580.650 byte**. Formato
NPZ non compresso (`np.savez`), array sintetici deterministici senza dati
wearable. Il contenuto atteso e il SHA dei byte finali sono calcolati
indipendentemente dal file Lustre, un file alla volta in memoria seekable.
Prima dei grandi file, verificare con un NPZ piccolo che il riferimento
BytesIO e il salvataggio per percorso della NumPy effettiva producano gli
stessi byte. Un sink hash non seekable non è un sostituto valido.

Per tre triplette `aged`, creare DB2/DB3 all'inizio e registrare la chiusura;
non riscriverli. Completare ciascuna con DB6 soltanto quando DB2/DB3 sono
chiusi da almeno **720 s**. Le tre triplette `fresh` sono invece scritte
consecutivamente prima della propria lettura. Misurare le durate effettive:
non accelerare artificiosamente, non ripetere un ciclo perché la scrittura
non ha impiegato 1–2 s. La copertura di quella velocità è un risultato.

Tre modalità, applicate dopo la chiusura del writer dell'intera tripletta:

- **A:** nessuna barriera aggiunta, come il percorso attuale.
- **F:** aprire ciascuna nuova uscita senza scriverla, eseguire `fsync` sul
  descriptor, controllare il risultato e chiuderlo prima del controllo.
  È una barriera post-producer mirata, non una modifica del suo `np.savez`.
- **S:** una chiamata a `os.sync()` dopo la chiusura, con durata registrata.

Ordine fisso: **fresh A, F, S; aged S, F, A**. Ogni ciclo legge nell'ordine
DB2, DB3, DB6. Totale sei cicli e 18 file sintetici, senza ripetizioni adattive.
Le attese per arrivare a 720 s non aumentano il tempo massimo del job.
`os.sync` agisce anche sugli altri file già creati: registrare questa
esposizione. Aged-A non è un controllo «mai sincronizzato». Questo disegno
prova riproduzione/integrità, non stima causalmente l'efficacia di sync.

## Osservazioni per file

1. **Prima osservazione identica al percorso originale:** `Path.stat()` →
   SHA completo tramite la funzione storica → `Path.stat()`. Non anteporre
   `lstat`, aperture o `fstat` aggiuntivi che potrebbero aggiornare gli
   attributi e nascondere il fenomeno. Conservare entrambi i record anche
   se la guardia originale rifiuterebbe; la diagnostica registra, non adotta
   una cache di produzione.
2. Registrare timestamp monotonic/UTC di inizio e fine, durata, dimensione,
   hash, campi stat **interi**: dev, ino, size, mode, uid/gid, nlink, atime_ns,
   mtime_ns, ctime_ns; elenco preciso dei campi cambiati e frazioni ns.
   `atime` è descrittivo, non criterio della guardia originale.
3. Dopo **2 s**, seconda lettura completa con descriptor singolo,
   `fstat` prima/dopo, conteggio dei byte e confronto del descriptor col
   percorso prima/dopo. Rifiutare symlink/sostituzioni. Confrontare i due
   SHA con l'oracolo. Esattamente due letture principali; niente loop.
4. Verificare struttura NPZ/CRC, schema e contenuto sintetico noto; registrare
   separatamente il costo. I file di produzione non sono modificati o
   riserializzati. Il codice candidato della finalizzazione è sottoposto
   agli stessi eventi osservati, senza sopprimere i rifiuti dell'originale.
5. Solo **dopo** le osservazioni primarie, se disponibile senza privilegi,
   raccogliere `statx` con sincronizzazione forzata e FID/layout. Indicare
   unsupported quando necessario; non usarlo prima del primo hash o come
   sostituto di un digest mancato. Non modificare tunable di statahead/AGL.

Una variazione di contenuto/oggetto, errore I/O, lunghezza inattesa, CRC o
oracolo incoerente chiude le prove produttive sintetiche con INTEGRITY_FAIL.
Sono esclusi da questa regola solo i mutanti CPU deliberati, etichettati e
confinati ai test locali prima del job.

## Input reali e percorsi dei lettori

Sei cache selezionate prima del job, nessuna scelta dopo i risultati:

1. DB6 del primo P1 fallito;
2. DB3 del P1-r1 fallito;
3. DB2 del P1-r1, con sidecar prodotto;
4. posa `val` del 300M completata;
5. EPN di P2 del 300M completata;
6. DB3 del controllo storico 30M a 19.618.

Claude raccoglie/verifica i pin disponibili prima di preparare il manifest;
non si inventano quelli mancanti. Per file falliti senza pin storico, le
impronte sono baseline diagnostiche e **non attestazioni di accettazione**.
Per cache con pin verificato, ogni lettura deve coincidere con quel pin.

Eseguire un giro prima dei grandi carichi e uno dopo i sei cicli, confrontando
anche lo stato attraverso l'intervallo. Provare sia la sequenza reale di
classificazione (`file_identity`, `check_unchanged`) sia quella della posa
(`verify_file`/`file_state` e confronto finale), usando i moduli congelati
senza fit. Salvare tutti i rifiuti e i due SHA; non generare receipt ordinarie
né correggere sidecar esistenti. Devono essere coperti sia i confronti
all'interno dell'hash sia quelli fra inizio e fine job.

## Test CPU obbligatori della politica candidata

Prima del job, prove con piccole fixture/mutanti e rapporto pubblicato:

- file invariato accettato; atime variabile da solo non lo rifiuta;
- cambiamento dei byte a size invariata rifiutato, anche se si ripristina
  il precedente mtime;
- append/troncamento/lettura corta rifiutati;
- sostituzione dell'inode o del percorso con **byte identici** rifiutata;
- dev, tipo, permessi/proprietà/link cambiati rifiutati;
- EIO e digest diverso da un pin rifiutati;
- seconda instabilità o più di una rilettura richiesta rifiutate;
- transizione limitata ai timestamp, SHA/oracolo/oggetto uguali e seconda
  lettura stabile: ammessa solo per finalizzazione delle uscite e con audit;
- su input pinati le guardie esatte restano in vigore; nessun catch-all;
- cambiamento dei byte, dell'oggetto o di un campo non ammesso durante la
  validazione NPZ rifiutato; i due confini della validazione sono entrambi
  coperti. L'eventuale sola transizione temporale segue la politica sopra,
  con massimo una rilettura aggiuntiva per cache nell'intero percorso candidato.

Non si cambia il codice condiviso congelato per far passare questi test.

## Regola di lettura e decisioni

Riportare per ogni modalità ed età: cicli completati, attributi cambiati,
hash/oracoli, rifiuti originali/candidati, durate e byte. Nessuna stima di
probabilità d'errore dai 18 file correlati, nessuna garanzia per 43 ore.

**METADATA_ONLY_REPRODUCED:** almeno una sequenza originale documenta una
variazione limitata a mtime/ctime; entrambi i digest e l'oracolo coincidono,
oggetto e size restano invariati. Tutte le fixture sono integre, la politica
candidata rifiuta i mutanti e accetta tutti i casi benigni coperti, con
audit completo. Questo sostiene la correzione circoscritta, non identifica
da solo il componente Lustre responsabile.

**READERS_CLEAR_IN_TEST:** entrambi i percorsi dei lettori restano stabili
nei due giri e nel confronto fra giri, rispettando i pin disponibili.
Questo requisito, insieme a METADATA_ONLY_REPRODUCED, consente di proporre
il pacchetto P1 corretto e il seguito dei lettori congelati. Il via richiede
comunque la decisione di Simone e il pacchetto coerente già previsto.

**READERS_AFFECTED:** una cache vecchia ha drift di stato anche a digest
uguale. Non rilasciare i fit invariati; specificare una correzione puntuale
dei lettori e rivedere costi/contratto, senza assimilarla al solo producer.

**INTEGRITY_FAIL:** differenza di byte, oggetto, size, pin, CRC o errore I/O.
Fermare la fase interessata, preservare le prove; niente fallback o rilancio
P1 per provare se la prossima volta passa. Coinvolgere il supporto storage
con le prove se il caso lo richiede; non modificare il protocollo scientifico.

**NOT_REPRODUCED / INCOMPLETE:** nessun drift riprodotto, matrice incompleta,
ambiente non confrontabile o tempo esaurito. La causa resta aperta; non
promuovere sync a soluzione né autorizzare automaticamente una terza
inferenza. Registrare cosa manca e proporre il prossimo accertamento
mirato, con decisione esplicita. Nessun nuovo job discende da un risultato
inconclusivo di questo protocollo.
