# Continuazione T100-G: interruzioni e conservazione (04/11/2026)

Chiusura operativa dei rilievi **C-M1 / C-M2** della review Claude `63fd9c5`,
`docs/reviews/claude_review_consolidation_20261104.md`. La continuazione scientifica
è approvata dalla review; nessun training, dose, seme, loss o criterio cambia qui.
Il piano resta `awaiting_claude_review` durante la preparazione. Il responsabile
può portarlo a `claude_review_converged` soltanto dopo verifica dei test e
**rilettura favorevole di Claude del diff delle correzioni**, come richiesto da
Simone nella consegna successiva alla review originale;
questo stato non sostituisce il gate operativo sulla quota né la pubblicazione
del piano congelato, il submitter e il ledger comuni.

## C-M1: una fase incompleta non è una fase da rilanciare

Restano tre segmenti: 19.618 → 30.000 → 39.237 → 49.046. Ogni job riserva
4 GPU, 32 CPU, 480 GiB e 10.800 s; il trainer si arresta entro il controllo
`--time-limit-s=10500`, lasciando circa 300 s per salvataggi, hash e copie.
Le stime sono rispettivamente 8.050, 7.162 e 7.606 s; il margine è una stima,
non una garanzia su I/O o congestione. Tetto invariato: 36 GPU-ore, 288 core-ore.

Il controller accetta una fase solo con `summary.steps == endpoint`,
`summary.stopped == "max_steps"`, `STOP` assente, checkpoint completo, continuità
del piano/codice, e copia persistente verificata. Non riusa mai una directory
di rapporto già esistente. I successori dipendono da `afterok`; una ricevuta
assente o non completa blocca comunque il controller.

In caso di limite di tempo, TIMEOUT SLURM, errore di hashing/copia o altra uscita:

1. Non ripetere `sbatch`, non eliminare output, non cambiare `summary.json`,
   `STOP` o le ricevute. Raccogliere `sacct` con ExitCode/risorse, log, launch,
   summary, config, nomi/dimensioni/date/SHA dei checkpoint realmente presenti.
   Riconciliare il ledger e fermare i successori rimasti bloccati; nessun retry
   automatico, nessuna conclusione scientifica da una fase incompleta.
2. **Recupero a training già completo:** se il target è stato raggiunto e lo
   stop è `max_steps`, ma il wrapper non ha finito hash/copia/ricevuta nei 300 s,
   preparare una revisione operativa per verificare e archiviare i file già
   presenti su un job CPU. Non ripetere alcun passo di training. La nuova
   ricevuta deve legare job originale, piano originale, revisione di recupero
   e SHA verificati; solo dopo si possono riabilitare i successori con pin nuovi.
3. **Recupero da checkpoint parziale:** è ammissibile solo con
   `summary.stopped == "limite di tempo"`, passo nell'intervallo della fase,
   `STOP` assente e checkpoint completo. `checkpoint_identity` deve verificare
   lo stato al passo effettivo: configurazione originale (max_steps resta il
   target della fase), quattro RNG, ottimizzatore, stream/worker/world size,
   data signature, pose stats, finitezza e SHA. Un semplice file presente dopo
   un kill non basta. Copiare la sorgente verificata in archivio persistente
   prima di riprendere, conservando intatti i file che documentano il fallimento.
4. Il recupero richiede **nuova revisione versionata del piano**, nuovo
   `plan_sha256`, output e ID logico distinti, pin del checkpoint parziale,
   preventivo residuo + consuntivo del tentativo fallito, test di continuità e
   commit/push verificato. Non si aumenta la dose e non si riavvia il warmup.
   Per riprendere dopo 39.237 si usa la normale ripresa del trainer dallo stato
   completo, non `--branch-from` (che correttamente vieta un nuovo ramo dopo la
   fase stabile). Il piano odierno non implementa né autorizza un bypass di
   quella guardia. La revisione deve esplicitare come trasferisce, su compute
   node, il checkpoint in una nuova directory di ripresa senza mutarne lo stato.
5. STOP, NaN, corruzione, identità cambiata o summary non verificabile richiedono
   diagnosi; non sono classificati come recuperi di limite di tempo. Se serve
   cambiare il protocollo scientifico, si torna alla review della modifica.

Questa è una procedura preregistrata, non un controller di retry generico. I job
oggi previsti mantengono le loro guardie strette e non consumano tentativi extra.

## C-M2: gli endpoint hanno una copia persistente prima della ricevuta completa

La sola copia in `$SCRATCH` è soggetta alla finestra di purge di 40 giorni.
`scripts/consolidation_continue.py`, esclusivamente tramite il launcher SLURM su
compute node, conserva copie byte-identiche in:

```
/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/continuation/checkpoints/
```

| Passo | File persistente | Momento |
|---|---|---|
| 19.618 | `checkpoint_step0019618.pt` | Prima del training del primo segmento |
| 30.000 | `checkpoint_step0030000.pt` | Prima della ricevuta del segmento operativo |
| 39.237 | `checkpoint_predecay.pt` | Prima della ricevuta dell'endpoint |
| 49.046 | `checkpoint_postdecay.pt` | Prima della ricevuta finale |

Ogni copia passa da un file `.partial` creato in esclusiva, flush/fsync, SHA256
e verifica della sorgente rimasta immutata. La pubblicazione del nome finale è
atomica senza sovrascrittura. Un `.partial` lasciato da un errore richiede la
diagnosi C-M1; non viene cancellato/rimpiazzato automaticamente. I segmenti
successivi possono soltanto verificare e riusare la copia già esistente del
19.618, mai sovrascriverla. Per i nuovi endpoint una destinazione esistente è
un errore. I checkpoint e le copie non entrano in Git.

Le ricevute registrano percorso persistente, byte, SHA256, istante UTC di
verifica, sorgente e `atime_ns`/`mtime_ns`/`ctime_ns` osservati. La data di purge
è indicata prudenzialmente come **mtime della sorgente + 40 giorni**, non come
promessa del filesystem; letture successive possono aggiornare atime. Il
19.618 ha la precedente stima 13/12/2026; le date dei nuovi endpoint saranno
calcolate dai file effettivi. Non ci si affida a un touch o alla sola stima:
la copia in WORK è il gate di conservazione e la sorgente preferita per i nuovi
pin delle estrazioni. `complete: true` non viene scritto se la copia manca o
il suo hash non coincide.

## Quota: gate prima del lancio e controllo prima delle copie

Le quattro copie occupano circa 1,8 GB, usando come riferimento i checkpoint
da circa 454 MB del tokenizer; il gate richiede **almeno 8 GiB e 1.000 inode**
liberi sotto i limiti di progetto WORK (margine per file temporanei e crescita).
È una verifica di quota Lustre di progetto, non del `df` dell'intero filesystem.
Se gli inode non hanno quota configurata, il requisito numerico non si applica.
Per i blocchi una quota non interpretabile o assente fa fallire il gate.

Prima di congelare/sottomettere il piano, il responsabile legge senza scritture:

```
lfs project -d /leonardo_work/IscrB_WearUsFM
lfs quota -p <ID appena letto> /leonardo_work/IscrB_WearUsFM
```

L'output della seconda chiamata deve essere numerico (non `-h`): si confronta
l'uso con il minore fra soft/hard limit positivi, senza contare la grace come
spazio disponibile. `retention.quota_preflight` va popolato prima del lancio con
`work_root`, `checked_at_utc` (ISO UTC), `project_output`, `quota_output`, usando
la lettura reale; `check_quota_evidence` ne verifica formato e margine. Prima della
sottomissione chiamarlo con `submission_time=datetime.now(timezone.utc)` controlla
anche età non negativa e massimo24ore (testato); il submitter generale non lo
chiama da solo, quindi va incluso nel controllo operativo della consegna. La
lettura dev'essere recente (entro 24 ore dalla sottomissione) e pubblicata nel
piano. A inizio job e subito prima della copia finale il wrapper ripete entrambe
le query con timeout di 20 s ciascuna e registra le risposte nelle ricevute.
Le 24 ore si verificano alla sottomissione, non all'avvio dopo un'attesa in coda:
il runtime ricontrolla la quota reale e non fallisce per la sola età del pin.
La quota è condivisa: una lettura positiva non è una prenotazione; errori
EDQUOT o query fallite bloccano il completamento senza retry cieco.

**Stato della preparazione:** `quota_preflight: null` è intenzionale e blocca
l'esecuzione. La prima query SSH in sola lettura del 04/11 non ha restituito
`project-id` né quota ed è stata interrotta (exit 255), senza scritture remote.
Nessuna quota è stata assunta o dichiarata verificata. La chiusura documentale
C-M2 e la review scientifica restano distinte da questo gate operativo.
