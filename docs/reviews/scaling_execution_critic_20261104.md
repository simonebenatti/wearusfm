# Review operativa dopo Claude — scaling 300M, 4 novembre 2026

**Verdetto: APPROVATO per la preparazione operativa consegnata.** Nessun rilievo
aperto nei cambi esaminati. Questa review non attesta l'esito reale di S3 né la
readiness del training scientifico a 5.000: quel lancio richiede ancora i quattro
gate completi, le evidenze reali, i pin e la review della catena concreta.

La review Claude `a988a72d96f209cd89bc7f7b85b088a619022264` rimane integra. Questa
nota copre la correzione F-1 e i tre nuovi launcher successivi alla sua consegna.
Nessun job, SSH, fit o commit eseguito dal revisore. Il launcher training e i suoi
test sono stati implementati da questo agente; la loro verifica qui è una
verifica d'implementazione, mentre F-1 e i launcher GPU/CPU hanno ricevuto una
lettura e riesecuzione indipendenti dall'autore.

## Riscontri

| Parte | Verifica e conseguenza |
|---|---|
| F-1, `scaling_compare.py` | Nel confronto fra ispezioni di fasi diverse viene esclusa soltanto `states`. Utenti, bersagli/supporto, SHA della cache e checkpoint, braccio e split restano confrontati. Cache, provenienza e ricevute vengono rihashate; `check_unchanged` resta attivo durante la singola esecuzione. Touch e ripristino identico sono accettati; corruzione e supporto diverso sono rifiutati. |
| Training | Soltanto due argomenti: piano relativo al checkout congelato e directory ricevute nuova. Una sola fase fresh 300M/5.000, senza override di dose, inizializzazione, resume o continuazione. Allocazione reale controllata tramite il wrapper congelato: 4 GPU, 1 nodo, 32 CPU, 480G, 19.800 secondi. |
| Estrazioni GPU | Cinque celle registrate: posa/P1/P2 300M e P1/P2 30M, teacher al passo 5.000. Nessuna nuova estrazione della posa 30M né job separato di sola validazione. Ogni cella richiede allocazione reale, durata e output corrispondenti; il controller congelato mantiene i controlli dei checkpoint e delle cache. |
| Letture CPU | Partizione `lrd_all_serial`, nodo `login08`, zero GPU, 30G; conteggi CPU e tempi verificati contro SLURM. Piani esterni al checkout dei lettori sono accettati solo con percorso canonico e SHA esplicito; il codice resta congelato. Ogni stage accetta esclusivamente i propri argomenti. |
| Posa | Prima della ridge servono validate sulle quattro cache, tre benchmark train-only, pin e provenienza comuni. Contabilità di validate e dei tre benchmark associata ai rispettivi job: `COMPLETED`, `0:0`, CPU/tempo coerenti e RSS positivo entro 30GiB. Il launcher ricalcola il preventivo e rifiuta costi oltre un'ora. |
| Sonde | Prima del fit, il piano deve coprire esattamente tre celle × cinque dataset × quattro fold, senza sovrapposizioni, con massimo 15 shard. Ogni shard usa 8 CPU, 30GiB e al massimo 120 minuti; somma massima 136 core-ore. Il lettore congelato controlla anche tempi/RSS e identità dei benchmark reali. |

La formula del training è stata ricalcolata sul report reale del preflight
`63526191` usando `scaling_train.check_preflight`: media dei 112 aggiornamenti
300M **2,7666786609 s**, rapporto **3,9085627137**, durata minima **19.800 s**.
Il timeout operativo del trainer è 19.500 s. Tetto training: **22 GPU-ore e
176 core-ore**, distinto dalle valutazioni.

Le cinque estrazioni sommano **8,25 GPU-ore e 66 core-ore**. Il confronto CPU
mantiene 30 minuti con otto thread: **4 core-ore**, evitando una riduzione non
misurata del tempo a 450 secondi. Le fasi CPU massime sommano 179 core-ore; con
checkout 1/3 e riserva 2/3 il totale resta **180 core-ore**. Il launcher controlla
allocazioni e piano shard; conteggio complessivo dei job, unicità degli invii e
consumo della riserva richiedono il submitter e ledger comuni. Un launcher senza
stato non impedisce da solo di sottomettere due volte lo stesso stage.

## Rilievi trovati e chiusi prima della consegna

1. **RSS e completamento dei benchmark posa.** La prima versione verificava la
   contabilità del solo validate. Ora richiede anche i tre accounting dei
   benchmark e `MaxRSS_bytes` reale, positivo e nei limiti. Queste misure non
   certificano il picco futuro di fit/predizione: il tetto resta 30GiB e un errore
   richiede diagnosi, senza aumento automatico né retry cieco.
2. **F-2: il runtime del comparatore includeva librerie aggiuntive.** Riprodotto
   localmente con le librerie reali: importare `scaling_pose_readout` dava due
   threadpool; importare poi `scaling_compare` ne aggiungeva un terzo OpenMP
   tramite sklearn. Osservare il solo runtime della posa nel padre non
   rappresentava il processo del comparatore. Il launcher ora preimporta
   `scaling_compare` nel guard e in ogni figlio prima del controller: moduli e
   ordine di caricamento sono comuni. Controprova locale con otto thread:
   snapshot posa e sonde entrambi a tre pool e identità BLAS uguale. Il confronto
   richiede poi versioni, thread e librerie uguali alla ricevuta posa pinnata.
3. **Bootstrap `runpy` e percorsi relativi.** Riprodotto un `ValueError` chiamando
   `source_hashes()` dopo `runpy.run_path('scripts/scaling_pose_readout.py')`:
   `__file__` risultava relativo, mentre i controller usano una radice assoluta.
   Il bootstrap ora risolve il percorso prima di `run_path`. La regressione usa
   il testo operativo effettivo, importa librerie vere, verifica `--help` e poi
   chiama `source_hashes` dai namespace caricati con `__file__` assoluto per tutti
   e tre i controller. Nessuna modifica ai sorgenti scientifici congelati.

## Verifiche eseguite

Tutti i comandi pytest hanno usato `PYTHONPATH=src:tests/cpu`,
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, l'interprete `~/.venvs/wearusfm-tok/bin/python`,
`-q -p no:cacheprovider`.

| Comando / perimetro | Esito di questa verifica |
|---|---|
| `tests/cpu/test_scaling_train_launcher.py` | 29 passati, 28,73 s; esecuzione dei guard Python/allocation reali, dispatch scientifico intercettato |
| `tests/cpu/test_scaling_compare.py -k pose_inspection` | 4 passati, 8,98 s; regressioni F-1 indipendenti |
| `tests/cpu/test_scaling_evaluation_launcher.py -k 'registered_extraction or unregistered_cells or endpoint_override or equivalent_slurm'` | 17 passati, 11,07 s; cinque celle, casi vietati, output e formato SLURM equivalente |
| `tests/cpu/test_scaling_readers_launcher.py -k 'pose_ridge_gate or compare_current_runtime or real_common_preload or total136corehour or stages_dispatch or actual_walltime'` | 29 passati, 27,23 s; gate posa/runtime, bootstrap reale, limite cumulativo shard e dispatch |

Suite complete degli autori, distinte dai rerun indipendenti: comparatore+lettura
41 passati/14,97 s; launcher estrazioni 41 passati/25,92 s; launcher CPU 72
passati/63,77 s e regressione finale del bootstrap assoluto 1 passata/3,55 s.
La verifica integrata finale sarà eseguita dal responsabile prima del commit.
`bash -n` dei launcher e `git diff --check` sono puliti.

Rihashati anche i **9 file** di `scientific_runtime_freeze.json` e i **103
sorgenti** del contratto del training: **zero differenze**. F-1 è l'unico cambio
del comparatore rispetto alla review Claude ed è esplicitamente registrato nel
nuovo manifest; i produttori dell'evidenza S3 restano invariati.

| File operativo verificato | SHA256 |
|---|---|
| `scripts/scaling_compare.py` | `a3610519b453a3d37747f5a4ce926b2164a40dd503546b22b6c45829085eb938` |
| `scripts/slurm/scaling_train.sbatch` | `619fe66db4f08e88f8d7f97e468a599f069d4a4cc182892295f74a3d591363ea` |
| `scripts/slurm/scaling_evaluation.sbatch` | `a3e880e42d9dc82aaedee2eb85265f750402da9c2492e617b0fd579be0382084` |
| `scripts/slurm/scaling_readers.sbatch` | `2da66c150fbc238dcf9b211a2eb672bb41adab8de96fc6f3f0c0ea5e716f6355` |

## Ordine e limiti della readiness

La precisazione S5 è necessaria: prima del training si congelano implementazioni,
test, matrici, launcher, formule, tetti cumulativi e condizioni di stop. Un
benchmark sulle cache 300M/5.000 non può precedere quel training. Dopo il checkpoint
completo e le estrazioni si misurano i benchmark sul train, prima dei fit; questi
restringono i preventivi entro i tetti già pubblicati. Nessun benchmark futuro è
presentato come osservato e nessuna ricevuta operativa viene sostituita con una
fixture. Se i tempi non rientrano, la fase si ferma per una decisione documentata.

I test locali non provano i picchi reali degli estrattori o dei fit, né l'identità
numerica su Leonardo: i guard runtime verificano quest'ultima al lavoro reale.
S3 era ancora in corso durante questa preparazione e il suo risultato non è
verificato da questa review. Non sono stati letti nuovi risultati scientifici;
nessun nuovo criterio, loss, seme, dose o autorizzazione a 19.618 è introdotto.
