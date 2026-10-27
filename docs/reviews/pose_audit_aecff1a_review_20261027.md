# Chiusura della review dell'audit posa — aecff1a

27/10/2026, GPT. Codice verificato: `aecff1a652096e8320f169d26dbea01d283ef29f`, ramo Claude `worktree-claude-training-engine`.
Letti sia il delta `e198a87..aecff1a` sia le correzioni del critico `ebb16e1..aecff1a`. HEAD osservato `2c8c57510f17a54586ae4dfbd72c0cd964c6bc1e`
aggiunge soltanto registrazione del lancio e comandi; E3 aggiornato in `e970833`.

**Verdetto: rilievi sostanziali della review precedente chiusi; favorevole a completare E3 e audit J2 gia' autorizzati.**
Nessun motivo emerso per interrompere i job o richiedere un nuovo ciclo di esperimenti prima dei loro risultati. Restano due dettagli P3
del sintetico, correggibili nella documentazione/etichettatura senza rilanciare simulazioni o job. La convergenza riguarda la qualita'
degli strumenti e della diagnosi: non e' ancora un risultato positivo di apprendimento di R3.

## Chiusura dei rilievi

| Rilievo | Riscontro sul codice corrente | Esito |
|---|---|---|
| R1, associazione raw EMG/processato | `pose_audit.py:126–160,417–418,434–436`: lettura raw dallo stesso membro del tar, confronto dei campioni alle coordinate delle finestre con tolleranza di quantizzazione; confronto lettura diretta/da blocco. Fixture unica attraverso entrambi gli ingest e negativi sostituzione/spostamento | Chiuso nel perimetro campionato |
| R1, controlli mancanti trattati come superati | Stati verificato/fallito/non verificato per controllo/registrazione; senza raw l'audit e' parziale, non verde (`469–472`) | Chiuso |
| R2, NaN/Inf e statistiche per lato | Oracle indipendente esclude righe non finite senza dilatazione; `store.norm(view)` fornisce le statistiche (`403,446`). Fixture integrata per lato/non finiti | Chiuso |
| Padding vacuo | Finestra da 80 patch insieme a quella da 160; assert su padding effettivo (`test_pose_chain_oracle.py:122,156–163`) | Chiuso |
| R3, argmin usato come allineamento | Scansione esplicitamente esplorativa, MAE/N su righe di feature comuni; lag operativo selezionato nello split interno e `None` se insufficiente (`304–338`); shift sintetico corretto a 200 ms/8 patch | Chiuso come controllo esplorativo |
| R4, movimento attraverso buchi | Differenze su indici originali e coppie adiacenti entrambe valide (`90–113`) | Chiuso |
| Selezione alpha | Split interno per finestre con gap, normalizzazione stimata sul solo inner fit (`210–231`) | Chiuso |
| Cronologia e disallineato | Tempo assoluto dal CSV quando disponibile; fallback lessicografico dichiarato e bordo salvato. Pairing con i vincoli di E2, MAE allineata/disallineata/gap appaiati ed esclusioni (`241–303`) | Chiuso con limiti sotto |
| Correzioni del critico | Confronto stem con precedenza corretta; guardia d'identita' produce registrazione fallita e lascia terminare il rapporto; fallback cronologico e lag operativo espliciti | Chiuso |
| Schedule e teacher sintetico | Config dbg costruita autonomamente nel test; LR identici al trainer in tutti i 1.300 aggiornamenti, dose 671. Teacher su LayerNorm(x), rappresentabile dalla testa | Chiuso |
| Conclusione esclusiva su H1 | Rettifica esplicita nelle decisioni: sintetico illustrativo, B_eff e q reali non misurati, encoder mobile/ancore assenti | Chiuso |

## Verifiche eseguite

- **Riesecuzione GPT: 20 test passati in 46,06 s**, sui tre file richiesti, src del checkout Claude e un thread BLAS:
  `test_pose_audit.py`, `test_pose_chain_oracle.py`, `test_pose_eval.py`. Nessuna modifica al codice Claude.
- Letture indipendenti parallele di catena/identita', split/ridge/ritardi e sintetico/risorse. Ulteriori **36 confronti dell'oracle** su
  12 start anche fuori griglia, M=0/50/280, NaN/Inf e tratti a zero: target e maschere concordano.
- Schedule sintetico ricalcolato: differenza massima dal trainer zero; dose dbg 671, R3 fino a 5.000 passi 4.019,5.
- JSON del sintetico: **96 esecuzioni = 32 combinazioni di parametri ×3 semi**. Mediane e primi checkpoint osservati sotto il 90%
  corrispondono ai numeri delle decisioni. Il primo checkpoint sotto soglia non e' il passo esatto di attraversamento.
- **Suite completa di Claude, log locale letto, non rieseguita da GPT:** `suite_cpu_audit_v3_20261027.txt` termina con
  `1462 passed, 1 warning in 1405.95s`, `rc=0`. Warning noto sul tensore convertito a scalare. Il log non include uno SHA nel contenuto:
  riferire questa evidenza come suite di Claude, senza spacciarla per una seconda verifica indipendente sullo SHA congelato.

## Due dettagli P3, non bloccanti

1. `docs/decisioni.md:4972`, «i semi differiscono al piu' di 0,01»: il JSON contiene un massimo di **0,019865** fra i ratio finali
   dei tre semi, per R3/q=0,5/B=28/P=1; con q=0,2/B=28/P=1 e' 0,018712. Correggere a circa 0,02 o riportare il massimo esatto.
2. `scripts/pose_head_synthetic.py:63,98`: **1-q e' il floor MSE assoluto di popolazione dovuto al rumore**, non esattamente il rapporto
   rispetto alla MSE iniziale del pannello finito. q e' nominale: la normalizzazione della norma di W non compensa esattamente la
   covarianza di LayerNorm. Rinominare `optimum_ratio_at_fixed_features` in `noise_floor_mse` e qualificare q; se si vuole un rapporto,
   esplicitare il denominatore. Varianza media del segnale/q nei tre semi circa 0,99949/0,99807/0,99670 (epsilon trascurato): effetto
   piccolo, nessuna riapertura della diagnosi o necessita' di ripetere le simulazioni per questo dettaglio.

## Limiti da mantenere nella lettura dei risultati

- Raw e letture del loader sono verificati sulle finestre campionate. La fixture non contiene gap nell'asse temporale: non equivale
  a certificare ogni percorso possibile o l'intero dataset. Il controllo esatto `time_axis` fra sidecar storici puo' segnalare una
  differenza di metadati: un eventuale fallimento va letto nel dettaglio prima di attribuirlo ai campioni.
- Nella lag scan sono comuni le righe delle feature; **target e baseline cambiano ancora con il lag** (`312–316`). La curva resta
  descrittiva e non identifica causalmente un errore temporale. Non trasformarla in un nuovo cancello di allineamento.
- `simultaneous_pairs` conta overlap fra mani opposte della stessa sessione/fase; non e' una garanzia universale contro duplicati.
  Se il contatore e' positivo, la ridge di quell'utente non sostiene una pretesa di indipendenza temporale senza ulteriori verifiche.
- Nessuna di queste prove dimostra ancora H1/H2/H3 nei dati reali. RMS negativa non prova catena errata; E3 negativa su R3c non
  esclude che il training con posa possa costruire una rappresentazione utile.

## E3 recuperata e stato dei job

Letto il controllo cache locale di Claude `~/wearusfm_local/pilot_posa/e3_cache_check_20261027.txt`:

- Val: 18.548 finestre, **1.923 recording effettivamente rappresentati**, 15 utenti; tutti i recording nello split val del CSV.
- Train: 1.907 finestre, **198 recording effettivamente rappresentati**, 40 utenti; tutti nello split train del CSV.
- **Intersezione utenti train/val zero**. Stesso percorso di checkpoint R3c postdecay, code SHA 41f6ac1 e hash delle statistiche
  `757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e`; per_side=false. Sono ora cache distinte utilizzabili per E3.
- Distinguere i recording selezionati (1.950/200) da quelli presenti nelle finestre della cache (1.923/198). Il checkpoint R3c
  non ha testa e riporta `checkpoint_pose_stats_sha256=None`: non e' un checkpoint di testa posa con normalizzazione addestrata.

**Snapshot live tramite leonardo-ops, 27/10/2026 alle 18:41–18:42 CET**, sola lettura `squeue/sacct/ls/tail`:

| Job | Attivita' | Stato |
|---|---|---|
| 62442491 | Vera estrazione train E3 | COMPLETED, exit 0, 4min12s |
| 62443474 | Rapporto E3 | PENDING, QOSMaxCpuPerUserLimit |
| 62447995 | Checkout audit aecff1a | PENDING, QOSMaxCpuPerUserLimit |
| — | Audit J2 effettivo | Non ancora sottomesso; catena locale attende checkout COMPLETED |
| 62429063 | R3c-e2 P1 pool | COMPLETED, 1h08m14s |
| 62429066 | R3c-e2 P1 dec | COMPLETED, 2h06m53s |
| 62429083 | R3c-e2 P2 pool | COMPLETED, 15min51s |
| 62429093 | R3c-e2 P2 dec | RUNNING, circa 18min |
| 62429103, 62429112, 62429117, 62429120 | Quattro sonde R3c-e1 | PENDING, QOSMaxCpuPerUserLimit |

Nessun fallimento nello snapshot. Il rapporto E3 e la directory risultati dell'audit non esistono ancora. Il file della catena locale
prevede correttamente sottomissione audit soltanto dopo checkout COMPLETED; il log del watcher non prova da solo che il processo sia
ancora attivo. GPT non ha modificato il watcher, lanciato/cancellato job o letto dati wearable grezzi dal cluster.

Risorse coerenti: rapporto 8 core/2h/**30 GB espliciti nel comando**, audit 4 core/1h/16GB, checkout 1 core/10min. Il launcher E3 nel
repo conserva il default 60GB: allinearlo in una manutenzione successiva, senza editare il checkout congelato del job corrente.
Consuntivo train 0,07 GPU-ore/0,56 core-ore + tetti rapporto 16/audit 4/checkout 0,1667 = **20,7267 core-ore**, entro 25,5;
residuo non impegnato circa 4,7733 core-ore e 0,60 GPU-ore. Le estrazioni errate precedenti restano rendicontate separatamente.

**Prossimo passo:** completare questa tranche e leggerne congiuntamente risultati e copertura. B da 5.000 passi e R3 completo restano
rinviati; nessun via aggiuntivo al training con questa review. La consegna tecnica e' sufficientemente consolidata per preparare il
passaggio scientifico, ma la scelta dell'esperimento successivo deve usare i risultati reali ancora mancanti.
