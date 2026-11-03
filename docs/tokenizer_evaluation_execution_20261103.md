# Tokenizer: esecuzione delle valutazioni dopo il preflight — 03/11/2026

**I valutatori della nuova campagna sono implementati e revisionati.** Il controller verifica il teacher effettivo, la ridge usa il supporto centrale comune di sei patch e le sonde usano 12.000 iterazioni. I vecchi controller B19618 e temporale restano invariati: le loro identità specifiche non sono quelle di questa campagna.

Questo documento prepara l'esecuzione autorizzata da Simone; non lancia job, non modifica il runtime e non prenota risorse. **Aggiornamento operativo del 03/11:** Simone autorizza anche il superamento del budget precedente; il disegno è quindi **tre bracci nuovi fino a 19.618 passi, con le valutazioni comuni**, senza ripiego sul B storico e senza seconda richiesta di consenso. L'esecuzione usa submitter e ledger comuni, secondo l'autorizzazione operativa vigente. Il preflight ha **tetto 2 GPU-ore**, distinto dall'ipotesi di 1 GPU-ora del precedente conto illustrativo. Dopo il preflight si sostituiscono le stime con tempi misurati e tetti espliciti. L'autorizzazione aggiuntiva riguarda questo confronto, non campagne o retry illimitati.

Fonti normative: [protocollo](tokenizer_window_nonlinear_protocol_20261102.md), [emendamento del 03/11](reviews/risposta_claude_tokenizer_20261103.md), [configurazioni aggiornate](../results/infra/tokenizer_review_20261103/comparison_configurations.json). Prevale l'emendamento: **raggio esplicito 0, JEPA 0**, EPN-612 domanda principale di classificazione distinta dalla posa. L'audit riguarda il checkout GPT `84408bf30bf79fcde7dff34138ba59d30ac9d1d3`; per le sonde convergenti è stato letto, senza integrarlo, il ramo Claude al commit `a59079a44b482b85c9ffa26efb2b4851ac6e68a6`.

## Matrice minima e dipendenze

| Momento | Per ciascuno di T25-L, T100-L, T100-G | Rapporti |
|---|---|---|
| 5.000 passi | Teacher posa: sottoinsieme train fisso e tutti i 15 utenti val | Tre ridge istantanee, un confronto appaiato comune; pannello student del training letto separatamente |
| 19.618 passi, fase autorizzata | Stessa estrazione teacher posa; teacher P1 DB2/DB3/DB6 e P2 EPN-612/UCI | Tre ridge; sonde lineari per i cinque dataset; confronti per dataset, P1 e P2 |

Sono **sei coppie train/val di posa** nell'intera matrice, non sei estrazioni aggiuntive student. Si possono eseguire train e val in sequenza nella stessa allocazione GPU: tre job a 5.000 e tre all'endpoint. All'endpoint si aggiungono tre estrazioni P1 e tre P2: **12 job GPU di valutazione** se si mantiene questa suddivisione. È una granularità proposta, non una lista già sottomessa.

La dipendenza è: preflight tecnico e piano delle allocazioni → training del braccio → ricevuta del checkpoint → estrazione → verifica delle cache → fit/report CPU → confronto quando tutti i bracci alla stessa dose sono completi. Il confronto rifiuta celle mancanti, duplicate o con dose diversa; non usa l'intersezione disponibile. Screening e endpoint hanno rapporti distinti. Non si sceglie un checkpoint sulla val e non si continua soltanto il vincitore. L'endpoint autorizzato non dipende dal segno dei risultati intermedi: il passaggio si arresta soltanto per problemi tecnici da risolvere, senza una nuova richiesta di consenso per la continuazione già approvata.

`scripts/evaluation_chain.py` gestisce già DAG, guardie dei checkpoint e submitter, ma il suo schema richiede run concluso per la guardia ordinaria. Con training separati a 5.000 e successiva ripresa ciò è coerente. Non collegare l'estrazione a 5.000 all'ultimo job lungo aspettandosi che parta durante quel job. Predicati del ledger, `sacct`, assenza di STOP, checksum e ricevute devono essere verificati anche nei runner: `afterok` da solo non prova l'identità del risultato. Dipendenze su job già completati si sostituiscono con evidenza verificata, senza mantenere ID scaduti in SLURM.

## Estrazione: componenti già utilizzabili

### Posa teacher, stessa selezione storica

Il comando scientifico esiste già. Esempio per il train di **un** checkpoint, dopo le guardie; `CODE_DIR`, `CKPT` e `POSE_OUT` devono provenire dal piano congelato, con percorsi assoluti:

```bash
python3 "$CODE_DIR/scripts/pose_eval.py" --stage extract \
  --checkpoint "$CKPT" --which teacher \
  --csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" \
  --processed-root "$SCRATCH/data/processed/emg2pose" \
  --pose-root "$SCRATCH/data/processed/emg2pose_pose" \
  --split train --n-users 40 --n-recordings 200 \
  --n-windows 10 --window-s 4 --seed 20261026 --batch 32 --device cuda \
  --out "$POSE_OUT/train_subset.npz"
```

Per val: medesimo comando con `--split val`, senza `--n-users` e `--n-recordings`, uscita `val.npz`. Non cambiare seme, ordinamento delle registrazioni, stima della scala di sessione, finestre del loader o preprocessing fra i bracci. Attesi 40 utenti/200 registrazioni/1.907 finestre train e 15 utenti/1.950 registrazioni/18.548 finestre val. Se i conteggi o una singola identità differiscono, fermare il confronto e diagnosticare; non correggere la selezione in base ai risultati.

`pose_eval.py` usa `R.sanity_config().loader`, non ricostruisce automaticamente il loader dal checkpoint. Il controller deve quindi verificare la parità dei campi di preprocessing e geometria con la ricetta congelata; non basta che il modello sia caricato correttamente. Le guardie di `pose_slot_extract.py` sono un riferimento già esistente per ordine, metadati, scale e finestre. Qui **non** si richiede uguaglianza delle feature con B storico: devono cambiare col nuovo encoder. Si richiede uguaglianza esatta di metadati, target, mask, std e pose_var, contro i riferimenti storici e fra tutti i nuovi bracci/endpoint.

L'estrattore calcola la media K64 e salva `lat` in FP16, come le cache storiche; il reader convertirà in FP32 prima della LayerNorm stateless, poi userà FP64 per la ridge. **LayerNorm FP32 non significa cache della media in FP32.** Questa è la variante minima coerente con il reader precedente. Se si decide di conservare la media in FP32, il cambiamento deve essere esplicito, uniforme nei tre bracci e fissato prima delle estrazioni; aumenta spazio e I/O e richiede un parametro dell'estrattore. Non farlo implicitamente dentro il report.

La presenza di `pred` nella cache teacher è la predizione della **testa teacher**, non quella student. La ridge primaria non usa quel campo. Il pannello student prodotto dal training resta una diagnosi distinta. Estrarre anche lo student sulla val è lavoro aggiuntivo non necessario alla matrice minima, da contabilizzare separatamente.

### Classificazione teacher, pool768

Il percorso `scripts/channel_readout.py --stage extract` carica già esplicitamente il teacher. Il comando P1 esistente è:

```bash
python3 "$CODE_DIR/scripts/channel_readout.py" --stage extract \
  --checkpoint "$CKPT" --task p1 --datasets ninapro_db2 ninapro_db3 ninapro_db6 \
  --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --splits "$CODE_DIR/splits/v1/splits_v1.json" \
  --features "$CLASS_OUT/p1" --batch 64 --device cuda
```

Per P2 usare lo stesso script con `--task p2 --datasets epn612 uci_emg`, uscita `"$CLASS_OUT/p2"` e questi ingressi:

```bash
--epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" \
--epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON" \
--uci-root "$WORK/data/raw/uci_emg"
```

Non serve un flag di inferenza per finestra o GELU: **si legge la configurazione del checkpoint**, senza override. L'estrattore conserva attualmente anche local e dec e calcola il decoder; il report usa soltanto `x_pool`, 768 feature FP32, senza PCA. Questa sovrapproduzione è il percorso minimo già verificato e la base dei tempi storici. `calibration_probe.py` ha già un'estrazione del solo pool via `fm_features.extract_features`, ma produce un formato di cache diverso: usarla richiede un adattatore e una prova di parità pool sui tre nuovi tokenizer. Nessun risparmio temporale è assunto prima della misura, e non è necessario aggiungere questa ottimizzazione per rendere eseguibile il piano.

Si conservano crop, filtro, scale, esclusione dei test ufficiali, montaggi e identità OOF della ricetta P1/P2. Il **crop6 è della posa**, non un nuovo taglio del pooling P1/P2. Il parametro storico `edge_guard_s=0.1` non va interpretato come prova che `pool_p1p2` escluda automaticamente quelle patch: il pooling usa il supporto valido dell'encoder. Cambiare questo supporto ora introdurrebbe una seconda modifica della ricetta di valutazione.

## Guardie nuove indispensabili

Il controller deve attestare prima e dopo ciascuna estrazione:

1. SHA del codice congelato, piano, configurazione completa e checkpoint; dose esatta 5.000 o 19.618; stato teacher presente, finito e caricato senza chiavi mancanti; `pose_stats` e manifest corretti. Scrivere in directory nuove, fuori dal codice, senza sovrascrivere cache pregresse.
2. Configurazione effettiva: `patch_ms=25`, `window_ms=null/100/100`, `tokenizer_activation=identity/identity/gelu`, D384/K64, gain e pesi dal checkpoint. Nessuna nuova calibrazione di `fe_gain` in valutazione.
3. **Rischio silenzioso della GELU:** `fm_features.load_model` prende soltanto i campi conosciuti da `FMConfig`. Un vecchio checkout può ignorare `tokenizer_activation` e caricare ugualmente i pesi, perché GELU non aggiunge parametri. Occorrono confronto della configurazione e assert sull'attivazione effettiva del tokenizer; `load_state_dict(strict=True)` da solo non basta. Il codice attuale conosce entrambi i campi, ma il launcher deve puntare proprio allo SHA aggiornato.
4. Provenienza dell'encoder `teacher` e dello schema di cache, hash degli ingressi, array e metadati, ambiente/precisione/thread, contatori delle finestre, assenza di modifiche al checkpoint. Salvare ricevuta completa soltanto dopo tutti i controlli.

Non riusare senza adattamento `pose_b19618_eval_guard.py`: impone vecchio SHA, nomi B/R3c, passo 19.618 e **student** per la posa. Non riusare `pose_temporal_readout.py` come CLI della nuova campagna: impone B/R3c student, ricevute storiche, crop4 e due lettori. Sono riutilizzabili le funzioni numeriche verificate, non quei pin.

## Rapporti CPU e moduli esecutivi

Implementati tre moduli separati, tutti con piano e provenienze verificati: `scripts/tokenizer_evaluation.py` (`--stage validate|extract-pose|extract-classification`), `scripts/tokenizer_pose_readout.py` (fase posizionale `validate|benchmark|ridge|compare`) e `scripts/tokenizer_probe.py` (fase posizionale `fit-shard|aggregate|compare`). Le estrazioni interne sono i comandi già descritti. Questi moduli non sottomettono job né avviano training. I pin checkpoint si completano dalle ricevute delle fasi; i piani CPU si congelano dopo l’intera griglia di cache della dose, senza placeholder. Queste dipendenze non impediscono di lanciare il training.

### Ridge

Riutilizzare `pose_temporal_readout.normalized_inputs` e `central_mask(..., offset=6)`; riutilizzare `pose_e3_b.sufficient`, `merge_stats`, `cross_validate` e `prepare_solver`. Occorre adattare esplicitamente slicing, dimensioni e report: le funzioni di livello superiore del reader temporale fissano `OFFSET=4`, quindi modificare soltanto la maschera sarebbe sbagliato. Niente monkeypatch di costanti globali.

Contratto numerico:

- Solo media K64 istantanea D384; LayerNorm FP32, eps 1e-5; niente lag, PCA, slot o scaler appreso.
- Fit e scoring con `6 <= t` e `t+6 < n_patches`, maschera della sola patch centrale. Target adiacenti non decidono la validità delle feature. Intercetta non penalizzata; statistiche e soluzione FP64.
- Due fold: utenti train ordinati alternati, stessi 40 utenti e stessi fold per tutte le celle. Alpha 10^-2…10^4, estensioni previste fino a tre per lato, pareggio all'alpha minore, selezione tramite SSE standardizzata supportata sui due fold. Minimo ancora al bordo: risultato incompleto da diagnosticare, nessuna selezione sulla val.
- Dopo il refit su tutto il train, valutazione sui 15 utenti disgiunti. Dinamiche definite dal `pose_var` originale della finestra intera ≥ **0.2574231028556824**, senza ricalcolare soglia o appartenenza sul crop.
- `E_dyn`: SSE delle predizioni e target centrati separatamente per finestra/giunto, divisa per la varianza totale supportata dei target. Almeno due istanti per cella finestra/giunto; accumulo pesato per elementi validi come `pose_dynamics`. Secondari MAE in radianti, correlazione centrata, rapporto delle varianze e distribuzioni per utente; tutti/statici separati.
- Per i tre contrasti T100-L−T25-L, T100-G−T100-L, T100-G−T25-L: stessi 2.000 draw utenti, seed 20261102. Salvare somme/denominatori per utente e hash dei draw; IC percentile 95%. Controllare supporto e denominatori identici, senza scartare silenziosamente repliche invalide. Coefficienti e predizioni fuori Git, report e hash in Git.

Il comando storico `pose_eval.py --stage report` **non** implementa questo estimando: applica E1/E2/E3 storici, altra griglia, campionamento degli istanti e regole proprie. Non usarlo come rapporto primario né trasferirne le etichette di superamento alle nuove dosi.

### Logistica convergente e confronto

Nel checkout GPT letto, `harness.probe.linear_probe` ha ancora `max_iter=3000` fisso e `channel_readout.py` non espone il nuovo audit. **Il nuovo `scripts/tokenizer_probe.py` realizza il port minimo del percorso numerico e dell'audit di Claude**, con commit e hash delle fonti nel modulo, senza modificare quelle API storiche. Le fasi sono `fit-shard`, `aggregate` e `compare`, con piano JSON, `--expected-code-sha` e `--expected-plan-sha256` obbligatori. Non simula una fase A storica né ricicla i gate B/R3c delle vecchie cache. Il runner `probe_convergence.py` della vecchia campagna non è una CLI generica per i nuovi checkpoint.

Ricetta da fissare: pool768 FP32, StandardScaler sul solo train del fold, logistica lbfgs/L2 con pesi bilanciati, tol 1e-4, **12.000 iterazioni**, C in [1e-3, 1e-2, 1e-1, 1, 10, 100], seme 0. Selezione sulla val interna, pareggio C minore, refit sul solo train del fold; quattro fold, sei candidati più refit: **28 fit/dataset, 140/braccio, 420 all'endpoint per tre bracci**. Registrare dtype effettivo, versioni e thread (riferimento storico 8) e tutti i candidati, inclusi quelli non convergenti. Warning residui rendono la convergenza non risolta: nessuna eliminazione selettiva o estensione automatica delle iterazioni.

Il nuovo modulo conserva in questo caso OOF, audit e numeri descrittivi, ma pubblica `reading_status=solver_unresolved`, `decision_valid=false` e le etichette di non inferiorità P1/P2 `unresolved_solver`. Le posizioni degli intervalli non diventano un verdetto valido in presenza di fit irrisolti. La non inferiorità a 1 pp riguarda P1/P2, non una nuova regola per ciascun dataset. Il controllo finale di fonti/piano precede la pubblicazione del rapporto completo.

Gli shard possono mantenere il disegno già verificato: DB2 fold 0–1, DB2 fold 2–3, DB3 tutti, DB6 tutti, P2 tutti. Si può separare EPN da UCI per operatività, purché ogni coppia dataset/fold compaia una sola volta. La metrica EPN è riportata separatamente come principale; UCI è secondaria. P1 pesa 8/13, 3/13, 2/13; P2 pesa 1/2, 1/2. Non rinormalizzare i pesi su un dataset mancante.

Il comparatore deve verificare parità esatta di righe OOF, target nativi (EPN può avere stringhe), soggetti, fold, supporto, ricetta e preprocessing. Predizioni di classi apprese ma assenti nei target di un held-out possono essere legittime e contano come errori: non rifiutarle tramite appartenenza alla sola `unique(y_OOF)`. Distinguere warning di scoring e di convergenza. Bootstrap utenti condiviso fra i tre bracci, entro dataset, 2.000 repliche/seed 20261102; riportare per dataset e aggregati. Il comparatore storico usa altri seed/numero di repliche e non basta cambiargli l'etichetta. Esplicitare classi mancanti nelle repliche ed eventuali metriche non definite, senza cambiare a posteriori la popolazione.

Per la non inferiorità P1/P2 usare precisamente l'emendamento: IC del delta candidato−controllo [L,U], L>−1 pp sostiene il margine; U<−1 pp sostiene perdita oltre margine; altrimenti irrisolto. EPN positivo non prova un effetto causale della sola frequenza di campionamento. Gli IC restano condizionati a un seme di pretraining e a una validazione di sviluppo già utilizzata.

## Budget conservativo da aggiornare con il preflight

Base verificata: [riconciliazione del 03/11](../results/infra/tokenizer_review_20261103/budget_reconciliation.json) e [audit precedente invariato](../results/infra/frontend_window100_20261102/budget_audit.json). Residuo massimo noto: **63,292500 GPU-ore**, circa **318,82 core-ore CPU-only**; non è una nuova lettura del saldo CINECA. Con l'autorizzazione aggiuntiva questo residuo è il riferimento contabile per rendere visibile lo sforamento, **non una barriera al confronto approvato**. I core allocati con le GPU si riportano separatamente e non si sottraggono una seconda volta dal tetto CPU-only.

| Voce osservata, per encoder | GPU-ore | Che cosa comprende |
|---|---:|---|
| B5k training | 4,237778 | 3.814 s su 4 GPU, riferimento per estrapolazione |
| B19618 posa train+val | 0,500000 | Student, usato soltanto come proxy temporale del teacher nuovo |
| B19618 teacher P1 | 0,229444 | Tutti e tre i dataset, estrazione channel_readout |
| B19618 teacher P2 | 0,466111 | EPN e UCI |
| B5k student val aggiuntiva | 0,435278 | Solo se richiesta oltre al teacher |

Valutazioni per endpoint completo: 1,195556 GPU-ore per encoder storico, non 2,2. Le 2,395 GPU-ore della tranche B19618 erano **due encoder**. Il timing del nuovo tokenizer può differire; i valori non sono tetti garantiti.

Scenari ancora illustrativi, tutti con preflight **2 GPU-ore**:

| Perimetro | Conto GPU-ore | Lettura |
|---|---:|---|
| Tre bracci a 5.000, training storico +15%, screening posa +25% | **18,495333** | 14,620333 training + 1,875 estrazioni + 2 preflight; prima del timing nuovo |
| Tre bracci a 19.618, training storico +15%, screening e finale ai tempi storici | **64,451007** | Supera già il residuo di 1,158507, senza margine sulle estrazioni |
| Stesso disegno, +25% anche sulle estrazioni | **65,722673** | Supera il residuo di 2,430173; ancora senza eventuali estrazioni student aggiuntive |

Il totale a 19.618 **include** i primi 5.000: non sommare i due training. Le valutazioni a 5.000 e finali sono invece due lavori distinti. Registrare l'extrabudget conservativo nel ledger senza contare sul fatto che i job finiscano prima dei tetti. La copertura aggiuntiva è autorizzata: non ridurre dose, numero di bracci o valutazioni per rientrare a posteriori nel saldo precedente, e non sostituire il terzo braccio con B storico.

Come base operativa prudente da affinare, una coppia train/val posa può avere 1 GPU/8 CPU/100 GB/45 min, P1 1 GPU/8 CPU/100 GB/30 min e P2 1 GPU/8 CPU/100 GB/45 min. Sono tetti proposti, non prove di sufficienza per W100. La matrice completa riserverebbe così **8,25 GPU-ore di sole estrazioni**; con training illustrativo +15% e preflight 2 il conto arriverebbe a **67,614340**, prima dell'arrotondamento dei tetti di training. Lo screening soltanto riserverebbe 2,25 GPU-ore di estrazioni. Nel piano esecutivo sostituire ogni riserva con GPU × walltime del job e verificare la somma totale, senza confondere atteso e tetto.

Per CPU: il controllo convergente di Claude ha consumato circa 27 core-ore di shard per **due** encoder, più aggregati/confronto; tre nuovi encoder danno una base di circa 40,5 core-ore di soli fit, non una promessa. Proposta prudente iniziale: per braccio DB2 2×60 min, DB3 30 min, DB6 30 min, P2 60 min, sempre 8 CPU/30 GB → **32 core-ore/braccio, 96 in totale**. Sei aggregati da 2 CPU/20 min e un confronto da 2 CPU/30 min aggiungono 5 core-ore. Due rapporti ridge, uno per dose, da 8 CPU/30 min aggiungono 8 core-ore; benchmark train-only 8 CPU/15 min e due validazioni 2 CPU/30 min aggiungono 4. Totale proposto **113 core-ore**, più checkout/guardie dichiarati nel piano. I fit sui nuovi embedding possono richiedere più tempo: misurare il benchmark e non trasformare i vecchi tempi in garanzia di completamento. Nessun retry automatico.

Prima del lancio scientifico occorrono: adattatore e test di parità/mismatch, prova sintetica che il checkpoint GELU sia ricaricato come GELU, fixture dei tre bracci e delle due dosi, selezione/target comuni congelati e piano DAG con risorse, hash e ricevute. I nuovi test devono verificare anche crop6 coerente tra feature/target/mask, assenza di lettura val per alpha, confronto appaiato su tre celle, convergenza attribuita ai fit e fallimento esplicito sulle celle mancanti. I nuovi checkpoint non esistono ancora: **i benchmark dei loro fit e delle loro estrazioni non sono un prerequisito al training**. Si possono allocare valutazioni conservativamente dai tempi storici e dal fattore di throughput del preflight, poi affinare i tempi quando le cache train sono disponibili, senza usare score val e senza cambiare la ricetta. Non occorre chiedere di nuovo l'autorizzazione per il confronto e l'extrabudget già approvati; restano vincolanti identità, tetti operativi dichiarati e perimetro del disegno. La conclusione di questo audit è un piano diretto per i tre bracci, non una richiesta di scegliere una variante ridotta.

Il preventivo esecutivo successivo al collaudo è in [training e risorse](tokenizer_training_execution_20261103.md): sostituisce i conti illustrativi sopra senza cambiare la matrice scientifica.
