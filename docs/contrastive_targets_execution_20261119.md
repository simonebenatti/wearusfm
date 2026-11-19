# Linea 3 — coorte RMS100/fast125, controlli da stato comune

19/11/2026. **REVIEW_REQUIRED. Nessun invio, checkout remoto o misura CUDA da questa consegna.**
Base `438c8d65`; review preparatoria Claude `94f5094224bde9cc076c093ef37e57f3f83dc49c`;
decisione scientifica in `faa4a8d7`, copiata byte per byte in
`reviews/contrastive_common_diagnostics_science_20261119.md`.
Simone ha affidato qui l'implementazione e la preparazione per **3-launcher**, unico operatore
per deployment, invii, stato e raccolta su Leonardo. La prima release riguarda solo l'audit.
Il presente documento e i due piani emendati prevalgono sui prontuari operativi del 16–18/11.

## Ricetta e criteri fissati prima delle nuove misure

C0/CN/CR conservano stesso modello, seed, stream, maschere, supporto della posa, dose e
schedule. I target sono RMS100 e spettro rapido125/v3, passo25ms; tokenizer100/GELU invariato.
Le λ sono da ricalcolare sul train nuovo, con la formula e i gate già prescritti.
La lettura C0−T100-G è **descrittiva**: cambiano impalcatura contrastiva e i due target;
il nuovo profilo numerico è dichiarato a parte. Non è un controllo di compatibilità né
una stima causale della sola impalcatura. `evaluation` resta byte-equivalente come oggetto
al piano scientifico di partenza; nessun cambiamento alle soglie downstream.

Il profilo candidato è **bf16 autocast, SDPA solo math, TF32 matmul/cuDNN spento**.
Le opzioni effettive sono registrate su tutti i rank e confrontate fra i sette run
(28 ricevute): profilo mancante rende inconcluso il confronto; flag differenti lo bloccano. Non si impone l'algoritmo deterministico
come sostituto del test: l'identità viene misurata. Il wrapper non cambia i default del trainer
o i checkout storici. Il benchmark di produzione deve misurare tempo e memoria a batch32
prima di proporre il profilo nei training lunghi. Memoria, tempo, esiti o supporto insufficienti
mantengono il blocco; nessun passaggio automatico a un altro backend.

### Parità a peso zero

Nel processo **full C0** intercettare lo stato subito prima degli update128 e256:
127 e255 update completati. Da quel medesimo studente, teacher EMA, FIFO, batch e
microbatch/mask reali, eseguire due backward per C0, CN e CR ripristinando RNG CPU e CUDA
per rank (si registrano anche Python/NumPy). I sei casi usano reducer nuovi uguali tra loro;
non copiano lo stato interno del reducer vivo. Non vengono sostituiti teacher o coda con fixture.
L'osservatore verifica che pesi, teacher, coda, gradienti e RNG del processo continuo restino intatti.

Richiedere gradienti ridotti **byte-identici entro e tra bracci, e fra rank**. Entrambe le loss
devono avere supporto e i pesi effettivi essere zero; controllare batch globale32 e provenienza.
Profilo richiesto, punto o supporto mancante rende inconcluso il relativo confronto.
Una disuguaglianza con evidenza completa fallisce. Nessun esito inconclusivo rilascia training.
Il rapporto lega sorgenti, config canonico, summary del C0, inizializzazione e provenienza.

Su tutte le256 loss e norme delle tre traiettorie indipendenti mantenere `rtol=1e-4`,
`atol=1e-6`, valori, scarti e conteggi delle violazioni **come descrittivi della deriva**.
Stream identico e pesi zero sono invarianti obbligatori; la prova decisiva è ora quella
sullo stato comune. Non si cancella né si riclassifica il fallimento storico.

### Ripresa a loss attive

I full CN e CR salvano il checkpoint nominato a544 e continuano **nello stesso processo**
fino a576. Solo nei config del collaudo la lista `keep_checkpoints` include544; la ricetta
scientifica lunga conserva5000/19618. Per ciascun braccio, due job distinti A/B avviano
quattro processi nuovi e copiano dagli stessi byte del checkpoint544; fanno solo545–576.
Config, hash dei byte iniziali, mapping delle origini e stato al confine sono controllati.
Non c'è un secondo addestramento indipendente del prefisso1–544.

Per tutti i rank e tutti i32 update: impronte separate RNG CPU/CUDA prima e dopo backward,
batch/microbatch, supporto delle loss, contatori, ordine e identità della coda. Al confine
student, teacher, momenti e gruppi Adam al prossimo LR e stato contrastivo coincidono.
Le tre coppie sono continuo/A, continuo/B, A/B. Su ognuna vale la regola storica:

- parametri student/teacher: errore aggregato ≤1% dello spostamento **dall'inizializzazione
  originale del primo membro**, con denominatore e pavimento originali;
- buffer, contatori, metadati, stream, RNG e ordine della coda: esatti;
- ogni singolo tensore mobile di Adam e delle chiavi della coda: errore relativo ≤1%.

Separatamente si pubblicano distanze rispetto al movimento da544 e per blocco.
Non sostituiscono il denominatore del gate. Una violazione conserva il blocco.
Un supporto richiesto mancante non è successo; un eccesso continuo/reload con reload concordi
richiede localizzazione prima di promuovere il profilo.

## Sequenza, costi e saldo

Il generatore riproducibile è `scripts/contrastive_targets_plan.py` (`--check` verifica senza scrivere).
Due piani: `results/infra/contrastive_targets_20261118/plan.json` e `preflight_plan.json`.
11 celle +6 checkout = **17 allocazioni**, nessun retry incluso.

| Fase | Job | Allocazione per job | Limite | GPUh totali | Coreh totali |
|---|---:|---|---:|---:|---:|
| Audit1024 batch |1|8CPU,30GiB|2h|0|16|
| Calibrazione512 batch senza update |1|4GPU,32CPU,240GiB|30min|2|16|
| Oracolo CUDA/NCCL e rank-exit |1|4GPU,32CPU,240GiB|30min|2|16|
| Full C0/CN/CR,576 update |3|4GPU,32CPU,240GiB|30min|6|48|
| Reload CN_A/CN_B/CR_A/CR_B,32 update |4|4GPU,32CPU,240GiB|30min|8|64|
| Lettura dei controlli e preventivo lungo |1|8CPU,30GiB|1h|0|8|
| Checkout audit/calibrazione/CUDA/full/reload/lettura |6|4CPU,2GiB|4min|0|1,6|
| **Tetto richiesto** |**17**|||**18**|**169,6**|

I checkout richiedono **tutti e tre** gli override `--cpus-per-task=4 --mem=2G --time=00:04:00`.
L'audit richiede `--time=02:00:00`. I job GPU mantengono1800s richiesti e1770s interni.
Il limite interno non autorizza il margine esterno. Somma dei walltime allocati7,9h;
non è una previsione di durata in coda. Dipendenze e limiti della QoS possono allungarla.
Con il billing storico uguale alle CPU,169,6coreh sono anche **ore locali**.
Il launcher controlla il billing reale prima dell'invio, senza assumere che sia immutato.

Consumo storico effettivo attestato:5,284444GPUh/51,371111coreh prima dei diagnostici;
diagnostici conclusi0,802222GPUh/6,456667coreh. **Totale effettivo6,086667GPUh/57,827778coreh**.
Il vecchio tetto14GPUh/129,333333coreh ha residuo aritmetico7,913333GPUh/71,505556coreh;
non equivale a una nuova autorizzazione né a risorse ancora prenotate. Il precedente preventivo
non eseguito per100/125 (14GPUh/137,066667coreh) è sostituito, non sommato al consumo.
Incremento della proposta:4GPUh/32,533333coreh. Storico effettivo + tutto il nuovo massimo:
**24,086667GPUh/227,427778coreh**, training escluso.

Margine SLURM storico:OverTimeLimit180s +KillWait180s. Sommando cautelativamente360s
per ognuna delle17 allocazioni:3,6GPUh/32,8coreh addizionali; inviluppo nuovo21,6GPUh/
202,4coreh. È una riserva di contabilizzazione, non tempo scientifico ammesso.
L'audit oltre7200s viene rifiutato anche se sacct dice COMPLETED. Si registrano consumo
richiesto, effettivo e margine separatamente.

Saldo di progetto attuale e scritture pendenti **non verificati da questa chat**:
3-launcher li acquisisce prima dell'invio e calcola residuo dopo fase e prenotazioni.
Quota: almeno12GiB nuovi più scritture pendenti; i percorsi di input sono letti, non duplicati.
Non si usa una quota del mattino come garanzia futura. Nessuna misura nuova di throughput
è ancora disponibile; i training lunghi e il downstream hanno preventivo separato.

Manutenzione registrata: **23/11/2026 ore08:00–25/11 ore08:00, Europe/Rome**.
3-launcher ricontrolla le reservation correnti. Se la fase non entra prima della chiusura,
resta in attesa per il riavvio; nessun accorciamento del test o retry implicito.

## Rilascio vincolato alla review e primo audit

La review nuova deve contenere un unico blocco `contrastive-release-review` con schema
`contrastive-target-release-review-v1`, status `APPROVED`, SHA completo `reviewed_commit`,
`source_sha256`, `science_plan_sha256`, `preflight_plan_sha256`. I valori sono quelli
**dopo questo emendamento**. `contrastive_release.py bind-review` controlla i byte originali
nel commit della review, quelli dei sorgenti/piani nel commit rivisto e la copia nel checkout.
I job portano il binding e le impronte. Una review del17/11 o94f50942, senza questo binding,
non rilascia la coorte. I piani non incorporano il loro stesso hash né quello della review:
le release sono file separati. L'autore/verdetto rimane responsabilità del revisore e
dell'operatore che ne legge la provenienza: non si pretende una firma crittografica d'autore.

Dopo review, pubblicare una release **solo audit**, checkout immutabile figlio del commit
rivisto con soli metadati operativi. `CODE_DIR` e `EXPECTED_SHA` pieni; source e piani
identici a quelli revisionati. Nome di deploy proposto `deploy/contrastive-targets-audit-20261119`,
checkout `$WORK/wearusfm_campaigns/contrastive-targets-audit-20261119`.
ID audit `CONTRAST-AUDIT-targets-v3-20261118`, checkout
`CONTRAST-CHECKOUT-audit-targets-v3-20261118`. Nessuno di questi è inviato qui.

Le fasi successive pubblicano pin solo dopo le ricevute effettive. Per calibrazione servono:
nuovo audit1024 batch valido e SHA del report; accounting JSON schema
`contrastive-audit-accounting-v1`, source/hashreport, ID logico e SLURM, pin del TSV sacct.
Il TSV deve riportare `JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw`,
con unica riga dell'allocazione, **COMPLETED,0:0,ElapsedRaw≤7200,8CPU,120min**.
Il gate è eseguito nella release, nel dispatcher e nel runtime della calibrazione;
la lettura finale ricalcola la stessa verifica. Nessuna eccezione TIMEOUT ereditata.

## Comandi per 3-launcher dopo la review

Nessuna sostituzione di placeholder con valori inventati. Gli esempi non sono stati eseguiti. I comandi di preparazione locale sono:

```bash
PYTHONPATH=src:scripts python scripts/contrastive_targets_plan.py --check
PYTHONPATH=src:scripts python scripts/contrastive_release.py --help
PYTHONPATH=src:scripts python scripts/contrastive_release.py bind-review --help
PYTHONPATH=src:scripts python scripts/contrastive_release.py make-job --help
```

L'operatore usa il binding della review per costruire il job audit dal suo ID e dai quattro
input pinnati già nel piano; output nuovo sotto la radice della coorte. Poi committa/pusha
metadati, verifica SHA, pubblica il deploy e usa `scripts/submit_job.py` e ledger comune.
Il checkout usa `scripts/slurm/campaign_checkout.sbatch` con gli override sopra e gli export `CAMPAIGN` e `CAMPAIGN_SHA`
del commit pieno già pubblicato (lo SHA è anche quello dichiarato al submitter); i launcher contrastivi ricevono il path relativo del jobJSON.
Schema del solo checkout audit, da completare dopo la review:

```bash
python3 scripts/submit_job.py \
  --id CONTRAST-CHECKOUT-audit-targets-v3-20261118 \
  --agente gpt --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 0 \
  --script-stdin scripts/slurm/campaign_checkout.sbatch -- \
  --cpus-per-task=4 --mem=2G --time=00:04:00 \
  --export=ALL,CAMPAIGN=contrastive-targets-audit-20261119,CAMPAIGN_SHA=<SHA_COMPLETO_PUBBLICATO>
```

`--agente gpt` è l'etichetta del submitter; l'operatore resta 3-launcher. Per i checkout
successivi `--dopo <ID_PREREQUISITO>` precede il separatore `--`. Lo script checkout non
accetta `--sha`: il submitter ricava lo SHA da `CAMPAIGN_SHA`.
I comandi definitivi di invio con tutti gli SHA e gli afterok vengono emessi nella consegna
post-review, prima del primo invio. Nessun riferimento mobile sostituisce lo SHA di checkout.

Fasi e ricevute: audit→calibrazione→CUDA→tre full→quattro reload→lettura.
I reload A/B dipendono dal full del proprio braccio e dal checkout con i pin544 appena misurati.
La lettura richiede tutte le sette esecuzioni complete, le tracce, le ricevute numeriche e
il report di parità C0; CPU e GPU devono terminare nei rispettivi limiti. Gli output mancanti
non diventano risultati negativi della loss. Ledger unico, un tentativo per ID, nessun retry.
Il checkout dei full è distinto da quello CUDA: la ricevuta CUDA deve esistere ed essere
pinnata prima di costruire i job full. I sei checkout congelano solo le nuove evidenze tra fasi; nessun cambio di sorgenti/piani.

## Verifica locale e limiti

`contrastive_local_validation.py` produce comandi completi, ambiente, hash di test e log,
source digest e codici di uscita; `--verify` ricontrolla ogni pin. I log della suite18/11
sono mantenuti e pinnati senza spacciarli per nuove esecuzioni. La selezione storica di121test,
priva del comando, è sostituita dal nuovo elenco completo riproducibile dei test pertinenti.
Nessuna CPU locale certifica CUDA/NCCL o i tempi al batch32: quelli restano gate sul cluster.

N-6: `validate_recipe` completa ancora `target_version` assente secondo il defaultv3;
è un limite preesistente, esplicitato. I config della nuova coorte sono scritti completi e
pinnati, e audit/calibrazione/release li vincolano; non si modifica qui il parser generico.
N-7: `run_frozen` verifica il checkout storico8905e346. I suoi artefatti alla punta restano
protetti da confronto Git: la validazione locale registra che quelli preesistenti sono
invariati; non attribuisce al lettore congelato una verifica del nuovo runtime.

Verifica locale finale del 19/11: **465 test passati in 102,59 s**, launcher validati e
piani rigenerati identici. Ricevuta con comandi/hash in `amendment_20261119/validation_final/`;
resoconto dei due tentativi precedenti conservati in `amendment_20261119/verification.json`.
Aggiunta compatibilità con le API CUDA di PyTorch 2.2, provata simulandole sul Mac;
nessuna esecuzione su CUDA o sul cluster dichiarata.
