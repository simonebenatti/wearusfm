# Linea 1 — Continuazione del 300M da 5.000 a 19.618 passi: piano operativo

Data: 06/11/2026. Autore del codice e operatore: Claude. Review: GPT. **Stato: in attesa della review di GPT;
nessun deploy, nessun checkout, nessun job inviato.**

## 1. Autorizzazione e perimetro

- Simone a Claude, 06/11 verso le 17:22: «si autorizzo maq non voglio una ennersima review di gpt prima di ogni
  lancio, ne facciamo una poi lancio.»
- Simone a GPT, registrato da GPT nel commit `2d58e98b019f588f9c1f809291d4e3bf36834c6f`: «va bene, approvo la
  continuazione del 300M a 19.618 passi. Preparami il prompt per claude, cosi' prepara il piano e poi claude lancia».
- Consegna di GPT seguita qui: `docs/prompt_claude_scaling_continuation19618_20261106.md` nello stesso commit.
- Dentro il perimetro: ripresa del checkpoint completo a 5.000 fino a 19.618 (14.618 aggiornamenti), stessi dati,
  seme, batch globale 32, quattro GPU, schedule originale; valutazione teacher di posa, P1 e P2 contro il 30M gia'
  esistente a 19.618.
- Fuori dal perimetro: passi oltre 19.618, nuove loss o architetture, altri semi, nuovo training del 30M,
  riestrazione del controllo, `boost_qos_bprod`, secondi segmenti di training, retry.
- Tetti (freeze v2, `continuation19618_budget_freeze_20261105_v2.json`): training 59 GPU-ore; estrazioni 7 GPU-ore
  (66,35 in tutto con le tolleranze); letture 350 core-ore solo CPU. Le CPU allegate ai job GPU si rendicontano a
  parte: 472 core-ore il training, 56 le estrazioni.

## 2. Che cosa vuol dire «ripresa esatta»

- Il trainer non cambia: i 103 sorgenti scientifici restano identici byte per byte al contratto
  `training_source_contract.json`. `scripts/scaling_continue.py` e' solo orchestrazione attorno all'ingresso
  `train_jepa.py`, come `scripts/scaling_train.py` a 5.000 (che resta invariato e vincolato a 5.000).
- La cartella del run a 5.000 e' una prova in sola lettura: i suoi 11 file sono nel piano con SHA-256 e dimensione,
  si verificano prima del training e di nuovo dopo. La tranche nuova si scrive in una cartella nuova, in cui il
  wrapper copia `checkpoint.pt` e i tre file d'identita' (`fixed_eval_set.json`, `pose_panel_set.json`,
  `origin_sessions.json`) e ne verifica l'uguaglianza byte per byte. Il trainer li ricontrolla e non li riscrive.
- La ricetta e' il comando validato a 5.000 con due soli cambi, `--out-dir` e `--max-steps 19618`, piu' il limite
  operativo `--time-limit-s 51300`. La configurazione differisce solo in `max_steps`: lo verificano il wrapper e la
  guardia di ripresa del trainer (`_check_resume_recipe`).
- Il trainer ripristina: student, teacher, AdamW (momenti e contatori), generatori casuali di ogni processo (CPU e
  CUDA), posizione dello stream dei dati, contatori degli allarmi. Lo schedule resta quello originale: 49.046 passi
  totali, warmup 1.962, decay da 39.237. Da 5.000 a 19.618 il learning rate e' sempre al picco: nessun nuovo warmup.
- Prove su CPU (`tests/cpu/test_scaling_continue.py`): una corsa fermata a un checkpoint conservato, copiata in una
  cartella nuova e ripresa e' uguale bit per bit a una corsa ininterrotta in student, teacher, momenti e contatori di
  AdamW, generatori casuali, record delle metriche dei nuovi passi (loss, norma del gradiente, learning rate,
  origine dei batch) e diagnostica fissa. Provato in un processo con 0 e 2 worker, in DDP a due processi con
  prefetch e, con la testa e il pannello di posa della ricetta reale, copiando esattamente i quattro file del job
  vero (i record del pannello coincidono, tempi di orologio esclusi). La cartella della prima fase resta identica.
- Limite dichiarato: l'uguaglianza bit per bit e' provata su CPU con modelli piccoli. Su GPU (bf16, kernel non
  deterministici) l'uguaglianza fra una corsa ripresa e una ininterrotta non si misura: nel budget non c'e' un
  collaudo GPU in piu'. Sul run reale si verifica: stato ripristinato dagli stessi byte, guardie di ripresa del
  trainer, record esattamente dei passi 5.001-19.618 con il learning rate dello schedule originale, contatori di
  AdamW avanzati di 14.618 esatti, insieme del pannello di posa invariato, cartella a 5.000 intatta.

## 3. Verifiche gia' fatte in sola lettura (06/11, 17:39-17:45)

Evidenze: `remote_listing.txt`, `remote_sha256.txt`, `remote_verification.json`, generate da
`make_training_plan.py` dalle letture grezze (senza job di altri progetti).

- Run a 5.000: esattamente 11 file; `checkpoint.pt` e `checkpoint_step0005000.pt` identici
  (`37279995d0d484e241b032eb9b6b23a31c16a5947a0e2cf0c575da0cb70641aa`); ricevute uguali alle copie nel repo.
- Controllo 30M a 19.618: checkpoint uguale al freeze v2; due cache di posa, cinque di classificazione, provenienze,
  ricevute e riferimenti di posa uguali agli inventari rivisti (19 pin). Conservazione su scratch: la scadenza piu'
  vicina e' l'08/12/2026 alle 01:27, oltre l'orizzonte dichiarato del 22/11.
- Ingressi del training uguali al contratto. File d'identita' uguali fra candidato e controllo.
- Scheduler: QoS `normal` senza limite di durata proprio, ammessa per l'account; `boost_usr_prod` fino a 24 ore.
- Alle 18:18: percorsi nuovi su scratch canonici e non esistenti, nome del checkout di campagna libero, run a 5.000
  ancora con 11 file, `$WORK` a 1006G su 1T senza limite rigido.
- Non verificato qui: il contenuto delle cache oltre SHA-256 e dimensione (lo rifa' lo stadio di valutazione).

## 4. Stadi, job e percorsi

| Stadio | ID logico | Risorse | Limite | Costo massimo |
|---|---|---|---|---|
| A0 checkout del codice | `SCALING-300M-cont19618-checkout-20261106` | `lrd_all_serial`, 2 CPU, 4G | 00:05:00 | 0 GPU-ore; 0,17 core-ore |
| A1 training 5.000 → 19.618 | `SCALING-300M-train19618-20261106` | `boost_usr_prod`, QoS `normal`, 1 nodo, 4 GPU, 32 CPU, 480G | 14:45:00 | 59 GPU-ore; 472 core-ore allegate |
| B0 checkout della valutazione | da fissare con i pin reali | `lrd_all_serial`, 2 CPU, 4G | 00:05:00 | 0,17 core-ore |
| B1 estrazioni: posa, P1, P2 | da fissare con i pin reali | 1 GPU, 8 CPU, 100G ciascuna | 2:30, 2:00, 2:30 | 7 GPU-ore; 56 core-ore allegate |
| B2 letture solo CPU | da fissare dopo i benchmark reali | `lrd_all_serial`, al massimo 8 CPU e 30 GiB | fit al massimo 3 ore | 350 core-ore con i checkout |

- B2 comprende: validazione della posa, 3 benchmark di posa e 15 di classificazione, ridge di posa, al massimo 15
  blocchi di fit, 6 aggregati, un confronto. I fit partono solo dopo i benchmark reali a 19.618.
- Campagna del submitter: `scaling-300m-20261104`, agente `claude`. Checkout:
  `$WORK/wearusfm_campaigns/scaling_300m_continuation19618_20261106_v1`, dal riferimento
  `deploy/scaling_300m_continuation19618_20261106_v1` (mai `main`).
- Piano del training: `training_plan.json` in questa cartella.
- Cartella nuova del run: `$SCRATCH/wearusfm_runs/runs/scaling_300m_20261104/T100-G-300M-19618-from5000`.
- Ricevute del training: `$SCRATCH/wearusfm_runs/results/scaling_300m_20261104/continuation19618_v1`. Stanno su
  scratch perche' `$WORK` e' a 1006G su 1T: si raccolgono sul Mac e si committano subito dopo il job.
- Log di SLURM: `$WORK/wearusfm_runs/logs/scaling_continue_<jobid>.out` (pochi kB; l'uscita del trainer va in
  `training.log` fra le ricevute).

## 5. Costi e tempi dello stadio A

- Atteso: 14.618 passi a 2,933 s (misurati sui 5.000) = 11 ore 55 minuti, piu' circa 20 minuti di verifiche prima e
  dopo: circa 49 GPU-ore. Massimo: 14 ore 45 minuti = 59 GPU-ore, cioe' tutto il tetto del training.
- Dopo il training, del tetto complessivo restano 7,35 GPU-ore per le estrazioni.
- Il trainer si ferma da solo a 51.300 s (14 ore 15 minuti); gli altri 30 minuti sono delle verifiche del wrapper.
- Attesa in coda: non stimabile. Il training a 5.000 ha aspettato 9 ore 41 minuti; un job di forma uguale della
  linea 3 e' in attesa dal 05/11 alle 15:18.
- Memoria: riferimento 37,75 GiB allocati (run a 5.000). Il cancello resta quello gia' usato: allocato entro 0,9
  della memoria della GPU, per ogni processo. Si registrano allocato e riservato di ogni processo; nel collaudo di
  produzione il riservato e' arrivato a 0,86 della GPU: e' un rischio noto, non un cancello nuovo.

## 6. Cancelli dello stadio A, nell'ordine in cui scattano

Prima del training, dentro il job (un fallimento qui costa solo l'attesa in coda):

1. checkout di campagna pulito allo SHA atteso; sorgenti scientifici e di esecuzione uguali ai pin;
2. allocazione reale letta da `scontrol`: account, partizione, QoS `normal`, 1 nodo, 4 GPU, 32 CPU, 480G, 14:45:00;
3. piano uguale al freeze v2: un solo segmento, 59 GPU-ore, stesso controllo;
4. ingressi del training uguali al contratto; firma dei dati indipendente dalla dose;
5. run a 5.000: inventario esatto degli 11 file, ricevuta, stato ripristinabile, momenti di AdamW corrispondenti ai
   parametri del modello nell'ordine dell'optimizer, contatore massimo uguale a 5.000;
6. controllo 30M a 19.618: ricevuta, identita' del checkpoint, 19 pin delle cache, conservazione su scratch calcolata
   dalle date reali dei file;
7. cartelle di uscita nuove: nessuna sovrascrittura possibile.

Dopo il training (nessuna ricevuta se uno fallisce):

8. insieme esatto dei 12 file attesi; fine per `max_steps` a 19.618 su 4 processi; nessuno `STOP`;
9. checkpoint con nome e finale con lo stesso stato allenato; contatori di AdamW avanzati di 14.618 esatti;
10. record delle metriche esattamente 5.001-19.618 sul learning rate originale; diagnostica fissa e pannello di posa
    ai soli multipli di 500; insieme del pannello uguale a quello del run a 5.000;
11. picchi di memoria di ogni processo entro il cancello e riconciliati con il riepilogo del trainer;
12. gli 11 file del run a 5.000, il controllo e gli ingressi identici a prima; codice del checkout invariato.

La ricevuta dichiara `scientific_endpoint: true`, `automatic_continuation: false`,
`further_training_authorized: false`.

## 7. Stop e recuperi

- Stop al primo cancello fallito, errore, `TIMEOUT` o sforamento: raccolta e diagnosi, nessun retry.
- Un job fallito dopo la creazione delle cartelle le lascia sul posto: lo stesso piano non puo' ripartire, perche' il
  wrapper rifiuta uscite esistenti. Un recupero tecnico richiede un piano nuovo con i suoi pin e una review.
- Se il trainer si ferma per limite di tempo lascia un checkpoint ripristinabile ma nessuna ricevuta. Il freeze
  prevede un solo segmento: un secondo segmento e' un ampliamento e lo decide Simone.
- Sforare un tetto o cambiare perimetro richiede una decisione nuova di Simone.

## 8. Corsia seriale, attese e manutenzioni

- Il training usa `boost_usr_prod`, non la corsia seriale. Il checkout A0 chiede alla corsia seriale 2 CPU per 5
  minuti. Alle 18:18 la linea 2 la occupa con un fit in corso e uno in attesa per il limite di CPU per utente: il
  checkout puo' partire solo quando si liberano 2 CPU. Prima dell'invio controllo la coda; non cancello ne' sposto
  job di altre linee. Serve che la linea 2 lasci passare quel job: lo chiedo a Simone.
- Attesa dei job: ciclo di shell con `sacct` in sola lettura (`~/wearusfm_local/line1/wait_jobs_final.sh`, gia' usato
  negli stadi finali della linea 1). Non e' un monitor AI: non invia, non cancella, non rilancia.
- Manutenzioni: 10/11 07:00-12/11 07:00 su 540 nodi e 12/11-14/11 su altri 180; gli altri nodi restano disponibili.
  Dal 23/11 08:00 al 25/11 08:00 tutto Leonardo e' fermo: un training che non parte entro il 22/11 verso le 17
  resta in coda fino al 25/11.

## 9. Raccolta dello stadio A

Dopo lo stato finale: `sacct -P` grezzo, log di SLURM, ricevute (`receipt.json`, `launch.json`, `summary.json`,
`config.json`, `rank_0..3.json`, `training.log`), SHA-256 e dimensioni dei file della cartella nuova e di quella a
5.000, consuntivo in GPU-ore e in core-ore allegate. Tutto sul Mac in questa cartella, con commit e push verificato.
L'esito tecnico e l'analisi scientifica finale vanno a GPT in documenti separati.

## 10. Stadio B: che cosa manca e come si rilascia

- Manca il codice della valutazione a 19.618: i wrapper di estrazione, lettura e confronto esistenti sono vincolati
  a 5.000. Lo scrivo con i suoi test mentre gira il training, senza toccare i wrapper a 5.000.
- I pin del checkpoint a 19.618 e delle cache nuove esistono solo dopo il training; i piani dei fit dipendono dai
  benchmark reali. Si pubblicano quando sono reali.
- Rilascio: Simone ha chiesto una sola review prima dei lanci; la consegna di GPT prevede una review dei rilasci
  successivi. Lettura di Claude, da confermare a Simone: questa review sblocca il training; il codice della
  valutazione passa da una sola altra review, fatta mentre il training gira, e i passaggi interni (benchmark, fit,
  aggregati, confronto) sono cancelli meccanici scritti nel piano rivisto, senza altre review.
- La lettura dell'endpoint e' gia' fissata in `endpoint19618_reading.md`.

## 11. Comandi

I comandi esatti del checkout e del training stanno in `training_submit_lines.txt`, pubblicato in un commit
successivo perche' contengono lo SHA di questo commit. Controllo locale del piano, senza cluster:

    PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -B \
      scripts/scaling_continue.py check-plan \
      --plan results/infra/scaling_300m_20261104/continuation19618_20261106/training_plan.json

Test (156 nuovi; il percorso a 5.000 resta coperto dai suoi test, invariati):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -B -c "import sys; \
      sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', \
      'tests/cpu/test_scaling_continue.py', 'tests/cpu/test_scaling_continue_launcher.py', \
      'tests/cpu/test_scaling_train.py', 'tests/cpu/test_scaling_train_launcher.py']))"
