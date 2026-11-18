# Prompt per Claude — review del collaudo 30M/300M, 04/11/2026

Ti chiedo una review scientifica e algoritmica della prima consegna della linea
scaling approvata da Simone. **Solo review: nessun esperimento, sottomissione,
allocazione o modifica al codice.** Riporta rilievi riproducibili e correzioni
proposte; GPT le implementerà. Non considerare un problema il fatto che i
valutatori della futura prova a 5.000 passi non siano ancora pronti: sono un
gate successivo, esplicitamente dichiarato.

Riferimento del codice: **`05d4bf0663481522a56d96c0b07847924cf5bd26`**, sul ramo
`codex/scaling-300m-20261104` di origin. Il commit successivo aggiunge soltanto
questo prompt. Worktree GPT separato:
`/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-scaling-300m`.

Base precedente: `f35d5bfa2c25738fba4da8260dee32acc69a059f`. Il diff della consegna:

```bash
git diff f35d5bfa2c25738fba4da8260dee32acc69a059f 05d4bf0663481522a56d96c0b07847924cf5bd26 -- scripts/scaling_preflight.py scripts/slurm/scaling_preflight.sbatch tests/cpu/test_scaling_preflight.py docs/scaling_300m_protocol_20261104.md results/infra/scaling_300m_20261104
```

## Perché lo facciamo

Simone vuole confrontare il 300M con il 30M usando la stessa architettura del
tokenizer oggi più promettente: **100 ms + GELU**, passo 25 ms, 64 slot. La prova
scientifica sarà un 300M nuovo a 5.000 passi contro il 30M già disponibile alla
stessa dose. Un risultato negativo così breve non basterà a escludere lo scaling;
19.618 passi sono un seguito possibile, non una continuazione automatica.

Questa consegna prepara soltanto il **collaudo tecnico**, necessario per misurare
memoria, throughput, stabilità e comportamento del clipping prima del preventivo
del training scientifico. Il preset 300M è già nel codice: D896, 14 teste,
3 blocchi locali, 18 del backbone e 2 del decoder, circa 292 milioni di parametri.
Non si carica un checkpoint 30M in forme diverse.

## File da leggere

- `docs/scaling_300m_protocol_20261104.md`: domanda, recipe, limiti, gate e seguito.
- `scripts/scaling_preflight.py`: nuovo controller del collaudo.
- `scripts/slurm/scaling_preflight.sbatch`: launcher congelato, quattro A100 su un nodo.
- `tests/cpu/test_scaling_preflight.py`: oracolo degli update, transizione di fase e guardie.
- `results/infra/scaling_300m_20261104/preflight_plan.json` e `resource_policy.json`.
- `docs/reviews/scaling_300m_preflight_critic_20261104.md`: review interna e tre rilievi già corretti.
- Le funzioni richiamate di `scripts/tokenizer_preflight.py` e
  `src/wearusfm/training/run.py`, più `scripts/train_jepa.py` per ricostruire la recipe.

## Disegno da verificare

I modelli 30M e 300M sono residenti **uno alla volta**, ciascuno sul medesimo prefisso
di 128 batch globali da 32, seed nominale 0, quattro rank e sei worker per rank.
Per ogni taglia si fanno 64 update con warmup nativo, poi 64 sullo stesso modello,
teacher e AdamW portando il LR al picco 1e-4. Il clipping resta 1. Il secondo tratto
è uno stress anticipato: non simula un modello allenato fino al warmup e non viene
contato come training scientifico. Nessun checkpoint è salvato o riutilizzato.

Restano uguali ancore 0,2 + posa 1, JEPA 0, maschere, preprocessing, target, batch,
gain iniziale del tokenizer e configurazione WSD totale 49.046 / warmup 1.962.
Si modificano solo i valori del preset LADDER. La soglia assoluta di rango 38,4
rimane quella approvata; non si moltiplica per la larghezza.

La parità richiesta è su batch effettivamente consegnati, segnali processati,
origini, maschere, target e supporti effettivi. Pesi di forme diverse e consumo RNG
del dropout non possono essere identici e non lo si pretende. Il banco del
tokenizer e il gain iniziali devono invece coincidere.

## Domande precise della review

1. La sequenza nativa/stress richiama correttamente l'update di produzione e
   mantiene gli stati di AdamW, teacher e RNG? L'oracolo a due passi consecutivi
   verifica davvero questa continuità, oltre alle singole fasi?
2. Il pairing dei dati e la distinzione dal RNG del modello sono corretti? Le
   guardie impediscono che un report troncato sia dichiarato completo?
3. La memoria è misurata per un modello residente e i tempi sono interpretati
   correttamente? Individua eventuali riferimenti che impediscano di liberare il
   primo modello o lavoro diagnostico erroneamente incluso/escluso nei tempi.
4. Gradiente globale, fattore di clip, update relativi e monitor QKV/MLP dei blocchi
   iniziale/finale sono informativi senza far passare lo stress per una prova di
   ottimalità del learning rate? Serve un controllo tecnico indispensabile ora?
5. Sono coerenti gate, deadline, provenance, output nuovi, pin, launcher e risorse?
   Un fallimento deve fermare la fase senza retry cieco o modifiche silenziose.
6. Il confronto scientifico successivo è ben delimitato: 300M5000 contro 30M5000,
   teacher, stessi supporti/fold e validatori espliciti D896/pool1792? Il protocollo
   riconosce correttamente che la maggiore dimensione della sonda fa parte del
   confronto e che i dati sono di sviluppo con un solo seme?

La review interna ha corretto: monitor iniziale limitato a una LayerNorm, controllo
incompleto delle liste di record e tempo T100-G5000 del preventivo. Ora si osservano
QKV/MLP primo/ultimo blocco, si impongono dose/step/fasi esatti e si parte dai
3.883,029760837555 secondi del summary del trainer. Verifica le correzioni senza
assumere sufficiente il verdetto del critico.

## Costi e stato

**Nessun job avviato.** Il collaudo ha tetto **2 GPU-ore** (4 GPU × 30 minuti) e
16 core-ore allocate; checkout separato, 0,333 core-ore. Deadline interna 25 minuti.
Il preventivo successivo è solo un inviluppo da restringere dopo benchmark:
24 GPU-ore training5000, 8,25 GPU-ore estrazioni e 180 core-ore CPU per lettori e
controlli. Il vecchio rapporto di velocità 300M/30M serve solo a stimare il costo
iniziale; non viene presentato come misura attuale.

Le guardie della vecchia campagna sono fissate a D384/pool768, tre bracci e dose
19.618 per la classificazione. Non saranno aggirate: il training scientifico non
parte finché la nuova lettura a D896/pool1792 non è pronta, revisionata e preventivata.
La posa 30M5000 esiste; la classificazione del controllo richiede nuove estrazioni,
senza riaddestrare il 30M. Nessun nuovo monitor o ampliamento alle loss.

Verifiche eseguite: **23 test passati in 12,99 secondi**, sintassi Bash e diff puliti.
Per rieseguire dal checkout del commit:

```bash
env PYTHONPATH=src:tests/cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_scaling_preflight.py tests/cpu/test_tokenizer_preflight.py
bash -n scripts/slurm/scaling_preflight.sbatch
```

Restituisci **APPROVATO / DA CORREGGERE**, distinguendo blocchi del collaudo,
miglioramenti facoltativi e requisiti del seguito scientifico. Per ogni blocco
indica file/riga, problema concreto, impatto e correzione minima. Registra la review
nel tuo worktree e nel registro condiviso, commit/push come concordato. Non lanciare
nulla: root GPT mantiene l'operatore unico e riprenderà dal tuo verdetto.
