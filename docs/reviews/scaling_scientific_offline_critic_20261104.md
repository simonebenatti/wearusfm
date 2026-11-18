# Review scientifica offline del seguito 300M — 04/11/2026

**Perimetro:** recepimento F5/F8 e S8 della review di Claude, controller per il futuro training a 5.000 e contratto delle estrazioni. Nessuna connessione al cluster, esecuzione scientifica, sottomissione o modifica del codice congelato del collaudo durante questa review.

**Verdetto: favorevole alla preparazione offline, non certificazione della readiness operativa a 5.000.** I gate mancanti restano bloccanti per quella fase. Il confronto a 19.618 rimane una proposta distinta; questa review non ne autorizza l'esecuzione.

## Regola scientifica S8

La regola implementata in `scripts/scaling_reading.py` concorda con il protocollo:

- EPN usa la differenza candidato meno controllo in punti percentuali: IC interamente positivo significa migliore, interamente negativo peggiore, altrimenti irrisolto.
- La posa usa `E_dyn(300M) − E_dyn(30M)`, quindi il segno favorevole è negativo. Il 2% è una riduzione relativa del punto stimato accompagnata da IC favorevole della differenza, non un nuovo margine di danno o una soglia da superare con tutto l'IC.
- Il margine di 1 punto riguarda soltanto P1/P2. IC inferiore strettamente sopra −1 sostiene la non inferiorità; IC superiore strettamente sotto −1 sostiene una perdita oltre margine; le uguaglianze e gli attraversamenti restano irrisolti.
- Almeno un primario migliore, nessuno peggiore e nessuna perdita P1/P2 oltre margine motivano una proposta di conferma a 19.618. Due primari irrisolti, senza quelle perdite, motivano la domanda sulla dose. Qualsiasi primario peggiore o perdita P1/P2 oltre margine mette in pausa questa ricetta, inclusi i risultati discordanti. Un gate di integrità, pairing o convergenza fallito precede la lettura scientifica.

La pausa è una regola operativa sull'investimento in questa ricetta, non un verdetto sulla possibilità dello scaling. L'incertezza a 5.000 non diventa equivalenza. La NI irrisolta non viene trattata come danno dimostrato. Nessun aggiustamento, sottrazione o soglia è ricavato dal contrasto storico T25-L/B; i secondari a dimensione pareggiata non consentono di cambiare riga della tabella.

Gli IC sono appaiati per soggetto, 2.000 estrazioni, seed 20261102, con popolazioni/supporti comuni entro ciascun endpoint. Restano marginali, condizionati a un solo seme di pretraining e ai dati di sviluppo: la regola multi-endpoint non possiede una garanzia congiunta del 95%.

Il lettore riceve riepiloghi già auditati: **non calcola gli IC e non prova la provenienza delle OOF**. Questi compiti spettano all'aggregatore futuro, richiesto dal gate di readiness. I tre booleani di qualità non sono da soli una certificazione scientifica.

La formula per i 14.618 passi aggiuntivi conserva il massimo fra tempo completo del trainer 300M a 5.000 diviso 5.000 e media dei 112 update del collaudo, moltiplica per 1,20, aggiunge 1.200 secondi e arrotonda ai 900 superiori, su quattro GPU. Vanno fissati prima dell'apertura degli score i tetti numerici distinti di training, estrazioni e lettori CPU; nessun aumento motivato da score promettenti. Questa contabilità non concede autorizzazione automatica a 19.618.

## Correzioni controllate

| Rilievo | Correzione e stato |
|---|---|
| F5: il test LR stress poteva ripetere soltanto la stessa funzione | Asserzione diretta `got["lr"] == cfg.lr`. Chiuso nei test, nessuna modifica del collaudo congelato. |
| F8: l'oracolo RNG poteva ereditare lo stato del wrapper | Lo stato di riferimento avanza dal proprio aggiornamento; confronto con `rng_after_sha256`, anche nel passaggio 63→64 con lo stesso AdamW e relativi momenti/EMA. Chiuso. |
| E1: preset `sanity` cercato in `LADDER` | Il guard usa esplicitamente `30m`/`300m`; test dei due modelli pubblici prima dell'I/O. Chiuso. |
| E2: il contratto training non copriva gli script estrattori | Pin separato delle cinque primitive, confrontate alle versioni storiche delle cache; controlli prima/dopo. Chiuso. |
| E3: piano estrazioni e contratto training potevano essere validi ma distinti | Binding di comparison, quattro input di training, checkpoint storico 30M e configurazioni complete dei due bracci al contratto verificato; test negativi sulle sostituzioni. Chiuso. |
| T1: sola estrazione non equivale a campagna valutabile | Il gate richiede sorgenti/risorse/test/review per ridge, sonde, audit di convergenza, controllo S5 e aggregatore appaiato, oltre alle estrazioni. Chiuso nel contratto; componenti mancanti restano da implementare. |
| T2: retention del controllo verificata solo fino al train | Il piano e la readiness condividono un orizzonte esplicito fino a estrazioni/confronto; la conservazione deve coprirlo e rispettare il limite dichiarato dello scratch. Chiuso. |

## Identità e integrità

Verifica indipendente locale: tutti i **103 file** di `scientific_source_sha256` corrispondono sia ai byte del checkout sia a `git show afb644ea10ecc2a716160b4d8a4904e805c366fb`. Le cinque primitive `pose_eval.py`, `channel_readout.py`, `calibration_probe.py`, `tokenizer_evaluation.py`, `tokenizer_pose_readout.py` sono identiche sia a `f2061b8bc91a6355213793a9719121135ad48fa2` (estrazione posa storica) sia a `b315dad` (classificazione storica).

Il nuovo wrapper ricostruisce il CLI originale, cambia preset/output/dose dichiarati e aggiunge soltanto il limite di tempo operativo. Mantiene schedule WSD 49.046, warmup 1.962, batch globale 32, quattro rank, sei worker/rank e input congelati. L'identità del codice non afferma identità dei pesi iniziali o del consumo RNG fra forme diverse.

L'endpoint viene letto con carico stretto del teacher, controllo della configurazione completa, finestra/attivazione e ripristino esatto del gain salvato, senza calibrazione sui dati di valutazione. I test piccoli usano checkpoint PyTorch reali, oltre a guard architetturali espliciti D384/D896. Le cache richiedono target/maschere/supporti identici, righe/etichette/soggetti/fold conservati, finitezza dei campioni reali e larghezze posa 384/896 e pool 768/1792. Le etichette testuali EPN non passano per `isfinite`.

I controlli ereditati su AdamW/RNG sono strutturali. **Non certificano da soli una ripresa numericamente completa a 19.618:** in particolare non associano ogni momento a ogni parametro del modello. L'inferenza dell'endpoint è controllata direttamente; una futura implementazione di resume deve avere una verifica dedicata.

## Limiti e lavoro ancora necessario

Il collaudo congelato resta `05d4bf0663481522a56d96c0b07847924cf5bd26`. F1–F4/F6 non sono dichiarati implementati da questa review: niente nuova attestazione di copertura delle forme peggiori, di unicità della residenza da una sola euristica di memoria o di runtime completamente confrontabile. I 128 update misurano fattibilità e diagnostiche della ricetta; non provano LR ottimale, convergenza scientifica o memoria massima del percorso di produzione.

Il training a 5.000 non è pronto al lancio soltanto perché questi script passano i test. Richiede risultati reali del collaudo, verifica del percorso di memoria con prefetch/pannelli/eval, conservazione del controllo, lettori e aggregatore completi (compreso S5), benchmark/costi, pin effettivi, review finale e allocazione corrispondente. Non è stato creato un report di readiness reale con campi sintetici. Le fixture sintetiche restano nei test temporanei.

## Validazione CPU

Regressione indipendente finale, eseguita una volta dopo conferma di stabilità dei file da parte degli autori: **248 test passati in 19,52 secondi**. Comando, nel worktree `wearusfm-scaling-300m`:

```bash
PYTHONPATH=src:tests/cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_scaling_preflight.py tests/cpu/test_tokenizer_preflight.py tests/cpu/test_scaling_train.py tests/cpu/test_tokenizer_train.py tests/cpu/test_scaling_evaluation.py tests/cpu/test_tokenizer_evaluation.py tests/cpu/test_scaling_reading.py
```

I test di F8 usano dropout muscolare 0,4, quindi il confronto RNG esercita consumo casuale effettivo. I test di estrazione impiegano file NPZ e checkpoint reali piccoli; quelli del controller isolano l'I/O esterno, senza una readiness artificiale nel codice operativo. La presente nota non sostituisce i gate GPU o i risultati scientifici.

Identità SHA-256 dei file revisionati e della suite nuova/modificata:

| File | SHA-256 |
|---|---|
| `scripts/scaling_train.py` | `3184f78ded0d3c33fbd9fd5b3c2fa0da71a98ed2139e21522ebd504edbb33194` |
| `scripts/scaling_evaluation.py` | `395849450dc24a804a71133e272a2b76dd502a5466d7ca3a2d79c42bb292323e` |
| `scripts/scaling_reading.py` | `13231e82664cfa6a815a8e87fb6c85380b70838d8746109d8af627b5a4f056d5` |
| `results/infra/scaling_300m_20261104/training_source_contract.json` | `14b044de1df7be3e55c836685342d94ae48f9d072dc6784b33cca197dab47df1` |
| `results/infra/scaling_300m_20261104/scientific_reading_policy.json` | `da58ec3a7965da8db29e97a1089ae1d1d7e652dba4250f24dcb1b8e608602960` |
| `tests/cpu/test_scaling_preflight.py` | `b026d65e7b5a3edebc8d2b18fb91f605ed4562daa4d1d936ccc79ec852ab2d91` |
| `tests/cpu/test_scaling_train.py` | `92a17ec447f0cd34a53a3ab9f5245bbdd07889dbbe0919ba86475ab4606fa149` |
| `tests/cpu/test_scaling_evaluation.py` | `2006f66ac82f92d59852ebe5f579f7acf14a6be1abe5941061fc189e10464f08` |
| `tests/cpu/test_scaling_reading.py` | `60f7978c81b885fdffcd6280c8ad6df12ede45e57eff2da70f9663bf67ce661f` |
