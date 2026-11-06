# Prompt per Claude — review del tokenizer 100 ms e della variante non lineare

Simone chiede una tua review scientifica e tecnica prima di provare il confronto preparato da GPT.
**Questa consegna è solo di revisione:** non modificare il codice, non lanciare job, non fare training,
estrazioni o nuovi esperimenti. Puoi leggere i risultati esistenti e rieseguire test CPU pertinenti.
GPT resta responsabile dell'implementazione e dell'esecuzione dopo il confronto con la tua review.

## Riferimento esatto

Codice e protocollo da revisionare: **`c052792f0a796ba72d2804c81b3e5721a3010d93`**, già pubblicato su
`origin/codex/r3-long-collaudo-20261028`. Base precedente: `7e1c26f2f4e882242fb46a247432f63f78b7b013`.
Il commit successivo che contiene questo prompt non modifica il codice scientifico da esaminare.
Aggiorna i riferimenti di origin e leggi dal tuo worktree, anche con `git show`: non serve fare merge
e non modificare il worktree GPT. Lavora e pubblica la review sul tuo ramo.

Leggi prima:

- `docs/tokenizer_window_nonlinear_protocol_20261102.md`: disegno, criteri proposti, limiti e fasi.
- `docs/piano_scientifico_20261102.md`: risultati già acquisiti e nuova priorità tokenizer.
- `results/infra/frontend_window100_20261102/comparison_configurations.json` e i tre `config_T*.json`:
  configurazioni complete, argomenti CLI e hash verificati.
- `results/infra/frontend_window100_20261102/budget_audit.json` e `verification.json`:
  residuo locale documentato e verifiche eseguite, con limiti espliciti.

Leggi direttamente il codice, poi confronta il tuo giudizio con
`docs/reviews/tokenizer_window_nonlinear_review_20261102.md`: la review del critico è favorevole
alla preparazione locale, ma non certifica utilità scientifica o costo dei nuovi training.

Diff pertinente:

```bash
git diff 7e1c26f c052792 -- scripts/train_jepa.py src/wearusfm/model/fm.py src/wearusfm/model/patch_tokens.py src/wearusfm/training/jepa.py src/wearusfm/training/run.py tests/cpu/test_frontend_window.py tests/cpu/test_tokenizer_activation.py tests/cpu/test_tokenizer_comparison.py
```

Controlla anche i percorsi invariati in `src/wearusfm/model/frontend.py`, mascheramento del segnale e loader,
per verificare le affermazioni sul supporto temporale. Il precedente R5 è documentato in
`results/pilot_obiettivi_20261024/README.md`; B storico in
`results/infra/pilot_posa/config_b_atteso_20261028.json` e nei rapporti B19618.

## Proposta da mettere alla prova

Tre nuovi training completi, a pari dose e dati, con obiettivo B invariato: ancore0,2 + posa1,
JEPA0, RVQ spento; nessuna nuova loss. Passo25ms in tutti:

| Braccio | Finestra nominale / overlap | Trasformazione prima del Perceiver |
|---|---|---|
| T25-L | 25 / 0 ms | Banco apprendibile → gain fisso → proiezione |
| T100-L | 100 / 75 ms | Banco apprendibile → gain fisso → proiezione |
| T100-G | 100 / 75 ms | Banco apprendibile → gain fisso → GELU → proiezione |

Il banco attuale è apprendibile ma lineare nell'EMG a pesi fissati. GELU rende non lineare la mappa
del segnale senza aggiungere parametri rispetto a T100-L. Tra T25-L e T100-L, invece, la regola
esistente di scala delle basi aggiunge5.184 coefficienti: non è una pura variazione della durata
a identico numero di parametri. L'anti-aliasing resta: supporto numerico circa225/300ms,
distinto dai25/100ms nominali;40token/s invariati. A200Hz i5/20campioni sono quelli nominali.

Tutti i bracci usano `masked_loss_radius=2` per le query delle ancore, anche con JEPA spento;
supporto e bersagli della posa rimangono invariati. Serve un nuovo riferimento T25-L perché B storico
aveva raggio0. Default software/checkpoint storici conservati. Non si sostituisce il tokenizer
dei vecchi checkpoint in inferenza e non si parte dai loro pesi.

Prima collaudo tecnico su batch train reali, poi screening a5.000 passi;19.618 è l'endpoint proposto
a pari dose, subordinato alla copertura nel preventivo completo. Lo schedule è fissato dall'inizio;
proseguire significa riprendere i propri checkpoint, non pagare due volte i primi5.000 passi.
Nessun nuovo training di questa proposta è stato lanciato.

## Punti della review

1. **La modifica affronta l'ipotesi giusta?** Valuta se100ms/75overlap e GELU sulle risposte integrate
   siano una prima prova utile di struttura locale dell'EMG prima del Perceiver. Distingui questa
   GELU da rettificare/quadrare il segnale prima dell'integrazione o da un encoder convoluzionale
   temporale: eventuali cancellazioni nell'integrale sono già avvenute prima della GELU; non stiamo
   dichiarando di aver implementato RMS o invarianza alla fase. Il modello
   ha già non linearità più avanti: quale informazione o interazione nuova può rendere accessibile
   questa posizione dell'attivazione? Se preferisci un intervento diverso, indica la modifica minima
   e la decisione che cambierebbe; non proporre automaticamente una ricerca di architetture.

2. **I tre bracci sono un confronto interpretabile?** Verifica necessità del riferimento nuovo,
   supporto comune delle query e conseguenze del raggio2: le maschere brevi possono perdere tutte
   le query. Distingui parità fra bracci da adeguatezza del compito risultante. Indica quali conteggi
   sui batch reali bastano a decidere prima dei training se il supporto delle ancore è utilizzabile.
   Valuta la conservazione delle quote raw, l'aumento delle basi e la scelta dell'obiettivo B.
   Considera R5a150ms come evidenza precedente, senza equipararlo al nuovo confronto né ignorarlo.

3. **Implementazione e garanzie.** Controlla posizione GELU dopo gain/prima proiezione, apprendibilità,
   default bit-identico, calibrazione del gain e variazione di scala introdotta da GELU. Verifica
   masking prima del frontend, limiti del prefiltraggio, supporto nominale contro code anti-alias,
   coerenza fra conteggi e loss effettiva, posa non erosa, compatibilità resume/branch/frozen.
   I test passano anche D8 dopo GELU sullo stesso segnale a diverse fs: non è una prova su tutto
   il corpus o su kernel già addestrati. Segnala difetti con riferimento e controesempio concreto.

4. **Dose e lettura dei risultati.** Valuta5k come screening e19.618 come endpoint proposto,
   identità dello stream e dei pesi iniziali condivisi, rischio di selezionare i checkpoint sulla val.
   Per le sonde frozen sono proposti teacher sia in regressione sia in classificazione; la testa
   student resta diagnostica separata. Valuta questa scelta e il bordo comune6patch, mantenendo
   gli angoli centrali25ms e la soglia dinamica storica. Nessun confronto diretto fra numeri con
   sorgenti/supporti diversi. Posa già supervisionata significa generalizzazione a utenti diversi,
   non trasferimento a una regressione nuova.

5. **Metriche e criteri prima dei risultati.** Controlla ridge/alpha train-only, errore centrato primario,
   MAE/correlazione/ampiezza secondarie, bootstrap appaiato per utente e sonde P1/P2 senza PCA.
   Il2%di miglioramento relativo e il margine1pp in classificazione sono **nuove proposte GPT**,
   non soglie storiche firmate. Valuta se siano utili e realistiche o se correggerle prima dei job.
   La val è già stata osservata, c'è un solo seme di pretraining e gli IC sono esplorativi:
   non trasformare un delta piccolo o un IC inconcludente in una promessa di miglioramento del FM.

6. **Fattibilità e decisione operativa minima.** Verifica il costo di tre nuovi training contro il
   beneficio informativo atteso. Stime preliminari12,7GPUh a5k e49,9GPUh totali a19.618,
   prima dei nuovi timing e delle valutazioni; non sommarle. Residuo locale massimo noto63,2925GPUh
   e circa318,82coreh CPU-only; i core allocati nei jobGPU sono contabilizzati separatamente.
   Il margine residuo non è un saldo CINECA aggiornato. Indica solo ciò che manca per rendere
   eseguibile il collaudo e congelare il piano: dati/supporti, tempi/memoria, valutatori, tetti.
   Distingui questi completamenti previsti da difetti bloccanti nella proposta.

Sono già passati129test distinti; non serve rieseguire l'intera suite. Se utile, gruppo mirato:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/cpu ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_frontend.py tests/cpu/test_frontend_window.py tests/cpu/test_tokenizer_activation.py tests/cpu/test_tokenizer_comparison.py
```

Eseguilo soltanto su una copia che contenga il codice del commit indicato; dichiara SHA, comandi
ed esiti effettivamente controllati. Non presentare test letti nei rapporti come rieseguiti da te.

## Consegna

Restituisci **concordo**, **concordo con modifiche**, oppure **propongo un altro passo**, motivando
se procederesti con questo collaudo e questo confronto. Per ogni rilievo separa: difetto dimostrato,
correzione indispensabile prima del training, completamento operativo già previsto, miglioramento
facoltativo. Non convergere per cortesia e non riaprire prove concluse senza una ragione concreta.

Salva la review in `docs/reviews/claude_review_tokenizer_20261102.md`, aggiorna entrambe le sezioni di
`docs/claude_GPT_shared` sul tuo ramo, committa e pusha; comunica a Simone verdetto e commit.
Non implementare correzioni e non lanciare esperimenti: GPT leggerà la review, risolverà gli eventuali
rilievi e procederà al collaudo nell'ambito dell'autorizzazione già ricevuta, senza richiedere un secondo
consenso per lo stesso test. Eventuali cambi sostanziali di obiettivo o di budget vanno segnalati come tali.
