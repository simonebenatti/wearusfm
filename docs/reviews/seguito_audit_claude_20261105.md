# Seguito all'audit Claude e al TIMEOUT scaling — 05/11/2026

Fonte: segnalazione diretta di Simone, registro Claude `ef0d987`, audit Claude
`docs/coordinamento_allineamento_claude_gpt_20261105.md` acquisito testualmente da quel commit.
Conferma indipendente SLURM del coordinamento alle **19:03 CET**.

## Campagne

| Linea | Riscontro | Conseguenza |
|---|---|---|
|1 scaling|63593508 TIMEOUT, 16:31:36–18:04:40, 5.584 s; 63593262 e 63593593 COMPLETED 0:0|STOP previsto dal piano76a3126: nessun retry o fit sotto il piano fermato. Owner avvisato e gia' impegnato nella raccolta/diagnosi e nella consegna del recupero a Claude.|
|3 LQ|63593091 COMPLETED 0:0 in3.970s;63593128 PENDING Priority|Completamento tecnico dei5.000; nessun giudizio di qualita' senza valutazioni. Continuazione gia' sottomessa, nessun duplicato.|
|2 EPN15|Quattro frozenC0 completi;quattro frozenC1 eottoFT/scratchC0 RUNNING;ottoC1 PENDING|Matrice ancora incompleta, nessun contrasto finale15epoche.|
|2 lungo|Il pacchetto di valutazione e' pubblicato a6f5d909|Implementato/testato e in attesa della review esterna Claude. Non e' ancora lanciato; approvazione di Simone gia' ricevuta per il lancio dopo convergenza.|

La chat1 ha pubblicato raccolta, diagnosi e prompt a Claude a **03cbd18**. Le evidenze aggiungono un secondo blocco: il preventivo
congelato sui13report completi supera gia'136core-ore, senza i due benchmark
mancanti. Questa e' una constatazione dell'owner pubblicata e in attesa di review,
non un nuovo budget approvato da coordinamento. Sono conservati training,14cache
e risultati tecnici completi; il piano di recupero non equivale a una ripartenza.
La diagnosi non dice che il300M rappresenta peggio del30M.

## Rettifica di copertura del precedente audit

L'audit c77d133 controllava separazione delle modifiche fra rami, identita' delle
review copiate, pubblicazione e associazioni nel ledger. Non aveva eseguito il
test storico B19618: il suo esito favorevole non implicava una suite verde.
Il rilievo D-5 di Claude e' accolto; non era registrato esplicitamente nella mia
consegna precedente. Origine c052792, non84408bf (ultimo autore del file).

La riproduzione indipendente conferma14test verdi con runtime congelato24139ab
e il fallimento con runtime corrente. Il test importava train_jepa.py dal ramo
attuale sebbene il launcher imponga e importi24139ab. I due campi aggiunti dal
tokenizer, model.tokenizer_activation=identity e jepa.masked_loss_radius=None,
sono assenti nello schema della configurazione B archiviata. La guardia rigorosa
rifiuta correttamente schemi diversi; non deve essere rilassata per far passare
il test. La correzione riguarda l'isolamento del runtime nel test.

## Punti di integrazione ancora aperti

- **D-1/D-2:** integrazione dedicata su main, un ramo alla volta, senza squash/rebase;
  unione testuale delle voci nelle due sezioni e delle decisioni, comprese le60righe
  Claude del29–30/10. Copiare singole review non sostituisce questa integrazione.
- **D-3:** chiuso dalla conferma di Simone riportata da Claude; main7616f33 resta
  la versione canonica delle autorizzazioni. I checkout attivi non si modificano.
- **D-5:** correzione e test separati in questa consegna; non ancora una dichiarazione
  che tutti i rami o la futura integrazione abbiano superato l'intera suite.
- **D-6:** mona_*,t1_*,t2_* usano l'account condiviso e consumano la sua quota;
  sono registrati nell'audit come campagna MONALISA, fuori dal ledger WearUsFM.
  Nessuna loro cancellazione o attribuzione automatica al budget scientifico
  WearUsFM. Il conto complessivo dell'account deve tenerne conto separatamente;
  l'eventuale ripartizione fra progetti rimane una decisione di Simone.
- **D-7:** approvazione del codice/launcher e rilettura di ricevute o piani operativi
  successivi sono coperture distinte. Non attribuire a Claude review interne GPT.
- **D-8:** la linea1 non include8ee6603; le differenze del piano scientifico non
  sono modifiche concorrenti. Da conservare esplicitamente nell'integrazione.

Nessun job, deploy, retry o modifica ai worktree delle tre linee da coordinamento.

## Correzione D-5 pubblicabile prima dell'integrazione

Commit separato **d2f2f08**, solo `tests/cpu/test_pose_b19618_launcher.py`.
Il test della ricetta archivia dal Git locale il solo runtime24139ab (src e CLI)
e lo importa in un subprocess isolato; attesta che il modulo provenga da quel
percorso. Restano i confronti esatti di configurazione, warmup, decay e LR.
Una regressione separata verifica che il runtime corrente accetti i default
neutri per il resume, che il guard storico rifiuti il nuovo schema grezzo e che
GELU o raggio esplicito diverso restino incompatibili con il checkpoint storico.
Nessuna normalizzazione nel launcher e nessuna modifica al codice di training.

Riproduzione indipendente: test originale con runtime corrente1fallito/13passati;
stesso test/launcher con runtime congelato14passati. Dopo la correzione, verifica
locale del coordinamento: **15passati in3,40s**, con PYTHONPATH=src, OMP/OPENBLAS/MKL=1,
PYTHONDONTWRITEBYTECODE=1 e pytest senza cache. La correzione e' sul ramo di
coordinamento; non e' ancora una correzione acquisita su ogni ramo o un merge su main.
La storia Git locale deve contenere il commit congelato per questo test storico.

Verifica aggiuntiva indipendente su copie scratch, con il solo test corretto e
sorgenti/CLI/config di ciascuna punta: consolidamento6f5d909 **15passati/5,08s**;
scaling03cbd18 **15passati/4,95s**; logcontrastcc81430 **15passati/4,95s**.
Nessuna modifica ai tre worktree; adozione del commit affidata ai rispettivi owner.
