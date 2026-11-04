# Screening della taglia 300M con T100-G — 04/11/2026

Simone ha approvato la linea parallela: prima collaudo tecnico, poi300M a5.000
passi contro30M alla stessa dose. L'estensione a19.618 è una possibile seconda
tranche, non un lancio già disposto. L'operatore dei job resta GPT root tramite
submitter e ledger comuni; l'agente scaling prepara in un worktree separato.
Nessuna automazione, multinodo, nuova loss o innesto del checkpoint30M nel300M.

Stato: codice del collaudo implementato e23test CPU superati; review indipendente
favorevole alla consegna aClaude. Simone ha chiesto la review diClaude prima dei lanci. **Non è ancora un via al training scientifico:** occorrono risultato del
collaudo, preventivo misurato e valutatori che trattino esplicitamente la larghezza
del300M. Nessun job è stato sottomesso da questa consegna.

## Domanda e confronto

Quanto cambia l'informazione accessibile con più parametri, nella ricetta T100-G
attuale e alla stessa esposizione? Il confronto è condizionato a dati, dose,
obiettivo, inizializzazione convenzionale e ottimizzazione fissati. Non stima una
legge universale di scaling, né quale sia il miglior modello possibile a300M.

| Proprietà | Controllo30M già disponibile | Candidato300M da zero |
|---|---:|---:|
| Parametri student+decoder+teste, senza EMA |28.362.013|292.054.847|
| Dimensione, teste, dimensione per testa |384,6,64|896,14,64|
| Blocchi locale/backbone/decoder |2/8/2|3/18/2|
| Tokenizer / passo / slot |100ms+GELU /25ms /64|identici|
| Loss |posa1 +ancore0,2; JEPA0|identica|
| AdamW / picco LR / clip globale |WD0,05 /1e-4 /1|identici, salvo stop tecnico|
| Schedule |WSD49.046, warmup1.962, decay finale20%|identico|
| Batch globale / rank / worker per rank |32 /4 /6|identici|
| Dataset, split, preprocessing, maschere, target |pin della campagna tokenizer|identici|
| Prima dose scientifica |5.000 aggiornamenti|5.000 aggiornamenti|

Solo i cinque valori del preset LADDER cambiano. L'allarme di rango conserva la
soglia assoluta38,4 approvata per tutte le taglie; non viene moltiplicato perD.
Il seed nominale0 è comune, ma forme/depth diversi producono pesi e dropout diversi:
non si promette parità dei pesi o del consumo RNG del modello. Il banco del tokenizer
e il gain iniziali, costruiti prima della proiezione dipendente daD, sono verificati
identici. Questo mantiene il punto di lavoro della GELU già usato, senza ricalibrarlo
in risposta ai risultati downstream.

## Collaudo tecnico train-only: nessun checkpoint

Script `scripts/scaling_preflight.py`, launcher `scripts/slurm/scaling_preflight.sbatch`,
piano `results/infra/scaling_300m_20261104/preflight_plan.json`. Usa le funzioni di
aggiornamento già verificate di tokenizer_preflight e del trainer di produzione.

I due modelli sono eseguiti **uno alla volta**, ciascuno da zero sullo stesso prefisso
di128batch globali da32. Non si sommano le loro memorie GPU. Per ogni taglia:

1.64aggiornamenti con lo schedule nativo: LR da1e-4/1962 a64×1e-4/1962.
2.64aggiornamenti consecutivi sullo stesso modello/AdamW, forzando solo il LR al
  picco1e-4 (warmup temporaneamente0). Lo stress anticipa il picco a pesi ancora
  giovani; **non simula un checkpoint a1.962passi** e non è esposizione scientifica.
3.Si eliminano tutti i pesi/stati; il run scientifico ripartirà da zero con warmup
  intero. Nessun checkpoint o score downstream viene scritto.

Il controllo30M viene rieseguito solo per misurare il rapporto dei tempi nelle
medesime condizioni hardware. Non è una replica scientifica del pretraining.
Ogni rank conserva hash dei batch, segnali processati, origine, maschere e target;
si verifica parità fra taglie, inclusi supporti reali del decoder e della posa.
Manifest esclusivamente pretraining; EPN/UCI esplicitamente rifiutati dal guard.
Pin dei file e stat delle sorgenti verificati prima/dopo. Output nuovo e fuori dal
codice, nessuna sovrascrittura o ripresa implicita.

Si misurano tempi forward/backward e clip/AdamW/EMA con massimi fra rank; esclusi
i primi8passi di ciascuna fase per i riassunti. Si riportano memoria allocata e
riservata, norma del gradiente prima del clip, fattore/frequenza di clipping,
norme di aggiornamento assolute e relative per frontend/proiezione/posa e matrici
QKV temporali/MLP del primo e dell'ultimo blocco del backbone, con nomi espliciti
nei report. Le norme relative omettono una divisione per zero: la
testa iniziale a zero non è un'anomalia. Sono statistiche diagnostiche, non un
criterio di uguaglianza fra architetture di diversa larghezza/profondità.

### Gate e lettura del learning rate

Gate automatici:128batch per taglia e4rank completi, parità degli ingressi e supporti,
pesi/gradienti/aggiornamenti finiti, gradienti e aggiornamenti non nulli nei gruppi
monitorati lungo il collaudo, memoria massima allocata≤90% della memoria GPU,
pin/sorgenti immutati. Scadenza interna25min su allocazione30min. Un fallimento
ferma quella fase; niente retry cieco, riduzione batch nascosta o scelta delLR suval.

**Il collaudo non dimostra che1e-4/clip1 siano ottimali.** Si esaminano anche loss,
clipping e dimensione relativa degli update: un clipping frequente da solo non
dimostra instabilità con AdamW, e non si impone che il300M abbia le stesse norme
del30M. Se è tecnicamente stabile, la prima prova mantiene questi iperparametri,
così misura la taglia nella stessa ricetta. Se le diagnostiche mostrano un problema
concreto, si scrive una calibrazione train-only separata prima di consumare5.000
passi. Non si sceglie un LR a posteriori dalla migliore accuratezza EPN o posa val.

Memoria: il valore include le piccole copie diagnostiche dei gruppi selezionati,
quindi è conservativo, ma128batch non esauriscono tutte le forme del corpus.
Timing: non include checkpoint/pannelli, e le diagnostiche danno tempo al loader;
la stima del training include overhead storico e margine, non solo kernel timing.

## Training scientifico5.000 e valutazioni necessarie

Il wrapper futuro ricostruirà il CLI T100-G cambiando solo preset300m e cartella
nuova; pin del codice scientifico, configurazione, input, schedule e stream.
Una ricevuta completa richiederà `summary.stopped=max_steps`, step5000, assenzaSTOP,
stato student/teacher/AdamW/RNG integro e checkpoint nominato immutabile, come nel
wrapper tokenizer. Nessuna selezione del checkpoint sullo sviluppo: si legge5000.

Per ogni taglia si usa teacherEMA, stessa popolazione/supporto di posa e stessi
split e fold P1/P2. La posa30M a5000 ha già cache e ridge; la classificazione30M a5000
richiede **nuove estrazioni**, non un nuovo training. Non si confronta300M5000 con
sonde30M19618. Gli score19618 esistenti non scelgono iperparametri della nuova prova.

| Lettura | Input30M /300M | Endpoint e controlli |
|---|---|---|
| Posa ridge istantanea |sequenza patch×384 /patch×896, media dei64slot|crop6, LNstatelessFP32, ridgeFP64, alpha solo40utenti train, stessi15utenti val; primario E_dyn, correlazione e varianza predetta come diagnostiche|
| Classificazione |pool768 /pool1792 FP32|stessa aggregazione backbone+locale, StandardScaler solo foldtrain, stesso C-grid, max_iter12.000 e audit convergenza; primario EPN-612, P1/P2/UCI secondari|
| Head posa/fixed panels |testa originale di ogni modello|train-only per dose/stabilità; non sostituiscono generalizzazione della ridge|

La diversa dimensione del readout fa parte della prova del modello più grande:
non isola «solo più profondità a uguale capacità della sonda». Nessuna PCA aggiunta
silenziosamente. Alpha eC si selezionano internamente e allo stesso modo; i benchmark
CPU precedono i fit, senza lettura degli score di sviluppo per fissare le risorse.

**Vincolo implementativo esplicito:** `tokenizer_evaluation.py`,
`tokenizer_pose_readout.py` e `tokenizer_probe.py` oggi verificano D384/pool768,
trebracci e, per la classificazione, solo19618. Le funzioni di estrazione sottostanti
usano la dimensione del checkpoint, ma le guardie di campagna non possono essere
bypassate. Si prepareranno controller/validatori della nuova campagna che registrino
D896/pool1792 e dose5000, con regressioni contro gli stessi calcoli aD384.
Il training scientifico non viene lanciato finché queste letture e i loro costi non
sono concretamente pronti e revisionati. Questo documento non afferma che lo siano.

La matrice pre/postbackbone e39.237/49.046 approvata per lo schedule lungo30M non
viene aggiunta automaticamente a questo screening. Qui resta la lettura storica.

## Risorse: collaudo fissato, seguito da misurare

Politica versionata: `results/infra/scaling_300m_20261104/resource_policy.json`.
Le cifre provvisorie tengono separati training, estrazioni e CPU. Non sono prenotazioni
né autorizzano a inviare fasi non ancora pronte.

| Fase | Risorse massime | Costo |
|---|---|---:|
| Checkout congelato |2CPU,4G,10min|0,333coreh|
| Collaudo30/300M sequenziale |4A100,32CPU,480G,30min|2GPUh +16coreh allocati|
|300M0→5000, stima provvisoria |4A100,32CPU,480G, fino6h|circa17GPUh attese da rapporto storico; tetto24GPUh da aggiornare sul benchmark|
| Estrazioni300M posa/P1/P2 +30M P1/P2 a5000 |1GPU/8CPU perjob; nessun train30M|inviluppo8,25GPUh, da verificare dopo benchmark|
| Ridge/sonde CPU e controlli |fino8CPU/30G; shard fino120min|inviluppo180coreh CPU-only, da restringere prima dei fit|

Il rapporto storico15/10 2,769/0,707≈3,92 è solo una stima iniziale: ricetta precedente.
Il T100-G30M5000 corrente registra3883,030s nel trainer; non confondere con una
misura SLURM completa. Dopo il collaudo si usa il rapporto misurato e il massimo
fra tempo perupdate300M e tempo completo30M×rapporto, +20% e1200s di margine,
arrotondato ai15min superiori. Il nuovo preventivo è pubblicato con commit/push
prima della sottomissione. CPU nei jobGPU contabilizzata separatamente.

L'estrazione300M e la ridge possono costare più del rapporto dei parametri: per la
ridge D896/D384 implica circa12,7× nel termine cubico, mentre il classificatore
riceve2,33× feature. I benchmark reali determinano gli shard; niente invii CPU
basati soltanto sull'etichetta «inferenza» o sull'assenza diGPU.

## Cosa autorizza a concludere e seguito19.618

Contrasto appaiato300M−30M a5000 su OOF o utenti posa comuni, bootstrap2000 con seed
20261102 come nella campagna precedente. IC condizionati a questo seme, checkpoint
e sonde. Soglia pratica posa2% relativo e margineNI1pp P1/P2 sono riferimenti
descrittivi mantenuti, non prove definitive di superiorità di una famiglia.
Un solo seme e dati di sviluppo; EPN positivo non isola causalmente la frequenza
di campionamento, posa su utenti nuovi resta lo stesso task supervisionato.

Un5000 negativo non boccia lo scaling: dose ancora iniziale, ottimizzazione non
dimostrata ottimale, rappresentazioni sondate con lettori specifici. Dopo i risultati
si propone il confronto a19.618 con il30M già disponibile. La decisione considera
stabilità, andamento train, misure downstream e costo; non richiede necessariamente
un IC positivo a5000. Il seguito conserva la stessa run e schedule, con altri14.618
passi; stima iniziale circa49,40GPUh aggiuntive, da sostituire con timing effettivo.
Il modello300M non viene proseguito implicitamente durante la lettura di5000.
