# Diagnostica dei rami PSD e wavelet dopo il 2C (protocollo `rami_diagnostica_v1`)

Proposta di Claude del 20/10/2026, punto 1 della consegna sui rami spettrali (Simone: «Procedi col punto 1», sul prompt preparato da GPT il
20/10, registro: «Prompt operativo per Claude: seguito dei rami spettrali», base 3b97655). Rientra nel punto 5 della tabella comune; i
verdetti storici del 2A e del 2C restano invariati (punto 1 storico chiuso). **Nessun job lanciato.** Prima del lancio: review del `critico`
(parte scientifica nuova), lettura congelata in `docs/decisioni.md`, preventivo e via di Simone. La review di GPT arriva dopo e non blocca.

## Domanda

Nel 2C (raw congelato, rami allenati 3.000 passi solo su emg2qwerty) i rami al passo 0 aiutavano di piu' di quelli allenati (descrittivo,
test ufficiali: famiglia B, P2, R+PW0 contro R +2,34 punti, R+PW contro R+PW0 -1,59; famiglia A, P1, R+PW contro R+PW0 -0,84).
**Il training peggiora i rami anche da soli, oppure il peggioramento emerge soprattutto quando si concatenano al raw?** La risposta dice
*dove* si vede il peggioramento, non *perche'* avviene: obiettivo troppo ristretto, ottimizzazione (LR, schedule), dominio di training e
geometria dell'embedding restano spiegazioni possibili per qualunque esito.

## Dati: solo cache gia' estratte, nessuna nuova estrazione

Verificato sul cluster in sola lettura il 20/10 (`~/wearusfm_local/reports/rami_diagnostica/cache_20261020.txt`):

| Cache | Cartella | Contenuto |
|---|---|---|
| Rami al passo 0 e allenati (2C) | `$WORK/wearusfm_runs/results/passo6/step2c_1410/<task>/<teacher>/branches_{init,trained}` | 2 x 128 colonne (PSD, wavelet), provenienza `step2c_frozen_branches_v1`, commit 1c94852 |
| Embedding raw (tranche downstream) | `$WORK/wearusfm_runs/results/passo6/step1_downstream_0910/<task>/<teacher>` | 768 colonne, `step1_downstream_4000_v1` |

Teacher dell'A/B dello Step 1: famiglia A (`A_seed0`, `A_seed1`) e famiglia B (`B_seed0`, `B_seed1`), gli stessi del 2C. Stessi ingressi,
finestre, soggetti e split del 2C: le cache dei rami sono gia' accoppiate a quelle raw (stesso hash degli ingressi e stesse righe, controllato
dal 2C e ricontrollato qui). Ogni file si confronta col suo SHA-256 registrato nella provenienza, e le righe lette con l'impronta di etichette e
soggetti registrata; identita' della provenienza (protocollo, commit, teacher, checkpoint, split) confrontata coi valori attesi.

**Copertura della validation interna: completa, nessuna estrazione necessaria.**
- **P1:** le cache hanno i ruoli `train_`, `val_` (soggetti di pretraining degli split firmati, 32 / 8 / 8 per DB2 / DB3 / DB6, gli stessi
  della S della calibrazione) e `test_`. La diagnostica legge solo `train_` e `val_`: **le righe di test non si aprono mai** (nei test CPU sono
  NaN e la sonda passa). Si controlla che l'unione di `train_` e `val_` sia esattamente il pretraining degli split e non tocchi il test.
- **P2:** le cache hanno tutti i soggetti di EPN-612 e UCI-EMG. Si tengono solo quelli fuori dai test di **tutti e tre** gli split ufficiali
  7:1:2 (semi 0, 1, 2; `calibration_probe.p2_internal_subjects`, la S della P2 dei bracci firmata il 16/10, stessi id e stesso ordinamento
  degli split di `probe_p2.py`): 314 soggetti di EPN-612 su 612 e 19 di UCI-EMG su 36 (verificato il 20/10 sulla provenienza della tranche
  downstream, `step1_downstream_0910/p2/B_seed0/provenance.json`: 91.800 e 662 finestre in tutto); le altre righe si scartano subito dopo la
  lettura, prima di qualunque calcolo. UCI-EMG interno e' piccolo (19 soggetti, circa la meta' delle 662 finestre) e pesa 0,5 nella S della
  P2: la sua parte e' la piu' rumorosa, da leggere coi suoi intervalli.

**Che cos'e' il passo 0** (precisazione di GPT, registro del 20/10): frontend fisici (PSD, banco di wavelet) e normalizzazione calibrata,
piu' gli encoder dei rami non allenati. Non e' la baseline dell'encoder casuale del punto 2 (quella riguarda il raw). Le cache dei rami al
passo 0 e allenati devono venire dallo stesso run del 2C (stessi SHA dei checkpoint finale e iniziale nelle due provenienze), completo e non
di collaudo: controllato.

## Insiemi di colonne

| Insieme | Colonne | Ruolo |
|---|---|---|
| PW0 | rami al passo 0 (256) | soli rami, non allenati |
| PW | rami allenati (256) | soli rami, allenati |
| R | raw (768) | controllo |
| RPW0 | raw + rami al passo 0 (1.024) | concatenazione, non allenati |
| RPW | raw + rami allenati (1.024) | concatenazione, allenati |

PSD e wavelet separati (P0, P, W0, W) non entrano in questa tranche: si aggiungono solo se l'esito lo chiede (128 colonne, costo minore).

## Metrica: la validation interna della calibrazione, invariata

`scripts/calibration_probe.py` (protocollo della S firmato per C1-C4 e per i bracci): 4 fold di soggetti col seme 20261015; per fold la sonda
lineare (`harness.probe.linear_probe`: logistica bilanciata) si addestra sui soggetti di train del fold, **normalizzazione sul solo train del
fold, C scelto sui soggetti di validazione del fold** (10% dei rimanenti), valutazione sui soggetti tenuti fuori. Ogni soggetto interno e'
predetto una volta. S per dataset = accuratezza bilanciata fuori fold; S della P1 = media coi pesi della P1 (8/13, 3/13, 2/13, dai conteggi
dei soggetti di test), S della P2 = media dei due dataset. Nessun numero dei test ufficiali entra in nulla.

## Confronti e lettura (congelati in `docs/decisioni.md` prima delle sonde)

Confronto accoppiato per famiglia (`arms_compare.s_comparison`: soggetti ricampionati insieme per i quattro modelli, 1.000 estrazioni, seme
20261009; i due teacher della famiglia come i due semi dei bracci), delta = insieme di arrivo - insieme di partenza, per compito:

| Confronto | Partenza -> arrivo | Domanda |
|---|---|---|
| `rami_da_soli` | PW0 -> PW | il training peggiora i rami da soli? |
| `concatenazione` | RPW0 -> RPW | il training peggiora la concatenazione? |
| `apporto_passo0` | R -> RPW0 | quanto aggiungono al raw i rami al passo 0 (descrittivo) |
| `apporto_allenati` | R -> RPW | quanto aggiungono al raw i rami allenati (descrittivo) |

Lettura di ogni confronto (la regola dei bracci e la sua speculare, nessuna soglia nuova): **«migliora»** se delta >= +0,01, limite inferiore
dell'IC al 95% > 0 e delta > 0 con tutti e due i teacher; **«peggiora»** se delta <= -0,01, limite superiore < 0 e delta < 0 con tutti e
due; altrimenti **«entro le soglie»**, cioe' **regola non superata**: per la stima sotto 1 punto, per l'intervallo che tocca lo zero o per
i teacher discordi. **Non esclude una differenza, neanche oltre 1 punto** (seconda review del `critico`): per esempio -1,5 punti con IC
[-3,0, +0,1] e' «entro le soglie». Esempio reale: nel 2C, famiglia A, P1 ufficiale, R+PW contro R+PW0 = -0,84 punti (IC -1,38..-0,39,
negativo con i due teacher) sarebbe «entro le soglie»: l'effetto che motiva questa domanda puo' stare sotto la soglia. Delta, IC e delta per
teacher si riportano sempre accanto alla lettura.

**Risposta per famiglia e compito (review del `critico` del 20/10):** solo quale dei due confronti chiave ha superato la regola, nella forma
«soli rami (PW0 -> PW): <lettura>; concatenazione (RPW0 -> RPW): <lettura>», con «peggioramento oltre le soglie», «miglioramento oltre le
soglie» o «regola non superata (stima, intervallo o accordo dei teacher insufficienti): non esclude una differenza, neanche oltre 1 punto». Nessun meccanismo (perdita d'informazione, compensazione del raw) e
nessun «soprattutto»: una differenza di esito rispetto a una soglia non e' una differenza stimata.

**Il confronto fra i due delta** (la parte comparativa della domanda) e' un descrittivo **senza soglia**: la differenza dei delta,
(RPW - RPW0) - (PW - PW0), media dei due teacher, con IC al 95% dallo stesso ricampionamento accoppiato dei soggetti per gli otto modelli
(righe e soggetti uguali per tutti gli insiemi). Negativa: la concatenazione perde piu' dei soli rami; positiva: meno. Si riporta col suo
intervallo, per dataset e per teacher.

Si legge per famiglia (A e B separate) e per compito, poi per dataset (descrittivo). Le letture orientano il punto 2 (obiettivo temporale)
e il punto 3 (informazione spaziale); non adottano e non escludono nulla. Due teacher per famiglia: incertezza condizionata a quei modelli.
**16 letture** (4 confronti x 2 famiglie x 2 compiti) **senza correzione per confronti multipli**, come nel 2C: i due confronti chiave
rispondono alla domanda, gli altri interpretano.

**Descrittivi (nessuna soglia):** S per dataset e per teacher; rango effettivo di ogni blocco (raw, rami al passo 0, allenati) sulle righe
interne; ridondanza col raw: CKA lineare fra raw e rami (passo 0, allenati) e fra rami al passo 0 e allenati, e R^2 fuori fold dei rami
predetti dal raw (ridge sulle feature standardizzate nel train del fold, alpha in {1, 100, 10.000} scelto sui soggetti di validazione del
fold, R^2 sui soggetti tenuti fuori). Un R^2 che sale col training dice che i rami allenati sono piu' leggibili dal raw (piu' ridondanti).

## Codice e test

`scripts/spectral_diagnostic.py` (fasi `probe` e `compare`), launcher `scripts/slurm/spectral_diagnostic.sbatch` (`lib_code_dir.sh`, radici
delle cache fisse, rapporti assoluti e fuori dal codice e dalle cache). Rapporti nel formato della calibrazione (S, SE, predizioni fuori fold
con le righe della cache): li legge `calibration_select.load_oof`, e `check_pairing` controlla l'accoppiamento prima di ogni confronto.
Test CPU in `tests/cpu/test_spectral_diagnostic.py` (righe di test della P1 mai lette, soggetti di test della P2 scartati, file cambiati,
provenienza di un altro teacher o di run diversi, righe permutate, rapporti non accoppiati, tutte le combinazioni di letture in `compare`,
differenza dei delta su un caso noto, R^2 del ridge su casi noti, CKA, piano della catena coi limiti) e in `tests/cpu/test_probe_launchers.py`
(radici, uscite e split non sovrascrivibili dagli argomenti; esito del confronto sempre in `<cartella>/esito.json`).

## Esecuzione e costo (CPU seriale, 0 GPU-ora)

Misure di riferimento: le sonde della S della P1 sullo stesso insieme di finestre (pretraining di DB2/DB3/DB6, ~82 mila) a 768 colonne hanno
impiegato 1 h 09 - 1 h 46 a 8 core (valutazione di C1, 19/10). Le altre durate sono **stime** (larghezza delle feature; per la P2 interna
nessuna misura diretta, stima dalle sonde ufficiali della P2 del 2C, 48-53 min a 1.024 colonne su tutti i dati).

| Job per teacher | Insiemi | Stima | Tempo massimo |
|---|---|---|---|
| P1-a | R, PW0, PW + ridondanza | ~2 h 10 | 4 h |
| P1-b | RPW0 | ~1 h 45 | 3 h 30 |
| P1-c | RPW | ~1 h 45 | 3 h 30 |
| P2 | R, PW0, PW, RPW0, RPW + ridondanza | ~1 h 50 | 4 h |

Quattro teacher: 16 job da 8 core su `lrd_all_serial` (30 GB), piu' il confronto (2 core, 30 min) e il checkout di campagna (1 core, 10
min). **Stima ~241 core-ora, tetto 481 core-ora (= 481 ore locali), 0 GPU-ora.** Nessuna misura diretta a 256 colonne ne' per la P2
interna: stime, coperte dal tempo massimo per job. **Budget:** ore locali CPU dell'account (saldo del 15/10: 1.257.249 usate su 2.000.000,
residuo ~742.700); nessun budget GPU toccato (finestra 1, calibrazione, passo 6, infrastruttura invariati). Non c'e' un budget CPU firmato
per il punto 5: il tetto di 481 ore locali e' quello da autorizzare.

**Limiti della seriale e convivenza con la finestra 1** (review del `critico` del 20/10): sulla seriale valgono al piu' 10 job per utente in
coda (`QOSMaxSubmitJobPerUserLimit`, rifiuto reale il 14/10) e 8 CPU per utente in esecuzione (`QOSMaxCpuPerUserLimit`): un job da 8 core
alla volta, ~30 h di calendario. Le sonde S della finestra 1 (valutazione di C2, poi C3-C4) girano sulla stessa seriale. Quindi:
- **lancio dopo la valutazione di C2** (sonde e selezione finite), non durante;
- sottomissione **a tranche con `scripts/evaluation_chain.py`** (piano dalla fase `plan`): la catena sottomette solo se i job seriali
  dell'utente in coda o in esecuzione, finestra 1 compresa, sono meno di 2 (`limits`; con 8 core per utente ne gira uno alla volta, il
  secondo e' solo quello pronto a partire: di piu' non accelera), e non mette afterok sui predecessori gia' finiti
  (correzione del 19-20/10); il confronto e' l'ultimo nodo, sottomesso solo quando tutte le sonde sono sottomesse. Si richiama a tranche;
- `--nice=10000` ordina solo i job **in attesa**: abbassa la priorita' della sonda della diagnostica in coda rispetto a un job della finestra
  1 messo in coda dopo, ma non interrompe quella in esecuzione. **Ritardo massimo su un job seriale della finestra 1: il resto della sonda in
  esecuzione, al piu' 4 h** (stima tipica 1 h 45 - 2 h 10), **se il nice prevale sul vantaggio d'eta' della sonda in attesa**. I pesi di
  priorita' di Leonardo non sono letti (seconda review del `critico`: `scontrol show config | grep -i Priority` o `sprio`, sola lettura, non
  fra i comandi ammessi a leonardo-ops; li puo' leggere Simone). **Caso peggiore**, se l'eta' prevale: anche la sonda in attesa passa
  davanti, fino a ~8 h (2 job da 4 h). Prima delle valutazioni di C3 e C4 la diagnostica non si richiama (nessuna tranche nuova): il ritardo
  resta al piu' quello dei due job gia' sottomessi.

Ordine: famiglia B, poi A (il segnale descrittivo piu' forte del 2C e' in B, P2); il confronto della sola B si puo' leggere prima (`compare
--families B`). Disco: rapporti e predizioni fuori fold **~250 MB** (npz non compressi, etichette di EPN-612 e soggetti come stringhe: ~7,5 MB
per insieme P2 e ~3 MB per insieme P1, stima del `critico`) in `$SCRATCH/wearusfm_runs/rami_diagnostica_<data>`, copiati sul Mac a fine
tranche; le cache si leggono soltanto.

## Sequenza di lancio (dopo review del `critico`, decisioni.md e via di Simone; dopo la valutazione di C2)

1. Riferimento `deploy/rami-diag-<data>` allo SHA del commit rivisto, preflight in sola lettura, checkout di campagna (job CPU).
2. Piano della catena: `python3 scripts/spectral_diagnostic.py plan --code-dir <checkout> --expected-sha <SHA> --out-dir
   $SCRATCH/wearusfm_runs/rami_diagnostica_<data> --out results/rami_diagnostica/piano_<data>.json --max-serial 2` (commit). Per teacher
   (B_seed0, B_seed1, A_seed0, A_seed1) i quattro job della tabella (`spectral_diagnostic.sbatch <OUT> probe --task p1 --arm <teacher> --sets
   R PW0 PW --redundancy`, `--time=04:00:00`; `... --sets RPW0` e `... --sets RPW`, `--time=03:30:00`; `... --task p2 --sets R PW0 PW RPW0
   RPW --redundancy`, `--time=04:00:00`), poi il confronto (`spectral_diagnostic.sbatch <OUT> compare`, `--time=00:30:00
   --cpus-per-task=2`), tutti con `--nice=10000`.
3. `evaluation_chain.py <piano> --submit` via leonardo-ops, a tranche (dry-run prima di ogni tranche), finche' tutto e' sottomesso; nessuna
   tranche nuova a ridosso delle valutazioni di C3 e C4.
4. Copia di rapporti ed esito sul Mac, commit in `results/rami_diagnostica/`, voce nel registro e in `docs/decisioni.md`; poi la proposta
   del pilot del punto 2 secondo l'esito.

## Review del `critico` (20/10/2026, su c631c6d): DA RIVEDERE, rilievi chiusi

Bloccanti: (1) la risposta attribuiva un meccanismo («il raw compensa»): ora riporta solo quale confronto supera la regola; (2) «non
peggiora» univa esiti diversi e il confronto fra i due delta non era stimato: risposta a tre valori per confronto, differenza dei delta con
IC come descrittivo senza soglia, nota sull'effetto sotto 1 punto (A/P1 -0,84); (3) il piano di sottomissione ignorava i limiti per utente
della seriale e la valutazione di C2: lancio dopo C2, tranche con `evaluation_chain.py` e `limits`, confronto come ultimo nodo, ritardo
massimo dichiarato, frase su `--nice` corretta; (4) l'uscita del confronto non era controllata: fissata a `<cartella>/esito.json`, il
launcher rifiuta uscite, radici e split negli argomenti e lo script non accetta abbreviazioni. Minori: numero dei test nuovi corretto
(6, non 7: suite 1218 PASS su c631c6d); disco ~250 MB; test su R^2 del ridge, su tutte le combinazioni di `compare` e sull'uscita del
confronto; identita' del run del 2C (stesso run per passo 0 e allenati, completo, non di collaudo); confronti multipli, significato del
passo 0 e budget dichiarati.

Seconda review (20/10, su 10d3731): DA RIVEDERE per un rilievo aperto in parte, chiuso: «entro le soglie» esce anche con stima oltre 1
punto e intervallo che tocca lo zero, quindi la frase e' ora «regola non superata [...]: non esclude una differenza, neanche oltre 1 punto»,
con un test sul caso. Minori: ritardo su C2 dichiarato come condizionato ai pesi di priorita' (non letti), col caso peggiore, e limite di coda
abbassato a 2 (stesso ritmo, meno attesa); soggetti di UCI-EMG verificati sulla provenienza (36, 19 interni, 662 finestre).
