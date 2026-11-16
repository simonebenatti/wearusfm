# Prompt per GPT: linea 1, l'estrazione p1 a 19.618 fallisce due volte al controllo d'identita' delle cache; review della diagnosi e della direzione (16/11/2026)

Ruoli della linea 1 (CLAUDE.md su main `e114a78`): Claude scrive il codice ed e' l'unico operatore (deployment, invii, stato/log,
raccolta); GPT fa ragionamento scientifico, valutazione delle direzioni e review, resta integratore su main e non esegue deploy,
invii, rilanci o raccolte della linea 1. Simone, 16/11, alla domanda «Approvi l'opzione B (diagnostica, controllo corretto,
nuovo rilancio di p1) senza review di GPT?»: «no prepaprami un prompt per gpt». **Nessun job parte prima della tua review e
della decisione di Simone.**

## 1. Fatti

Ramo d'esecuzione `claude/scaling-300m-exec-20261106` su origin, punta `3143e52`. Registro sul ramo
`claude/review-scaling-300m-20261104`, punta `0ca4e40`, sezioni 1 e 2 del 16/11. Voci del 16/11 in `docs/decisioni.md`.

- **Stato dello stadio B** della continuazione 300M a 19.618:
  - training 63979977 `COMPLETED 0:0` (11/11);
  - estrazioni del 16/11 dal checkout congelato `scaling_300m_evaluation19618_20261106_v1` a `df2de35`: pose 64571999 e p2
    64572001 `COMPLETED 0:0`, con le ricevute nel pacchetto ma non ancora raccolte;
  - p1 64572000 `FAILED 1:0`.
- **Dove fallisce p1.** Dopo l'estrazione: il sottoprocesso e' uscito con codice 0 e le tre cache sono scritte. Si ferma nel
  controllo d'identita' delle uscite, `scaling_evaluation19618.execute` → `tokenizer_preflight.file_identity`: stat, sha256, stat, e
  confronto di inode, size, mtime_ns e ctime_ns.
  - Errore: «input changed while hashing» su `ninapro_db6.npz` (2,19 GB, scritto in ~2 s, controllato ~9 s dopo).
  - Diagnosi: `results/infra/scaling_300m_20261104/continuation19618_20261106/failure_64572000_diagnosis_20261116/README.md`
    (commit `90ce085`).
- **Il rilancio unico.** Approvato da Simone senza la tua review: «Sì, approvo» a «Approvi il rilancio unico di p1 senza review
  di GPT?». Stessa sottomissione, con ID `SCALING-300M-eval19618-p1-r1-20261116` e cartella `.../T100-G-300M/p1-r1`.
  - Modifica solo lato Mac in `scripts/scaling_plans19618.py`, commit `7c5c886`. Record:
    `results/infra/scaling_300m_20261104/continuation19618_20261116_p1_relaunch/README.md`.
  - Il cancello aggiunge cinque controlli: fallimento diagnosticato, cartella intatta, riga uguale salvo la cartella, unico job,
    tetti dello stadio.
  - Verifica a quattro lenti (subagent `critico`) con scettici; sei mutanti uccisi. Test dei tre file dello stadio B: 200
    passati. Suite completa su `7c5c886`: vedi §5.
- **Il rilancio 64596915** (commit `ca18841`): `FAILED 1:0` dopo 29:53, nello stesso controllo ma su `ninapro_db3.npz`, scritto
  12 min 16 s prima del controllo. db2 e' passato, db6 non e' stato raggiunto.
  - Prove in `results/infra/scaling_300m_20261104/continuation19618_20261106/evidence/failure-p1-r1-64596915/` (commit `3143e52`):
    README, log SLURM, `stat`, `lfs getstripe -c` (tutte le cache hanno una stripe), `sacct`.

| Tentativo | Cache | Scritta | Controllata | Esito |
|---|---|---|---|---|
| 64572000 | `ninapro_db2` | 10:05:15 | 10:20:59-10:21:07 | superato |
| 64572000 | `ninapro_db3` | 10:08:33 | 10:21:07-10:21:08 | superato |
| 64572000 | `ninapro_db6` (2,19 GB) | 10:20:57-10:20:59 | 10:21:08-10:21:11 | fallito |
| 64596915 | `ninapro_db2` | 15:46:01-15:46:02 | 16:01:29-16:01:36 | superato |
| 64596915 | `ninapro_db3` (0,49 GB) | 15:49:19-15:49:20 | 16:01:36-16:01:39 | fallito |
| 64596915 | `ninapro_db6` (2,19 GB) | 16:01:28-16:01:29 | non raggiunto | — |

- **Osservazione.** I due fallimenti cadono 7-12 s dopo la scrittura di db6 (2,19 GB in 1-2 s). Nella stessa finestra sono
  passati tre controlli. A 19.618 sono passati 7 controlli su 9. Al 5.000 (05/11, prima delle manutenzioni del 10-14/11) non e'
  fallito nessun controllo, con tempi simili.
- **Lettura di Claude, non verificata.** Sul nodo che ha appena scritto molti dati, gli attributi riportati da Lustre restano
  instabili per qualche secondo, anche per file vecchi della stessa cartella. Il campo cambiato non e' noto. La lettura del
  mattino (corsa con lo scarico del file controllato) e' rettificata dal secondo caso.
- **Consumo.**
  - Stadio: 4,714444 GPU-ore su 7 (pose 2,036111; p1 0,522778 + 0,498056; p2 1,6575) e 37,715556 core-ore su 56.
  - Riga p1 del freeze (2 GPU-ore, 16 core-ore, un job): 1,020833 GPU-ore.
  - Solo CPU dello stadio B: 349,917778 su 350, dopo l'emendamento dei checkout approvato il 16/11.

## 2. Proposta di Claude (opzione B), non approvata

1. **Diagnostica.** Un job su un nodo Booster: al massimo 30 min, 1 GPU allocata e non usata, 8 CPU, 0,5 GPU-ore. Fa tre cose:
   - riproduce la scrittura dei file di p1, cioe' 1,9, 0,5 e 2,2 GB con l'ultimo scritto in 1-2 s;
   - ripete il controllo attuale sui file della cartella, con e senza `os.sync()` prima, registrando ogni coppia di `stat`;
   - rilegge piu' volte, senza scritture, le cache gia' esistenti, per capire se il problema tocca anche i lettori.
   Le uscite vanno su `$SCRATCH/wearusfm_runs/diag/...`.
2. **Controllo corretto** nel codice di valutazione, con test e checkout nuovo del codice:
   - `os.sync()` dopo il sottoprocesso d'estrazione;
   - se gli attributi cambiano durante l'hash, una seconda lettura completa dopo una pausa;
   - si accetta solo con attributi stabili nella seconda lettura e lo stesso sha256; entrambe le letture restano nella provenance.
3. **Rilancio di p1** da quel checkout, con ID e cartella nuovi, al massimo 2 GPU-ore. Lo stadio arriverebbe a 6,764444 GPU-ore su
   7 (54,115556 core-ore su 56) con il margine; la riga p1 a 3,070833 GPU-ore sui 2 del freeze.
4. **Emendamento CPU.** Il sesto checkout aggiunge 0,266667 core-ore con margine: il totale solo CPU va a 350,184444 su 350.

## 3. Cosa ti chiedo

1. **Diagnosi.**
   - E' plausibile?
   - Ci sono cause alternative: statahead, dimensione pigra sul MDT, ctime/mtime fra MDT e OST, client cambiato con le
     manutenzioni?
   - Cosa misurare per distinguerle: campo che cambia, frequenza, dipendenza dalla scrittura, tipo di nodo.
2. **Criteri della diagnostica, fissati prima di guardare.** Condizioni, numero di cicli, cosa conta come riproduzione e cosa come
   effetto della correzione, tetti. Claude li congela in `docs/decisioni.md` prima del job.
3. **Correzione.**
   - Il criterio d'integrita' basta, o serve altro? Per esempio due sha256 completi sempre, oppure nessun confronto di ctime.
   - Dove va: solo nell'estrazione (`scaling_evaluation19618.py` e' un file derivato con sostituzioni dichiarate, verificate da
     `scaling_derive19618.py`), oppure in `tokenizer_preflight.file_identity` e `check_unchanged`, condivisi con i lettori? Va
     verificato se il contratto delle sorgenti del training ne fissa i byte.
4. **Commit d'estrazione.** Due strade:
   - commit misti: p1 da un commit nuovo, pose e p2 da `df2de35`. I piani dei lettori portano `extraction_code_sha` per cache; lo
     strumento sul Mac oggi ne vuole uno solo, e i lettori sono da controllare;
   - riestrazione di tutte e tre col codice nuovo, che supera i 7 GPU-ore dello stadio e richiede un emendamento.
5. **Lettori.** Conviene indurire anche il loro `check_unchanged` prima delle ~43 ore di coda seriale, o procedere e fermarsi al
   primo fallimento?
6. **Budget.** Come trattare questi punti:
   - la riga p1 oltre i 2 GPU-ore;
   - la diagnostica, se dentro o fuori il tetto dello stadio;
   - l'emendamento CPU;
   - la registrazione dei tre.
7. **Alternative.** Strade che non ho visto, compresa una che non tocchi il codice congelato del cluster.

## 4. Come rispondere

- Review in `docs/reviews/` sul tuo ramo, con la voce nel registro; push su origin e verifica dello SHA remoto. Non modificare i
  rami di Claude.
- Verdetto esplicito: B approvata, B con modifiche (quali), oppure un'altra direzione. Scrivi i criteri della diagnostica in modo
  che si possano congelare cosi' come sono.
- Nessun deploy, invio, rilancio o raccolta della linea 1. Dopo la tua review e la decisione di Simone, Claude implementa, fa
  rivedere cio' che va rivisto ed esegue.

## 5. Stato dei test

Suite completa `tests/cpu` sul commit `7c5c886`, avviata verso le 14:50 del 16/11: in corso alle 16:17 (96%, nessun fallimento fin li'); esito finale nel registro e in `docs/decisioni.md`.
