# Review Claude: prova mirata sulla ripresa InfoNCE, diagnostico approvato su `8c72655c`

Rilettura del 27/11/2026 delle correzioni di GPT. La prima review del 21/11/2026 (commit `fc3eef6b`, release chiusa su
`9d34dc69`) è conservata in fondo, sotto «Storico». Revisore: Claude. Linea 3: scienza e codice a GPT. Claude non ha
lanciato nulla. Il verdetto storico resta `FAILED_REQUIRES_REVIEW`: questa approvazione riguarda il solo diagnostico e
non abilita training.

## Esito del 27/11/2026

**Diagnostico approvato sul commit `8c72655c3d894c82fd4918ed523baf0ac9c69e23`. Il blocco di review è emesso più sotto.**

- **I due difetti bloccanti sono corretti.** B-1 e B-2 sono chiusi nel codice. Li ho provati anche sul cluster, in sola
  lettura: sul torch vero di Leonardo e sui checkpoint storici veri.
- **I nove punti sono recepiti.** Le tre regole nuove (copie inattive dei generatori, due bias scalari, recupero del
  solo lettore) sono scritte in `decisioni.md`, nel protocollo e nel piano prima della misura.
- **Nessun rilievo bloccante.** Cinque revisori indipendenti hanno prodotto 24 rilievi. Dopo il voto di quattro scettici
  sui due classificati «importanti», sono tutti minori o note. Non chiedo un altro giro.
- **Che cosa approvo.** Una sola esecuzione del pacchetto: un checkout CPU e un job da 4 GPU per 15 minuti. Non approvo
  training. La lettura scientifica dell'esito spetta a GPT.
- **Tetto richiesto.** 1,0 GPU-ora e 8,2667 ore locali; con i margini di chiusura di SLURM l'inviluppo è 1,4 GPU-ore e
  11,8667 ore locali. A piena allocazione restano 12,26 GPU-ore e 114,0011 ore locali sul tetto firmato.

## Richiesta

- **GPT, tramite Simone, il 27/11:** rileggere il solo diff `9a2ce7e7..8c72655c` e vincolare il commit completo
  `8c72655c3d894c82fd4918ed523baf0ac9c69e23`. «Se approvi il diagnostico, pubblica il nuovo blocco sul commit corretto e
  restituisci lo SHA della review. Il lancio spetterà a 3-launcher.»
- **Simone, il 27/11, in chat con Claude:** «in realta farai tu il lancio. Anxi quando è tutto pronto, passa alla chat
  0-leggi le tabelline e lei si occupera di fare il lancio su leonardo». La chat Claude indicata ha per titolo
  «0- leggi le tabelline».

## Metodo

- **Lettura mia** del diff (13 file, 713 righe aggiunte e 65 tolte), del prompt aggiornato, della voce del 22/11 di
  `decisioni.md`, del protocollo e del piano.
- **Impronte ricalcolate** con `--package` su un clone pulito: pacchetto `e9310d67…702d` e piano `4f538cf1…853b`, come
  nel messaggio di GPT. Il digest dei sorgenti di produzione è invariato: `40d39f56…`.
- **Ricevuta dei test riprodotta:** 211 passati nei 12 file.
- **Cinque revisori indipendenti**, uno per parte: confronto; orchestrazione e lettore; pacchetto, binding e protocollo;
  test e mutazioni; critico. Due scettici per ciascuno dei due rilievi che un revisore aveva classificato «importante».
- **Letture su Leonardo il 27/11**, in sola lettura, descritte sotto. Nessun file scritto, nessun job.
- **Rilettura della bozza** contro le fonti da parte di un verificatore indipendente: circa 130 affermazioni, nessun
  errore di fatto, sette imprecisioni corrette qui.

## Chiusura dei rilievi del 21/11

| Rilievo | Correzione in `8c72655c` | Come l'ho verificata |
|---|---|---|
| B-1, lettura dei vettori | `observe.py:145` passa `str(tensor_path)` a `torch.load(…, mmap=True, weights_only=True)` | Sul torch 2.2 vero: con `Path` esce il `ValueError` del 21/11; con `str` 14 viste sintetiche, salvate nello stesso schema a gruppi dell'osservatore, si caricano uguali alle originali. È l'unico `torch.load` del pacchetto; i checkpoint a 576 passano da `contrastive_target_resume.checkpoint`, che usava già `str` |
| B-2, confronto a 576 | `warm_endpoint_differences` (`compare.py:95`): le tre copie inattive sono riportate a parte e devono coincidere con la gamba ordinaria; ogni altra differenza non in virgola mobile dà `INCONCLUSIVE` | Funzioni copiate dal commit ed eseguite sui checkpoint veri: esito atteso, controlli negativi compresi. Un revisore: 28 casi sintetici nel formato vero, tutti corretti |
| 1, due bias scalari | `screen_reduced_gradients` (`compare.py:123`): solo per gli ID 253 e 257 il denominatore comprende la media dei moduli dei quattro gradienti locali ordinari; il rapporto originale resta nel report | Simulazioni di due revisori, sotto. Il modello vero ha solo questi due parametri di un elemento |
| 2, recupero del solo lettore | Regola in `decisioni.md`, nel protocollo e nel piano | Letta. Due imprecisioni di testo, sotto fra i rilievi minori |
| 3, test del confronto | `test_cn_restart_end_to_end.py`: nove casi passano dalla riga di comando del confronto, con il controllo del torch 2.2 simulato e payload realistici. I rifiuti di `load_observation` hanno un `match` specifico | Eseguiti. Restano deboli su alcune guardie, sotto |
| 4, timeout | `subprocess.TimeoutExpired` è nella tupla (`contrastive_cn_restart.py:178`) | Un revisore ha simulato `run()` in 10 casi con il vero `checked_command`: ogni errore scrive `execution_failure.json` e i tempi |
| 5, preventivo | `--gpu-ore 1.0` nel comando di invio | Letto nel protocollo |
| 6, controllo della norma | La norma è descrittiva: deve solo essere un numero finito e non negativo; l'identità resta l'impronta sha256 | Letto; un revisore ha provato zero e zero negativo |
| 7, costanti del piano | `test_plan_matches_worker_and_readout` lega piano e costanti del codice | Eseguito |
| 8, `--validate-only` | Comando per esteso nel protocollo | Letto |
| 9, raccolta | `phase_timings.jsonl` con i tempi per fase; file `.err` e `lfs quota` nel protocollo | Letto e simulato |
| Frase sulle altre differenze a 576 | Nel protocollo e in `decisioni.md`: ogni altra differenza non in virgola mobile dà `INCONCLUSIVE` | Nel codice è un ritorno anticipato con i percorsi nel motivo |

Fra `9d34dc69` e `8c72655c` non cambiano le due gambe (`contrastive_cn_restart_leg.py`), lo script SLURM e `src/`.
Nell'osservatore cambia solo `load_observation`. Le verifiche del 21/11 su gambe, riscaldamento e hook restano quindi
valide.

## Prove nuove su Leonardo, 27/11, in sola lettura

**Funzioni corrette sui checkpoint veri** (continuo, CN_A, CN_B; 24 s sul nodo di login):

- Lo schema dei generatori è valido nei tre file: `torch_rng` di 5.056 byte, quattro `cuda_rng` di 16 byte, quattro
  `rank_rng` con le due voci attese.
- Ho usato CN_A come gamba riscaldata simulata e CN_B come gamba ordinaria. Contro il continuo la lista grezza contiene
  `cuda_rng.1`, `cuda_rng.2` e `cuda_rng.3`; la lista scientifica è vuota. Ognuna delle tre copie risulta uguale
  all'ordinaria e diversa dal continuo. I 1.990 tensori in virgola mobile diversi sono attesi: CN_A non è riscaldata.
- Controlli negativi: una copia dell'ordinaria alterata di un bit resta squalificante (`cuda_rng.2`); una gamba
  riscaldata uguale al continuo anche nelle tre copie viene rifiutata, perché non coincide con l'ordinaria.
- CN_B contro CN_A: payload esatto.

**Chiamata corretta sul torch vero** (`2.2.0a0+git6c8c5ad`, file tenuto in sola memoria, niente su disco):

- Con `Path` e `mmap=True`: `ValueError: f must be a string filename in order to use mmap argument`.
- Con `str`, `mmap=True` e `weights_only=True`: 14 viste, in precisione singola e doppia, che condividono lo storage del
  gruppo come nel file scritto dall'osservatore. Tipo, forma, offset nello storage e byte sono uguali agli originali.

**Stato prima del lancio**, letto alle 15:08:

- La cartella di uscita e il checkout `cn-restart545-20261121` non esistono, come richiesto.
- Il repository canonico è su `main` a `8dad7e82`, pulito. Il commit `8c72655c` non c'è ancora: serve il push del
  riferimento di deploy.
- Quota di progetto su WORK: 991 G usati su 1 T. Il margine è di circa 33 G contro i 4 GB richiesti; va riletto subito
  prima dell'invio, perché la quota è condivisa.
- Nessuna prenotazione di manutenzione generale: le quattro prenotazioni attive sono di altri account (non ne ho letto i
  flag) e due nodi della partizione risultano in manutenzione. In coda per l'utente c'è un solo job, di un'altra linea.
- Nel ledger comune non ci sono voci della linea 3 dopo il 21/11 alle 15:24 e nessuna per i due ID della prova.

## Simulazioni sui due bias scalari

- **Caso realistico** (un revisore, 200.000 estrazioni, quattro addendi in precisione singola, quattro ordini di
  somma). Con il rapporto originale la soglia di 1e-6 viene superata per solo arrotondamento nell'1,6-1,9% dei casi.
  Con il denominatore nuovo il massimo è 2,3e-7 e la soglia non viene mai superata.
- **Caso sfavorevole** (un altro revisore, 200.000 casi, metà con somma quasi nulla). Rapporto originale: 97.208
  superamenti. Rapporto nuovo: nessuno, massimo 2,34e-7. Il limite analitico è 3,58e-7.
- **Eccessi veri.** Scarti di 1e-3 e di 5e-6 su entrambi i bias vengono ancora rilevati.
- **Tensori di 5 e 6 elementi** (ID 255 e 259), fuori dalla regola: massimo simulato 5,8e-7, mai sopra la soglia.

## Rilievi minori e note

Nessuno richiede modifiche prima del lancio. Il pacchetto non va toccato: ogni modifica cambierebbe le impronte. I punti
2, 4 e 8 sono note; gli altri sono rilievi minori.

1. **Test ancora deboli su alcune guardie.** Un revisore ha rifatto da zero un catalogo di 164 mutazioni: 78
   sopravvivono ai test, 9 su 45 nelle parti nuove. Il test completo non esercita `HYPOTHESIS_UNSUPPORTED` per gradienti
   locali diversi o per assenza di scarto, `PARTIAL_SUPPORT` al 545 e il caso di una copia inattiva diversa sia dal
   continuo sia dall'ordinaria. Il revisore li aveva classificati «importanti»; i quattro scettici li hanno riportati a
   minori, perché il codice è corretto: sei casi aggiuntivi eseguiti sul codice non modificato passano tutti.
2. **Ritorno anticipato.** Se a 576 la gamba riscaldata ha una differenza non in virgola mobile non ammessa, il report
   esce `INCONCLUSIVE` senza le misure del 545. È coerente con la regola registrata. I file grezzi restano su WORK.
3. **Tempi fuori dalla protezione.** Le righe di inizio e fine fase sono scritte fuori dal `try`. Un errore di disco
   in quel punto chiuderebbe il job senza `execution_failure.json`. L'esito resterebbe «nessun risultato», non
   un'attribuzione sbagliata.
4. **Errore imprevisto del confronto.** Un'eccezione fuori dalla tupla catturata lascia la causa solo nel log di SLURM:
   per questo vanno raccolti entrambi i file `.out` e `.err`, come il protocollo già chiede.
5. **Testo del recupero.** `decisioni.md` nomina solo l'errore del lettore; il protocollo aggiunge l'interruzione
   operativa del confronto. La frase sulla fedeltà dell'ordinaria non dice che cosa fare se il falso rifiuto viene dal
   lettore. Lettura mia, da confermare con GPT prima di usarla: vale il testo più stretto. Il recupero si applica solo
   con le due gambe complete a 576, i dati grezzi integri e un errore dimostrato nel solo lettore. Ogni caso dubbio va a
   Simone come decisione.
6. **Motivo con percorso vuoto.** Se cambiasse l'insieme delle chiavi di primo livello del payload, il motivo
   dell'`INCONCLUSIVE` riporterebbe un percorso vuoto. È solo una questione di leggibilità.
7. **Binding e commit successivi.** `--bind-review` accetta il commit di review indicato anche se il mio ramo è poi
   avanzato. Va usato esattamente lo SHA che restituisco.
8. **Frasi superate nel protocollo.** Il protocollo nomina 3-launcher, `--agente gpt` e la pianificazione rispetto al
   fermo del 23-25/11. Fa parte del pacchetto e non si può aggiornare senza cambiare l'impronta: vedi sotto.

## Avvertenze per chi lancia

- **Operatore.** Per l'indicazione di Simone del 27/11 il lancio lo esegue la chat Claude «0- leggi le tabelline», non
  3-launcher. Il cambio si applica ai comandi al momento dell'invio: `--agente claude` e invocazione diretta del
  submitter condiviso, i cui byte coincidono con quelli del commit. ID, risorse, variabili, criteri e ledger restano
  quelli del protocollo.
- **Commit di release.** Lo creo io con il `--bind-review` di GPT non modificato, su un ramo di Claude figlio di
  `8c72655c`, per non aggiungere un passaggio dopo il cambio di operatore; lo SHA è nel registro. Deve aggiungere il
  solo `release_binding.json`. Non deve modificare i sei file del pacchetto né il piano, altrimenti il binding viene
  rifiutato.
- **Riferimento di deploy.** Prima del job di checkout il commit di release va pubblicato su `leonardo` con un
  riferimento `deploy/…`. Il protocollo non lo scrive per esteso.
- **Validazione sul login.** Il blocco di `--validate-only` inizia con `set -eo pipefail`: va eseguito come script,
  per esempio con `ssh leonardo bash -s`, salvando stdout e stderr. Incollato in una shell interattiva la chiuderebbe
  al primo errore.
- **Un solo tentativo per ID.** Nessun rilancio. Un timeout o un errore di esecuzione si raccoglie e torna a GPT e a
  Simone.

## Limiti

- **Nessuna GPU.** L'hook su NCCL, il ramo `libcuda` per l'identità della GPU, i sette gruppi sulla scheda e i tempi
  reali si vedono solo in esecuzione. Il disegno li protegge: se la gamba ordinaria non riproduce CN_A o i gruppi non
  sono quelli attesi, l'esito è `INCONCLUSIVE`, non un'attribuzione sbagliata.
- **Confronto intero sui dati veri.** `compare()` non può girare prima della misura: gli servono le due gambe nuove.
  Sui checkpoint veri ho provato le funzioni del confronto a 576; il percorso completo è provato su cartelle sintetiche.
- **Catena su CPU rifatta con un modello più semplice.** L'attrezzatura del 21/11 non era più su disco. Un revisore ha
  rifatto la catena con 4 processi Gloo e 270 parametri sintetici: esito `SUPPORTED`, scarto relativo massimo 1,22e-7.
  La prova con il trainer vero è quella del 21/11.
- **Contabilità.** La base di 4,74 GPU-ore e 47,3322 ore locali viene dal consuntivo di GPT. Oggi non l'ho ricavata di
  nuovo dall'accounting di SLURM; il ledger non mostra job della linea 3 successivi.
- **Prove.** Rapporti dei revisori, sintesi e script del workflow, script e uscite delle letture su Leonardo del 27/11
  e rapporto del verificatore della bozza sono nella cartella
  `claude_review_contrastive_cn_restart_probe_20261121_evidence/rilettura_20261127/`.

## Blocco di review

```cn-restart-review
{"schema":"cn-restart-review-v1","status":"APPROVED_FOR_DIAGNOSTIC","reviewed_commit":"8c72655c3d894c82fd4918ed523baf0ac9c69e23","package_sha256":"e9310d67b8304c064e09d71b0ec55d37b57abf3b4e2090c2d02d4def586f702d","plan_sha256":"4f538cf183f6e3daf7bd30a5612eecdbbd07106f64d5f348a2f3e759e877853b"}
```

## Prompt per GPT

```text
Review Claude delle correzioni alla prova mirata InfoNCE (diff 9a2ce7e7..8c72655c): docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md sul ramo origin/worktree-claude-training-engine, prove in _evidence/rilettura_20261127/. Lo SHA completo del commit di review te lo passa Simone.

Esito: diagnostico APPROVATO su 8c72655c3d894c82fd4918ed523baf0ac9c69e23. Il blocco di review è emesso in copia unica, con le impronte ricalcolate (pacchetto e9310d67…702d, piano 4f538cf1…853b). Il verdetto storico resta FAILED_REQUIRES_REVIEW; nessun training approvato.

Che cosa ho verificato:
- B-1: sul torch 2.2.0a0 vero di Leonardo, con Path esce il ValueError, con str + mmap + weights_only i vettori si caricano uguali agli originali (file in sola memoria, niente su disco).
- B-2: warm_endpoint_differences e validate_rng_schema copiate dal commit ed eseguite sui checkpoint veri. CN_A come riscaldata simulata e CN_B come ordinaria: lista grezza cuda_rng.1-3, lista scientifica vuota, ogni copia uguale all'ordinaria e diversa dal continuo. Controlli negativi corretti.
- I nove punti: recepiti. 211 test riprodotti. Due simulazioni indipendenti sui bias 253 e 257: con il denominatore nuovo nessun superamento da solo arrotondamento, eccessi veri ancora rilevati.

Rilievi minori, senza richiesta di modifica (il pacchetto non va toccato, cambierebbero le impronte):
1. test deboli su alcune guardie (78 mutazioni su 164 sopravvivono, 9 su 45 nelle parti nuove); il test completo non esercita HYPOTHESIS_UNSUPPORTED per locali diversi o per assenza di scarto, PARTIAL_SUPPORT al 545, copia inattiva diversa sia dal continuo sia dall'ordinaria. Sei casi aggiuntivi sul codice non modificato passano;
2. il ritorno anticipato INCONCLUSIVE a 576 lascia fuori dal report le misure del 545;
3. le righe dei tempi sono scritte fuori dal try;
4. un errore imprevisto del confronto lascia la causa solo nel log di SLURM;
5. testo del recupero: decisioni.md nomina solo l'errore del lettore, il protocollo anche l'interruzione operativa; non è detto che cosa fare se un falso rifiuto della fedeltà dell'ordinaria viene dal lettore. Io leggo il testo più stretto; conferma o correggi prima che serva;
6. se cambiasse l'insieme delle chiavi di primo livello del payload, il motivo dell'INCONCLUSIVE riporterebbe un percorso vuoto;
7. --bind-review accetta il commit di review indicato anche se il mio ramo è poi avanzato: va usato lo SHA esatto.
L'ottavo punto del file riguarda le frasi superate del protocollo (operatore e fermo del 23-25/11), qui sotto.

Cambio di operatore deciso da Simone il 27/11, in chat con me: «in realta farai tu il lancio. Anxi quando è tutto pronto, passa alla chat 0-leggi le tabelline e lei si occupera di fare il lancio su leonardo». Il protocollo, che nomina 3-launcher e --agente gpt, fa parte del pacchetto e non viene modificato: il cambio si applica ai comandi al momento dell'invio (--agente claude). Per non aggiungere un passaggio, il commit di release con il solo release_binding.json lo creo io con il tuo --bind-review non modificato, su un ramo di Claude figlio di 8c72655c; trovi lo SHA nel registro. Se vuoi portarlo sul tuo ramo è un avanzamento semplice: non rifare il binding, uscirebbe un secondo commit di release.

A te resta la lettura scientifica quando i risultati saranno raccolti: report.json, phase_timings.jsonl, ricevute per rank, i due log di SLURM e l'accounting.
```

## Storico: review del 21/11/2026, commit `fc3eef6b`

Il testo che segue è la review del 21/11 com'era al commit `fc3eef6bf023147284512e89ee8b833b9eb63f08`, con i soli
titoli abbassati di livello. Descrive lo stato di allora sul candidato `9d34dc69`: release chiusa e blocco non emesso.
È superato dall'esito del 27/11 in testa al file.

**Titolo originale:** Review Claude: prova mirata sulla ripresa InfoNCE (`9d34dc69`), release chiusa per due correzioni

21/11/2026. Revisore: Claude. Linea 3: scienza e codice a GPT, lanci a 3-launcher. Claude non ha lanciato nulla.
Il verdetto storico resta `FAILED_REQUIRES_REVIEW`.

### Richiesta (21/11)

Prompt di GPT passato da Simone (`docs/prompt_claude_cn_restart_probe_20261121.md` a `9a2ce7e7`): rivedere il solo diff
`918245b6..9d34dc690cf60fcbfe405279dc17546655fa3f5b` del diagnostico a due riprese InfoNCE. Simone, in chat con Claude: «Se approvi il
diagnostico, pubblica il nuovo blocco di review previsto nel prompt e restituisci lo SHA completo. Il lancio spetta poi a
3-launcher.»

Metodo:
- **Lettura mia** dei quattro script, del lanciatore, del piano, del protocollo e della voce di `decisioni.md`.
- **Impronte ricalcolate.** `--package` restituisce `package_sha256 b0640ca6…` e `plan_sha256 9d5811b6…`, come nel prompt.
  Il digest dei sorgenti di produzione è invariato: `40d39f56…`.
- **Letture in sola lettura su Leonardo**, descritte in fondo: sorgente e API del torch del cluster, e la funzione di
  confronto di GPT eseguita sui checkpoint storici veri.
- **Sette revisori indipendenti**, uno per parte, più due scettici per ogni rilievo serio: 41 analisi in tutto.
- **Prova della catena intera su CPU** con il trainer vero e 4 processi Gloo.
- **Rilettura della bozza** contro le fonti da parte di un verificatore indipendente: circa 95 affermazioni, nessun
  errore di fatto, 14 precisazioni applicate qui.

### Esito (21/11)

**Diagnostico non approvato in questo stato. Il blocco di review non è emesso e la release resta chiusa.**

- **Il disegno è approvato.** Due riprese dallo stesso fork, osservazione per rank e per tensore al 545, criteri e
  falsificatori registrati prima della misura: è l'attuazione corretta del limite del 20/11.
- **L'implementazione delle due gambe regge.** Le celle eseguono la vera ripresa CN_A, il riscaldamento è a effetto nullo
  e l'osservatore non altera la riduzione. È verificato con il codice vero, su CPU.
- **Due difetti bloccanti stanno nel solo confronto finale.** Con il codice attuale il report uscirebbe `INCONCLUSIVE`
  in ogni caso, dopo aver consumato le GPU. Ognuno si corregge in poche righe.
- **Nove correzioni non bloccanti** conviene farle nello stesso commit, perché due di esse toccano regole che vanno
  fissate prima della misura.
- **Prossimo passo.** Corretti i punti, rileggo solo il diff ed emetto il blocco sul commit nuovo.

### Difetti bloccanti (21/11)

#### B-1. La lettura dei vettori fallisce sul torch del cluster

- **Dove.** `scripts/diagnostics/contrastive_cn_restart_observe.py:144`:
  `torch.load(tensor_path, map_location="cpu", mmap=True, weights_only=True)`, con `tensor_path` di tipo `pathlib.Path`.
- **Che cosa succede.** Il torch `2.2.0a0+git6c8c5ad` di Leonardo, con `mmap=True`, accetta solo una stringa:
  `serialization.py:1012-1014` solleva `ValueError("f must be a string filename in order to use mmap argument")`.
  `compare.py` cattura l'eccezione e scrive il report con stato `INCONCLUSIVE`.
- **Prova.** Il sorgente letto sul cluster. Più revisori hanno riprodotto il fallimento sul Mac, simulando quel
  controllo su osservazioni prodotte dal codice vero con Gloo. È lo stesso errore del job 64706211 del 17/11.
- **Perché i test non lo vedono.** Il torch 2.14 del Mac accetta i `Path`. Sette test di rifiuto di `load_observation`
  usano `pytest.raises(ValueError)` senza `match`: in cinque di essi l'errore del torch 2.2 passerebbe per il rifiuto
  atteso.
- **Correzione.** `torch.load(str(tensor_path), …)`. Due revisori hanno verificato che basta.

#### B-2. Il confronto a 576 della gamba riscaldata non può mai passare

- **Dove.** `scripts/diagnostics/contrastive_cn_restart_compare.py:241` e `:248`:
  `require(not endpoints["warm"]["metadata_or_schema_different"], …)`.
- **Che cosa succede.** Il checkpoint salva `torch.cuda.get_rng_state_all()` del rank 0, cioè lo stato dei generatori di
  tutte e quattro le GPU viste da quel processo (`run.py:1482`). Tre di queste GPU il rank 0 non le usa.
  - Un training partito da zero risemina tutti i generatori con un seme per rank (`run.py:1463`).
  - Una ripresa li semina con `cfg.seed` (`run.py:1383`) e poi ripristina solo quello della propria GPU
    (`run.py:1446-1450`).

  Quindi `cuda_rng.1`, `cuda_rng.2` e `cuda_rng.3` di qualunque ripresa differiscono da quelli del continuo, per
  costruzione e senza alcun effetto sulla traiettoria. La gamba riscaldata è una ripresa: la riga 248 solleva
  `ValueError` e il report esce `INCONCLUSIVE`, prima ancora di leggere i gradienti del 545.
- **Prova sui dati veri.** Ho eseguito su Leonardo, in sola lettura, la funzione `checkpoint_differences` del commit
  candidato sui checkpoint storici:

| Coppia | Tensori in virgola mobile diversi | `metadata_or_schema_different` | `exact_payload` |
|---|---|---|---|
| Continuo e ripresa CN_A | 1.990 (270 studente, 270 teacher, 540 momenti, 910 della coda) | `cuda_rng.1`, `cuda_rng.2`, `cuda_rng.3` | falso |
| Riprese CN_A e CN_B | 0 | nessuno | vero |

- **Conseguenza anche senza la riga 248.** `exact_payload` resterebbe falso e un recupero completo del continuo sarebbe
  classificato `PARTIAL_SUPPORT`: `SUPPORTED` è irraggiungibile.
- **Correzione proposta.** Nel solo confronto fra gamba riscaldata e continuo, trattare questi tre percorsi come metadati
  di processo noti:
  - escluderli dall'identità scientifica e riportarli a parte;
  - richiedere che coincidano con quelli della gamba ordinaria, che deve riprodurre CN_A;
  - continuare a confrontare `cuda_rng.0`, `torch_rng` e `rank_rng`.

  La regola va scritta in `decisioni.md` e nel protocollo prima della misura, con la prova storica qui sopra. Va scritto
  anche che ogni altra differenza non in virgola mobile a 576 continua a rendere la prova inconclusiva: oggi questa
  regola è nel codice ma non nei criteri registrati. È coerente con il criterio già registrato: «Le differenze di sola
  serializzazione vengono separate».
- **Perché i test non lo vedono.** `compare()` non è mai eseguito per intero dalla suite, e nessun payload di test ha
  questa forma.

### Da correggere nello stesso commit (21/11)

Nessuno di questi punti basterebbe a chiudere la release. I primi due toccano però regole che dopo la misura sarebbero
post hoc.

1. **Soglia di 1e-6 e parametri scalari.** Il limite è relativo alla sola norma del gradiente ridotto. Il modello ha due
   parametri di un solo elemento, `anchor_heads.rms.bias` (ID 253) e `anchor_heads.env.bias` (ID 257). Su un solo
   elemento la somma dei quattro rank può cadere per caso vicino a zero, e allora il solo ordine della somma supera
   1e-6. Tre scettici su quattro stimano con una simulazione un rischio dell'1-2,5% di un `HYPOTHESIS_UNSUPPORTED`
   dovuto al solo arrotondamento; il quarto giudica lo scenario non plausibile. Per i tensori con più elementi il
   rischio è trascurabile: massimo simulato 6,4e-7, tipico 5e-8.
   Proposta: aggiungere al denominatore la scala degli addendi, per esempio la norma di `Σ_r |g_r| / 4`, già
   calcolabile dai vettori locali. In alternativa, riportare in forma descrittiva un eccesso sui tensori di un elemento.
   La scelta spetta a GPT e va registrata prima della misura.
2. **Regola per rieseguire il solo confronto.** I due difetti bloccanti stavano nel confronto su CPU, che legge soltanto
   file. Conviene registrare prima del lancio che un `INCONCLUSIVE` dovuto al solo confronto, con le due gambe complete,
   si chiude rieseguendo su CPU un confronto corretto e rivisto, sugli stessi dati grezzi, con criteri invariati e
   senza nuove GPU. Senza questa frase, correggere il lettore dopo la misura sarebbe contestabile.
3. **Test del confronto.** Un revisore ha applicato 84 mutazioni alle guardie dei quattro script: 54 sopravvivono ai
   test. Serve almeno un test di `compare()` su cartelle sintetiche, con il controllo del torch 2.2 simulato e un payload
   con `rank_rng` e `cuda_rng` realistici, un caso per esito. Servono anche i `match` nei rifiuti di `load_observation`.
   Un revisore ha già costruito in simulazione un confronto completo con gli artefatti storici veri: si può riusare.
4. **Timeout non registrato.** `checked_command` solleva `subprocess.TimeoutExpired`, che non è fra le eccezioni
   catturate in `contrastive_cn_restart.py:174`: allo scadere `execution_failure.json` non viene scritto. Basta
   aggiungerla alla tupla.
5. **Preventivo nel comando di invio.** Il comando registra `--gpu-ore 0.59`, e il protocollo dichiara la scelta:
   l'estremo superiore della stima storica. La regola del progetto è però GPU allocate per walltime massimo richiesto: 4
   GPU per 15 minuti fanno `--gpu-ore 1.0`.
6. **Controllo della norma.** `observe.py:163` ricalcola la norma con tolleranza relativa 1e-12. È ridondante rispetto
   all'impronta sha256 e un falso rifiuto renderebbe la prova inconclusiva: conviene toglierlo o allentarlo.
7. **Costanti del piano.** `relative_gradient_roundoff_limit`, `gradient_norm_floor`, numero di parametri, di rank e
   passi stanno nel piano ma il codice usa copie proprie. Basta un test che ne verifichi l'uguaglianza.
8. **Comando di `--validate-only`.** Il protocollo lo cita senza riportarlo per esteso.
9. **Raccolta.** Il protocollo chiede i tempi, ma i tempi per gamba non vengono registrati. Vanno raccolti anche il file
   `.err` del job e, prima dell'invio, lo spazio su WORK con `lfs quota`: servono circa 4 GB.

### Verifiche positive, per punto del prompt (21/11)

**1. Le due celle eseguono la vera ripresa CN_A.**
- **Riga di comando.** Gli argomenti ricostruiti coincidono con quelli di `contrastive_job.build_command` per la ripresa
  storica. Il runner riconosce la fase `resume_a` e il fork `aa5952f0…` dalla release pinnata.
- **Cartella di uscita.** Il payload del checkpoint non dipende dalla cartella: CN_A e CN_B, in cartelle diverse, hanno
  lo stesso file.
- **Ordine delle chiamate.** La traccia di produzione registra batch, generatori e confine prima del riscaldamento e i
  generatori «dopo» alla fine del backward vero. L'hook è montato prima del primo backward del processo. Il primo
  clipping reale avviene dopo il ritorno dal backward del 545.
- **Riscaldamento a effetto nullo.** Con la vera funzione di accumulo del trainer, il runtime contrastivo vero e DDP
  Gloo a 2 processi, le varianti senza hook, ordinaria e riscaldata danno gradienti locali, studente, teacher,
  ottimizzatore, runtime e generatori identici bit per bit; cambia solo lo stato DDP. Con il trainer intero a 4 processi
  lo stato al confine coincide prima e dopo il riscaldamento. Una scansione del codice non trova attributi dei moduli
  modificati dal forward.

**2. L'hook osserva la riduzione vera.**
- **Prima e dopo.** Il buffer è copiato prima che l'hook di riferimento lo divida e lo riduca; il vettore ridotto è
  confrontato con il gradiente effettivo del parametro prima del clipping.
- **Copertura.** I 270 parametri sono coperti una volta per rank, compresi gli 8 in doppia precisione. Il modello vero ha
  114.202.292 byte in precisione singola e 68.480 in doppia, come i gruppi di costruzione storici.
- **Sette gruppi.** CN_A e CN_B, dopo la ricostruzione, registrano la stessa composizione a sette gruppi del continuo: il
  controllo sulla gamba riscaldata è soddisfacibile.
- **Hook e percorso nativo.** Su Gloo a 4 processi, con tipi misti e più update, la riduzione con l'hook è identica bit
  per bit a quella senza. Su NCCL non è verificabile qui: lo protegge il controllo di fedeltà della gamba ordinaria.
- **Torch 2.2.** Sul cluster ho letto le chiamate più a rischio: i metodi di `GradBucket`, il controllo e la
  registrazione dell'hook, l'hook di riferimento, gli hook dell'ottimizzatore. Esistono. Per le altre un revisore non
  trova incompatibilità con la 2.2, senza lettura sul cluster. Manca l'attributo `uuid` delle proprietà CUDA: gira
  quindi il ramo di ripiego con `libcuda`. Il comportamento effettivo dell'hook sulla 2.2 si vedrà solo in esecuzione.

**3. Le regole di lettura corrispondono ai criteri registrati.**
- **Classificazione.** Gradienti locali diversi, eccesso oltre la soglia o nessuno scarto danno `HYPOTHESIS_UNSUPPORTED`.
  Mancato recupero al 545 o a 576 dà `PARTIAL_SUPPORT`. Fedeltà dell'ordinaria, confine, gruppi, posizionamento e
  coerenza fra rank portano a `INCONCLUSIVE`.
- **Nessuna autorizzazione.** Il report ha sempre `training_approved` falso e `FAILED_REQUIRES_REVIEW`, anche quando è
  inconclusivo.
- **Sugli archivi veri.** Le funzioni del confronto passano sulle tracce, sulle metriche e sui profili storici: una
  ripresa soddisfa le uguaglianze di traccia contro il continuo; CN_A e CN_B hanno gli otto campi di gradiente identici
  in tutti i 32 update.
- **Payload vero.** Non contiene foglie numpy. La configurazione nel checkpoint coincide con `config.json`.

**4. Pacchetto, binding, pin e risorse.**
- **Binding.** Simulato in un clone privato: accetta solo il blocco unico ed esatto. Rifiuta blocco doppio, impronta
  sbagliata, commit abbreviato o non antenato, file modificati dopo la review e il vecchio blocco.
- **Pin storici.** Fork, riferimento ordinario, continuo e job CN_A coincidono con gli artefatti. I 31 file archiviati
  che il confronto rilegge sono identici alle copie remote.
- **Tempo.** Stima dai dati storici: da 450 a 620 s contro circa 830 s utili. Il margine è adeguato, non ampio.
- **Costi.** I conti sono esatti: 1 GPU-ora e 8,2667 ore locali al massimo; inviluppo 1,4 e 11,8667; residuo 12,26
  GPU-ore e 114,0011 ore locali sul tetto firmato.

**5. Test.** La ricevuta si riproduce: 170 test passati, 32 nei quattro file nuovi. Sono stabili, ma deboli (punto 3
sopra).

### Prova della catena intera su CPU (21/11)

Un revisore ha eseguito l'intera sequenza sul Mac, con 4 processi Gloo, il trainer vero su un modello piccolo (106
parametri, gruppi DDP rimpiccioliti per averne più d'uno) e le funzioni vere del diagnostico. Ha chiamato le funzioni
del confronto una per una, classificazione compresa: `compare()` intero richiede i checkpoint del cluster.

| Passo | Esito |
|---|---|
| Continuo con hook di riferimento contro continuo senza hook | identici bit per bit |
| Ripresa ordinaria contro continuo | diversi dal primo update dopo il fork |
| Gradienti locali, ordinaria contro riscaldata | identici |
| Scarto ridotto relativo massimo per tensore | 9,5e-8, sotto la soglia di 1e-6 |
| Ripresa riscaldata contro continuo | identici bit per bit: gradienti, checkpoint finale, metriche |
| Gruppi osservati nella riscaldata | quelli ricostruiti, uguali al log DDP del continuo |
| Classificazione | `SUPPORTED` |

Su CPU l'intervento ricostruisce quindi il continuo esattamente. È un'indicazione a favore dell'ipotesi e della
fattibilità del riscaldamento, non una misura: modello piccolo, Gloo al posto di NCCL, nessuna aritmetica bf16.

### Note, senza richiesta di modifica (21/11)

- **Identificazione della GPU.** Il ramo `ctypes` non è mai stato eseguito su CUDA e gira al 545, subito prima del
  backward misurato. Le firme sono corrette e il driver è già inizializzato: gli scettici lo giudicano un rischio minore.
  Non conviene anticiparlo, perché cambierebbe l'ordine di inizializzazione rispetto alla ripresa storica.
- **`SUPPORTED` dice che il riscaldamento basta.** Non dimostra che la causa sia la sola composizione dei gruppi. Il
  protocollo lo dichiara già.
- **Zeri con segno.** Una differenza dei soli zeri con segno nel ridotto conta come «nessuno scarto». È irrilevante in
  pratica.
- **Binding sul cluster.** Lì `require_binding` verifica la coerenza interna del binding; l'autenticità rispetto al mio
  ramo è verificata al momento del `--bind-review`. È il limite già noto del meccanismo precedente.
- **Ordine delle gambe.** Con un timeout si perde la gamba riscaldata, cioè l'intervento.

### Letture su Leonardo, in sola lettura (21/11)

Nessun file scritto, nessun job. Il confronto dei checkpoint ha richiesto 24 s; le altre letture sono brevi e non
cronometrate.

- **Torch del cluster.** `serialization.py`, `default_hooks.py`, `distributed.py`, `optimizer.py`; attributi di
  `GradBucket` e delle proprietà CUDA; Python 3.11.6.
- **Checkpoint storici.** Continuo, CN_A e CN_B aperti con `mmap`. Ho eseguito la funzione `checkpoint_differences`
  copiata senza modifiche dalle righe 35-72 del commit candidato. Tipi delle foglie: solo tensori, numeri, stringhe,
  booleani e `None`.
- **Requisiti del confronto.** Configurazione del payload uguale a `config.json`; `summary.json` con 576 passi e
  `max_steps`; `start.json` con le impronte attese; fork presente; cartella di uscita assente.

### Limiti (21/11)

- **Nessuna GPU.** NCCL, bf16 su CUDA e il ramo `libcuda` non sono verificabili sul Mac.
- **Sette gruppi sulla GPU.** Che il riscaldamento sul cluster riproduca proprio i sette gruppi del continuo è
  plausibile ma non verificabile qui. Se non li riproduce, l'esito è `INCONCLUSIVE`, non un'attribuzione sbagliata.
- **Tempi.** La stima viene dai job storici: l'overhead reale della strumentazione non è misurato.
- **Soglia e scalari.** Il rischio dell'1-2,5% è una stima per simulazione, non una misura.
- **Prova su CPU.** Modello piccolo, Gloo, senza bf16: non anticipa l'esito sul cluster.
- **Prove.** Rapporti e script dei revisori, uscite delle letture su Leonardo e sintesi del workflow sono nella cartella
  `claude_review_contrastive_cn_restart_probe_20261121_evidence/`.

### Blocco di review (21/11)

Non emesso. Lo aggiungo, in un'unica copia, quando il commit corretto supera la rilettura del diff.

### Prompt per GPT (21/11)

```text
Review Claude della prova mirata InfoNCE (diff 918245b6..9d34dc69): docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md sul ramo origin/worktree-claude-training-engine, prove nella cartella _evidence/ accanto. Diagnostico NON approvato in questo stato: blocco di review non emesso, release chiusa. Disegno approvato, gambe e osservatore verificati con il codice vero. Nessun lancio da parte mia.

Due difetti bloccanti, entrambi nel solo confronto CPU: con il codice attuale il report uscirebbe INCONCLUSIVE in ogni caso.
- B-1: contrastive_cn_restart_observe.py:144 chiama torch.load(tensor_path, ..., mmap=True) con un pathlib.Path. Il torch 2.2.0a0 del cluster accetta solo str (serialization.py:1012-1014, ValueError; stesso errore del job 64706211). Correzione: str(tensor_path).
- B-2: contrastive_cn_restart_compare.py:248. Il payload salva get_rng_state_all() del rank 0 (run.py:1482); un run da zero risemina per rank (run.py:1463), una ripresa semina con cfg.seed (run.py:1383) e ripristina solo la propria GPU (run.py:1446-1450). Quindi cuda_rng.1, .2 e .3 di ogni ripresa differiscono dal continuo per costruzione. Ho eseguito checkpoint_differences del commit candidato sui checkpoint veri, in sola lettura: continuo contro CN_A dà metadata_or_schema_different = [cuda_rng.1, cuda_rng.2, cuda_rng.3]; CN_A contro CN_B è esatto. Correzione proposta: nel solo confronto riscaldata/continuo escludere quei tre percorsi dall'identità scientifica, riportarli a parte, richiedere che coincidano con la gamba ordinaria e continuare a confrontare cuda_rng.0, torch_rng e rank_rng. Regola da scrivere in decisioni.md e nel protocollo prima della misura.

Nello stesso commit, senza un altro giro:
1. soglia 1e-6 e i due parametri scalari (ID 253 anchor_heads.rms.bias, ID 257 anchor_heads.env.bias): rischio stimato dell'1-2,5% di un HYPOTHESIS_UNSUPPORTED da solo arrotondamento; proposta: scala degli addendi al denominatore, oppure eccesso descrittivo per i tensori di un elemento; scelta tua, da registrare prima della misura;
2. registrare prima del lancio che un INCONCLUSIVE dovuto al solo confronto, a gambe complete, si chiude rieseguendo su CPU un confronto corretto e rivisto sugli stessi dati, a criteri invariati;
3. un test di compare() su cartelle sintetiche con il controllo del torch 2.2 simulato e payload con rank_rng e cuda_rng realistici, un caso per esito; match nei rifiuti di load_observation (oggi 54 mutazioni su 84 sopravvivono);
4. subprocess.TimeoutExpired nella tupla dell'except di contrastive_cn_restart.py:174;
5. --gpu-ore 1.0 nel comando di invio (4 GPU per 15 minuti);
6. togliere o allentare il controllo della norma a 1e-12 in observe.py:163;
7. un test che leghi le costanti del piano a quelle del codice;
8. il comando esatto di --validate-only nel protocollo;
9. tempi per gamba registrati, raccolta anche del file .err e controllo dello spazio su WORK con lfs quota (circa 4 GB).
Nella regola per B-2 scrivi anche che ogni altra differenza non in virgola mobile a 576 resta causa di INCONCLUSIVE.

Verificato e a posto: riga di comando della vera ripresa CN_A, riscaldamento a effetto nullo con la vera funzione di accumulo del trainer, hook identico al percorso nativo su Gloo, copertura dei 270 parametri, sette gruppi soddisfacibili, regole di lettura conformi ai criteri registrati, binding e pin, costi (1 GPU-ora, 8,2667 ore locali), tempo stimato 450-620 s su circa 830 utili. Su CPU con 4 processi Gloo e il trainer vero su un modello piccolo la ripresa riscaldata ricostruisce il continuo bit per bit e la classificazione dà SUPPORTED: è un'indicazione, non una misura (Gloo, senza bf16, funzioni del confronto chiamate una per una).

Pubblica il commit corretto con le nuove impronte di pacchetto e piano: rileggo solo il diff ed emetto il blocco sul commit nuovo.
```
