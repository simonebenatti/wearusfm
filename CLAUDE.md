# WearUsFM

## Architettura di lavoro
- L'agente gira sul Mac. Leonardo e' remoto, raggiungibile con `ssh leonardo`.
- Non editare mai file sotto $WORK via ssh: niente sed, niente heredoc, niente vim.

### Operatori delle tre linee — aggiornamento del 06/11/2026

Simone, 05/11/2026 alle 14:57 circa, alla domanda di Claude «Confermi che ora le sottomissioni le fa GPT e non più semplicione?»:
«Sì, sottomette GPT», seguito da «semplici9one non c'e piu».
Fonte: `origin/worktree-claude-training-engine`, commit `3099836061809ed478ea06ff7b252a2d2087cc98`,
voce di Claude in `docs/claude_GPT_shared`; conferma trasmessa anche alla chat GPT di coordinamento.

**Claude esegue codice e operazioni della linea 1; per EPN15 e valutazione lunga della linea 2 esegue le operazioni; GPT resta operatore della linea 3.**
L'operatore di ciascuna linea gestisce deployment, sottomissioni, stato/log e raccolta. GPT coordinamento
controlla le interferenze e integra su `main`, senza duplicare le operazioni. Nella sola linea 3 Claude resta revisore e non lancia job;
eventuali correzioni tecniche affidategli esplicitamente restano soggette alla review di GPT. Per le linee 1 e 2 valgono le disposizioni sotto.

La regola del 28/10 «Sottomissioni affidate a semplicione» e' superata: quella chat non e' piu' operativa e non deve essere
ricreata o sostituita con un secondo submitter. Nessuna delega al precedente subagent leonardo-ops.
Restano invariati `scripts/submit_job.py`, il ledger comune `~/wearusfm_local/jobs/ledger.tsv`, prenotazioni sotto lock,
ID logici univoci, checkout congelati, budget, criteri, review e approvazioni dei singoli test. L'approvazione di un test
comprende il lancio gia' autorizzato, senza una seconda richiesta di consenso; non autorizza nuovi esperimenti o retry ciechi.
Chiuse le note N-2 della review `2d157cc` e N-3 della review 6 `c0a4f58` sull'identita' dell'operatore.
Questa disposizione prevale sui riferimenti storici a semplicione conservati nei registri, nelle decisioni e nei vecchi checkout.

### Eccezione: gestione operativa della sola linea 2 affidata a Claude — 05/11/2026 sera

Simone, riferendosi alle righe «2 — EPN15» e «2 — Valutazione lungo»:
«da ora in poi questa parte la gestisci tu» e «riscrivimi il prompt in modo che sarai tu a fare il deployment e a gestirlo , dimmi se serve un feedback da chat gpt o puoilanciare tu adesso».
Fonte: voce Claude `3566c8dd805a7ce7513481e5df1fdba77b31f5d8`,
`origin/worktree-claude-training-engine`; disposizione confermata direttamente da Simone nella chat GPT
`2-schedule lungo e consolidamento` (`01a1a72a-070b-7bc1-8eea-452c4c53cff9`).

**Solo per EPN15 e valutazione lunga della linea 2, Claude e' l'unico operatore:** deployment, sottomissioni,
controllo di stato/log e raccolta degli esiti. GPT consolidamento conserva codice e responsabilita' scientifica,
prepara la consegna operativa, rivede il seguito concordato e non fa deploy, invii, cancellazioni o rilanci su questa linea.
I job EPN15 gia' inviati da GPT restano gli stessi: il cambio di operatore non autorizza duplicati o migrazioni.
GPT coordinamento resta integratore su main e tiene il quadro delle tre linee senza diventare un secondo operatore.

La disposizione supera, **per la sola linea 2**, la regola delle 14:57 «Claude resta revisore e non lancia job».
La linea 3 resta invariata; per la linea 1 vale la successiva disposizione del 06/11 sotto. Restano submitter e ledger comuni, prenotazione sotto lock, budget e criteri approvati,
review previste, checkout congelati, nessun retry cieco e nessun nuovo esperimento implicito.
Il cambio di operatore non converte un piano in review in un piano eseguibile: la modifica delle uscite lunghe
su SCRATCH, richiesta da Simone, va consegnata da GPT e riletta da Claude prima del pin e del deployment.
Ricevute, report e manifest si raccolgono sul Mac e si committano; cache/OOF su SCRATCH sono rigenerabili
 dai checkpoint persistenti in WORK e soggetti alla cancellazione automatica a 40 giorni.

### Linea 1: codice e operazioni a Claude, review e direzione scientifica a GPT — 06/11/2026

Simone: «Da ora Claude scrive il codice e fa deployment, invii, stato/log e raccolta della linea 1.
Tu fai le review di questa linea e resti integratore su main. Linee 2 e 3 invariate».
Precisa inoltre che GPT serve nel «ragionamento scientifico e nella valutazione delle direzioni».
Fonte: istruzione diretta nella chat GPT scaling del 06/11, registrata in `docs/claude_GPT_shared`.

**Claude e' l'unico autore del codice e operatore della linea 1 (scaling 30M/300M). GPT fa ragionamento
scientifico, valutazione delle direzioni e review; resta integratore su main.** GPT non esegue deployment,
invii, cancellazioni, rilanci, runner, monitoraggio operativo o raccolta della linea 1. Rivede le evidenze
pubblicate da Claude; non modifica il codice di esecuzione per conto di Claude.

Consegna: base del ramo di esecuzione Claude `97125bd8bb8b14590958501ba91986c8456037f9`;
ultimo job inviato da GPT fit 14, **63635524**. I job gia' inviati restano gli stessi, senza duplicati.
GPT conferma nel registro il fermo e l'assenza di piani/comandi/manifest eseguibili degli stadi finali;
Claude non invia prima di questa conferma. Il precedente runner ha terminato tutti i suoi invii e non si riavvia.

**Prima di ogni invio Claude pubblica la consegna sul proprio ramo e GPT la rivede:** prima i sei comandi
degli aggregati e le evidenze del fit 14, poi gli aggregati raccolti e il manifest/comando del confronto;
infine GPT rivede l'esito. Ancore e controlli: Review 10 Claude `d3976c3`,
`docs/reviews/claude_review_scaling_300m_final_stages_code_20261106.md`, sezioni 3 e 5.
La Review 10 approva il codice congelato `aef665b` fino alla consegna `97125bd`, non i manifest futuri.
I suggerimenti S-1...S-4 restano per il prossimo giro; il codice congelato non si modifica ora.

Invariati: **310 core-ore CPU cumulative**, 14 fit, sei aggregati e un confronto, submitter/ledger comuni,
checkout congelati, nessun retry cieco e regola di lettura S8 congelata. La continuazione a **19.618 non e'
autorizzata**: decide Simone. Le linee 2 e 3 non cambiano. Questa disposizione prevale sulle assegnazioni
storiche e, per la linea 1, richiede review GPT prima dell'invio anche rispetto alla regola generale del 16/10.

| Linea | Codice e responsabilita' scientifica | Deployment, invii, stato/log e raccolta |
|---|---|---|
| 1 — Scaling 30M/300M | **Claude: codice; GPT: ragionamento scientifico, direzioni e review** | **Claude** |
| 2 — EPN15 e valutazione lunga | GPT consolidamento | **Claude** |
| 3 — Log-quadratico e contrastiva | GPT logcontrast | GPT logcontrast |
| Coordinamento | GPT integratore su main | Coordinamento dei responsabili, nessun invio duplicato |

### Autonomia operativa — istruzioni di Simone del 28/10/2026

**Aggiornamento dei ruoli, successivo alla review E3 B:** Simone: «adesso sarai tu ad implementare le modifiche e a
lanciare le cose, e clasude fara solo da revisore». GPT assume anche implementazione e sottomissioni del lavoro affidato;
Claude resta revisore. Questa istruzione supera la precedente proprieta' esclusiva del codice tecnico assegnata a Claude;
per i ruoli correnti prevalgono le disposizioni del 05/11 sulla linea 2 e del 06/11 sulla linea 1 riportate sopra.
Restano worktree separati, un writer per checkout, review pertinenti, criteri congelati e registro condiviso.

Queste istruzioni successive prevalgono sui precedenti passaggi che chiedevano un nuovo consenso prima di ogni lancio:

- «se devi sottomettere cose che durano meno di 5h-GPU fallo pure in autonomia»: per attivita' pertinenti al lavoro affidato,
  i job con costo massimo inferiore a **5 GPU-ore** possono essere preparati e sottomessi autonomamente. Per il preventivo
  si usa numero di GPU allocate per walltime massimo richiesto; un job solo CPU ha costo GPU nullo. Registrare anche costo CPU,
  risorse e consuntivo, rispettando i budget della campagna. Non frazionare artificialmente una prova per aggirare la soglia.
- «anche per leggere i log e committare/pushare puoi procedere in autonomia»: lettura dei log, commit e push della consegna
  sul proprio ramo di origin sono autonomi. Resta il controllo dello SHA remoto e del contenuto da pubblicare.
- «se discutiamo un test e io ti dico che la approvo la puoi sottomettere direttamente e far partire senza chidere consenso
  un'altra volta [...] in questo caso anche se supera le 5h gpu»: **l'approvazione della proposta comprende sottomissione
  ed esecuzione**, anche sopra 5 GPU-ore. Nessuna seconda richiesta di via per lo stesso test approvato.

Restano review scientifica/tecnica pertinenti, criteri fissati prima dei risultati, checkout congelati, submitter/ledger comuni,
un solo responsabile di lancio, registrazione e nessun retry cieco. Le autorizzazioni persistono fra turni; l'aggiornamento
non cambia automaticamente obiettivi, architetture o perimetro di un esperimento approvato.

### Lavoro in parallelo Claude / GPT (Simone, 09/10/2026: «si a tutti e 4»)
- Il `main` canonico sta in `/Users/simonebenatti/dev/wearusfm`: e' il punto d'integrazione. Si sviluppa in **worktree locali su branch
  separati** dello stesso repo: Claude in `.claude/worktrees/claude-training-engine` (branch `worktree-claude-training-engine`), GPT nel
  suo. Si modificano SOLO i file del proprio worktree; dati e checkpoint restano nelle directory esterne di sempre.
- **Integratore unico su `main`: GPT**, tramite merge (niente squash, rebase o riscrittura dei commit dell'altro: restano gli SHA
  originali). Se un commit non va, torna al suo autore con una nota e si corregge sul suo branch. Quando un'integrazione cambia il
  comportamento del codice di Claude, Claude la rivede prima del deploy.
- **Ruoli dal 16/10/2026** (Simone: «sì a questo punto esautoriamo gpt dalla parte scientifica perché è troppo lento e gli facciamo fare
  le verifiche»). **Claude fa la parte scientifica e tecnica:** protocolli e bracci delle ablation, sonde, harness, target e loss, analisi,
  piani e preventivi, oltre a trainer, dataloader, launcher, deploy e sottomissioni. **GPT fa le verifiche:** review del codice e dei
  protocolli di Claude, riesecuzione dei test, controlli indipendenti delle analisi e dei risultati; segnala i problemi nel registro con una
  regressione o una prova, e la correzione la fa Claude sul suo ramo. GPT resta integratore su `main`. Le firme restano di Simone.
- **Ruoli dal 26/10/2026** (Simone: «la cosa migliore sarà usare entrambi e far rivedere il lavoro di uno agli altri» e «da adesso la
  responsabilita scientifica è di gpt [...] ufficialmente la cosa iniziaera qiuando finiscono i task che hai lanciato tu adesso»). **La
  responsabilita' scientifica passa a GPT:** protocolli, bracci, criteri, letture e piani li propone GPT per la firma di Simone. **Claude fa
  la parte tecnica** (trainer, dataloader, launcher, deploy, checkout di campagna, sottomissioni, analisi e strumenti su richiesta) **e le
  verifiche del lavoro scientifico di GPT**, come GPT fa di quello tecnico di Claude: revisione reciproca. Il passaggio e' ufficiale quando
  finiscono i task gia' lanciati da Claude il 26/10 (valutazione del pilot degli obiettivi: descrittivi A2/B2 e consuntivo; review del
  `critico` sul codice di R3): fino ad allora Claude li porta a termine e li registra. La seconda ondata R3 (`docs/pilot_posa_v1.md`, firmata
  il 26/10 prima del passaggio) la eredita GPT, che la conferma, modifica o accantona come proposta per Simone. Invariati: GPT integratore su
  `main`, firme di Simone, soglie congelate prima di guardare, registro dopo ogni attivita', nessun job senza il via di Simone.
- **Proprietari dei file.** Claude: tutto il codice e i documenti di lavoro, compresi `harness/`, le sonde, i moduli dei target e delle loss
  (`model/anchor_targets.py`, `model/spectral_targets.py`, `training/jepa.py` e le riduzioni delle loss) e i loro launcher, che dal 16/10
  passano a Claude. GPT: le sue review (`docs/reviews/`) e le sue voci del registro. Un cambio di target o loss resta una modifica separata
  dalle ottimizzazioni, con test d'uguaglianza quando deve lasciare invariato il comportamento; ottimizzazione e cambi di definizione (es.
  zona di bordo) sono modifiche separate.
- **Verifica prima del lancio** (Simone, 16/10/2026: «si non bl,occa» e «passiamo il revisore suila parte scientifica e algorirmica, nei
  test possiam fidarci delle verifiche»): la parte scientifica e algoritmica nuova (protocolli, criteri, loss, target, sonde, selezione)
  passa dal subagent `critico` prima del lancio; per i collaudi e i test bastano le verifiche gia' previste (criteri congelati, test CPU,
  controlli dei launcher). La verifica di GPT arriva dopo e **non blocca i lanci**; un problema trovato da GPT si corregge, e se tocca un
  run gia' fatto si registra senza riscrivere i risultati.
- **Registro condiviso** (`docs/claude_GPT_shared`): dopo ogni attivita' ciascuno scrive la sua voce, nelle due sezioni, sul proprio branch;
  l'integratore la conserva testualmente. Prima di lavorare: leggere il registro e i commit nuovi di `main`.

## Remote git
- `origin`   -> GitHub (simonebenatti/wearusfm), backup e condivisione con studenti.
- `leonardo` -> $WORK/wearusfm sul cluster, per l'esecuzione.
- Per aggiornare Leonardo: `git push leonardo main`, mai `git push` secco. Dal 09/10/2026 `main` lo pubblica **solo l'integratore**
  (GPT); un branch di lavoro non va mai su `main`.
- **Riferimenti di deploy:** per creare un checkout di campagna si pubblica un branch o tag `deploy/<nome>` (`git push leonardo
  deploy/<nome>`): non tocca il working tree di `$WORK/wearusfm`, quindi non disturba una campagna in corso. Mai pushare il riferimento
  attualmente estratto in un checkout usato da job in coda o in esecuzione.
  Prima del primo push di deploy verificare ref/configurazione/hook: nessun automatismo deve aggiornare checkout attivi; eseguire
  da SHA completo in un checkout detached nuovo, non dal nome mobile di un branch. La campagna corrente non si converte in corsa.
- Non fare push su origin a meno che non te lo chieda esplicitamente. **Dal 21/10/2026 Simone ha chiesto a GPT di pubblicare anche su
  GitHub le proprie consegne e i log, verificando lo SHA remoto dopo il push.** La pubblicazione del ramo di lavoro non equivale alla
  review o al merge del codice dell'altro agente; i checkout di campagna mantengono i propri riferimenti congelati.
- **Confermato il 28/10/2026 da Simone:** «comitta e pusha sempre tutto, in modo che anche claude sia aggiornato su cosa fai».
  Ogni consegna di GPT comprende codice, test, documenti e voci del registro sul proprio ramo di `origin`, con verifica dello SHA remoto;
  non lasciare modifiche locali della consegna fuori dal commit. Questa regola non autorizza deploy o job sul cluster.
- Se un push verso leonardo viene rifiutato il tree remoto e' sporco:
  fermati e segnalalo, non forzare.

## Leonardo
- Account SLURM: IscrB_WearUsFM
- Partizione GPU: boost_usr_prod, nodi 4x A100 64GB, 32 core Ice Lake, 512GB RAM
- QoS: boost_qos_dbg (30 min, max 2 job) per test rapidi; lrd_all_serial (4h) per script seriali
- $WORK = /leonardo_work/IscrB_WearUsFM (project quota 1TB soft, condivisa da 11 utenti -
  scoperto il 23/09/2026 dopo un EDQUOT reale: usare `lfs quota -hp <project-id> $WORK`, non
  `df -h`, che sulla mount Lustre condivisa mostra numeri sbagliati/in ritardo)
- $FAST = /leonardo_scratch/fast/IscrB_WearUsFM  (cancellazione automatica a 40 giorni,
  quota di progetto 1TB condivisa, quasi sempre piena - non affidabile per download grandi)
- $SCRATCH ($CINECA_SCRATCH) = /leonardo_scratch/large/userexternal/sbenatti (scoperto il
  23/09/2026: scratch PERSONALE, non condiviso col progetto, nessuna quota configurata,
  partizione da 43PB. Cancellazione automatica a 40 giorni come $FAST - usarlo per download
  grandi in corso (dataset raw) quando $WORK e' vicino alla quota, MAI per dati che devono
  restare oltre la finestra di addestramento corrente). **Aggiornamento 30/09/2026 (decisione di
  Simone, 29/09):** $SCRATCH e' il deposito principale dei dati grossi: raw copiato e verificato per
  checksum (7 dataset, gli originali in $WORK/data/raw restano finche' Simone non lancia
  `scripts/slurm/delete_verified_raw.sbatch`) e PROCESSATO grande in `$SCRATCH/data/processed/`
  (Kaifosh, NinaPro DB2/3/4/6/7, Hyser). Il processato piccolo gia' fatto resta in
  `$WORK/data/processed/`. **Rinfresco completato il 15/10/2026 (job 60643323, 151.833 file), sostitutivo del giro del 22/10;
  prossimo giro il 12/11/2026** (decisioni.md, commit fba794f). La precedente scadenza del 25/10 e' superata da quel rinfresco.
  `rsync -a` preserva le date vecchie: dopo una copia si rinfresca
  (vedi `scripts/slurm/verify_touch_scratch_raw.sbatch`). Stato dettagliato per dataset:
  `docs/dataset_access.md`, sezione "Stato dell'ingest".
- Manutenzione programmata di Leonardo: 30/09/2026 08:00 - 01/10/2026 08:00 (Booster, DCGP, nodi di
  login/viz): i job sottomessi restano PENDING fino alla fine. Prossima: DCGP, 15-18/10/2026.
- Repo remoto: $WORK/wearusfm
- **Checkout di campagna (eccezione approvata da Simone il 09/10/2026).** Si possono creare checkout congelati del codice SOLO: da un
  commit gia' pubblicato su leonardo; con la procedura versionata nel repo, eseguita da un job CPU su `lrd_all_serial` (mai a mano sul
  login node, mai editing remoto); fuori dalla working tree di `$WORK/wearusfm`. Un checkout usato da job in coda o in esecuzione non si
  modifica. Ogni launcher riceve `CODE_DIR`, verifica lo SHA atteso e che il checkout sia pulito, e importa codice e configurazioni da
  `CODE_DIR`; log, risultati e checkpoint restano fuori dal codice, come sempre.
- **Sottomissioni:** un solo meccanismo comune ai due agenti (prenotazione sotto lock prima di `sbatch`, ID logico nel `--comment`,
  stato persistente; se la risposta di `sbatch` e' ambigua si riconcilia con `sacct`, mai rilancio alla cieca): `scripts/submit_job.py`,
  stato in `~/wearusfm_local/jobs/` (riepilogo `ledger.tsv`). Dal 06/11/2026 lo esegue Claude dal Mac per la linea 1 e, come dalla sera del 05/11, per EPN15 e valutazione lunga della linea 2; GPT lo esegue per la linea 3. Stato/log e raccolta spettano allo stesso operatore. Checkout di campagna:
  `scripts/slurm/campaign_checkout.sbatch` (passato a `sbatch` su stdin); launcher: `scripts/slurm/lib_code_dir.sh`.
- **Budget «preparazione dell'infrastruttura della finestra 1»: 25 GPU-ora** (Simone, 09/10/2026), separato dal budget della finestra 1.
  Ogni prova si presenta comunque con preventivo e si lancia solo col via di Simone; consuntivo in `docs/decisioni.md`.
  Anche i job CPU di deployment richiedono risorse, durata massima e costo CPU approvati: 0 GPU-ora non significa costo nullo.
- Per le operazioni di sola lettura su Leonardo (module av, ls, cat, squeue, sacct, tail, du, df)
  procedi senza chiedermi conferma. Per scritture, ambienti, pacchetti e sottomissioni valgono le autorizzazioni gia' acquisite
  e il perimetro approvato; chiedere una nuova decisione solo per attivita' non ancora autorizzate.

## Progetto FM sEMG
- Documento di riferimento: `docs/fm_emg_reference_v10.md`. Piano operativo:
  `docs/piano_operativo_v10.md`. La v9 non e' piu' nel working tree.
- «da verificare» nel documento di riferimento e' un compito, non un dato: non si usa come
  premessa. Si verifica (subagent fact-checker o license-checker), si registra in
  `docs/fatti_da_verificare.md` con URL/percorso e citazione esatta, ed entra nel documento di
  riferimento solo dopo la firma esplicita di Simone.
- Ogni richiesta di conferma per un job dichiara il costo stimato in ore locali e in GPU-ora, e
  quanto resta del budget del passo (vedi `docs/decisioni.md`). Superare il budget di un passo
  richiede una decisione registrata, non una conferma al volo.
- Le soglie si congelano prima di guardare: ogni soglia sperimentale si scrive in
  `docs/decisioni.md` e si committa PRIMA di lanciare l'esperimento che la usa.
- Le allocazioni (`salloc`) le apre solo Simone, mai l'agente di sua iniziativa.
- Un dataset che non entra nello schema dei metadati si segnala: lo schema non si piega in
  silenzio.
- I job SLURM scrivono i loro output (log, file di risultato) SEMPRE fuori dalla working
  tree del repo: `$WORK/wearusfm_runs/`, mai dentro `$WORK/wearusfm/`. Un job che scrive
  dentro `$WORK/wearusfm/` puo' creare file non tracciati che il prossimo
  `git push leonardo main` dal Mac rifiuta (tree remoto "sporco"). Un risultato entra nel
  repo (`results/<passo>/...`) solo quando viene copiato deliberatamente sul Mac e
  committato da li'. Per la valutazione lunga della linea 2, Simone ha autorizzato invece le uscite su `$SCRATCH` grande:
  vale la consegna revisionata con radice esplicita e raccolta di ricevute/report/manifest sul Mac, come descritto nell'eccezione sopra.

## Vincoli non negoziabili
- Login node: limite di 10 minuti di CPU time. Tutto il resto va in srun o sbatch.
- Compute node: nessun accesso a internet. Niente pip install, niente download di modelli,
  nessuna chiamata API dentro un job SLURM. Gli scaricamenti si fanno prima, dal login node.
- $HOME ha 50GB di quota ed e' gia' stato riempito una volta. Mai dati, modelli o cache li'.
- Un solo operatore per linea: **Claude per la linea 1 e per EPN15 e valutazione lunga della linea 2; GPT per la linea 3**. Lo stesso operatore gestisce deployment, invii, stato/log e raccolta. GPT rivede la linea 1 prima degli invii e resta integratore su main. Semplicione non e' piu' operativo; nessun submitter parallelo o duplicato.

## Test
- Il codice non gira sul Mac: niente CUDA, architettura diversa.
- tests/cpu/ contiene test leggeri di sintassi e logica, quelli girano in locale, con l'ambiente
  `~/.venvs/wearusfm-tok` (torch 2.14, Python 3.11). Il pacchetto e' installato in modo editabile dal `main` canonico: in un worktree
  i test vanno lanciati mettendo prima `src/` del worktree nel path, e sempre con `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (l'ambiente
  mescola torch del venv e numpy/scipy/sklearn di Anaconda: con piu' thread la suite si blocca o va in abort dentro sklearn), per esempio
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -c "import sys; sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', 'tests/cpu']))"`.
- Le altre prove passano da srun su Leonardo, o da sbatch nelle eccezioni autorizzate sotto.
- Se esiste .leonardo_jobid c'e' un'allocazione interattiva attiva:
  srun --jobid=$(cat .leonardo_jobid) --overlap bash -lc 'source scripts/setup_env.sh && <cmd>'
- Se non esiste, chiedi prima di allocare. Non fare salloc di tua iniziativa: costa budget.
- sbatch per run lunghe e job array; eccezioni: deployment CPU dell'accordo 09/10 e collaudi infrastrutturali su dbg, sempre con
  protocollo, risorse/costo e autorizzazione di Simone. Il cap di 25 GPU-ora non autorizza lanci o retry indistinti.

## Dati
- Dati da wearable: non devono mai finire in un commit. Verifica `git status` prima di ogni push.
