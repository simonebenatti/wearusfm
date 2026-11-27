export const meta = {
  name: 'verifica-consegna-lancio-cn-restart',
  description: 'Verifica indipendente della sequenza operativa e del brief di consegna del lancio della prova mirata InfoNCE, con controprova dei rilievi seri',
  phases: [
    { title: 'Revisione', detail: 'cinque revisori, uno per dimensione', model: 'opus' },
    { title: 'Controprova', detail: 'scettici sui rilievi bloccanti o importanti', model: 'opus' },
  ],
}

const S = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe2'
const SEQ = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/contrastive_cn_restart_probe_launch_20261127.md'
const BRIEF = S + '/brief_launcher.md'
const RW = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release'
const REVIEW = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md'

const COMUNE = `
## Contesto
Progetto WearUsFM, linea 3. GPT ha preparato un diagnostico («prova mirata InfoNCE al passo 545»): un job di checkout CPU e un job da 4 GPU per 15 minuti sul cluster Leonardo (CINECA). Il revisore Claude lo ha approvato e ha creato il commit di release b58a66f695c8744dfa5fe633829cafe13a5a1fc4 (codice di GPT 8c72655c3d894c82fd4918ed523baf0ac9c69e23 più il solo file release_binding.json). Simone, il responsabile del progetto, ha deciso che il lancio lo esegue un'altra chat Claude, intitolata «0- leggi le tabelline», che gira su un modello più piccolo e con meno contesto. Riceverà SOLO due testi: un messaggio di consegna (il «brief») e un file con la sequenza operativa. È ammesso un solo tentativo per ciascun ID di job: un errore nei due testi può bruciarlo. Il tuo compito è trovare errori, omissioni e ambiguità nei due testi PRIMA della consegna. Non devi correggerli.

## I due testi da verificare
- Sequenza operativa: ${SEQ}
- Brief (bozza del messaggio; il segnaposto @SEQ_SHA@ verrà sostituito con lo SHA del commit che contiene la sequenza): ${BRIEF}

## Fonti, tutte in sola lettura
- Worktree git del commit di release (HEAD = b58a66f6…, pulito): ${RW}
  - protocollo di GPT: docs/contrastive_cn_restart_probe_20261121.md (la sezione «Risorse e passaggio operativo» e i «Comandi che 3-launcher completerà dopo la review»)
  - piano: results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/plan.json; binding: release_binding.json nella stessa cartella
  - codice: scripts/diagnostics/contrastive_cn_restart.py, contrastive_cn_restart_leg.py, contrastive_cn_restart_compare.py, contrastive_cn_restart_observe.py; scripts/contrastive_release.py, contrastive_job.py, contrastive_run.py, contrastive_split_benchmark.py, contrastive_prepare.py, contrastive_target_resume.py
  - script SLURM: scripts/slurm/cn_restart_probe.sbatch, scripts/slurm/campaign_checkout.sbatch, scripts/slurm/lib_code_dir.sh
  - submitter: scripts/submit_job.py (identico byte per byte a quello condiviso in "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py")
  - regole registrate: docs/decisioni.md (voci del 21/11 e del 22/11 sulla prova mirata InfoNCE)
- Review del revisore (parte nuova in testa al file, fino al titolo «Storico»): ${REVIEW}
- Prova a secco dei due comandi di invio, fatta con un ssh finto e un registro temporaneo: ${S}/prova_a_secco_invii/ (chiamata_1_argomenti.txt e chiamata_1_stdin.txt per il checkout; chiamata_4_argomenti.txt e chiamata_4_stdin.txt per il job GPU; registro/ledger.tsv e registro/<ID>/state.json)
- Letture in sola lettura fatte su Leonardo il 27/11: ${S}/leonardo/stato_pre_lancio_20261127.out.txt, ${S}/leonardo/reservation_coda_20261127.out.txt, ${S}/leonardo/confronto_corretto_veri_20261127.out.txt, ${S}/leonardo/torch22_mmap_str_memoria_20261127.out.txt
- Ledger comune dei job (sola lettura): ~/wearusfm_local/jobs/ledger.tsv
- Regole del progetto: il file CLAUDE.md del progetto è già nel tuo contesto.

Fatti già verificati dal revisore, che puoi usare: dentro una shell non interattiva aperta con ssh su Leonardo la variabile WORK vale /leonardo_work/IscrB_WearUsFM e la funzione module esiste; i due checkout precedenti della linea 3 sono durati 39 e 41 s con 4 CPU; la QoS boost_qos_dbg ammette 30 minuti, 2 job in esecuzione e 4 sottomessi per utente; il repository canonico sul cluster è su main a 8dad7e82, pulito, senza hook propri, con receive.denyCurrentBranch=updateInstead; il commit 8c72655c non è ancora sul cluster.

## DIVIETI ASSOLUTI
- Non usare ssh, scp, rsync, né git push, git fetch o git pull verso alcun remoto.
- Non eseguire scripts/submit_job.py in nessuna forma (nemmeno con opzioni di prova) e non eseguire sbatch, squeue, sacct, scontrol.
- Non modificare nessun file dentro i worktree git: né la sequenza, né il brief, né il worktree di release (nemmeno file temporanei o __pycache__: se esegui Python usa sempre l'opzione -B o PYTHONDONTWRITEBYTECODE=1 e non lanciare pytest dentro il worktree di release).
- Scrivi soltanto nella tua cartella di lavoro e nel tuo rapporto, indicati sotto.
Puoi leggere tutto, usare git in sola lettura (log, show, diff, merge-base, cat-file, ls-tree), grep, e piccoli script bash o Python locali che non toccano la rete (per Python: ~/.venvs/wearusfm-tok/bin/python -I -B).

## Che cosa restituire
Scrivi in italiano semplice. Per ogni rilievo indica: gravità («bloccante» = seguendo i testi alla lettera il lancio fallirebbe o brucerebbe un tentativo, oppure violerebbe una regola o un'autorizzazione; «importante» = rischio concreto di errore dell'operatore o informazione sbagliata che può portarlo fuori strada; «minore» = imprecisione che non cambia l'esito; «nota» = osservazione senza richiesta), dove (file e riga o passo), la prova (fonte con percorso e riga, o comando eseguito e sua uscita), e la correzione proposta in una o due frasi. Non segnalare questioni di stile. Elenca anche le verifiche positive più importanti che hai fatto davvero (non quelle che supponi) e i punti che non hai potuto verificare.
`

const SCHEMA = {
  type: 'object',
  properties: {
    sintesi: { type: 'string' },
    rilievi: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, gravita: { type: 'string', enum: ['bloccante', 'importante', 'minore', 'nota'] },
      titolo: { type: 'string' }, dove: { type: 'string' }, prova: { type: 'string' }, correzione: { type: 'string' },
    }, required: ['id', 'gravita', 'titolo', 'dove', 'prova', 'correzione'] } },
    verifiche_positive: { type: 'array', items: { type: 'string' } },
    punti_aperti: { type: 'array', items: { type: 'string' } },
  },
  required: ['sintesi', 'rilievi', 'verifiche_positive', 'punti_aperti'],
}

const VERDETTO = {
  type: 'object',
  properties: {
    regge: { type: 'boolean' },
    gravita_corretta: { type: 'string', enum: ['bloccante', 'importante', 'minore', 'nota', 'infondato'] },
    motivo: { type: 'string' },
  },
  required: ['regge', 'gravita_corretta', 'motivo'],
}

const DIMENSIONI = [
  { key: 'd1-comandi', compito: `
## La tua parte: esattezza dei comandi
Verifica riga per riga ogni comando della sequenza (passi 1-8) e ogni comando citato nel brief.
- Sintassi e quoting in zsh sul Mac e in bash sul nodo di login: heredoc con delimitatore fra apici, apici attorno a 'bash -s', la tilde nei percorsi di ridirezione, l'argomento "--export=ALL,…" che attraversa il submitter, ssh e la shell remota. Usa i file della prova a secco per vedere che cosa arriva davvero a ssh e confrontalo con ciò che lo script sbatch si aspetta.
- Refspec e verifica del push del riferimento di deploy.
- Corrispondenza esatta con i comandi del protocollo di GPT, salvo le differenze dichiarate nella sequenza (--agente claude, invocazione diretta del submitter condiviso al posto di mac_launcher_command.py, due righe date, SHA scritto per esteso). Ogni altra differenza è un rilievo.
- Corrispondenza dei parametri con plan.json (allocation, logical_ids, output), con le righe #SBATCH e con i controlli dentro gli script sbatch (per esempio la stringa TimeLimit, CODE_DIR, il numero di argomenti).
- Uscite attese dichiarate nella sequenza e nel brief (stringhe come «checkout di campagna pronto: … @ SHA», "new_binding_valid": true, "execution_started": false, COMPLETED 0:0): confrontale con il codice che le produce.
- I controlli del passo 1 (comandi git e grep) danno davvero l'esito descritto? Attenzione ai codici d'uscita (per esempio grep -c con zero corrispondenze) e a come un operatore li leggerebbe.
Puoi eseguire bash -n sul corpo dell'heredoc e sugli script sbatch e provare localmente frammenti innocui, sempre fuori dai worktree.` },
  { key: 'd2-codice-cluster', compito: `
## La tua parte: il codice che girerà sul cluster
Percorri il codice eseguito nei passi 4, 5 e 6, partendo dai file del worktree di release, e cerca ogni precondizione che la sequenza non dichiara o non controlla e che farebbe fallire un passo sul cluster.
- campaign_checkout.sbatch: che cosa richiede e che cosa produce.
- lib_code_dir.sh (code_dir_setup): controlli su HEAD, pulizia e sola lettura.
- contrastive_cn_restart.py --validate-only: require_binding, package (git diff --quiet con evidence_commit), review_fields (git show del reviewed_commit), commands (checked_pin, validate_review_binding del job storico e ciò che richiede, radici dei dati, cartella di uscita, variabile WORK), validate_argv in contrastive_cn_restart_leg.py, contrastive_run.py --validate-only. Che cosa serve nell'ambiente? Che cosa scrive? Può scrivere dentro il checkout in sola lettura?
- Per ogni comando git che il codice eseguirà sul cluster, verifica che i commit richiesti siano antenati del commit di release (quindi presenti sul cluster dopo il push del riferimento deploy): usa git merge-base --is-ancestor nel worktree di release.
- cn_restart_probe.sbatch e run(): controlli iniziali, validate_allocation, creazione della cartella di uscita, scadenze.
- Codici d'uscita e file prodotti dal job GPU nei vari esiti (confronto concluso con status INCONCLUSIVE; fase fallita; timeout; errore prima delle fasi): la sezione «Esiti possibili del job» della sequenza li descrive correttamente?
- Che cosa la validazione sul login NON copre e potrebbe fallire solo nel job GPU: elencalo, distinguendo ciò che è già dichiarato nella sequenza («Che cosa non si può verificare prima») da ciò che manca.
Non hai accesso al cluster: ragiona sul codice e sulle letture del 27/11.` },
  { key: 'd3-operatore', compito: `
## La tua parte: l'operatore che esegue alla lettera
Prima passata: leggi SOLO il brief e la sequenza, senza aprire le altre fonti, mettendoti nei panni di un operatore che esegue alla lettera e ha meno contesto. Scrivi nel tuo rapporto, passo per passo, che cosa faresti e dove ti fermeresti con un dubbio.
Seconda passata: usa le fonti per sciogliere i dubbi e classifica ciò che resta.
Cerca in particolare:
- istruzioni ambigue o che ammettono due letture;
- criteri di arresto mancanti o valori attesi non specificati;
- passi in cui un'esecuzione letterale produce un effetto indesiderato (file scritti nel posto sbagliato, worktree di release sporcato, comando lanciato dalla cartella sbagliata, variabile non definita);
- come si attende la fine dei due job, che cosa fare se un job resta in coda a lungo o se il certificato ssh scade prima della raccolta;
- dove salvare le uscite operative e che cosa esattamente committare, su quale ramo e in quale percorso;
- incoerenze fra brief e sequenza (numeri, nomi, ordine dei passi, vincoli presenti in uno solo dei due testi);
- informazioni che servono all'operatore e non ci sono (per esempio come ricavare il jobid, come leggere il log del checkout, come fare la raccolta).` },
  { key: 'd4-regole', compito: `
## La tua parte: regole del progetto e autorizzazioni
Confronta brief e sequenza con le regole del progetto (CLAUDE.md, già nel tuo contesto) e con le autorizzazioni citate.
- Riferimenti di deploy e checkout di campagna; un solo operatore per linea e la disposizione della linea 3 del 27/11; submitter e ledger comuni; nessun retry cieco; dichiarazione dei costi in ore locali e GPU-ora con il residuo del budget; output dei job fuori dalla working tree del repository sul cluster; limiti del nodo di login (10 minuti di CPU); niente editing remoto; allocazioni interattive.
- Dati da wearable mai in un commit: guarda l'elenco di raccolta del passo 7 e stabilisci quali file finirebbero nel repository. Controlla se file dello stesso tipo sono già archiviati nel repository da GPT per la ripresa storica CN_A (nel worktree di release: results/infra/contrastive_targets_20261118/reloads_execution_20261121/CN_A) e che cosa contengono, in particolare origin_sessions.json, pose_panel_set.json e fixed_eval_set.json: sono identificativi e metadati o segnali?
- Citazioni: le frasi attribuite a Simone e a GPT sono identiche nella review, nella sequenza e nel brief? Qualche frase attribuisce a Simone o a GPT qualcosa che le fonti non dicono?
- Numeri di costo (1,0; 8,2667; 1,4; 11,8667; 12,26; 114,0011; 4,74; 47,3322; 18; 169,6): rifai i conti a partire dal protocollo e dal piano.
- Il brief chiede alla chat destinataria qualcosa che eccede l'indicazione di Simone («in realta farai tu il lancio. Anxi quando è tutto pronto, passa alla chat 0-leggi le tabelline e lei si occupera di fare il lancio su leonardo») o che una regola vieta? Dice chiaramente che il messaggio viene da un'altra chat e non da Simone?
- Il campo --agente claude: è coerente con il significato della colonna «agente» del ledger e con le regole? Guarda il ledger e il submitter.` },
  { key: 'd5-completezza', compito: `
## La tua parte: che cosa manca
Confronta la sequenza, punto per punto, con ciò che il protocollo di GPT richiede a chi lancia (sezione «Risorse e passaggio operativo» e «Comandi che 3-launcher completerà dopo la review») e con le regole registrate in docs/decisioni.md.
Per ciascuna richiesta del protocollo di' se compare nella sequenza, in quale passo, e se è formulata in modo equivalente: assenza di prenotazioni o uscite preesistenti, saldo netto degli impegni, billing corrente, quota e scritture concorrenti, spazio per osservazioni e checkpoint, slot debug e reservation, autenticazione dello SHA del deploy e dei riferimenti storici, completamento e verifica del checkout (COMPLETED 0:0, SHA, pulizia), validazione sul login con stdout, stderr e ricevute conservati, variabili del job GPU, nessun invio implicito dal validatore, raccolta di phase_timings.jsonl, JSON delle interruzioni, inventari e impronte per rank, i due file SLURM con SHA e accounting, tensori diagnostici conservati su WORK con impronte e percorsi, lettura scientifica del nuovo esito, recupero del solo lettore.
Segnala anche: ciò che la sequenza aggiunge e che contraddice il protocollo; passi del protocollo ormai superati (per esempio la pianificazione rispetto al fermo del 23-25/11) che la sequenza dovrebbe dichiarare tali; informazioni della review del revisore (sezione «Avvertenze per chi lancia» e «Limiti») che non sono arrivate nella sequenza o nel brief.` },
]

const risultati = await pipeline(
  DIMENSIONI,
  (d) => agent(
    COMUNE + d.compito + `

## Dove scrivere
Cartella di lavoro tua: ${S}/agents2/${d.key}/ (creala). Rapporto completo: ${S}/reports2/${d.key}.md. Restituisci l'oggetto strutturato richiesto.`,
    { label: `revisione:${d.key}`, phase: 'Revisione', schema: SCHEMA, model: 'opus' }
  ),
  async (review, d) => {
    if (!review) return { key: d.key, review: null, verified: [] }
    const seri = review.rilievi.filter((f) => f.gravita === 'bloccante' || f.gravita === 'importante')
    const scelti = seri.slice(0, 4)
    if (seri.length > scelti.length) log(`${d.key}: ${seri.length - scelti.length} rilievi seri oltre il quarto non passano dalla controprova`)
    const lenti = [
      'Lente dei fatti: il fatto affermato dal rilievo è vero? Controllalo sulle fonti, riga per riga, e prova a confutarlo.',
      'Lente delle conseguenze: ammesso che il fatto sia vero, cambierebbe davvero il comportamento dell\'operatore o l\'esito del lancio? Considera che cosa l\'operatore leggerà e farà in pratica.',
    ]
    const verified = await parallel(scelti.map((f) => () =>
      parallel(lenti.map((lente, i) => () => agent(
        COMUNE + `
## La tua parte: controprova di un rilievo
Un revisore ha segnalato il rilievo qui sotto sui due testi. Il tuo compito è provare a CONFUTARLO. ${lente}
Se non riesci a confutarlo, di' che regge e indica la gravità corretta secondo le definizioni date sopra (oppure «infondato» se il rilievo è falso). Non proporre rilievi nuovi.

Rilievo ${f.id} (gravità dichiarata: ${f.gravita})
Titolo: ${f.titolo}
Dove: ${f.dove}
Prova portata dal revisore: ${f.prova}
Correzione proposta: ${f.correzione}

## Dove scrivere
Cartella di lavoro tua: ${S}/agents2/controprova-${d.key}-${f.id}-${i + 1}/ (creala solo se ti serve). Rapporto: ${S}/reports2/controprova-${d.key}-${f.id}-${i + 1}.md. Restituisci l'oggetto strutturato richiesto.`,
        { label: `controprova:${d.key}:${f.id}:${i + 1}`, phase: 'Controprova', schema: VERDETTO, model: 'opus' }
      ))).then((votes) => ({ finding: f, votes: votes.filter(Boolean) }))
    ))
    return { key: d.key, review, verified: verified.filter(Boolean) }
  }
)
return risultati.filter(Boolean)
