export const meta = {
  name: 'rilettura-finale-consegna-sola-lettura',
  description: 'Rilettura finale, con agenti che possono solo leggere file, della sequenza operativa definitiva e del brief di consegna del lancio',
  phases: [
    { title: 'Rilettura', detail: 'tre lettori in sola lettura', model: 'opus' },
    { title: 'Controprova', detail: 'uno scettico in sola lettura per ogni rilievo serio', model: 'opus' },
  ],
}

const S = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe2'
const SEQ = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/contrastive_cn_restart_probe_launch_20261127.md'
const BRIEF = S + '/brief_launcher_v2.md'
const RW = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release'

const COMUNE = `
## Il tuo ruolo in questo incarico
Ignora, per questo incarico, il compito abituale di verifica di fonti esterne: qui devi rileggere due testi operativi contro file locali. Hai solo lo strumento per LEGGERE file (Read). Non cercare sul web, non ti serve. Non puoi e non devi eseguire comandi.

## Contesto
Progetto WearUsFM, linea 3. GPT ha preparato un diagnostico («prova mirata InfoNCE al passo 545»): un job di checkout CPU e un job da 4 GPU per 15 minuti sul cluster Leonardo. Il revisore Claude lo ha approvato e ha creato il commit di release b58a66f695c8744dfa5fe633829cafe13a5a1fc4. Simone, il responsabile, ha deciso che il lancio lo esegue un'altra chat Claude («0- leggi le tabelline»), che gira su un modello più piccolo e con meno contesto. Riceverà SOLO due testi: un messaggio di consegna (il «brief») e un file con la sequenza operativa. È ammesso un solo tentativo per ciascun ID di job: un errore nei due testi può bruciarlo.

Una prima verifica, fatta da cinque revisori, ha prodotto 52 rilievi. La sequenza è stata riscritta per recepirli. Durante quella verifica uno dei revisori ha eseguito per errore il comando di invio del checkout con un ssh finto: nessun job è arrivato a Leonardo, ma nel registro comune dei job l'ID del checkout è rimasto prima «ambiguo» e poi, dopo la riconciliazione fatta dal revisore, «assente». Per questo la sequenza definitiva invia il checkout con l'opzione --dopo-assenza-verificata.

## I due testi da rileggere (versioni definitive)
- Sequenza operativa: ${SEQ}
- Brief (il segnaposto @SEQ_SHA@ verrà sostituito con lo SHA del commit che contiene la sequenza): ${BRIEF}

## Fonti da leggere con Read (percorsi completi)
Codice e documenti al commit di release:
- protocollo di GPT: ${RW}/docs/contrastive_cn_restart_probe_20261121.md
- piano: ${RW}/results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/plan.json
- driver: ${RW}/scripts/diagnostics/contrastive_cn_restart.py
- submitter: ${RW}/scripts/submit_job.py
- script SLURM: ${RW}/scripts/slurm/cn_restart_probe.sbatch ; ${RW}/scripts/slurm/campaign_checkout.sbatch ; ${RW}/scripts/slurm/lib_code_dir.sh
- runner (la stampa di --validate-only è attorno alle righe 318-327): ${RW}/scripts/contrastive_run.py
Prove raccolte dal revisore il 27/11:
- blocco A eseguito su Leonardo: ${S}/leonardo/blocco_A_20261127.out.txt
- blocco B eseguito su Leonardo: ${S}/leonardo/blocco_B_20261127.out.txt
- nessun job su Leonardo dopo l'incidente: ${S}/leonardo/dopo_incidente_20261127.out.txt
- file storici su WORK confrontati con l'archivio: ${S}/leonardo/storici_su_work_20261127.out.txt
- stato del registro per l'ID del checkout dopo la riconciliazione: /Users/simonebenatti/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json
- prova a secco dei due invii (ssh finto, registro temporaneo): ${S}/prova_a_secco_invii/chiamata_1_argomenti.txt (checkout) e ${S}/prova_a_secco_invii/chiamata_4_argomenti.txt (job GPU)
- prova a secco del checkout con --dopo-assenza-verificata su una copia del registro: ${S}/prova_a_secco_assenza/argomenti.txt e ${S}/prova_a_secco_assenza/registro/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json
Rapporti della prima verifica (i 52 rilievi):
- ${S}/reports2/d1-comandi.md ; ${S}/reports2/d2-codice-cluster.md ; ${S}/reports2/d3-operatore.md ; ${S}/reports2/d4-regole.md ; ${S}/reports2/d5-completezza.md
Le regole del progetto (CLAUDE.md) sono già nel tuo contesto.

## Che cosa restituire
Scrivi in italiano semplice. Per ogni rilievo indica: gravità («bloccante» = seguendo i testi alla lettera il lancio fallirebbe, brucerebbe un tentativo o violerebbe una regola; «importante» = rischio concreto di errore dell'operatore o informazione sbagliata; «minore» = imprecisione che non cambia l'esito; «nota» = osservazione senza richiesta), dove (passo o riga), la prova (file e riga letti) e la correzione proposta in una o due frasi. Niente questioni di stile. Elenca anche le verifiche positive più importanti che hai fatto davvero e ciò che non hai potuto verificare leggendo.
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
    non_verificabile: { type: 'array', items: { type: 'string' } },
  },
  required: ['sintesi', 'rilievi', 'verifiche_positive', 'non_verificabile'],
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

const LETTORI = [
  { key: 'f1-comandi-e-codice', compito: `
## La tua parte: comandi, parametri e codice
Rileggi ogni blocco di comandi della sequenza (passi 1-8) contro il codice e il protocollo.
- I due comandi di invio (passi 4 e 6): confrontali argomento per argomento con i comandi del protocollo di GPT e con ciò che il submitter fa (lettura degli argomenti, --script-stdin, composizione della riga di sbatch, --dopo-assenza-verificata, funzione reserve e tabella TRANSIZIONI). Confrontali con i file della prova a secco. Le differenze dichiarate nella sequenza sono: --agente claude, invocazione diretta del submitter condiviso, SHA scritto per esteso, --dopo-assenza-verificata nel solo checkout. Ogni altra differenza è un rilievo.
- Con lo stato attuale del registro (leggi state.json), il comando del passo 4 verrà accettato dal submitter? E quello del passo 6?
- Parametri: corrispondono a plan.json (allocation, logical_ids, output) e alle righe #SBATCH e ai controlli degli script sbatch?
- Blocco della validazione (passo 5): è il blocco del protocollo più le sole tre righe dichiarate? Le uscite attese descritte nella sequenza corrispondono a ciò che stampano lib_code_dir.sh, contrastive_run.py e contrastive_cn_restart.py?
- Blocchi A e B e blocco di verifica del checkout: i criteri scritti nella sequenza corrispondono alle uscite vere dei blocchi A e B del 27/11?
- Esiti del job GPU descritti al passo 6: corrispondono a ciò che fanno cn_restart_probe.sbatch e run() nel driver?
- Passo 7: i percorsi remoti e i nomi dei file corrispondono a quelli che il codice scrive (cartella di uscita del piano, log degli script sbatch)?` },
  { key: 'f2-operatore-letterale', compito: `
## La tua parte: l'operatore che esegue alla lettera
Leggi PRIMA soltanto il brief e la sequenza, mettendoti nei panni di un operatore con meno contesto che esegue alla lettera. Annota, passo per passo, dove resteresti con un dubbio o dove un'esecuzione letterale produrrebbe un effetto indesiderato.
Poi usa le fonti per sciogliere i dubbi e classifica ciò che resta. Cerca in particolare:
- istruzioni che ammettono due letture, criteri di arresto mancanti, valori attesi non specificati;
- segnaposto da sostituire a mano (per esempio un jobid) non spiegati con chiarezza;
- incoerenze fra brief e sequenza: numeri, nomi, ordine dei passi, vincoli presenti in uno solo dei due testi;
- che cosa farebbe l'operatore se Simone non risponde alla richiesta di via; che cosa può fare intanto;
- se la descrizione dell'incidente e dell'opzione --dopo-assenza-verificata è abbastanza chiara da non spingere l'operatore a toglierla, a ripeterla nel job GPU o a usare altre opzioni di rilancio;
- se è chiaro dove salvare le uscite, che cosa committare, su quale ramo e in quale worktree, e che cosa NON committare.` },
  { key: 'f3-rilievi-chiusi-e-fatti', compito: `
## La tua parte: i 52 rilievi sono stati recepiti? I fatti dichiarati sono veri?
1. Leggi i cinque rapporti della prima verifica. Per ciascun rilievo classificato bloccante o importante (dal revisore o dagli scettici), e per i minori che toccano comandi o criteri di arresto, di' se la sequenza definitiva e il brief lo chiudono, in quale punto, oppure se resta aperto. Segnala come rilievo solo ciò che resta aperto e conta.
2. Verifica sulle prove i fatti dichiarati nei due testi: la descrizione dell'incidente (orari, stati, nessun job su Leonardo: confronta con state.json e con dopo_incidente_20261127.out.txt), lo stato attuale del registro, le letture di quota, saldo e coda (blocco B), lo stato del repository canonico (blocco A), il confronto dei 27 file storici, le impronte del submitter, i numeri di costo (1,0; 8,2667; 1,4; 11,8667; 12,26; 114,0011; 11,86; 110,4011; 4,74; 47,3322; 18; 169,6: rifai le sottrazioni).
3. Verifica che le frasi attribuite a Simone siano identiche nei due testi e che nessuna frase gli attribuisca più di quanto è citato.
4. Regole del progetto: riferimenti di deploy, un solo operatore, submitter e ledger comuni, nessun retry cieco, output fuori dalla working tree del repository sul cluster, dati da wearable mai in un commit, limiti del nodo di login, registro condiviso, commit mai su main. La sequenza le rispetta?` },
]

const risultati = await pipeline(
  LETTORI,
  (d) => agent(COMUNE + d.compito + `\n\nRestituisci l'oggetto strutturato richiesto.`,
    { label: `rilettura:${d.key}`, phase: 'Rilettura', schema: SCHEMA, model: 'opus', agentType: 'fact-checker' }),
  async (review, d) => {
    if (!review) return { key: d.key, review: null, verified: [] }
    const seri = review.rilievi.filter((f) => f.gravita === 'bloccante' || f.gravita === 'importante')
    const scelti = seri.slice(0, 5)
    if (seri.length > scelti.length) log(`${d.key}: ${seri.length - scelti.length} rilievi seri oltre il quinto non passano dalla controprova`)
    const verified = await parallel(scelti.map((f) => () => agent(
      COMUNE + `
## La tua parte: controprova di un rilievo
Un lettore ha segnalato il rilievo qui sotto sui due testi. Il tuo compito è provare a CONFUTARLO leggendo le fonti: il fatto è vero? Se è vero, cambierebbe davvero il comportamento dell'operatore o l'esito del lancio? Se non riesci a confutarlo, di' che regge e indica la gravità corretta secondo le definizioni date sopra (oppure «infondato»). Non proporre rilievi nuovi.

Rilievo ${f.id} (gravità dichiarata: ${f.gravita})
Titolo: ${f.titolo}
Dove: ${f.dove}
Prova portata: ${f.prova}
Correzione proposta: ${f.correzione}

Restituisci l'oggetto strutturato richiesto.`,
      { label: `controprova:${d.key}:${f.id}`, phase: 'Controprova', schema: VERDETTO, model: 'opus', agentType: 'fact-checker' }
    ).then((v) => ({ finding: f, votes: v ? [v] : [] }))))
    return { key: d.key, review, verified: verified.filter(Boolean) }
  }
)
return risultati.filter(Boolean)
