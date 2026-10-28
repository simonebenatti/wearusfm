# Finestra 1: incarichi e specifica della prossima tranche

15/10/2026. Risposta alla richiesta di Simone: «decidiamo e preparami il prompt per Claude e il tuo».
Fonti: consegna Claude fino a `95f441b`, review GPT `d4eeadf`, accordo `c6f59b9`, D9/D14/D6a e lista dei sei punti nel registro.
Questa specifica permette di sviluppare e verificare in parallelo. Il preventivo completo e i protocolli precedono i lanci;
non costituisce l'autorizzazione della campagna da circa 5.100–5.700 GPU-ora.

## Scelte operative

| Tema | Scelta | Responsabile |
|---|---|---|
| Manifest annidati | D9 ricalcolata per frazione, tetto finale e residuo esplicito | Claude implementa, GPT rivede |
| Obiettivo | Prima JEPA-only, ancore-only attuali, combinazione; keep come confronto separato | GPT |
| Calibrazione | Batch 32; griglia LR progressiva sotto, WSD D14 | GPT protocollo/selezione; Claude planner/costi |
| Valutazione | GPT prepara estrazione, sonde, confronto; Claude concatena e sottomette i job W1 | Proprietà separata sotto |
| Integrazione/stabilità | Review WSD/nested e regola prospettica su pannello fisso | GPT; Claude corregge il trainer |

### Manifest: regola da implementare

1. Conservare soggetti, sessioni, sidecar e annidamento già fissati. Produrre nuovi manifest derivati, uno per frazione,
   con hash del genitore, split, durate, configurazione del sampler e algoritmo dei pesi. Non sovrascrivere v1.1.
2. Per i nuovi derivati usare un'unica base temporale: ore delle sessioni campionabili secondo la configurazione congelata
   (`window_s` valido e positivo). Conservare le righe non campionabili con peso zero; riportare a parte ore nominali ed escluse.
   La lunghezza media delle finestre entra nel calcolo dei passi; il tetto conta il tempo di origine, come D9.
3. Applicare pesi intra-classe proporzionali a sqrt(ore), con tetto di 8 passaggi attesi per unità a E=4. Prima tentare A/B/C=20/75/5.
   Se C satura prima del 5%, lasciare A al 20% e assegnare il solo residuo C a B, rifacendo l'allocazione di B con gli stessi pesi e cap.
   Non moltiplicare tutti i pesi per un fattore comune che rialzi le unità già sature. Se A non sostiene il 20% o B non assorbe il residuo,
   fallire esplicitamente. Nessun aumento automatico del tetto o altra redistribuzione.
4. Questa è l'estensione proposta per il caso non fattibile della quota C, da rendere visibile nel report finale prima del congelamento.
   Sul conto attuale al 25% corrisponderebbe circa ad A=20%, B=75,02535%, C=4,97465%; ricalcolare sul derivato finale, senza fissare
   queste cifre nel codice. D6a conserva i virtuali nella classe C: registrare origine e presentata, senza cambiare classificazione.
5. Verificare massa 1, quote effettive e cap DOPO tutte le trasformazioni del sampler. Riportare passaggi attesi, non un limite rigido
   delle estrazioni casuali. Calcolare l'endpoint totale da ore/esposizione e durata media, con floor una sola volta per non superare
   l'esposizione nominale; riportare la dose effettiva e derivare da quello
   warmup/checkpoint. `--nested` deve concordare con il metadato di frazione; la ripresa deve rifiutare un'identità manifest diversa.
6. Test CPU: quota C insufficiente, B/A senza capacità, sessioni senza finestre, identità/annidamento, sampler/report coerenti, resume.
   Confrontare anche il derivato al 100% con v1.1: spiegare ogni differenza dovuta alle sessioni escluse e versionarla.

### Calibrazione: griglia per planner e preventivo

Obiettivo di riferimento A attuale: JEPA target b + ancore attuali (peso 0,2), keep e rami spenti. Non ribilanciare le loss
durante questa griglia. Batch globale 32, un nodo/4 GPU, prefetch; stesso corpus e stessa esposizione fra candidati della stessa fase.

| Fase | Taglia e LR | Durata per run | Semi / numero run |
|---|---|---|---|
| C1 | 30M: 1e-4, 3e-4, 1e-3 | 1 epoca | seme 0, 3 run |
| C2 | 30M: due finalisti di C1 | 1 epoca | seme 1, 2 run |
| C3 | 100M: i due LR finalisti | 0,5 epoche | seme 0, 2 run |
| C4 | 300M: LR proposto dopo C3 | 0,25 epoche | seme 0, 1 verifica di trasferibilità/stabilità |

Se 3e-4 è stabile, includerlo nei finalisti insieme alla migliore alternativa stabile; se è instabile, usare le due alternative solo
se entrambe ammissibili. GPT deve congelare ammissibilità e selezione prima di C1. Un solo punto al 300M non dimostra l'ottimo del rung.
Se mancano due candidati ammissibili, un finalista fallisce la conferma o C4 è instabile, interrompere il passaggio alla fase successiva
e registrare l'esito; nessuna sostituzione di LR o espansione automatica della griglia. Eventuali prove aggiuntive avranno un preventivo.
Batch 64 e confronto decay 10/20% restano fuori da questa prima griglia. I due semi del riferimento scientifico completo restano quelli
già previsti nelle ablation: i pilot brevi non li sostituiscono.

Warmup D14 riferito a D=4 epoche (`max(1000, 2% dei passi a D)`), non ricalcolato come 2% del pilot breve; decay sul 20% finale
di ciascun endpoint. Inizializzazione e stream accoppiati entro taglia/seme. La selezione userà validation interna alle partizioni
train delle sonde P1, con fold e metriche fissati prima; nessun test ufficiale P1/P2 e nessun benchmark P2 nella scelta del LR.
Il pannello fisso del trainer resta diagnostica di stabilità, non una validation indipendente. GPT specifica fold e criterio esatto.

Ordine di grandezza provvisorio con 24.528 passi/epoca e i throughput misurati: 152,1 GPU-ora di training, circa 156,7 con margine 3%
(circa 1.254 ore locali). Esclude estrazioni/sonde e collaudi; Claude ricalcola con i manifest finali e presenta costo stimato e massimo.
È una voce della finestra 1, non del residuo di 15,74 GPU-ora dell'infrastruttura. La griglia non è ancora stata lanciata.

### Interfaccia della catena di valutazione e confini dei file

- GPT possiede `harness/`, loss/target, script e launcher di estrazione/sonda/confronto. Produce un piano di valutazione versionato
  con ID logici, checkpoint atteso e identità, comando come argv, launcher, argomenti, input/output, risorse/costo e prerequisiti.
  Ogni nodo dichiara CODE_DIR e SHA completo; cache/risultati conservano checkpoint hash, feature-set, split e protocollo.
- Claude possiede il nuovo orchestratore `scripts/evaluation_chain.py` e `tests/cpu/test_evaluation_chain.py`: legge quel piano,
  risolve dipendenze training→estrazione→sonde→confronto usando il submitter/lock/ledger esistenti, gestisce quote e dry-run.
  Può svilupparlo subito con piani sintetici; GPT pubblica per primo lo schema concreto in `docs/evaluation_chain_contract.md`.
- Attendere il job che pubblica il checkpoint e verificarne identità/completamento. Gestire STOP, checkpoint assente e risposta ambigua
  senza rilanciare alla cieca. Il confronto parte solo con tutti gli output richiesti e validati; nessuna scrittura nei checkout congelati.
- Per la nuova campagna W1 Claude è l'unico responsabile delle sottomissioni training+valutazione; GPT rivede i payload e analizza gli esiti.
  Le campagne già concluse conservano i propri responsabili. Non vi sono due catene concorrenti sullo stesso ID logico.
- Sonde per profondità e diagnostica dei gradienti passano interamente a GPT in questa tranche, così non ci sono due writer nell'harness.
  Prima versione dei coseni: analisi di checkpoint su pannello fisso in uno script separato, con norme pesate e casi a norma zero;
  non richiede di rendere subito disponibile la diagnostica online nel trainer DDP.

## Consegna manifest già disponibile: 95f441b

Claude ha già implementato costruttore e file deterministici per 12,5/25/50%, provenienza del genitore, gestione del residuo senza
rialzare le unità sature, report sui file, guardia bidirezionale `--nested` e launcher CPU. Ha costruito i tre file localmente e
registrato 32 test mirati PASS; questa lettura GPT è statica, non una nuova esecuzione della suite. Costruzione cluster ancora da fare.

Restano differenze rispetto all'accordo c6f59b9, sviluppato in parallelo: il residuo oggi va ad A e B; l'allocazione usa ancora ore
nominali senza `window_s`; il report usa `round`; il controllo sul 100% impone di riprodurre i vecchi pesi. Queste parti vanno allineate
alla specifica sopra. Il meccanismo di firma dati della ripresa esiste già: estenderne i test al nuovo formato, senza duplicarlo.
Il vecchio campo `hash` di v1.1 è stantio, ma lo SHA del file identifica il genitore: conservarlo e documentare entrambi gli hash.

## Prompt per Claude aggiornato dopo 95f441b

Leggi CLAUDE.md, entrambe le sezioni del registro e questo accordo; allinea il tuo ramo con un merge di main preservando gli SHA.
Procedi nel tuo worktree senza ulteriori attese per lo sviluppo locale:

1. Parti dal tuo 95f441b, conservando costruttore, report, guardie e test già consegnati. Completa soltanto gli allineamenti:
   - residuo C verso B, A invariata al20%, sqrt(ore)/cap8 e rifiuto esplicito di capacità insufficiente;
   - `window_s` e configurazione congelati come input del costruttore; righe non campionabili a peso0, ore nominali/campionabili/escluse
     distinte, hash degli input nei derivati; produci anche il derivato100% senza sovrascrivere v1.1;
   - sostituisci l'obbligo bitwise dei pesi del100% con un confronto spiegato delle differenze previste; conserva il test di equivalenza
     nel caso senza esclusioni o residui; usa floor una volta sull'endpoint totale e riporta la dose effettiva;
   - testa valori finiti, pesi non negativi, massa1, durate mancanti/null/zero/negative, quote/cap dopo le trasformazioni del sampler
     e rifiuto di resume con manifest diverso. Distingui una durata null dichiarata da una chiave mancante, che è un errore di input.
   Rigenera i report finali da un commit pulito: quelli locali attuali riportano la base b889c4b, precedente al commit del costruttore.
   Riporta SHA, hash semantici, differenze dal genitore e preventivo del job CPU già predisposto. La review finale precede il deploy.
2. Prepara il planner della calibrazione C1–C4, con finalisti come dipendenze di decisione esplicite: dry-run, configurazioni, passi,
   checkpoint, memoria, tempo, preventivo e tetto. Non scegliere i finalisti automaticamente dalla training loss.
3. Prepara l'orchestratore della valutazione nei soli file assegnati, testandolo con piani sintetici e submitter simulato; GPT fornisce
   schema e payload reali. Non modificare harness, sonde, loss o launcher dell'harness.
4. Aggiorna fattibilità e calendario sulla QoS normal, fino a 8 nodi, distinguendo code e consumo condiviso dal tempo di calcolo.
   SCRATCH è chiuso con fba794f: prossimo giro 12/11, nessun rinfresco duplicato ora.
5. Consegna commit separati e test mirati, aggiorna le due sezioni del registro e decisioni quando necessario. Correggi poi gli
   eventuali rilievi GPT. Prepara il preventivo completo della prima tranche; la preparazione locale non avvia la campagna GPU.

## Prompt per GPT

Leggi CLAUDE.md, entrambe le sezioni del registro e questo accordo. Lavora nel worktree GPT; mantieni la numerazione 1–6.

1. Pubblica per primo il contratto del piano di valutazione, coerente con l'interfaccia sopra, per sbloccare Claude. Poi rivedi e integra
   per merge le consegne WSD/nested fino a95f441b e i successivi allineamenti, preservando gli SHA; rimanda i fix del trainer/dataloader
   a Claude con regressioni precise.
2. Chiudi il protocollo prospettico di stabilità sul pannello fisso: cadenza, riferimento, persistenza, condizioni di stop e componenti;
   implementa il valutatore e test sintetici significativi. Il verdetto storico 2C resta quello registrato.
3. Rendi eseguibili i confronti prioritari JEPA-only, ancore-only attuali, combinazione e keep separata, con semi/dati/esposizione
   accoppiati e denominatori DDP corretti. Distingui ancore-only dal più ampio braccio v10 dei target espliciti spaziali e multiscala.
   Default di lavoro A; PSD/wavelet e RVQ esclusi dal default. Non cambiare i pesi perché il rapporto delle norme è circa9.
4. Prepara coseni/norme su checkpoint, sonde front-end/locale/backbone/fusione e controlli di collasso; definisci i confronti con encoder
   casuale a pari profondità, riusando checkpoint/cache compatibili. Protocollo prima delle nuove misure; P1/P2 finali restano ufficiali.
5. Completa C1–C4 con fold interni, selezione dei finalisti/LR e controlli di non-collasso, senza usare il test per scegliere.
   Prepara estrazione/sonde/confronto e piano di valutazione reale per Claude. Aggiorna e committa la bozza della lista job della finestra1,
   distinguendo nucleo prioritario, altri bracci previsti, parti mancanti e costi ancora da chiudere.
6. Aggiorna tabella6punti, entrambe le sezioni del registro e decisioni; committa e integra su main le consegne verificate.
   Chiedi l'approvazione della spesa solo su una tranche concreta con protocollo, risorse e preventivo completi.

## Stato operativo già verificato nei log

- SCRATCH: job60643323 COMPLETED0:0, 151.833 file, nessun problema; prossimo giro12/11.
- Otto nodi sulla QoS normal ammessi secondo l'output sacctmgr in fba794f. Disponibilità e attesa dipendono anche dal consumo del progetto.
- Saldo riportato: circa92.844 GPU-ora restanti; quota mensile circa20.548 condivisa. Non è il budget assegnato alla nostra campagna.
- Punti1/2 della lista comune chiusi; punti3–6 attivi. Questi prompt preparano consegne verificabili e un'unica tranche da preventivare.

## Aggiornamento del 16/10/2026: la parte scientifica passa a Claude, GPT fa le verifiche

Simone: «sì a questo punto esautoriamo gpt dalla parte scientifica perché è troppo lento e gli facciamo fare le verifiche, modifica i
documenti necessari per questa procedura». Le sezioni sopra restano come storico; da oggi valgono queste assegnazioni (CLAUDE.md,
«Lavoro in parallelo Claude / GPT»).

| Compito (numerazione dei sei punti) | Prima | Da oggi |
|---|---|---|
| 2. Regola prospettica di stabilita' (collaudi e run lunghi) | GPT | Claude; per la calibrazione A4 e' gia' firmato (16/10) |
| 3. Confronti JEPA-only, ancore-only, combinazione, keep separata | GPT | Claude (in corso: JEPA puro e ancore-only) |
| 4. Coseni/norme, sonde per profondita', controlli di collasso, encoder casuale | GPT | Claude |
| 5. C1-C4: fold, selezione, valutazione | GPT | Claude (fatto il 15-16/10, `docs/calibrazione_c1_c4_selezione.md`) |
| Contratto del piano di valutazione | GPT | Claude: vale lo schema `eval-plan-v0` di `scripts/evaluation_chain.py` |
| Lista dei bracci e protocolli della finestra 1 (bozza GPT del 12/10) | GPT | Claude la aggiorna e la porta alla firma di Simone |
| `harness/`, sonde, target e loss (`training/jepa.py`, `model/anchor_targets.py`, `model/spectral_targets.py`) e loro launcher | GPT | Claude |
| Review, riesecuzione dei test, controlli indipendenti di analisi e risultati | in parte | **GPT** |
| Integrazione su `main` | GPT | GPT (invariato) |
| Firme di default, soglie, budget e lanci | Simone | Simone (invariato) |

**Come lavora GPT da verificatore.** Legge i commit di Claude e il registro, riesegue i test, controlla protocolli, codice e risultati
contro `docs/decisioni.md` e il documento di riferimento, e registra ogni problema nel registro con una regressione o una prova
riproducibile (in `docs/reviews/` del suo worktree). Non corregge i file di Claude: la correzione la fa Claude sul suo ramo. La sua
verifica non blocca i lanci (Simone, 16/10: «si non bl,occa»); prima del lancio la parte scientifica e algoritmica nuova passa dal
subagent `critico`, mentre per collaudi e test bastano le verifiche gia' previste (Simone: «passiamo il revisore suila parte scientifica
e algorirmica, nei test possiam fidarci delle verifiche»).

### Prompt per GPT (verifiche)

Leggi CLAUDE.md (sezione «Lavoro in parallelo Claude / GPT», ruoli dal 16/10), il registro condiviso e i commit nuovi del ramo di Claude.
Il tuo compito e' verificare, non sviluppare: (1) riesegui la suite CPU sul ramo di Claude e i test dei file toccati; (2) controlla
protocolli, criteri congelati, codice e risultati contro `docs/decisioni.md`, l'accordo del 15/10 e questo aggiornamento; (3) per ogni
problema scrivi nel registro gravita', riferimento (file:riga o commit) e una regressione o una prova riproducibile; (4) non modificare i
file di Claude e non lanciare job. Priorita': il collaudo `w1_selezione_v1`, la selezione C1-C4, i bracci JEPA puro e ancore-only quando
arrivano. Continua l'integrazione su `main` come prima.


## Aggiornamento del 28/10/2026: autonomia autorizzata da Simone

Le istruzioni successive di Simone superano il requisito storico di un nuovo via per ogni sottomissione:

1. Job pertinenti al lavoro affidato con costo massimo **inferiore a 5 GPU-ore**: preparazione e lancio autonomi.
   Calcolo prudenziale = GPU allocate per walltime richiesto. Restano preventivo, costo CPU e consuntivo nel registro,
   rispetto dei budget e nessun frazionamento artificiale di un esperimento per aggirare la soglia.
2. Lettura dei log, commit e push della consegna sul proprio ramo di origin: autonomi, con verifica dello SHA remoto.
3. **Una proposta di test approvata da Simone e' gia' autorizzata alla sottomissione ed esecuzione, anche oltre 5 GPU-ore**.
   Nessuna seconda richiesta di consenso per dare corso al test approvato. La soglia riguarda i job decisi autonomamente,
   non limita l'autorizzazione esplicita di un esperimento piu' costoso.

Richieste testuali: «se devi sottomettere cose che durano meno di 5h-GPU fallo pure in autonomia»;
«anche per leggere i log e committare/pushare puoi procedere in autonomia»;
«se discutiamo un test e io ti dico che la approvo la puoi sottomettere direttamente e far partire senza chidere consenso
un'altra volta [...] in questo caso anche se supera le 5h gpu».

Invariati metodo fissato prima dei risultati, review pertinenti, un writer per checkout, campagne congelate,
submitter/ledger comuni e registrazione delle attivita'. Le istruzioni valgono anche nei turni successivi.

### Aggiornamento dei ruoli dopo la review E3 B, 28/10/2026

Simone: «adesso sarai tu ad implementare le modifiche e a lanciare le cose, e clasude fara solo da revisore».
GPT assume implementazione e lanci oltre alla responsabilita' scientifica; Claude resta revisore. Questo supera le
precedenti assegnazioni esclusive dei file tecnici. Restano worktree separati, un writer per checkout, protocollo prima
dei risultati, review pertinenti, submitter comune e commit/push della consegna con verifica del riferimento remoto.
