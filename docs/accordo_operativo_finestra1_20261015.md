# Finestra 1: incarichi e specifica della prossima tranche

15/10/2026. Risposta alla richiesta di Simone: «decidiamo e preparami il prompt per Claude e il tuo».
Fonti: consegna Claude fino a `fba794f`, review GPT `d4eeadf`, D9/D14/D6a e lista dei sei punti nel registro.
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

## Prompt per Claude

Leggi CLAUDE.md, entrambe le sezioni del registro e questo accordo; allinea il tuo ramo con un merge di main preservando gli SHA.
Procedi nel tuo worktree senza ulteriori attese per lo sviluppo locale:

1. Implementa i manifest derivati secondo la regola sopra, incluso residuo C→B con cap invariato. Correggi il report delle epoche,
   aggiungi le verifiche CPU e consegna report leggibile con quote/passaggi finali, differenze da v1.1 e provenienza.
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
   per merge le consegne WSD/nested fino a fba794f, preservando gli SHA; rimanda i fix del trainer/dataloader a Claude con regressioni precise.
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
