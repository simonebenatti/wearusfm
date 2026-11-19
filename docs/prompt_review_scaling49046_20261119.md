# Review unica: continuazione300M19.618→49.046 e valutazione

Pacchetto sul ramo `codex/scaling-49046-launcher-20261118`. Lo SHA completo e
il confronto con origin sono nella consegna del launcher. Simone19/11 ha
chiesto alla chat scientifica di rivedere il piano e, se favorevole, farlo
partire senza altro consenso. Non chiedo una nuova decisione sullo scaling.
Nessun job/deploy49k è stato eseguito. La chat launcher rimane operatore unico.

Leggi prima `docs/scaling_49046_execution_20261119.md`, poi:

1. `scaling_continue49046.py`, launcher e `scaling_build49046.py`: due segmenti,
   torch-first, intero stato, sorgente immutabile, pin effettivi e ricezione del
   predecessore. Piani in `results/infra/scaling_49046_20261118/evaluation/training/`.
2. `scaling_plans49046.py`, `scaling_operations49046.py`, `scaling_terminal_builders.py`:
   gate→lines, prerequisiti, quote, ledger, attestazione, piani futuri solo da prove
   effettive. Segmento2 inviato dopo raccolta segmento1, non accodato in anticipo.
3. Derivazione dei cinque valutatori con hash delle sorgenti19k, controllo30M
   originale e finalizzazioneP1-r2 nel percorso di estrazione reale. Launcher
   estrazioni legge il piano dal checkout metadata corretto e richiede entrambe
   le taglie; protocolli budget/launcher coerenti.
4. Archivio persistente dei tre checkpoint, quota/retention, risorse e
   `launcher/control_checkpoint_actual_verification.json`, `live_limits_20261119.json`.
5. Test: `launcher/verification.json` e stdout. I test dei launcher eseguono shell
   e Python embedded reali con scheduler/GPU simulati; il test del decay esegue
   un trainingCPU piccolo reale. Nessuna attestazione di bit-identitàGPU.

Chiusure dei rilievi della pre-review:
- stdout sacct separato dal piano predecessore; test segmento2 e mutazioni.
- Piano estrazione nel checkout-extraction con percorso assoluto e hash;
  sei celle richieste anche nel guard embedded reale.
- Contratto estrazioni6job/14GPUh allineato al budget. Controllo non riaddestrato,
  cache riestratte esplicitamente: adapter linea2 non certificato in questa consegna.
- Protocollo risorse posa identico fra builder e launcher.
- `EVIDENCE_DIRS` definito, due trainingstage registrati, documenti iniziali
  assegnati al checkoutCODE. Percorsi raccolta→costruzione provati.
- Prerequisiti cross-stage e costo attivi nel percorso ordinario, non soltanto
  nel vecchio profiloP1. Nessuna riga da cancello fallito/vecchio/mutato.
- Archivio è un job concreto incluso nella catena e nel conto.
- Primari invariati. Secondario dose×taglia deliberatamente descrittivo,
  comparableFalse e nessunIC in assenza di certificazione inter-dose completa.
- Piano training canonico unico consumato; mirror obbligatoriamente uguali e
  attestati. Costo119GPUh ricalcolato dai due job effettivi.

Budget: training119GPUh richieste/119,4 con margine; sei estrazioni14/14,3;
CPU-only massimo347,8 su400, inclusi cinque checkout e archivio. CPU allegate
ai jobGPU1069,6coreh con margine, separate. I tetti di fase non si estendono
con la riserva. PreservazioneWORK≈13,06GiB, quota45,95GiB libera nella lettura
19/11, gate fresco minimo30GiB; SCRATCH minimo20GiB e orizzonte19/12.

Risposta attesa: APPROVATO o DA RIVEDERE, rilievi bloccanti separati dai suggerimenti.
Con APPROVATO il launcher esegue la catena con i cancelli meccanici senza altre
review per invio né consenso di Simone. Primo errore reale/TIMEOUT/costo fuori
piano ferma lo stadio; nessun retry automatico. Nessuna prosecuzione oltre49.046,
nuovo seme/target/loss/finetuning o training30M autorizzato da questo pacchetto.


Verifiche: distinguere la regressione estesa881pass/1fallimento storico legato
al ledger (riprodotto su2df4ca68) dalla verifica finale mirata sui byte finali.
Non è dichiarata una suiteCPU intera verde. Il vecchio test2worker locale era
stato interrotto in un giro più ampio separato; nessun effetto sui job remoti.
Ultima correzione di robustezza: dimensioni dei checkpoint futuri tratte dalle
ricevute effettive, non hardcoded dal19k; la fixture esercita una dimensione
diversa. Budget e scienza invariati.


Esito finale:250test passati in178,24s, source-bound; inventario operativo
e integrità dei103file scientifici pubblicati.56ID massimi univoci e assenti
dal ledger alla preparazione. Nessun49k inviato/deployato.
