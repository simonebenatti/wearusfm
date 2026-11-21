# Rilascio del controllo30M indipendente — 21/11/2026

**APPROVATO**, nessun rilievo bloccante residuo sul codice
`abfaf64e28827793720097254ff2d0271d744a1d`, ramo
`codex/scaling-control30m-20261121`. Simone ha autorizzato preparazione,
verifica e lancio tramite l'unico launcher della linea; non serve un nuovo via.

La modifica è operativa: tre estrazioni30M già previste sono anticipate usando
il checkpoint49.046 esistente. Nessun nuovo training, dato, loss, seme, criterio
o algoritmo. Due revisori indipendenti hanno controllato adapter/join e gate.
Rieseguiti insieme i test nuovi e i cinque file della regressione del rilascio
storico: **359 passati in240,20s**, exit0, sorgenti invariati durante il test.
Il critico ha inoltre rieseguito85test adapter/bridge, tutti passati.

Il loader viene derivato mediante una sola sostituzione dichiarata della guardia
sullo schema, da un sorgente verificato perSHA256. Richiede `control_only is True`,
solo30M, dose49.046 e nessun checkpoint300M fittizio. Riusa l'esecuzione originale,
compresa la guardiaP1-r2, e ripristina il loader anche su errore. Launcher e test
coprono entrambe le classificazioni e la posa, mantenendo risorse e log.

Gate→lines conserva pin, quota, predecessori, stato del ledger e scadenza1800s.
ID e output sono gli stessi del piano originale: il successivo gate comune
riconoscerà i tre job30M e ammetterà soltanto i tre300M mancanti. Il bridge conserva
per ciascun braccio il proprio codice/piano, pretende gli stessi input comuni,
verifica i file30M fra raccolta anticipata e finale e genera piani compatibili
con i lettori87da975. Nessuna riscrittura di cache o provenienze.

Un rilievo operativo è stato corretto prima del commit: la rilettura accounting
richiedevaRSS mentre il collector lo ammetteva assente per questi stadi. Ora sono
coerenti (`need_rss=False`), con regressione che accetta il dato assente ma rifiuta
un accounting alterato. Il valore viene conservato quando presente.

Le7/7,15GPUh del30M restano comprese nelle14/14,3totali. L'unico costo aggiunto è
un checkout2CPU/5min:0,266667coreh con margine. Nuovo massimoCPU-only
**348,066667/400**, fasecheckout6job con tetto1,6. Il consuntivo finale deve includere
il costo effettivo del checkout aggiunto, oltre ai cinque conteggiati dal controller
storico. Nessun allargamento degli altri tetti.

I sorgenti originali del training, controller, estrattori, lettori e submitter sono
immutati. Il300M continua nel checkout87da975; il nuovo codice sta in un checkout
separato. I futuri commit metadata della campagna originale devono rimanere
documentali: il launcher userà il bridge dal nuovo worktree per generare i piani
nel vecchio, senza fondere i nuovi script nel checkout storico.

Evidenze, pin e test:
`results/infra/scaling_49046_20261118/evaluation/evaluation/control30m_early_20261121/review/`.
Procedura e risorse: `docs/scaling_control30m_early_execution_20261121.md`.
Nessun job è stato inviato dalla chat scientifica. Il launcher può procedere con
gate vivi, pubblicazione delle righe e preparatorelocale previsto daAGENTS,
poi checkout e tre estrazioni, raccolta e pubblicazione senza altri consensi.
Primo errore reale: stop e diagnosi, nessun retry automatico.
