# Review GPT dell'implementazione A1 — 06/11/2026

**APPROVATO sul piano tecnico e scientifico. Nessun difetto bloccante trovato.**
Verifica successiva richiesta da Claude, non un nuovo gate GPT prima dei lanci.
Riferimento: `2dea7ca037b55023c50254735ae69eb08945ec72`, ramo
`origin/claude/consolidation-long-exec-20261105`; disegno approvato da Claude
in `7ceedbb5d891168f585229e826baf6535f35c1cc`, registro `e0e915a...`.
Baseline del codice eseguito: `166fb8653b53439507183d76e3e36a79dd28d166`.

Review eseguita su checkout locale detached del commit esatto. Nessun accesso
operativo al cluster, deploy, sottomissione, cancellazione, fit o inferenza.
Nessuna modifica al codice, ai piani o al ledger di Claude.

## Esito sui punti richiesti

1. **Definizioni emendate.** Il diff reale cambia provenienza degli input,
   ammissione e confezione dei job, non i calcoli scientifici. Tutto `src/`,
   `consolidation_long_classification.py`, `consolidation_long_pose.py` e il
   dispatcher `consolidation_long.py` sono byte-identici: 103 file verificati.
   `BENCHMARK_POLICY` e `resource_gate` originari restano invariati. Il gate A1
   riusa i loro costi per fold, applicando la suddivisione e la formula di W
   concordate; non riduce margini, griglia, dose o copertura.

2. **Compatibilita' limitata.** Verificati i 227 sorgenti originali contro i due
   commit di fase e i pin delle 28 ricevute. Il manifest e' fissato nel codice
   per SHA e dimensione: `983df48465531014275b6f5f677e6907a1219f26fbeef529b6489ea85c473c95`,
   65.919 byte. `verified_input` richiede pin esatto, tipo, source e ricetta
   originali, codice/piano della fase e campi di identita'. Il cache manifest
   e la source incorporata nelle NPZ posa conservano le provenienze originarie.
   Tutti i nuovi risultati richiedono invece la nuova source piena
   `77e09626271b1aeded8d7f3df77ef9557115e903a9aca8e280e3f3259527ac88`.
   Non e' introdotta un'accettazione generica di ricevute con source diverse.

3. **Elenco chiuso dei sorgenti.** Le regioni comprendono decoratori, commenti,
   spazi precedenti e fine file; l'ordine delle istruzioni e' vincolato. La
   garanzia su cio' che sta dentro le funzioni dichiarate modificabili e sul
   modulo nuovo e' la review del diff con il pin pieno del nuovo codice,
   non una prova automatica di equivalenza semantica. Questa distinzione e'
   dichiarata correttamente. Non ho trovato cambi scientifici nelle regioni
   effettivamente modificate.

4. **Test degli stadi chiusi.** Sostituire i due test esecutivi di estrazione e
   benchmark della classificazione con test di rifiuto e' coerente: questa
   versione non deve piu' eseguire quegli stadi. Il testo dei due rami resta
   congelato per hash; i moduli scientifici e i test di parita' restano coperti.
   Non e' la rimozione di una funzionalita' che A1 promette di eseguire.

5. **Pin delle fasi.** `pin_phase.py` sostituisce correttamente lo snippet
   precedente. `fit --dry-run` ricostruisce 69 shard ammessi e la posa dalle
   ricevute originali. Verifica aggiuntiva dell'agente operativo su 72 ricevute
   sintetiche: `aggregate` produce i manifest 69+3 e rifiuta un SHA di fase
   errato; `summary` accetta le due comparazioni della fase corretta. I pin non
   sono autoreferenziali: il commit della fase precede il commit documentale
   che registra il suo SHA. Nuovi report e confronti restano sotto una sola
   source; gli input vecchi non vengono riscritti.

6. **Risorse e corsia.** I 69 ID, output e W coincidono con la proposta GPT.
   Confermati 1.276 coreh per i fit e 41 per le altre allocazioni future:
   1.317 future, 1.359,218889 includendo il consumo, oppure 1.451,666667
   conservando anche i tetti delle fasi concluse; tetto 1.711,666667 invariato.
   Nessuna GPU nuova. Posa per prima e corsia gestita da Claude secondo la
   disposizione di Simone; il prerequisito relativo allo scaling va riletto
   dall'operatore, non assunto da uno snapshot storico.

## Verifiche eseguite da GPT

- **342 test mirati passati in 25,51 s**, sugli undici file della consegna,
  compreso l'emendamento. Non ho rieseguito tutta la suite CPU: la claim dei
  2.456 test resta quella di Claude e non viene presentata come verifica mia.
- `build_inputs.py --check`: manifest riprodotto con hash e byte attesi.
- `apply_to_plans.py --check`: sorgente, 103 job di calcolo, 48 ID sostituiti,
  conteggi, costi e tetti coerenti.
- `pin_phase.py fit --dry-run`: gate originale negativo per 15 shard, nuovo
  gate positivo per 69; posa 583,268940615 s per dose.
- `bash -n` del launcher: passato. Nessuna modifica nel checkout di verifica.
- Due revisioni indipendenti: compatibilita' degli input e diff scientifico;
  pinning, matrice e rilascio operativo. Entrambe senza blocchi materiali.

## Margine di tempo e limite preesistente

Il margine minimo oltre la riserva e' **40,479991 s**, nella vista post DB6.
La distanza completa fra P e W e' **640,479991 s**: restano i 600 s di riserva,
oltre ai fattori prudenziali gia' inclusi in P. Non e' una garanzia contro
variabilita' del solver ai checkpoint successivi. TIMEOUT conserva la regola
di arresto, senza rilanci o variazioni automatiche.

Confermo il limite preesistente: `compare-pose` e `summary` creano la directory
prima di tutti i controlli specifici sugli input. **Non richiedo una modifica
del runtime A1 per questo punto.** I pin delle fasi e i controlli in sola lettura
su completezza, provenienza, hash degli artefatti e assenza dell'output riducono
gli errori operativi. Il preflight dei fit, da solo, non esercita tutti i
controlli finali dei confronti: non va descritto come copertura completa.
Un rifiuto tardivo resta da diagnosticare conservando log e directory, senza
riutilizzare automaticamente un ID o un'uscita parziale. Il riordino dei
controlli puo' essere una manutenzione futura fuori dalla source congelata.

## Stato e autorizzazione gia' impartita

Nel commit esaminato `plan.amendment.decision` e' null: il rifiuto del rilascio
ordinario e' verificato dai test. GPT non ha modificato quel campo e non ha
usato la decisione fittizia dei test come autorizzazione reale.

Simone ha gia' affidato a Claude anche l'implementazione di questo emendamento,
rispondendo «questo meglio se lo fa claude, risparmiamo un passaggio, ok ?» al
passaggio che assegnava implementazione e consegna a GPT e lanci a Claude.
La mia interpretazione nel contesto delle autorizzazioni persistenti e' che
Claude debba completare la catena A1 entro ricetta e tetto gia' approvati dopo
i controlli previsti, senza una nuova richiesta del medesimo via. La registrazione
deve citare le parole reali e il contesto, senza inventare una frase «Approvo A1»
mai pronunciata. Questa review non aggiunge un altro passaggio operativo GPT.

## EPN15

Confermata la rettifica documentale separata `c04d10b`: «nessun guadagno risolto»
al posto di «resta dov'era». Numeri e risultati non cambiano. La verifica di
EPN15 e' gia' chiusa in `bb08cd0`; nessuna azione aggiuntiva richiesta.
