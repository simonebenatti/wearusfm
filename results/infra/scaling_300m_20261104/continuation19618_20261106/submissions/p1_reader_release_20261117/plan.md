# Lettori 19.618 dopo P1-r2 — rilascio operativo 17/11/2026

Simone: «ok, collega il controller a P1-r2 e procedi con validazione e benchmark,
poi lancia i fit se è tutto a posto. Senza chiedermi di nuovo il consenso,
comitta e pusha tutto».

GPT e' l'operatore di questa catena autorizzata; submitter e ledger restano
quelli comuni. Il successo P1-r2 e' accettato in 267e692. Il rilascio operativo
collega il controller locale alle tre estrazioni effettive, poi esegue i
controlli gia' previsti senza chiedere di nuovo autorizzazione a ogni passaggio.

## Codice, documenti e responsabilita'

Il codice numerico e il dispatcher eseguiti sul cluster rimangono quelli del
checkout `scaling_300m_evaluation19618_20261106_v1`, commit
`df2de35c05381e7a5a69edde409171b210a3b5d8`. Il controller Mac aggiornato ha
una review indipendente e un profilo esplicito per la catena P1-r2. Il
percorso storico e i due tentativi falliti rimangono riproducibili.

I tre checkout documentali residui (`plans`, `budgets`, `inputs`) discendono
da df2de35 sul ramo `codex/scaling-reader-metadata-20261117`. Vi si aggiungono
soltanto documenti/evidenze sotto i percorsi gia' ammessi. Nessuna eccezione
generica a `release_check` o `rereview_check`, nessun nuovo checkout del codice
numerico. L'identita' del controller Mac rivisto e' registrata separatamente
nei verbali, senza attribuirla al codice remoto congelato.

P1-r2: job 64620434, codice e4ebdbfa, ricevuta f5bc05c3. Posa/P2: job 64571999/
64572001, codice df2de35. Tutti a 19.618, checkpoint 300M d7ebe9d8. Il join
verificato mantiene l'identita' per cache e le sette cache del controllo 30M.
Gli ingressi di riferimento sono pubblicati in
`results/infra/scaling_300m_20261104/p1_recovery_readout_20261117/reader_inputs/`.

## Sequenza e tetti

| Fase | Job | Risorse per job | Tetto core-ore con margine |
|---|---:|---|---:|
| Checkout plans | 1 | 2 CPU, 4 GiB, 5 min | 0,266667 |
| Validazione posa | 1 | 2 CPU, 30 GiB, 30 min | 1,10 |
| Benchmark posa | 3 | 8 CPU, 30 GiB, 10 min | 5,20 |
| Benchmark classificazione | 15 | 8 CPU, 30 GiB, limiti 10–120 min gia' congelati | 76,666667 |
| Checkout budgets | 1 | 2 CPU, 4 GiB, 5 min | 0,266667 |
| Ridge posa | 1 | 8 CPU, 30 GiB, 60 min | 8,40 |
| Fit classificazione | <=15 | 8 CPU, 30 GiB, <=3 h; copertura 60 celle/dataset/fold | <=246 |

Restano prenotati gli aggregati e il confronto futuri con l'ultimo checkout,
nei tetti congelati. Questa consegna operativa procede fino ai fit: non
introduce un nuovo obiettivo scientifico o altri training/estrazioni.
La ridge resta il lettore della posa previsto, senza variazioni di loss o dati.

L'inviluppo globale CPU-only e' **351 core-ore**. Riserva storica prudente
4,792778 + residuo massimo 345,166667 = 349,959445. Quel residuo comprende gia'
i tre checkout documentali; non vengono ricontati. I job completati accettati
sostituiscono la propria prenotazione con il consumo reale; job non iniziati
o attivi restano al massimo. I tetti di fase restano invariati: massimo 15
shard, 3 h ciascuno, 240 core-ore richieste per i fit; nessun ampliamento implicito.
Zero GPU per questa catena CPU. Margine scheduler 180 s per job.

## Controlli operativi

- Rifiuto di ingressi che non discendono dall'accettazione P1-r2, di un diverso
  checkpoint/produttore e di qualunque adozione dei P1 falliti.
- Vecchio code-checkout Claude riconosciuto dal ledger; nuovi job GPT con ID
  specifici della stessa campagna. Nessuna falsificazione dell'operatore.
- Validazione conclusa prima dei benchmark. Budget/fit soltanto dopo tutti i
  benchmark richiesti, completi, convergenti, stesso contesto numerico e RSS
  nel limite. Nessun risultato scientifico scelto durante i benchmark.
- Ogni invio segue un gate fresco: checkout/pin/pubblicazione, evidenze,
  assenza uscite e duplicati, quota WORK >=5 GiB, ledger, budget globale e di fase,
  massimo 4 job della linea insieme, massimo 10 job condivisi nella QoS seriale.
- Tranche in catena afterok; la successiva attende COMPLETED 0:0 di tutta la
  precedente. PENDING/RUNNING/COMPLETING significano attesa, non fallimento.
- Stop al primo errore reale, rifiuto, TIMEOUT o sforamento; raccolta e diagnosi,
  nessun retry. Non si toccano job o prenotazioni delle altre linee.
- Evidenze, piani e budget pubblicati prima del loro utilizzo; ogni SHA remoto
  verificato. I documenti gia' distribuiti non vengono riscritti.

Il lavoro viene seguito nel turno corrente, senza automazioni o monitor
persistenti in background. Il ledger consente la ripresa senza duplicare job
in caso di interruzione della chat.
