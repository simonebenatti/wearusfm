# Consegna operativa a 1-30/300 launcher — 17/11/2026

Simone: «ecco manda tutto alla chat 30/300 launcher, va avanti lei per queste cose»;
titolo confermato **1-30/300 launcher**, ID `01a1e915-4dc5-7d90-bd1e-371ccd3b8560`.
Questa consegna trasferisce il ruolo operativo alla chat launcher. La chat scientifica
ha fermato gli invii e non mantiene runner o monitor in background; non cancella job.
Il lavoro resta quello già autorizzato: completare validazione/benchmark e lanciare
ridge e fit se tutti i cancelli passano, con commit/push e senza ulteriori consensi.

## Stato certo al passaggio

| Fase | Job | Esito verificato |
|---|---|---|
| P1-r2 | 64620434 | Accettato in 267e692, tre cache integralmente rilette, 4,62 GB |
| Checkout piani | 64655487 | COMPLETED 0:0, raccolto, 30 s |
| Validazione posa | 64659520 | COMPLETED 0:0, raccolto, 42 s, 2,46 GiB |
| Benchmark posa 30M / 300M / pareggiato | 64665960 / 64665968 / 64665970 | COMPLETED 0:0, raccolti, 27 / 24 / 23 s |
| Benchmark classificazione 30M DB2 / DB3 | 64667499 / 64667505 | COMPLETED 0:0, 842 / 163 s; raccolta canonica di stadio ancora da fare |
| Benchmark classificazione 30M DB6 | **64669464** | RUNNING alle 10:44:16 UTC; unico invio della seconda tranche prima dello stop |
| Altri benchmark classificazione | 12 | Non inviati al passaggio, verificare sempre ledger e scheduler |
| Budget dei fit / checkout budgets / ridge / fit | — | Non rilasciati né inviati |

**Non eseguire per intero `probe_benchmark_tranche02_lines.txt`: la prima riga è già
64669464.** Alla richiesta di trasferimento è stato interrotto l'orchestratore locale
che scorreva quella tranche. Il ledger è stato riletto due volte: gli ID 30M EPN612,
30M UCI e 300M naturale DB2 non esistono; nessun secondo invio è confermato. Prima di
proseguire riconciliare ancora questi ID con squeue/sacct e il ledger, senza duplicati.
Attendere COMPLETED 0:0 di DB6; poi un nuovo gate genera le sole righe mancanti.

## Due worktree, un solo writer da ora: launcher

- Operazioni, controller, evidenze e registro:
  `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-p1-recovery`,
  ramo `codex/scaling-p1-recovery-20261116`.
  Ultimo rilascio prima della consegna: **1d8d26245a946d03477623dd3f2dedda14ab800a**,
  già su origin e SHA remoto verificato. La consegna aggiunge solo documenti/ricevute.
- Metadata da distribuire:
  `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-reader-metadata`,
  ramo `codex/scaling-reader-metadata-20261117`,
  **60389a7157b0e7e7c2cd58c6fd9e0e387c5dd6dc** (già distribuito come plans).
- Numerica e dispatcher congelati sul cluster:
  `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_evaluation19618_20261106_v1`,
  **df2de35c05381e7a5a69edde409171b210a3b5d8**. Nessuna modifica.

Riutilizzare i worktree disponibili, controllando HEAD/albero; nessun reset di lavoro
altrui. La chat scientifica non vi scrive più dopo il commit di consegna.

## Pacchetto e review

`P = results/infra/scaling_300m_20261104/continuation19618_20261106`.

Leggere:
- `P/submissions/p1_reader_release_20261117/plan.md`: autorizzazione, risorse e sequenza;
- `docs/reviews/gpt_scaling_p1_reader_release_20261117.md` e audit indipendente affiancato al piano;
- `scripts/scaling_plans19618.py`, `scripts/scaling_p1_reader_bridge.py`,
  `scripts/scaling_p1_reader_budget.py`: controller Mac già rivisto, **277 test superati**;
- `P/evidence/checkout-plans`, `pose-validate`, `pose-benchmark`: evidenze raccolte;
- `P/submissions/p1_reader_release_20261117/`: gate, righe, ricevute, audit e snapshot del passaggio;
- `results/infra/scaling_300m_20261104/p1_recovery_readout_20261117/acceptance.json`:
  accettazione P1-r2 e provenienza mista, SHA `0b1f538757dc8f39e31425dfc20d6b7ac039820c2bc34c5c65e1351fce46c9ce`.

I 103 sorgenti scientifici e i file numerici/dispatcher/submitter sono invariati.
La nuova guardia riguarda soltanto il produttore P1-r2 e non è stata causalmente
esercitata da una transizione di timestamp: non dichiarare dimostrata la causa Lustre.
La validazione già fatta riguarda le quattro cache **posa**, non la qualità scientifica
e non tutta la finitezza dei latenti; benchmark/fit/predict verificano le patch lette.

## Ripresa concreta

Nel worktree operativo, usare Python wearusfm-tok e **sempre il profilo esplicito**:

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B scripts/scaling_plans19618.py --p1-reader-recovery gate probe-benchmark
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B scripts/scaling_plans19618.py --p1-reader-recovery lines probe-benchmark
```

Non fare il gate di nuova tranche mentre un benchmark già inviato dello stesso stadio
è attivo: attendere prima tutta la tranche. Gate/lines vincolano controller, piani,
evidenze, checkout e ledger; validità 1800 s. Pubblicare gate e righe esatte prima
dell'uso, verificarne lo SHA su origin. Inviare una riga alla volta tramite il comune
`scripts/submit_job.py` (operatore gpt), fermandosi se l'esito non è `sottomesso`.
Nessun invio diretto con sbatch, nessun retry o nuovo ID sostitutivo.

1. Riconciliare 64669464 e gli ID non inviati indicati sopra. Finire i 15 benchmark
   classificazione, a tranche ammesse dal gate: massimo 4 job della linea e 10 della
   QoS seriale. Il limite condiviso è 8 CPU per utente; non toccare le altre linee.
2. Tutti COMPLETED 0:0: `collect probe-benchmark`. Il comando raccoglie l'intero
   stadio e rifiuta evidenze incoerenti; non creare a mano la directory canonica prima.
   Pubblicare sacct grezzo accanto agli accounting, report e pin.
3. `budgets`: costruttori congelati dai report reali. Verificano tempi, memoria,
   convergenza, contesto numerico, copertura e limiti; **stop** a qualsiasi rifiuto.
   `budget-table` nel profilo ricalcola l'inviluppo cumulativo.
4. Copiare i nuovi documenti e le evidenze richieste dai pin nel ramo metadata,
   discendente da 60389a7: **solo aggiunte** in `P/readers`, `P/evidence`,
   `P/submissions` o documentazione. Non copiarvi controller/helper Mac e non modificare
   documenti già distribuiti. Commit/push verificato di entrambi i rami.
5. `gate checkout-budgets --sha <nuovo-SHA-metadata> --reviewed df2de35c05381e7a5a69edde409171b210a3b5d8`,
   poi `lines checkout-budgets --sha <stesso-SHA>`. Deploy solo nuovo riferimento
   `deploy/scaling_300m_budgets19618_20261106_v1`; checkout comune CPU, mai sul login.
   Raccogliere `collect checkout-budgets` solo dopo COMPLETED 0:0.
6. Gate e lines per `pose-ridge` e `probe-fit`, budget reale entro tutti i tetti;
   pubblicazione prima degli invii. Proseguire per tranche con gli stessi controlli,
   senza chiedere un nuovo consenso per il lavoro già approvato. Registry in entrambe
   le sezioni, risultati e decisioni aggiornati; commit/push con SHA remoto verificato.

Questa consegna arriva ai fit. Aggregati, confronto e checkout inputs rimangono
prenotati nel budget, senza estendere implicitamente il perimetro operativo richiesto.

## Budget e pin da conservare

- Tetto CPU-only cumulativo **351 core-ore**, non il vecchio 350 del freeze.
- Riserva storica prudente **4,792778**; include costi storici già accettati. Non
  ricontare i tre checkout residui: sono già nell'inviluppo.
- Massimo iniziale 349,959444667; dopo checkout/validate/posa raccolti:
  **343,597222444/351**. Le due prove classificazione concluse sono ancora prenotate
  al massimo finché manca la raccolta canonica: non sono costi reali aggiuntivi.
- Fit: massimo 15 shard, <=3 h ciascuno, <=240 core-ore richieste, <=246 con margine
  scheduler di 180 s/job. Copertura esatta 60 celle/dataset/fold. Nessun ampliamento.
- Ridge: 8 CPU, 30 GiB, 3600 s congelati. Costruttore dai benchmark posa: 600 s
  derivati (577,26 s con margine), quindi conforme; non ridurre l'allocazione congelata.
- Zero GPU per i lettori. GPU P1/training precedenti sono un consuntivo separato.
- Plans: pose SHA `70483ba8c3e091a5f620f8d7621f3551e81c5b204647522b70cd93546fc73197`,
  probe SHA `4bbfd06515c2a935b09d8befc3686ce1173b84bc7230a6540844827d7a76acf6`.
- P1 producer e4ebdbfa, ricevuta f5bc05c3; posa/P2 producer df2de35. Checkpoint
  300M a 19.618 `d7ebe9d87a08882ae0c1c836b62988ea21c7f1246998a5f82613878a0499387e`.

## Note operative

Il profilo usa SSH con ProxyCommand=none. Se l'autenticazione del certificato scade,
la normale connessione configurata `ssh leonardo` ha ripristinato l'accesso; poi si
rifà soltanto la lettura del gate, lasciando traccia del rifiuto. Non è un retry di job.
DNS negato dal sandbox e permessi sul ledger richiedono l'eventuale escalation tecnica
degli strumenti, non una nuova autorizzazione scientifica di Simone. Mai aggirare i
controlli. Il ledger comune è `~/wearusfm_local/jobs`.

Primo rifiuto di un gate su contenuti/risorse, errore reale o TIMEOUT: fermare quella
fase, raccogliere e diagnosticare, senza rilanci o modifiche al codice congelato.
Non rinominare/riusare i P1 falliti e non riestrarre controlli. Nessun nuovo training,
seme, dose, loss o criterio scientifico.
