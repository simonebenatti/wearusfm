# Prompt per Claude 2 — rilettura del diff runtime TCN

Claude 2, ho corretto la tua review operativa `527f8dae716398203f8f2d7413d93494f020e185`
sul ramo `codex/pose-temponet-implementation-20261119`. La base è
`2f982114719dc2abd723bdbf0147d7fef34a731c`; il nuovo commit dei contenuti e tutti gli hash
sono in `results/infra/pose_temponet_execution_20261119/delivery_pin.json`.
Il commit successivo contiene soltanto il pin della consegna. Stato **REVIEW_REQUIRED**.
Nessun job, deployment o accesso a Leonardo; l'operatore resta **2-TCN head launcher**.

Rileggi soltanto il diff incrementale, senza agenti, come hai proposto. I riferimenti sono:
- `docs/reviews/pose_temponet_runtime_response_20261119.md`: risposta B-1–B-4, I-1–I-6 e minori;
- `docs/pose_temponet_execution_20261119.md`: sequenza operativa aggiornata;
- `results/infra/pose_temponet_execution_20261119/revision_runtime/`: piano precedente,
  piano rigenerato, produttore della verifica, log e ricevuta finale;
- `tests/cpu/test_pose_temponet_e2e_chain.py`, più test operational e runtime_fixes.

Verifica in particolare:
1. PYTHONPATH anteposto, tempi arrotondati al minuto fino a `_allocation`, agente submitter valido,
   callback che legge `pair.points` e hash iniziali emessi prima degli aggiornamenti.
2. Stato e artefatto del gate combinato, anche negativo. `commands --stage controls` richiede
   `--closed-reference-gate-output`: salva prima di stampare qualsiasi riga di invio.
3. Benchmark sullo scoring produttivo, identità preriscaldata, costo di snapshot in memoria
   distinto dagli artefatti effettivi; margine WLS L0/L13 con fit su y* e valutazione su y.
4. Commit e hash del piano verificati prima del checkout; nuova quota pinnata dopo i benchmark,
   dispatcher verso il launcher congelato e review legata all'identità del codice.
5. Catena dei 24 stadi attraverso run_stage con vere ricerche e rifit, poi inferenza/metriche/raccolta.
   Ambiente e dati sono sintetici, CUDA è simulata su CPU e la ricetta è abbreviata. Il vero
   qualify_controls è eseguito ma solo la sua ammissione finale viene forzata per attraversare
   la catena: esito originale e forzatura sono dichiarati. Nessuna funzione run_selection o
   run_refits viene sostituita. Questo non qualifica i controlli sui dati veri.

Ricetta scientifica v3, proposta JSON, soglie e tetti restano invariati. Nel rapporto sono
ora espliciti la soglia assoluta E_dyn e l'esito della seconda R-1. Le minori rinviate sono
limiti operativi dichiarati, non modifiche silenziose. La ricevuta distingue il test storico
preesistente dal nuovo pacchetto e non dichiara una suite interamente verde se quello resta fallito.

Pubblica il verdetto sul solo nuovo diff, gli eventuali blocchi con prova e le note residue;
registra nelle due sezioni, committa, pusha e verifica lo SHA remoto. Nessun lancio da parte tua.
Dopo convergenza la consegna va a 2-TCN head launcher entro i gate già fissati.
