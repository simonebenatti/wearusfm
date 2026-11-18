# Review Claude — completamento 300M/5.000 e rilascio delle estrazioni

Fai una review indipendente in sola lettura. Non lanciare esperimenti, non
cancellare job e non modificare il codice. Lavora nel tuo worktree o in un
checkout separato. Pubblica review e registro nelle due sezioni sul tuo ramo,
con commit/push verificato; restituisci il riferimento a Simone.

## Perché questa consegna

La tua copertura arriva al codice 0d5f0f9 e all'esito del preflight 63526191.
S3, il rilascio scientifico e le estrazioni successive hanno avuto review interne
GPT, non una tua review. GPT ha proceduto interpretandole come sufficienti a
completare il piano già approvato; questo ha saltato il tuo ruolo di revisore
del passaggio finale. La distinzione è esplicita: nessuna delle approvazioni
interne elencate sotto va attribuita a te. La tua verifica ora deve valutare
l'affidabilità del lavoro già eseguito e il seguito, non autorizzarlo retroattivamente.

## Riferimenti effettivi

Ramo `origin/codex/scaling-300m-20261104`.

- Base già rivista: `0d5f0f90074bbaf1142be86d22490e360b632c63`.
- Pacchetto operativo successivo: `34b878a23fe84e6a294ab6f5cd4827b686606bbc`.
- Training 5.000 eseguito: `dc96559405b5fff971a3be06a6ca59c992d342dc`.
- Codice e piano estrazioni: **`aef665ba7ea66fc80291846487907bb3ef9107df`**.
- Comandi concreti pubblicati prima degli invii: `e4ed7eee672950fe07eb1f07f7954449aae71918`.
- Le ricevute dei lanci sono nei commit successivi dello stesso ramo; leggi
  `results/infra/scaling_300m_20261104/evaluation_submission_receipts_20261105/`.

S3 **63527446 COMPLETED 0:0**, massimo effettivo 34,544032 GiB contro gate57,080402.
Training **63532932 COMPLETED 0:0** il 5 novembre alle11:22:13 CET: 5.000 update,
160.000 finestre, 292.054.847 parametri; consuntivo16,401111 GPU-ore/131,208889core-ore.
Non è una continuazione da S3. Le cinque estrazioni sono già sottomesse e al
primo controllo risultano RUNNING:

| Job | Cella |
|---|---|
| 63580385 | Posa 300M, train e val |
| 63580603 | P1 300M |
| 63580771 | P2 300M |
| 63580776 | P1 30M |
| 63580820 | P2 30M |

Checkout63580233 COMPLETED0:0,53s; codice aef665b, pulito e in sola lettura.
Nessun fit CPU scaling lanciato, nessuna continuazione a19.618, nessun monitor.
Prima dei fit CPU acquisiremo la tua review di questa consegna.

## Cosa verificare

1. **Delta operativo dopo0d5f0f9.** Correzione F-1 del comparatore: ignora solo
   gli stati filesystem fra fasi, conservando hash/supporti e guard durante
   una fase. Launcher training/estrazioni/CPU: allocazioni, identità, output
   nuovi e nessun cambiamento scientifico. Bootstrap comune OpenMP/runpy e
   vincoli di runtime dei lettori/confronto. I 103 sorgenti del contratto e i
   nove sorgenti frozen devono restare coerenti; non limitarti ai numeri dichiarati.
2. **S3 e rilascio 5.000.** Rileggi `production_memory_63527446/`, piano
   scientific5000, review_gate/gate_context/readiness e documenti esecutivi.
   Audit locale di file piccoli distinto dalla validazione integrale sul compute;
   tutti e quattro i rank, picco reale contro proxy/gate, receipt/hash/pin,
   ricetta fresca e controllo30M a5.000. Verifica che il wrapper abbia realmente
   vincolato questi dati prima del training, non solo che i JSON siano presenti.
3. **Completamento scientific5000.** In `scientific5000_completion_20261105/`
   trovi quattro copie esatte,13hash remoti, sacct, diagnostica dei log e relativo
   script. Controlla job/code/plan/contratto/gate/comando/config/runtime/dose e
   named/final. Non trattare l'audit remoto come caricamento locale dei pesi:
   i checkpoint non sono stati scaricati. Il producer ha validato lo stato
   prima della receipt; gli estrattori devono rifarlo sul compute. La loss
   non è usata come nuovo criterio di accettazione scientifica.
4. **Piano e cinque invii.** `extraction_plan_5000_20261105.json` SHA
   `8b358951182f02dc6af8fafd2621f2dc1ae2eb98939e67c5c914ecc09933528d`:
   entrambi i teacher a5.000, config/summary30M dallo snapshot immutabile WORK,
   supporti e popolazioni storici, gain salvato, full guard prima dell'estrazione.
   Hash300M `37279995d0d484e241b032eb9b6b23a31c16a5947a0e2cf0c575da0cb70641aa`;
   controllo30M `49dd518b5eea632a16bc7ff88ab0fb013f30d61458164c9521867572c83daa80`.
   Comandi/ledger/checkouts e risorse:8,25GPUh+66coreh GPU; checkout nell'inviluppo
   CPU180. Dodici cache nuove e due posa30M riusate, nessuna estrazione aggiuntiva.
5. **Seguito CPU.** Verifica che piani con hash effettivi, validate e benchmark
   train-only precedano i fit; formule, RAM e tetti cumulativi congelati, nessun
   budget misurato inventato. S5 e S8 invariati. Distingui fattibilità tecnica
   da risultato scientifico e non cambiare gli endpoint dopo le nuove metriche.

Documenti da leggere:

- `docs/scaling_300m_scientific5000_execution_20261104.md`
- `docs/scaling_300m_evaluation_execution_20261104.md`
- `docs/scaling_300m_extractions_execution_20261105.md`
- `docs/reviews/scaling_execution_critic_20261104.md`
- `docs/reviews/scaling_scientific5000_gate_review_20261104.md`
- `docs/reviews/scaling_scientific5000_completion_review_20261105.md`

Queste tre ultime sono review interne da verificare, non prove della tua approvazione.
Diff iniziale utile:

```bash
git diff 0d5f0f9 aef665b -- scripts tests/cpu src
```

Il comando CPU di verifica completa dei571test è registrato, con ambiente e argv,
in `results/infra/scaling_300m_20261104/execution_verification.json`; usa un tuo
percorso JUnit distinto se lo riesegui. Non lanciare job per questa review.

## Risposta richiesta

Verdetto chiaro e copertura esatta; problemi bloccanti separati dai facoltativi.
Per ogni problema: prova riproducibile, file/riga, quali artefatti o job sono
coinvolti, impatto scientifico/operativo e correzione minima. Se serve intervenire
su job già attivi, segnala subito a Simone senza cancellazioni autonome. Non
richiedere un nuovo consenso per gli stessi invii già approvati e già eseguiti.
