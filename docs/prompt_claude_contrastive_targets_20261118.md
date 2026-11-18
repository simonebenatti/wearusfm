# Review mirata dell'adeguamento target della linea 3

Ramo `codex/contrastive-target-windows-20261118` su origin. La ricetta 100/125 è quella
del tuo `0d91297`, rivisto e integrato separatamente; qui verificare soltanto l'adeguamento
contrastivo. Main pubblicato a `f546c9e4`, integrato nel ramo con `afb89b45`.
Nessun lancio da questa consegna. La proposta TCN resta distinta.

Diff scientifico da `e388a1de` a `0efdc29c` (non riesaminare tutta main):
- `scripts/contrastive_prepare.py`: lettura storica25/50 con helper stretto, baseline
  verificata, nuova ricetta100/125/v3 esplicita nei tre bracci e calibrazione.
- `scripts/contrastive_job.py`: audit7200s, lettura3600s, GPU1800s; overrideSLURM2h
  per il solo audit. Nessuna modifica ai limiti GPU o ai criteri numerici.
- `scripts/contrastive_benchmark_readout.py`: confronto col primo update storico
  solo descrittivo anche al passo1, perché i target sono cambiati.
- Test nuovi e test storici spostati al proprio reader congelato senza riscrivere ricevute.

Leggi `docs/contrastive_targets_amendment_20261118.md` e
`results/infra/contrastive_targets_20261118/{plan,preflight_plan,local_validation}.json`.
Suite CPU completa sul codice `65bc6022`: 3,723 casi distinti passati; log originali del disco pieno e recuperi nella ricevuta `full_suite/verification.json`. Il merge di main aggiunge solo la review e le evidenze, senza modifiche ai sorgenti.
I successivi commit della diagnostica da stato comune, fino a `d9020c20`, sono conservati
nel merge `65bc6022` con sorgenti/pin identici: non sono una nuova release dei diagnostici.

Verifica in un unico passaggio:
1. Nessuna reinterpretazione del T100-G storico;100/125 esplicito su tutte le celle nuove.
2. Nessuna accettazione automatica di vecchi audit, λ, calibrazioni o benchmark per la nuova coorte.
3. Confronti CN−C0/CR−C0 appaiati nella nuova ricetta; C0−T100-G con due fattori aggiuntivi dichiarati.
4. Coefficiente storico di λ dichiarato come prior; entrambe le λ da ricalcolare, gate invariati.
5. Preventivo: 13 allocazioni, 14 GPUh / 137,066667 coreh; audit di 2 h coerente col dispatcher.
6. Il cambio target non sana la parità/ripresa bf16: la diagnostica già preparata va conclusa
   nel proprio checkout, prima del nuovo rilascio. Nessun retry cieco o soglia nuova implicita.

Riporta prima eventuali blocchi dimostrati; note non bloccanti insieme, senza chiedere giri
ripetuti su parti invariate. Non copiare autorizzazioni e ricevute della vecchia coorte
in una nuova release. I costi di training scientifico si misurano dopo il preflight.
