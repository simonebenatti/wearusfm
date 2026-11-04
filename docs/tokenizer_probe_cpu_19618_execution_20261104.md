# Tokenizer: sonde CPU all'endpoint 19.618 — 04/11/2026

Le sei estrazioni P1/P2 dei tre teacher sono concluse con COMPLETED 0:0:
T25-L 63491935/63491937, T100-L 63491941/63491943 e T100-G 63491947/63491948.
I risultati scientifici delle sonde restano da calcolare; le estrazioni non sono
già un confronto di accuratezza.

## Identità e griglia congelata

Piano `results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json`,
SHA256 **dac83e0d3206da9f0c9c9cd3290cc1e7430fa40d496ba73bb01915cc0b573010**.
Costruttore e validatore originali `tokenizer_probe.make_plan/validate_plan`
eseguiti con successo sui 15 pin reali: nessun placeholder e nessuna modifica
del lettore scientifico. Il checkout CPU comune sarà
`tokenizer-endpointcpu-20261104`, congelato e pubblicato prima dei lanci.

Verificati i 15 SHA256 integrali delle cache, le sei ricevute complete remote e
le 15 provenienze: teacher effettivo, dose19.618, checkpoint invariati, codice
estrattore453ce6c496745f2d73aa3007406721586e6dd767 e piano estrazioni
a01799f892d473769a8770beefafd96d0a46896968bee98d2c5de3376eb9e660.
Geometrie T25-L/T100-L/T100-G e attivazioni identity/identity/gelu coerenti.
I lettori ricalcolano integralmente i pin e la finitezza pool768 prima dei fit,
e ricontrollano i pin dopo i fit.

Le firme byte degli array NPY y/s/subjects (e all_subjects per P2) coincidono
con i riferimenti storici e fra bracci. Mantengono dtype e ordine nativi:
le etichette EPN sono Unicode, senza conversione a numeri. Soggetti:
DB2 32, DB3 8, DB6 8, EPN314, UCI19; test ufficiali esclusi. Ricostruite e
confrontate le assegnazioni dei quattro fold prima di costruire il piano.

La raccolta `completed_19618/classification/collection_summary.json` contiene
copie JSON derivate delle ricevute, con l'inventario raw_inputs omesso; il piano
usa sempre le ricevute integrali remote e i loro hash, mai il riepilogo locale.
Un taglio dell'output di trasporto ha richiesto di recuperare soltanto la
frontiera DB3/DB6 di T100-L: provenienze originali, tre firme NPY DB3 e un hash
NPZ DB6. La riparazione documentata non ha alterato dati o job.
Nessuna cache NPZ, feature o predizione è salvata in Git.

## Risorse prima dell'invio

La politica `probe_resource_policy_19618.json` contiene ID logici unici,
output assoluti nuovi e argomenti strutturati del launcher per tutti i 22 job.

| Fase | Job | Risorse per job | Tetto core-ore |
|---|---:|---|---:|
| DB2 fold0–1 e2–3, tre bracci | 6 |8CPU/30G/60min|48|
| DB3 e DB6, tre bracci |6|8CPU/30G/30min|24|
| P2 EPN+UCI, tre bracci |3|8CPU/30G/60min|24|
| Aggregati P1/P2, tre bracci |6|2CPU/30G/20min|4|
| Confronto appaiato completo |1|2CPU/30G/30min|1|
| **Totale** |**22**|**0GPU**|**101**|

Le 101core-ore sono la quota sonde già prevista nella matrice autorizzata,
non un'aggiunta da contare due volte. Il checkout comune e la posa sono
contabilizzati nella politica comune. Base temporale: circa27core-ore di
fit storici per due encoder, proxy40,5 per tre; i tetti prudenti sono96 per
i fit, più5 per aggregati/confronto. Sono stime empiriche, non garanzie.

Non esiste una fase benchmark nell'API delle sonde. Nessun job sintetico extra
è introdotto per simularla. Dopo il checkout verificato, avviare in parallelo
i cinque shard T25-L; leggere il primo completamento per identità, completezza,
tempo e audit prima di liberare gli altri due bracci. Questa verifica operativa
non usa gli score per selezionare ricetta o bracci; il tempo di un DB6 corto non
prova la sufficienza del tetto DB2/P2. Continuano a valere le stime per tipo e
i tetti pubblicati, con lettura dei tempi omologhi appena disponibili.

Rispettare al massimo10job seriali RUNNING+PENDING nel contesto comune,
lasciando spazio a validazione/benchmark/ridge posa. Scaglionare le sottomissioni
con il submitter e ledger comuni; nessun duplicato. Aggregati dopo i rispettivi
shard, confronto dopo tutti e sei gli aggregati completi.

## Ricetta e comandi

Pool768 FP32, StandardScaler sul solo train del fold, logistica lbfgs/L2,
class_weight balanced, tol1e-4, max_iter12000, C=[.001,.01,.1,1,10,100],
seme0. Quattro fold, sei candidati più refit per fold: **420 fit complessivi**.
Selezione tramite val interna, pareggio al C minore e refit sul solo train.

Il lancio usa `scripts/submit_job.py`; gli argomenti di sbatch condivisi sono:
`--export=ALL,CODE_DIR=<checkout verificato>,EXPECTED_SHA=<SHA pubblicato>`,
numero CPU e walltime della singola riga della politica. Poi:
`<checkout>/scripts/slurm/tokenizer_probe.sbatch`,
`results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json`,
lo SHA256 del piano sopra e l'array `launcher_tail` del job.
Ogni ID logico e dipendenza sono nella politica; non interpolare score o
modificare dinamicamente la ricetta. Nessun job è stato sottomesso da questa
preparazione.

## Lettura finale

Confrontare tutti i bracci alla stessa dose: EPN separata come endpoint primario,
UCI secondaria; P1 e P2 con i pesi congelati e bootstrap utenti appaiato.
Gli IC sono condizionati al singolo seme di pretraining e alla validazione di
sviluppo già utilizzata. Eventuali candidati non convergenti rimangono nell'audit:
lettura solver_unresolved e nessun verdetto di non inferiorità valido.
Non allungare iterazioni o tetti a posteriori, non eliminare candidati difficili.
Errori tecnici richiedono diagnosi, senza retry cieco.

I report endpoint ancora assenti non vanno confusi con lo screening5k.
Nessun nuovo training, nuova loss, altro seme o monitor automatico riattivato.

