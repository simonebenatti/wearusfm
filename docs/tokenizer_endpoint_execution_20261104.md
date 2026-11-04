# Tokenizer: estrazioni all'endpoint 19.618 — 04/11/2026

Simone ha chiesto «han finito, provvedi / procedi». La campagna completa, incluse
valutazioni, sottomissioni dirette e superamento del vecchio budget, è già autorizzata.
Nessun nuovo consenso: si avviano le nove estrazioni residue della matrice pubblicata.
Le altre tre, posa teacher5k, sono già concluse; non si ripetono. Monitor automatico
spento come richiesto, nessuna nuova automazione.

## Integrità e consuntivo dei training

| Braccio | Job continuazione | Stato | Secondi | GPUh continuazione | GPUh totale da zero |
|---|---:|---|---:|---:|---:|
| T25-L |63386315|COMPLETED0:0|11051|12,278889|16,538889|
| T100-L |63386337|COMPLETED0:0|11329|12,587778|16,950000|
| T100-G |63386403|COMPLETED0:0|11334|12,593333|16,944444|

Totale training dei tre bracci **50,433333GPUh** sul tetto66, includendo una sola
volta i primi5k. Conclusione il03/11 fra23:22:52 e23:27:35CET. Tutte le ricevute
complete, dose19618, summary max_steps, quattro rank e batch globale32; training
congelato afb644ea10ecc2a716160b4d8a4904e805c366fb verificato esplicitamente.
Configurazioni integrali identiche alla ricetta salvo max_steps; tre STOP assenti.
Verificati15hash remoti: nove receipt/config/summary, tre checkpoint19618 e tre5k.
I checkpoint nominati5k sono invariati. Nove file raccolti identici byte per byte.

Evidenze: results/infra/tokenizer_execution_20261103/19618/ e
results/infra/tokenizer_evaluation_20261103/verification_19618.json.
La strict load del teacher, gain e GELU è ricontrollata dal controller dentro ogni
estrazione prima dell'inferenza; non si caricano checkpoint grossi sul login node.

## Piano e allocazioni congelate

Piano results/infra/tokenizer_evaluation_20261103/extraction_plan_19618.json,
SHA256 a01799f892d473769a8770beefafd96d0a46896968bee98d2c5de3376eb9e660.
Conserva integralmente input, selezioni e riferimenti della fase5k; aggiunge i pin
effettivi del checkpoint nominato19618 e degli snapshot di ricevuta/configurazione.
Nessun cambiamento ai valutatori scientifici o al launcher già revisionati.

| Famiglia, tre bracci ciascuna | Risorse per job | Tetto complessivo |
|---|---|---:|
| Posa train+val |1GPU/8CPU/100G/45min|2,25GPUh|
| P1: DB2/DB3/DB6 |1GPU/8CPU/100G/30min|1,50GPUh|
| P2: EPN-612/UCI |1GPU/8CPU/100G/45min|2,25GPUh|

**6GPUh e48coreh allocate conGPU**, già nelle8,25GPUh di valutazioni complessive.
Nuovo checkout tokenizer-eval19618-20261104:2CPU/4G/10min, **0,333333coreh CPU-only
aggiuntive**. Policy con ID, output e risorse: resource_policy_19618.json nella
stessa cartella del piano. Output sotto SCRATCH/.../tokenizer_evaluation_20261103/
19618/<ARM>/pose|p1|p2; inventario iniziale: radice19618 e checkout assenti, coda
utente vuota, nessun ID logico endpoint già nel ledger. Ricontrollo prima degli invii.

Commit/push verificato precede deploy su riferimento nuovo e checkout CPU congelato.
Solo dopo checkout COMPLETED0:0/SHA/pulizia/sola lettura verificati si inviano le nove
estrazioni via scripts/submit_job.py, senza dipendenze da ID di training ormai conclusi.
P1 ha override esplicito30min; posa/P2 esattamente45min. Il launcher rifiuta risorse
e combinazioni errate. Nessun nuovo job validate separato: ogni estrazione include
le proprie guardie prima/dopo e una ricevuta scritta soltanto a completamento.

## Seguito della stessa matrice

Dopo cache complete e provenienze/parità verificate: piano ridge19618 sulle sei
cache posa, e piano sonde convergenti sulle15cache di classificazione. Posa crop6,
classificazione pool768 senza PCA e senza nuovo crop. Encoder teacher congelati,
si addestrano soltanto i lettori. Benchmark train-only e preventivi prima dei fit,
entro i tetti pubblicati; nessun retry cieco o ampliamento scientifico.

Lo screening5k, raccolto separatamente oggi, è descrittivo e non seleziona quali
bracci valutare all'endpoint. Posa ed EPN-612 rimangono domande distinte; P1/P2
aggregati secondari. Nessun nuovo training FM, loss, seme o dose.
