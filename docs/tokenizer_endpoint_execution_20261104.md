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

## Correzione operativa prima delle estrazioni

Il primo checkout63491851 è FAILED2:0 dopo3s: GPT ha sottomesso il job mentre il
git push versoLeonardo era ancora in corso. Il launcher ha rifiutato il commit
non ancora disponibile, prima di creare cartelle. Nessuna GPU, nessuna estrazione,
costo2CPU×3s=0,001667coreh. Errore di sequenza dell'operatore, non dei training.

Prima del recupero: push concluso0, riferimento remoto e oggetto commit453ce6c
verificati, checkout destinazione ancora assente. Si ripete solo il checkout con
nuovo ID TOKENIZER-eval19618-checkout-r1-20261104, stesso codice/runtime e percorso,
2CPU/4G/10min=0,333333coreh massime. Costo fasecheckout aggiornato:0,335000coreh
fra consumo fallito e nuova riserva. Recupero diagnosticato, nessun retry cieco;
la diagnosi viene pubblicata prima del secondo invio. Le nove estrazioni non sono
state sottomesse e non vengono duplicate. Attendere esplicitamente il completamento
di ogni push/comando dipendente, anche quando il tool restituisce una sessione attiva.

## Nove estrazioni sottomesse e avviate

Checkout-r1 **63491904 COMPLETED0:0**,18s,0,010000coreh. Totale dei due tentativi
checkout0,011667coreh. SHA453ce6c496745f2d73aa3007406721586e6dd767 verificato sul
checkout pulito e in sola lettura; nessuna modifica successiva al runtime.

| Braccio | Posa | P1 | P2 |
|---|---:|---:|---:|
| T25-L |63491925|63491935|63491937|
| T100-L |63491940|63491941|63491943|
| T100-G |63491945|63491947|63491948|

Tutti i nove job risultano **RUNNING** al primo controllo dopo gli invii del04/11,
intorno alle08:21CET. Invii tramite submitter/ledger, nessun duplicato. Ricevute
originali in results/infra/tokenizer_evaluation_20261103/submission/, indice
submissions_19618.json, snapshot squeue e attestazione checkout nella stessa cartella.
Ridge e sonde CPU non ancora sottomesse: attendono le cache complete con pin reali.
Launcher sonde preparato e revisionato separatamente,18test passati; non modifica
il checkoutGPU corrente. Monitor automatico resta spento.

## Estrazioni concluse e lettori CPU19618 —04/11

Tutti i nove job sono **COMPLETED0:0**, ultimo termine08:50:01CET. Consuntivo
**3,581111GPUh /28,648889coreh allocate conGPU** contro6GPUh prenotate.
Le dodici estrazioni dell'intera matrice (tre5k+nove19618) sono dunque concluse,
**5,103611GPUh** complessive. Evidenze in completed_19618/sacct_completed.txt.

Lettori ancora da eseguire: nuova checkout comune tokenizer-endpointcpu-20261104,
2CPU/4G/10min=0,333333coreh. Posa validate/benchmark/ridge entro7coreh; sonde15shard,
6aggregati e confronto entro101coreh. Totale fase108,333333coreh, zeroGPU.
Rispetto alla matrice precedente si aggiungono esplicitamente2coreh per un secondo
benchmark posa e0,333333coreh per il checkout comune; autorizzazione extrabudget
già acquisita. Il costo atteso dei fit sonde è circa40,5coreh dai precedenti,
non il tetto96. Policy comune endpoint_cpu_resource_policy_19618.json.

Piani specifici: docs/tokenizer_pose_cpu_19618_execution_20261104.md e
docs/tokenizer_probe_cpu_19618_execution_20261104.md; nessuna modifica scientifica.
Prima inviare validate+benchmark e i cinque shard T25-L pianificati in parallelo.
Leggere timing e ricevuta tecnica del primo concluso prima di inviare gli altri
due bracci, sempre in gruppi entro10 job seriali RUNNING+PENDING. I tempi storici e i tetti
conservativi pubblicati restano la base del preventivo; uno shard rapido non
garantisce la durata di quelli piu grandi. Nessun uso degli score per adattare
la ricetta o scegliere bracci: tutti15shard restano previsti.
Ridge soltanto dopo lettura/pubblicazione del gate train-only. Aggregati/confronto
soltanto dopo i loro ingressi completi; non basta una cache senza ricevuta.
Monitor resta spento. Nessun nuovo training, loss, dose, seme o scelta adattiva.
