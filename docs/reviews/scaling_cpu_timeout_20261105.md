# Scaling 30M/300M: TIMEOUT dei benchmark CPU, 05/11/2026

La catena è **ferma**, senza fit downstream avviati. Il job 63593508 è terminato
in TIMEOUT; inoltre le misure già disponibili rendono insufficiente il tetto
di 136 core-ore dei fit. Non è un risultato sulla qualità delle rappresentazioni.
Simone chiede log e prompt per Claude; si riparte dopo convergenza della review.

Codice numerico congelato `aef665ba7ea66fc80291846487907bb3ef9107df`;
piani pubblicati in `1bde83e`, ultimo stato precedente in `76a3126`.
Nessuna modifica a `src/`, `scripts/`, `tests/` o ai checkout remoti.

## Esiti e raccolta

Stati definitivi raccolti alle 18:44:27 CET, dopo la ripresa richiesta da Simone.

| Job | Vista del lettore | Stato | Elapsed padre | Core-ore effettive | Picco RSS |
|---|---|---|---:|---:|---:|
|63593262|30M, 768 feature|COMPLETED 0:0|2.018 s|4,484444|2,634 GiB|
|63593508|300M, 1.792 feature|TIMEOUT|5.584 s|12,408889|3,839 GiB|
|63593593|300M, proiezione fissa a 768 feature|COMPLETED 0:0|2.194 s|4,875556|2,669 GiB|

Il padre del TIMEOUT riporta `ExitCode=0:0`, ma lo stato è TIMEOUT e il batch
è CANCELLED 0:15: non è un successo. Il limite richiesto era 5.400 s;
il costo usa tutti i 5.584 s osservati, senza troncarli e senza sommare i passi
batch/extern alla stessa allocazione. La causa precisa dei 184 s oltre il limite
non è attestata; non occorre attribuirla per contabilizzare il consumo.

In `results/infra/scaling_300m_20261104/cpu_readers_20261105/probe_benchmark/`:
30 file originali con hash e dimensioni ricalcolati (13 benchmark, 14 audit e tre
log), manifest della raccolta, raw `sacct -P` e accounting accanto per ogni job.
`audit_collected.py` riproduce offline `diagnosis.json`; non legge cache né score.

| Vista | DB2 | DB3 | DB6 | EPN612 | UCI |
|---|---|---|---|---|---|
|30M naturale|Completo|Completo|Completo|Completo|Completo|
|300M naturale|Completo, parent TIMEOUT|Completo, parent TIMEOUT|Completo, parent TIMEOUT|Solo audit parziale|Non iniziato|
|300M matched|Completo|Completo|Completo|Completo|Completo|

Sono presenti 83 fit di benchmark conclusi: 78 nei 13 report e cinque nell'audit
parziale EPN612. Tutti dichiarano convergenza secondo la regola congelata,
nessun warning o eccezione, nessun raggiungimento di 12.000 iterazioni.
EPN612 parziale contiene C=0,001/0,01/0,1/1/10; manca C=100. L'audit viene
persistito dopo ciascun fit: non identifica l'istruzione esatta interrotta.
L'assenza di `failed.json` non annulla il TIMEOUT.

Nei 13 report: piano/hash/contenuto, ricetta, codice, supporto dichiarato fra
viste, audit e runtime numerico coincidono; tutti train-only, fold0, nessuna
selezione C. Runtime Python3.11.6, NumPy1.26.4, SciPy1.12.0, sklearn1.4.1.post1,
pool a8thread; hash `1ac8481eb8b53fbdd423404a927cca02e908197f16ed6494268b56e5ddc24b1c`.
La verifica offline non sostituisce il futuro controllo vivo di cache e runtime.

## Diagnosi tecnica e budget

Il 300M naturale completa i tre NinaPro in circa 59,4 min di tempo interno ai
report; poi EPN612 resta incompleto nel job da90min. Il log termina con la
cancellazione per limite di tempo. Non risultano OOM o traceback; il massimo
RSS è3,839GiB su30GiB. L'evidenza dimostra insufficienza del walltime aggregato;
non dimostra divergenza del solver o inferiorità scientifica del300M. Non è
misurato quanto abbia inciso la contesa sul nodo condiviso.

C'è un secondo blocco indipendente. Applicando la formula congelata ai soli
13 report completi:

`sum[1,5 * (somma tempi sei C + massimo tempo C) * somma_f max(1,n_f/n_0)] * 8/3600`

si ottengono **148,134869 core-ore**, già oltre136. Sono esclusi i120s di overhead
per shard ed EPN612/UCI300M mancanti. È un limite inferiore del **preventivo
prescritto**, non una misura del futuro consumo effettivo. Nessun frazionamento
può far rispettare136 mantenendo quella formula. La stima iniziale era troppo
bassa: occorre correggerla esplicitamente, senza ridurre C, fold o convergenza.

Il consumo CPU-only raccolto è **22,027778 core-ore**: 21,768889 per questi tre
job più0,258889 di checkout/validazione/benchmark posa già conclusi. Zero GPU
in questa fase. Le CPU associate a training/estrazioni GPU sono contate altrove.
Il tentativo fallito resta incluso in ogni futuro consuntivo/preventivo.

## Quanto si conserva e proposta per la review

Checkpoint dei FM a5.000 e14cache di feature sono già verificati e conservati:
non servono nuovo training o nuove estrazioni. I13 benchmark completi vanno
conservati; i tre figli completi del parent TIMEOUT possono essere candidati al
riuso soltanto con verifica e decisione operativa esplicita, mantenendo lo stato
fallito del padre. L'audit EPN parziale non è un benchmark completo né uno stato
del solver ripristinabile. Nessun fit downstream era partito.

Proposta da far rivedere, **non rilasciata per invio**:

1. Recuperare soltanto EPN612 e UCI300M naturale con due nuovi output/ID.
   EPN ricomincia i sei C del suo benchmark; UCI inizia il proprio. Il controller
   è già per-dataset, ma serve un dispatcher operativo separato e revisionato:
   il launcher attuale impone tutti e cinque i dataset nello stesso job.
2. Conservare codice numerico aef665ba, piano, cache,8CPU/8thread,30GiB,
   login08, bootstrap/librerie, fold0, sei C e tutte le tolleranze.
   Proporre per il recupero **EPN120min + UCI10min**, tetto17,333333core-ore,
   zeroGPU. Sono tetti di allocazione proposti, non tempi attesi garantiti;
   EPN non ha ancora una misura completa. Nuovo errore/TIMEOUT = nuovo stop.
3. Dopo15report completi, calcolare il piano esatto dei fit con la stessa
   formula, RSS e copertura, pubblicando nuovo tetto fit e cumulativo CPU.
   Il vecchio136 è già impossibile: non viene alzato silenziosamente.
   Valutare il numero di shard rispetto al limite15 solo sulla copertura
   completa; mantenere2h/shard e non assumere che lo splitting basti.
4. Review di Claude del recupero e delle modifiche operative prima degli invii;
   quindi checkout pubblicati/pinnati, submitter/ledger comuni, raw accounting.
   Nessun nuovo consenso per ripetere il piano già autorizzato una volta chiuso
   l'emendamento concordato; nessun rilancio nell'attuale consegna.

La posa è pronta con preventivo660s, ma resta ferma con il resto della catena,
come prescritto da O-6/O-8 e dal piano CPU. S8, endpoint, seme, dose e criteri
scientifici restano invariati. Il budget condizionale19.618 è storico congelato:
questa nuova evidenza di fattibilità va segnalata nella sua eventuale revisione
prima degli score, non presentata come autorizzazione della continuazione.
