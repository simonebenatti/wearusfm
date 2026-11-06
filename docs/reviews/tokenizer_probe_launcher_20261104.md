# Launcher delle sonde tokenizer — verifica operativa, 04/11/2026

Implementato `scripts/slurm/tokenizer_probe.sbatch` per le sole fasi gia' revisionate
di `scripts/tokenizer_probe.py`: `fit-shard`, `aggregate`, `compare`. Nessuna
modifica del lettore, della ricetta scientifica o delle estrazioni. La verifica
qui riguarda il launcher; non sostituisce la review scientifica gia' registrata.

Contratto:

```
tokenizer_probe.sbatch <plan relativo a CODE_DIR> <SHA256 del plan> <fase> <output assoluto nuovo> <argomenti della fase>
fit-shard: <T25-L|T100-L|T100-G> <db2-01|db2-23|db3|db6|p2>
aggregate: <braccio> <p1|p2> <quattro shard.json P1 oppure uno P2>
compare: <sei report aggregati JSON>
```

Gli shard corrispondono esattamente alla politica congelata: DB2 fold 0–1 e 2–3,
DB3 e DB6 fold 0–3, P2 EPN612+UCI fold 0–3. Il lettore richiede tutti i quindici
pin delle cache, teacher a 19.618 passi e sole feature `x_pool` float32 larghe768;
controlla inoltre ricetta, sorgenti, identita', supporti, convergenza e copertura.
Il launcher non espone un override per student, dose o feature.

| Fase | CPU | Memoria massima | Walltime massimo |
| --- | ---: | ---: | ---: |
| DB2 fold 0–1 oppure 2–3 | 8 | 30 GiB | 60 min |
| DB3 oppure DB6 | 8 | 30 GiB | 30 min |
| P2 EPN612+UCI | 8 | 30 GiB | 60 min |
| Aggregato P1 oppure P2 | 2 | 30 GiB | 20 min |
| Confronto comune | 2 | 30 GiB | 30 min |

I default SBATCH coprono DB2/P2; per le altre fasi il submitter deve passare gli
override della tabella. Il wrapper verifica partizione CPU `lrd_all_serial`, un
nodo/task, CPU, memoria e walltime effettivo da `scontrol`; imposta tutti i thread
BLAS/OpenMP e nasconde CUDA. `code_dir_setup` verifica checkout pulito, congelato,
non scrivibile e SHA atteso prima e dopo il comando scientifico. Piano interno
al checkout con hash obbligatorio, output nuovo e canonico fuori dalle directory
di codice, report di ingresso assoluti esistenti. Non sottomette altri job.

**Preventivo prima dei fit.** L'API delle sonde non ha una fase `benchmark`.
Il benchmark della ridge misura un algoritmo diverso e non dimostra il tempo
delle regressioni logistiche. Il preventivo scientifico gia' pubblicato in
`docs/tokenizer_evaluation_execution_20261103.md` usa circa27 core-ore storiche
per due encoder, quindi circa40,5 per tre come proxy, contro un tetto di96 per
gli shard, piu'4 per gli aggregati e1 per il confronto. Lo stesso documento
autorizza allocazioni conservative dai tempi storici, da affinare quando le
nuove cache sono disponibili, senza cambiare ricetta o usare score val. Questo
giustifica il preventivo, non garantisce che i nuovi embedding convergano entro
il walltime. Per misurare i tempi senza aggiungere una nuova API si possono
leggere gli audit dei primi shard pianificati, mantenendo gli stessi job nel
conteggio e senza ripeterli come "benchmark". Nessun rilancio automatico, aumento
silenzioso dei tetti o nuovo benchmark scientifico viene introdotto dal launcher.

Verifiche locali: `bash -n` e18 test del contratto CLI con chiamate SLURM/moduli
simulate, senza fit: cinque shard, aggregazione/confronto e rifiuto di risorse,
fase, braccio, pin o destinazione non ammessi. I test eseguono realmente il
precontrollo Python dei percorsi/hash e hanno fatto correggere la compatibilita'
con array vuoti e `set -u` nel Bash3 del Mac. Non verificano il funzionamento
di SLURM o dell'ambiente CINECA. Nessun SSH, lancio o commit eseguito dal subagent.

Seconda review indipendente favorevole: verificata corrispondenza con l'API reale,
copertura4shardP1/1P2/6aggregati, limiti e firma numerica conservata dai fit8thread
anche durante aggregazione/confronto2thread. Nessun bug bloccante, nessuna ripetizione
dei18test necessaria. Root ha letto il launcher; nessun job CPU ancora sottomesso.
