# Correzione Path/mmap e rilancio del collaudo — 18/11/2026

Simone autorizza esplicitamente applicazione della correzione, aggiornamento dei riferimenti,
nuovo CUDA e, solo se superato, i cinque benchmark. Nessun nuovo consenso operativo richiesto.
La precedente esecuzione 64706211 resta FAILED 1:0; nessun suo risultato numerico viene riutilizzato.

## Perimetro tecnico

Sei conversioni `str(path)` nei caricamenti `torch.load(..., mmap=True)` di CUDA, controllo di
ripresa e lettore finale. Stessi file letti, argomenti, precisione e contenuti. Nessun cambiamento
in src/, trainer, modello, loss, ottimizzatore, dati, maschere, schedule, dose, selezione o soglie.
Test dei sei callsite eseguiti con il contratto osservato di Torch 2.2 e checkpoint sintetico:
contenuti, dtype, stride, stato optimizer, coda e RNG uguali alla lettura di riferimento.

## Riutilizzo esatto dell'evidenza, senza cambiarne la provenienza

Audit 64652559 e calibrazione 64701482 restano prodotti dai sorgenti `ab2d6460…`. La rettifica
operativa dell'audit in timeout resta limitata al suo rapporto completo già autorizzato.
Gli originali non vengono riscritti. Ricevuta di calibrazione originale SHA
`f57d8731a1434f592a6360547ad604cdedb517cc47e278c5c3839384df000b5e`;
checkpoint originale SHA `0756921a0e59c0d93cf5821793a8457f6becee7e55a7b68ea5ab1ec80e17d50a`.

Un manifest esplicito contiene l'elenco completo dei vecchi e nuovi sorgenti e gli SHA esatti
(di audit, config, summary, metrics e contenuto canonico dell'intera ricevuta). Il validatore
ricostruisce entrambi i digest, verifica i file reali, il piano scientifico e il perimetro dei
cinque file interessati, compreso se stesso. Qualunque altro file scientifico modificato,
aggiunto o mancante è rifiutato. Il manifest è incluso e pinnato nella release e nel checkout.

Il lettore ricalcola tutti i valori della calibrazione. `source_sha256` rimane quello del
produttore reale; `reader_provenance` descrive il lettore corrente. Solo dopo l'autenticazione
dell'intera ricevuta storica si ammette questa unica differenza nella provenienza del lettore.
I λ restano InfoNCE 0,11942686122572264 e RnC 0,33044627364850215, stessi passi 257–288.
La nuova coorte porta il digest corrente e la ricevuta originale; configurazioni byte-identiche.
CUDA, probe, benchmark e riprese devono attestare esclusivamente il nuovo digest: nessuna
compatibilità generica per vecchi risultati. Il solo consenso non sostituisce questi gate.

## Rilascio e risorse

Nuovi checkout, ID e output; submitter e ledger comuni. Prima CUDA con intero probe a quattro rank,
confronti numerici FP32/bf16 e STOP. Solo dopo COMPLETED 0:0, rapporto valido, hash verificati e
lettura positiva pubblicata: tre benchmark C0/CN/CR a 576 update e due riprese CN/CR 544+32.
Ogni cella conserva 4 GPU, 32 CPU, 240 GiB, 30 minuti: tetto 2 GPU-ore / 16 core-ore.
Nessun nuovo training scientifico a 5.000 o 19.618, nessun monitor o automazione.

Consuntivo precedente completo: 0,934444 GPU-ore / 16,128889 core-ore. Con sei celle GPU,
due checkout residui da 4 CPU × 240 s e lettura finale da 8 core-ore: massimo
**12,934444 GPU-ore / 120,662222 core-ore**, entro **14 / 129,333333**. Nessun uso garantito
oltre il walltime. In caso di fallimento: conservare l'evidenza, fermare le fasi dipendenti,
nessun retry cieco o cambio delle soglie.

Release concreta, manifest, risultati delle verifiche e comandi sono in
`results/infra/contrastive_comparison_20261116/compatibility_20261118/` e
`releases/mmap_20261118/`. Review indipendente tecnica; nessuna nuova review attribuita a Claude.
