# Terza review GPT: esito finale scaling 30M/300M a 5.000

**ESITO ACCETTATO. Nessun problema tecnico bloccante e nessun job da rifare.**
La lettura scientifica indipendente e il confronto con Claude sono in
`gpt_scaling_scientific_final_20261106.md`. Il risultato sostiene la proposta
S8 di conferma a 19.618; questa review non autorizza quella continuazione.

Consegna rivista: ramo `claude/scaling-300m-exec-20261106`, tip
`7fdb2e85945b23930de5347cc8999492b5d12bec`: esecuzione `1d12ebe`, collector e
selftest `6712a6c`, esito raccolto `7fdb2e8`. Ancore: Review 10 `d3976c3` §5
e rilascio GPT `0130821`. Nessun deploy, SSH, invio, cancellazione o modifica
al codice di produzione eseguito da GPT; consultati artefatti locali pubblicati.

## Esecuzione conforme al rilascio

- Pre-deploy 14:26:36, deploy del solo riferimento metadata
  `deploy/scaling_300m_comparison_inputs_20261106_v1` a
  `2b7b4551c6059d5456e6e12d171ef687eb0818f4`, gate-checkout 14:29:48.
- Checkout `63671405`: sottomesso 14:30:13, COMPLETED 0:0, 40 s, 2 CPU, 4G.
- Gate-compare 14:35:02 dopo il completamento del checkout; guardia degli
  argomenti sul dispatcher reale: un caso accettato e tre alterazioni rifiutate.
- Confronto `63671773`: sottomesso 14:36:12, COMPLETED 0:0, durata 193 s,
  8 CPU, 30G; fine 14:39:43, MaxRSS 3.143.208 K. Nessuna dipendenza o retry.
- Ricevute, argv e hash dei launcher coincidono con i due comandi approvati.
  Letture della coda con exit intero 0. Il dispatcher esegue davvero il controllo
  del runtime a otto thread prima del comparatore e lo riverifica alla fine.
- La regola temporanea «Due fit per volta» della linea 2 e' cessata con l'invio
  del confronto alle 14:36:12, coerentemente con il rilascio, non col checkout.

## Integrita' e verifica locale

Snapshot `outcome_state_20261106_1441.json` SHA256
`bbdc2aaf8dfc1877dfde1580db799a944a83a365c3bd99dce7a20e069e19e56d`.
Report SHA256 `4c10e7bb3007bdb46b5a1ce975b5531700ad9ac0aa0c413442de410c78b0c664`,
821.191 byte. Hash delle copie e raw sacct verificati; manifest `162da7fb...`,
hash canonico `bfa36e7f...`, nove pin invariati. Sorgenti confrontate anche con
i blob di `aef665ba7ea66fc80291846487907bb3ef9107df`.

Rieseguita offline la validazione: **28/28 PASS**. Selftest: **2/2 PASS in
11,29 s**. Riletto il collector: letture remote e copie locali, nessun invio
o scrittura remota; non e' stato eseguito `collect`. Lo strumento e' adeguato
all'accettazione di questa consegna. Le prove sintetiche non sostituiscono
le ricevute e i log reali, che sono stati verificati separatamente.

Report completo, passo 5.000, tre esiti di qualita' veri, reading_valid vero,
2.000 repliche/seme 20261102. Soggetti classificazione 314/32/8/8/19,
15 utenti della posa in ciascun gruppo. Nessuna replica con classi mancanti
o classe singola. I punti coincidono con gli aggregati; S8 ricalcolata
corrisponde. `continuation_authorized` resta falso.

Audit aritmetico separato, senza importare il progetto: **112 confronti
numerici PASS**, scarto massimo 1,42e-14. Ricalcolati punti e IC delle celle
classificative, 21 contrasti classificativi e nove contrasti di posa dai
conteggi per soggetto/SSE pubblicati. Bootstrap tramite matrici di molteplicita',
implementazione diversa dalla riduzione per indici del codice scientifico.
Script: `scaling_final_independent_20261106_check.py`; output compatto senza
identificativi in `final_outcome_review_20261106/independent_numbers.json`.

Questo audit non ricarica OOF o predizioni dal cluster e non riesegue i fit.
L'identita' di quegli artefatti si fonda sulla catena dei pin e sul comparatore
congelato completato; l'aritmetica dei riepiloghi e' stata verificata a parte.
Il report originale contiene conteggi e identificativi dei soggetti dei
dataset, nessun segnale; nessuna nuova copia di OOF o segnali in questa review.

## Consuntivo e seguito

Checkout 0,022222222 e confronto 0,428888889 core-ore, somma 0,451111111.
Cumulativo **125,540555556 / 310 core-ore CPU-only**. Perimetro concluso:
14 fit, sei aggregati, un checkout del manifest e un confronto.

Nessuna correzione al report congelato o al codice numerico. Una piccola
rettifica riguarda esclusivamente le descrittive post hoc del documento Claude:
conteggi di utenti migliorati/peggiorati/pari su EPN, affetti da confronti
floating point con zero. Dettaglio e riproduzione nella review scientifica;
nessun impatto su score, IC, S8 o accettazione tecnica. La decisione sul
seguito a 19.618 rimane di Simone.
