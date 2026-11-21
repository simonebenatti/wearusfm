# V1 - numeri, tabelle, riferimenti, citazioni (verificatore indipendente)

Script: verifica/v1-numeri-riferimenti/{dist.py,steps.py,trend.py,traces.py,params.py}. Fonti primarie: preflight_readout.json (sha256 ricontrollato),
metrics.jsonl e tracce del clone 918245b6, codice del clone, lettura Leonardo 21/11, decisioni.md, proposta faa4a8d7.
Circa 125 affermazioni controllate; la grande maggioranza dei numeri torna. Problemi:

1. Verdetto, "Dopo uno o due update identici": CR non ha alcun update identico dopo 545 (546 gia' 3,4e-6); CN ne ha uno (546). [errore]
2. 2.2 / 3.2 / prompt, "Update identico" al 546 e "al 546 tutto identico in InfoNCE": al 546 CN i coseni contrast_global_cos_nce_base e
   cos_rnc_base differiscono fra continuo e riprese (0,5889655845844823 vs ...827; -0,3373571680204288 vs ...290; ~5e-16), A=B. Le norme
   coincidono, non tutto. [eccesso]
3. 2.2 "3-7 volte nei 32 update, fra 0,02 e 0,04 decadi per update": la regressione di log10 di grad_norm dà 0,037 (CN) e 0,040 (CR)
   dec/update su 548-576, cioe' x11-13; 3-7 viene da rapporti fra medie di finestre (geo 3,2 CN, 5,3 CR; mediane terzili 6,2 / 4,6).
   0,02-0,04 mescola campi diversi (total, jepa, pose). I due numeri non sono coerenti fra loro. [imprecisione]
4. 2.5 "il contatore di iterazione della traccia avanza di uno per update": avanza 0..10 solo per 545-555 nelle riprese, poi resta 10;
   nel continuo resta 500 per tutti i 32 update (campionamento del logger DDP). [imprecisione]
5. 2.5 "le riduce con un solo all_reduce di un vettore in ordine fisso": e' un all_reduce per obiettivo (tre: base, nce, rnc),
   contrastive_objectives.py:427-433. [imprecisione]
6. 2.5 "I gradienti locali dell'encoder quindi coincidono": dedotto da norme e coseni dei vettori gia' ridotti con all_reduce, calcolati
   con autograd.grad per obiettivo (non i gradienti del backward); copertura solo SHARED_PREFIXES (tokenizer, identity, local, pool, backbone):
   non decoder, anchor_heads, pose_head, proiettore. La localizzazione "nella riduzione DDP" resta piu' forte di cosi'. [eccesso]
7. 2.2 "Sono somme non deterministiche che non entrano nel gradiente" + "a ogni update": jepa, log_rms ecc. sono loss che entrano nel gradiente;
   non deterministico e' il valore registrato (index_add), il cui gradiente e' deterministico. jepa differisce A/B in 29/32 update, log_rms 31/32 in CR. [imprecisione]
8. 2.4 e prompt "nessuna sta fra zero e 0,2%" / "almeno lo 0,2%": in CR una chiave vale 0,1997%. [imprecisione]
9. 2.3 / 2.6 / prompt "1,6 volte": e' il rapporto delle mediane (1,58); media 1,47, media geometrica 1,43. Statistica non dichiarata. [imprecisione]
10. Richiesta/Fonti "lettura di GPT a adb6e3d6": i conti 997-1.049 s, 1,11-1,17 GPU-ore e il tetto 2 GPU-ore confermati in 3.2 stanno nella sezione
    aggiunta a 918245b6 (diff adb6e3d6..918245b6, +8 righe). [imprecisione]
11. 2.7 n.4 «Gli altri tensori rientrano nella soglia» non letterale: GPT scrive "Gli altri tensori controllati di Adam e della coda rientrano nella soglia". [imprecisione]
12. Richiesta "Domanda per Simone: ..." e' una parafrasi non dichiarata; il prompt chiede se gli artefatti bastano o se spendere GPU, e nomina tre ipotesi. [imprecisione]
13. 2.6 "Le norme calcolate fuori da DDP coprono ... non il proiettore": neanche decoder, anchor_heads, pose_head. [imprecisione]
14. 2.2 "Un errore di scala o di stato darebbe almeno 1e-7": stima presentata come fatto. [eccesso]
15. 2.3 mediana chiavi dopo il fork 0,53%: e' sulle 910 chiavi diverse; sulle 1.001 post-fork 0,52% (CR 0,40% invariato). [imprecisione minima]

Fonti secondarie (non rifatte): 5.965 conteggi (f-confutazione), riproduzione 864a86de solo con chiavi torch 2.2 (a), coda per update di inserimento
(simulazione c; coerente con 5.070=4.979+91 e 5.040=4.979+61 che ho ricontato), sonda DDP torch 2.14 (d), stima 0,3 core-ore, dispositivo AdamW (b).
