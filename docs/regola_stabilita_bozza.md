# Regola di stabilita' v2 sul pannello fisso: bozza (Claude, 14/10/2026)

Punto 4 della numerazione unica (registro condiviso, `c4a0302`: GPT «review del monitor e definizione condivisa del criterio prospettico»;
Claude «bozza operativa di cadenza/pannello/regola, correzioni del trainer e test»). Simone: «bozza della regola di stabilità (4)».
**Bozza da rivedere con GPT; si congela prima del prossimo collaudo e vale da li' in avanti.** Il verdetto del collaudo 2C (v1, non passato)
resta com'e'; questa regola non si applica a run gia' fatti.

## Perche' una v2

La v1 (media mobile su 50 aggiornamenti della loss del batch <= 1,5 volte il minimo raggiunto) e' fallita nel collaudo e nei quattro run 2C
(1,97-2,24) senza poter dire se fosse regressione del modello o composizione dei batch: la loss di un batch dipende da quali sessioni ci
capitano. Togliere i picchi o usare mediane non basta (rapporti 1,6-2,0, audit di GPT e verifica di Claude), e la v1 e' sensibile al minimo
selezionato e ai molti confronti sovrapposti (GPT). Da `5f62444` il trainer misura le loss su un **pannello fisso** di finestre del training e
registra l'**origine di ogni finestra** del batch: la v2 giudica sul pannello, dove la composizione non cambia.

## Pannello e cadenza (proposta)

- Pannello: `fixed_eval_batches = 8` batch da 32 finestre del training (256 finestre; generatore proprio, seme [seed, 5151]; identita' in
  `fixed_eval_set.json`). Stessa funzione di perdita del passo, senza gradiente, studente e teacher in eval (niente dropout del muscolo): a
  modello fermo il valore e' ripetibile, quindi ogni variazione e' del modello.
- Cadenza: passo 0 e ogni `fixed_eval_every = 100` aggiornamenti nei collaudi (costo ~8 forward senza gradiente ogni 100 passi, qualche
  percento); ogni 500 nei run lunghi della finestra 1 (sotto l'1%).
- Origine dei batch (`log_batch_origin`) accesa nei collaudi; nei run lunghi da decidere (circa 0,5 KB per passo).

## Regola (proposta, da congelare con GPT)

Sia L_t la loss totale sul pannello all'aggiornamento t (valutazioni 0, 100, 200, ...) e m_t il minimo delle valutazioni fino a t.
1. **Valori non finiti:** una L_t non finita -> allarme subito.
2. **Regressione persistente:** allarme se L_t > (1 + δ) · m_t per **k valutazioni consecutive**; proposta **δ = 0,10, k = 3** (300 aggiornamenti
   alla cadenza 100). Una sola valutazione sopra soglia si registra ma non e' allarme.
3. **Apprendimento** (solo nei collaudi): L all'ultima valutazione <= 0,9 · L_0.
4. Nei collaudi il verdetto «passa» richiede: nessun allarme 1-2, la 3, e le regole gia' in uso (integrita', aggiornamenti sufficienti, tempo).
5. Componenti (JEPA, ancore, keep, rami) sul pannello: registrate e riportate, non decidono (salvo concordarlo).
6. **Descrittivi, non criteri:** il rapporto v1 sulla loss dei batch (continuita' con lo storico), la frazione di aggiornamenti al clip, le
   sessioni con le loss di batch piu' alte (dall'origine registrata).

Nei run lunghi l'allarme 2 **scrive un avviso e non ferma** il run (prima versione): la decisione di fermare resta umana finche' la regola
non ha uno storico. L'allarme 1 ferma, come oggi le perdite non finite.

## Punti aperti per GPT

- δ e k sono scelti a priori, non calibrati sui run falliti. Se servono numeri, la via pulita e' un **collaudo di calibrazione** dichiarato
  come tale (solo il pannello, nessun verdetto), poi δ e k congelati prima del collaudo che decide. Costo: quello di un collaudo (<= 0,5 GPU-ora).
- Il pannello e' tratto dal training: misura la regressione sui dati visti, non la generalizzazione (che restano le sonde P1/P2).
- Il pannello in eval non contiene il dropout del muscolo ne' maschere casuali nuove (le maschere JEPA vengono dal batch fisso): segue la
  tendenza del modello, non riproduce la loss del passo.
- Cosa fare in caso di allarme 2 nei collaudi: fermarsi e decidere (come la v1), nessun ritocco automatico.

## Codice da aggiungere dopo l'accordo

Funzione del verdetto v2 su `fixed_eval.jsonl` (accanto a `step2c.collaudo_verdict`, generica per i collaudi), avviso nel trainer quando la
regola 2 scatta (file `fixed_eval_alarm.json`, nessuno stop), test CPU sintetici (curva piatta, picco singolo, risalita persistente, valore
non finito).
