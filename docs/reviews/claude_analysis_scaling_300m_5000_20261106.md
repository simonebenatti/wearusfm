# Analisi scientifica di Claude: 300M contro 30M a 5.000 passi (linea 1, scaling)

**Data:** 06/11/2026. **Autore:** Claude, operatore della linea 1. **Stato:** analisi separata dalla validazione tecnica.
Rettificata il 06/11/2026 dopo la terza review di GPT: due conteggi della sezione 7 e due formulazioni (sezione 11).
Non autorizza nulla: la continuazione a 19.618 la decide Simone; ragionamento scientifico e direzioni restano di GPT, che
fa una lettura indipendente. Simone, alla domanda «Vuoi anche la mia analisi scientifica finale, come chiede GPT?»:
«Sì, anche la tua».

## 0. Fonti

- Report del confronto: ramo `claude/scaling-300m-exec-20261106`, commit `7fdb2e85945b23930de5347cc8999492b5d12bec`,
  `results/infra/scaling_300m_20261104/final_stages_20261106/collected_outcome_20261106_1441/report.json`
  (sha256 `4c10e7bb3007bdb46b5a1ce975b5531700ad9ac0aa0c413442de410c78b0c664`, 821.191 byte).
  Job `SCALING-compare-20261106` = 63671773, `COMPLETED 0:0`, codice congelato `aef665b`, manifest `162da7fb...`.
- Validazione tecnica, fatta prima di aprire gli score: 28 controlli superati, nello stesso commit
  (`comparison_outcome_reading_20261106.txt`). I tre esiti di qualita' sono veri e la lettura S8 e' valida.
- Regole fissate prima dei risultati: `docs/scaling_300m_protocol_20261104.md`, punti S5, S7 e S8, e
  `scripts/scaling_reading.py` congelato. Stanno nel ramo di GPT `codex/scaling-300m-20261104` e nel ramo di esecuzione,
  che ne deriva; non in questo ramo di review.
- Numeri di questo documento: `docs/reviews/claude_analysis_scaling_300m_5000_20261106_numbers.py <report.json>`
  (solo lettura del report). Gli score li ho aperti dopo la fine del confronto e dopo la validazione tecnica.

## 1. In breve

- **EPN-612 (primario): il 300M e' migliore del 30M di 1,92 punti**, IC95 [1,33; 2,53]. Esito S8: «migliore».
- **Posa (primario): nessuna differenza misurata.** `E_dyn` 0,89993 contro 0,89982, delta +0,00011, IC95
  [−0,0034; +0,0038]. Esito S8: «irrisolto». Nessun segnale pratico (la soglia era una riduzione del 2%; qui e' −0,013%).
- **P1 e P2: nessuna perdita.** Non inferiorita' sostenuta per entrambi (limiti inferiori 0,13 e −0,38, sopra −1).
- **Raccomandazione della tabella S8, fissata prima dei risultati: `propose_19618_confirmation`**, cioe' «Proporre 19.618
  come conferma alla dose più lunga». E' una proposta: `continuation_authorized` e' falso.
- **Il controllo a dimensione pareggiata e' nullo su EPN:** 300M compresso a 768 valori contro 30M: −0,12 [−0,68; +0,40].
  Il guadagno compare solo con la lettura larga (1.792 valori contro 768). Lettura mia: il dato non dice se il 300M ha
  imparato rappresentazioni migliori o se conta la larghezza della sonda. Il protocollo lo aveva previsto (punto 4).

## 2. Che cosa e' stato confrontato

- Tre celle, stessa dose di 5.000 passi, teacher EMA: `T100-G-30M-natural` (controllo), `T100-G-300M-natural` (candidato),
  `T100-G-300M-matched` (300M compresso con una proiezione casuale fissa 896→384 per blocco, decisa prima dei risultati).
- Il 30M e' il checkpoint a 5.000 passi della corsa gia' esistente; il 300M e' una corsa nuova. Un seme per taglia.
- Classificazione: sonda logistica sul pool (768 valori per 30M e compresso, 1.792 per il 300M naturale), quattro fold a
  soggetti separati, 420 fit; bootstrap appaiato per soggetto, 2.000 repliche, seme 20261102. Soggetti: EPN-612 314, NinaPro DB2 32,
  DB3 8, DB6 8, UCI 19. P1 = 8/13 DB2 + 3/13 DB3 + 2/13 DB6; P2 = meta' EPN-612 e meta' UCI.
- Posa: ridge istantanea su 15 utenti di validazione. `E_dyn` = errore centrato diviso per la varianza del target sulle
  finestre dinamiche (lettura mia della definizione nel codice congelato): 1 vuol dire non spiegare nulla del movimento.
- Soggetti di sviluppo, non di test.

## 3. Lettura fissata prima dei risultati (S8)

| Misura (300M naturale meno 30M) | Delta | IC95 | Esito S8 |
|---|---|---|---|
| EPN-612, punti percentuali | +1,917 | [+1,329; +2,527] | **migliore** (limite inferiore > 0) |
| Posa `E_dyn`, rapporto d'errore | +0,00011 | [−0,00339; +0,00378] | **irrisolto** |
| P1, punti percentuali | +0,601 | [+0,135; +1,026] | non inferiorita' sostenuta (limite inferiore > −1) |
| P2, punti percentuali | +1,301 | [−0,383; +2,721] | non inferiorita' sostenuta (limite inferiore > −1) |

- Riga applicata della tabella S8: «Almeno un primario migliore, nessuno peggiore; nessuna perdita P1/P2 oltre margine» ->
  «Proporre 19.618 come conferma alla dose più lunga. Segnalare separatamente se il guadagno di posa è sotto il 2% e se
  la NI è ancora irrisolta.»
- Segnalazioni richieste dalla riga: il guadagno di posa e' sotto il 2% (non c'e' guadagno: riduzione relativa −0,013%);
  la non inferiorita' non e' irrisolta, e' sostenuta per P1 e per P2.
- La regola non autorizza la continuazione e non trasforma un «irrisolto» in equivalenza.

## 4. Controllo a dimensione pareggiata (S5, secondario fissato prima dei risultati)

| EPN-612, punti percentuali | Delta | IC95 |
|---|---|---|
| 300M naturale meno 30M | +1,92 | [+1,33; +2,53] |
| 300M compresso meno 30M | −0,12 | [−0,68; +0,40] |
| 300M naturale meno 300M compresso | +2,04 | [+1,67; +2,43] |

- Il protocollo aveva scritto come leggerlo: «Un beneficio che sopravvive alla compressione sostiene che non dipende
  soltanto dalla maggiore larghezza della sonda; non isola in senso assoluto «quantità di informazione» e «capacità». Un
  nullo può dipendere dalla proiezione, quindi non esclude informazione nel300M naturale.»
- Qui il beneficio **non** sopravvive alla compressione. Quindi (lettura mia): il risultato non sostiene che il 300M sia
  migliore a parita' di larghezza della sonda; non dimostra nemmeno il contrario, perche' una proiezione casuale puo'
  buttare via informazione utile. Le due spiegazioni restano entrambe in piedi.
- Per regola (S8) il secondario aiuta a interpretare e non cambia la riga della tabella.
- Stesso disegno negli altri dati: P1 compresso meno 30M +0,18 [−0,24; +0,59]; P2 +0,99 [−0,26; +2,12];
  UCI +2,10 [−0,30; +4,29] (qui il compresso ha il punto piu' alto del naturale, con intervalli larghi: 19 soggetti).

## 5. Altri dataset e compiti (secondari)

Accuratezza bilanciata in percentuale; delta = 300M naturale meno 30M, in punti, con IC95.

| Dati | 30M | 300M naturale | 300M compresso | Delta | IC95 | Caso |
|---|---|---|---|---|---|---|
| EPN-612 (314 soggetti, 6 classi) | 76,31 | 78,23 | 76,19 | +1,92 | [+1,33; +2,53] | 16,7 |
| NinaPro DB2 (32, 49 classi) | 16,17 | 16,89 | 16,43 | +0,72 | [+0,07; +1,34] | 2,0 |
| NinaPro DB3 (8, 49 classi) | 2,12 | 2,53 | 2,49 | +0,41 | [−0,27; +0,88] | 2,0 |
| NinaPro DB6 (8, 7 classi) | 31,98 | 32,40 | 31,57 | +0,41 | [−0,83; +1,69] | 14,3 |
| UCI (19, 6 classi) | 78,63 | 79,31 | 80,73 | +0,68 | [−2,59; +3,42] | 16,7 |
| P1 | 15,36 | 15,96 | 15,54 | +0,60 | [+0,13; +1,03] | |
| P2 | 77,47 | 78,77 | 78,46 | +1,30 | [−0,38; +2,72] | |

- «Caso» e' l'accuratezza di chi tira a indovinare (100 diviso il numero di classi): la aggiungo io per dare la scala.
- Tutti i delta puntuali del 300M naturale sono positivi; solo EPN-612, DB2 e P1 hanno l'intervallo sopra zero.
- DB3 resta vicino al livello del caso in tutte e tre le celle (gli intervalli delle celle comprendono il 2,04% di chi
  tira a indovinare fra 49 classi) ed e' incerto: questo non prova che l'informazione manchi. Il suo delta e' stimabile
  ma non risolto. P1 pesa DB2 per 8/13, quindi il suo +0,60 e' quasi tutto DB2.
- Sono secondari, senza correzione per confronti multipli: da soli non sostengono una conclusione.

## 6. Posa

Rapporto d'errore (piu' basso e' meglio), 15 utenti; delta = 300M naturale meno 30M.

| Gruppo di finestre | 30M | 300M naturale | 300M compresso | Delta | IC95 |
|---|---|---|---|---|---|
| Dinamiche (`E_dyn`, primario) | 0,89982 | 0,89993 | 0,89931 | +0,00011 | [−0,00339; +0,00378] |
| Tutte | 0,92382 | 0,92481 | 0,92763 | +0,00099 | [−0,00243; +0,00467] |
| Statiche | 1,04455 | 1,04996 | 1,07009 | +0,00541 | [+0,00031; +0,01199] |

- Sulle finestre dinamiche il delta e' quasi zero e 9 utenti su 15 hanno un delta favorevole al 300M, 6 sfavorevole:
  nessuna direzione. «Irrisolto» non vuol dire equivalenza; pero' l'intervallo e' stretto (lettura mia dell'IC95): esclude
  variazioni relative oltre circa ±0,4%, molto sotto la soglia pratica del 2%.
- Sulle statiche il rapporto e' sopra 1 per tutti (la ridge fa peggio di una previsione senza variazioni nel tempo: e'
  una misura centrata, non il confronto con una media globale) e il 300M e' un po' peggiore:
  +0,0054 con il naturale, +0,0255 [+0,0192; +0,0335] con il compresso (sfavorevole in 15 utenti su 15). E' un secondario,
  su un gruppo dove la lettura non funziona per nessuna cella: lo segnalo, non ne traggo una conclusione.
- In tutte le celle la lettura lineare spiega circa il 10% del movimento sulle finestre dinamiche. La taglia, a 5.000
  passi, non sposta questa misura.

## 7. Descrittive post hoc (calcolate dopo aver visto i risultati; non cambiano la lettura S8)

- EPN-612 per soggetto, 300M naturale contro 30M: 199 soggetti meglio, 100 peggio, 15 uguali; mediana +2,00 punti.
  Con il compresso: 148 meglio, 153 peggio, 13 uguali; mediana 0,00. (Rettificato: prima 200, 101, 13 e 148, 155, 11;
  vedi sezione 11.)
- EPN-612 per classe (recall in percentuale, 30M -> 300M naturale): fist 66,4 -> 70,8; pinch 72,6 -> 75,8;
  waveIn 78,8 -> 81,2; waveOut 80,4 -> 81,7; open 60,3 -> 60,6; noGesture 99,4 -> 99,4. Il guadagno sta soprattutto in
  fist e pinch; open resta la classe piu' debole in tutte le celle.
- In termini di errore: su EPN-612 si passa dal 23,7% al 21,8% di errore bilanciato, circa l'8% in meno in senso relativo.

## 8. Che cosa i dati sostengono e che cosa no (lettura mia)

Sostengono:
- a parita' di dose (5.000 passi) e con la lettura a larghezza naturale, il 300M classifica i gesti di EPN-612 meglio del
  30M di circa due punti, in modo diffuso fra i soggetti;
- nessuna perdita su P1 e P2 oltre il margine di un punto;
- nessun peggioramento dei due primari: la riga della tabella S8 e' quella della proposta di conferma.

Non sostengono:
- un miglioramento della posa: a 5.000 passi il 300M non legge il movimento meglio del 30M;
- che il guadagno su EPN venga da rappresentazioni migliori a parita' di larghezza: il controllo compresso e' nullo;
- una stima del rumore fra corse: c'e' un solo seme per taglia e gli intervalli coprono solo il campionamento dei soggetti
  (S7: «servono repliche di pretraining indipendenti per misurare la variabilità fra run»). Un delta di due punti puo'
  contenere una parte di variabilita' fra corse che questo disegno non misura;
- una previsione su 19.618: il protocollo dice che la prosecuzione «risponde alla domanda sulla dose; non misura la
  variabilità fra training indipendenti».

Costo, per chi decide (dal collaudo del 04/11, job 63526191, `docs/decisioni.md` nel ramo di GPT): il 300M ha un rapporto
dei tempi di 3,9086 rispetto al 30M e un picco di memoria di 28,4 GiB contro 13,1. Il preventivo della seconda tranche e'
quello congelato da GPT prima degli score (`docs/scaling_300m_continuation_budget_freeze_20261105_v2.md`, ramo di GPT);
non l'ho ricalcolato.

## 9. Limiti

- Un seme per taglia; il 30M e' una corsa precedente, il 300M una corsa nuova.
- Soggetti di sviluppo. Intervalli marginali, nessuna garanzia congiunta del 95% su piu' misure (S8).
- Solo letture lineari (sonda logistica, ridge): dicono che cosa e' leggibile linearmente, non tutto cio' che il modello sa.
- Una sola proiezione fissa per il controllo pareggiato: il suo nullo vale per quella proiezione.
- Dose iniziale: 5.000 passi, contro i 19.618 a cui il 30M e' gia' disponibile.
- Livelli assoluti bassi su NinaPro (DB3 vicino al caso) e sulla posa (`E_dyn` 0,90): su quei dati le differenze sono piccole
  perche' entrambe le taglie leggono poco.

## 10. Domande aperte per GPT e Simone (non sono proposte di esperimento)

1. La proposta S8 e' «conferma a 19.618». Il primario migliorato e' uno solo, e il suo controllo pareggiato e' nullo:
   basta per investire nella seconda tranche, oppure prima serve capire se il guadagno e' larghezza della sonda?
2. Sulla posa le due taglie coincidono a 5.000 passi: e' la misura su cui la domanda sulla dose resta piu' aperta
   (lettura mia).
3. La variabilita' fra corse non e' misurata: quanto pesa su un delta di due punti lo puo' dire solo una replica.
4. La continuazione resta non autorizzata: «decide Simone».

## 11. Dopo la terza review di GPT (06/11/2026, commit `5e150092f9588e94dab4fad93e170699b6c8416d`)

Esito della review tecnica: «ESITO ACCETTATO. Nessun problema tecnico bloccante e nessun job da rifare.» La lettura
scientifica di GPT, scritta prima di aprire questo documento, e' in `docs/reviews/gpt_scaling_scientific_final_20261106.md`
(ramo `codex/scaling-300m-20261104`): «Convergiamo su tutti i punti che decidono il seguito».

Rettifiche a questo documento e allo script dei numeri. Il report congelato non cambia e nessun job e' da rifare.

- **Sezione 7, conteggi per soggetto su EPN-612: errore mio, trovato da GPT.** Avevo scritto 200 meglio, 101 peggio, 13
  uguali con il naturale e 148, 155, 11 con il compresso. I valori giusti sono 199, 100, 15 e 148, 153, 13. Causa: lo
  script confrontava con zero differenze di medie in virgola mobile, e quattro pareggi (due per contrasto) uscivano come
  differenze di circa 1e-16. Ora lo script usa frazioni esatte. Ricontrollato da me su tutti i dataset: cambia solo
  EPN-612; mediane, medie, score, intervalli e lettura S8 restano uguali.
- **Sezione 5, DB3.** Avevo scritto «a questa dose nessuna taglia legge quei dati» e «Il suo delta non si interpreta»:
  troppo. Gli intervalli delle tre celle comprendono il livello del caso, ma questo non prova che l'informazione manchi;
  il delta e' stimabile, solo non risolto (precisazione di GPT, che condivido).
- **Sezione 6, finestre statiche.** «La ridge fa peggio della media» era impreciso: il rapporto e' una misura centrata
  nel tempo e il riferimento 1 e' una previsione senza variazioni temporali, non una media globale (precisazione di GPT;
  ricontrollato sulla definizione in `scripts/pose_dynamics.py` del codice congelato).

Risposta di GPT alla domanda 1 della sezione 10: no, non serve spiegare prima il controllo pareggiato; «Non userei il
nullo dopo compressione come nuovo veto a posteriori». Sono d'accordo (lettura mia): la regola fissata prima dei
risultati e' S8, e usare ora il controllo pareggiato come condizione sarebbe una regola scritta dopo i dati. Il nullo
resta un limite d'interpretazione: se la continuazione si fa, lo stesso controllo va riletto a 19.618.

Raccomandazione di GPT: «una sola continuazione 300M fino a 19.618», stessa ricetta, motivata dal guadagno in
classificazione, con la posa come domanda aperta. Non e' autorizzata: decide Simone.
