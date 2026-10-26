# Diagnosi aggiornata del pretraining dopo la chiusura delle attivita' (Claude, 26/10/2026)

Richiesta di Simone del 26/10/2026 («Produci una diagnosi aggiornata dopo la chiusura [...] Mantieni invariati criteri e verdetti gia'
congelati. I descrittivi servono a capire il meccanismo, non a scegliere retroattivamente un vincitore»), con la revisione di GPT del 26/10
(`/Users/simonebenatti/Documents/ChatGPT/FM per EMG/reviews/scientifica_2026-10-26/proposta_scientifica.md`) come revisione critica da
verificare sui dati e sul codice. Dal 26/10 la responsabilita' scientifica e' di GPT: questo documento e' la chiusura e la verifica di Claude,
non una nuova proposta scientifica; l'unica proposta operativa e' l'emendamento di R3 (`docs/emendamento_pilot_posa_v1.md`), chiesto da Simone.

Convenzioni: S = validazione interna in punti percentuali (P1: NinaPro DB2/DB3/DB6 cross-soggetto; P2: EPN-612 + UCI-EMG); delta [IC95]
bootstrap accoppiato sui soggetti; **fatto** = numero letto da un file del repo con la regola congelata; *interpretazione* = lettura di Claude,
marcata. Nessun verdetto congelato cambia.

## 0. Stato della chiusura (al momento di scrivere; si aggiorna alla fine)

| Attivita' gia' autorizzata | Stato |
|---|---|
| Pilot, criterio §5 (6 letture `pool` a 2 epoche) | CHIUSO il 26/10 00:24: nessun braccio passa (`results/pilot_obiettivi_20261024/criterio.json`) |
| Pilot, descrittivi `dec` per canale a 2 epoche (A2) | 5 sonde su 8 finite (P1 dei 4 bracci; P2 di R0); R1/R2/R5 su P2 in coda: lettura appena finiscono |
| Pilot, endpoint a 1 epoca e pendenza 1 -> 2, coerenza R0-e1/C1, per dataset (B2) | 8 sonde su 16 sottomesse, in fila sulla seriale (una alla volta): lettura domani |
| Consuntivo `sacct` di training e valutazione | training 176,5 GPU-ora (fatto); valutazione a code vuote |
| Codice R3/R3c | test CPU verdi (44 della posa, 197 mirati, suite 1.430 prima delle correzioni, rilanciata), review del `critico` chiusa con 15 correzioni (`d8650c6`); **codice verificato, nessuna ipotesi scientifica verificata** (nessun job) |

## 1. Tabella unica per ipotesi

| # | Ipotesi | Esperimento | Risultato (fatti) e incertezza | Che cosa si conclude | Che cosa resta irrisolto | Prova minima |
|---|---|---|---|---|---|---|
| H1 | **Il readout medio (`pool`) nasconde informazione che l'encoder ha** | Readout per canale `local` e `dec` a k 768 contro `pool`, A, due semi, P1 e P2 (`results/readout_canale_20261023/confronti/`) | P1: `local` -2,00 [-2,60; -1,36] «peggiora», `dec` -0,42 [-1,20; +0,33]; P2: `local` -2,49 [-4,59; -0,66] «peggiora», `dec` +1,33 [-0,93; +3,54]; dentro EPN-612 `dec` +1,41 [+1,03; +1,83]; larghezza intera = k 768 (P2 `dec` +0,06 [-0,35; +0,38]) | Due readout lineari per canale, **che mediano ancora il tempo**, non battono il medio con la regola; a montaggio condiviso (EPN-612) il decoder per canale guadagna 1,4 punti | Un readout **temporale** o **non lineare** non e' stato provato: il limite del readout non e' escluso in generale (GPT, punto 2: concordo) | Sulle cache gia' estratte, CPU: (a) `[pool; dec]` concatenati con PCA nel fold, (b) MLP identica su A, ancore-only e casuale con iperparametri scelti nei fold. ~30 core-ore, 0 GPU-ora. Una testa temporale richiede una nuova estrazione delle sequenze (~2 GPU-ora) |
| H2 | **Il termine JEPA aggiunge poco o nulla alle ancore** | Ancore-only contro A, 30M, 4 epoche, due semi (`results/infra/w1arm_20261021/valutazione/esito_confronto_objective.json`); per canale (`A_vs_AO_*`) | S: P1 +0,51 [+0,16; +0,88] (semi +0,46 / +0,56), P2 -0,40 [-1,51; +0,75] (semi -1,52 / +0,72); DB2 +0,72 [+0,26; +1,13], EPN-612 -0,63 [-1,00; -0,27]; per canale `local` P1 +0,20 [+0,01; +0,39] (A meglio), `dec` P1 -0,78 [-1,12; -0,45] (ancore-only meglio), P2 +0,65 [-0,01; +1,26] / -0,46 [-1,52; +0,47]; JEPA-only collassa nei due semi (rango 70 -> 38 in 3.500 passi) | Nessuna differenza oltre la soglia pratica di 1 punto; segni opposti per compito e per readout; senza ancore il JEPA collassa | **Non equivalenza dimostrata** (GPT, punto 4: concordo): differenze piccole esistono (DB2, EPN-612, `dec`). «Peso morto» era piu' forte dei dati: la formulazione corretta e' «nessun beneficio robusto oltre la soglia a 30M e 4 epoche» | Nessuna nuova prova sul JEPA: e' la direzione su cui concordano GPT e Claude. R3c (ancore-only a 2 epoche) nella seconda ondata e' il controllo che serve |
| H3 | **Il modello e' troppo piccolo** | Rango effettivo nei primi 4.000 passi a 30M / 100M / 300M (`results/infra/w1cal_c4_20261022/diagnostiche_erank_taglie.json`); C4 sospesa | Rango ~67-87 alle tre taglie (dim 384 / 640 / 896); a fine training A 144-146 su 384; R1 163 ma S peggiore di R0 | Il rango non cresce con la larghezza nei primi passi; **il rango non misura l'utilita'** (R1 lo mostra) | Lo scaling non e' escluso scientificamente: non c'e' una S a 300M (GPT, punto 5: concordo che rango e non collasso non dimostrano utilita', ne' il contrario) | Nessuna, finche' non c'e' un beneficio replicabile da scalare (accordo con GPT) |
| H4 | **I target sono alla scala sbagliata (25 ms)** | R2 (target per regione nascosta) e R5 (banco a 150 ms) contro R0, 2 epoche, un seme (`results/pilot_obiettivi_20261024/`) | R2: P1 -0,27 [-0,72; +0,16], P2 +0,55 [-1,53; +2,66]; R5: P1 -0,53 [-1,07; -0,02], P2 +0,35 [-2,14; +2,68], con EPN-612 -2,45 [-2,97; -1,92] e UCI +3,15 [-1,78; +7,80] | Queste due ricette non aiutano; R5 sposta i dataset in direzioni opposte | **Le ancore osservano gia' 25 / 50 / 200 / 500 ms** (`anchor_targets.py`: RMS 25 ms, bande 200 ms, inviluppo 500 ms, multi-scala 50 / 500 ms): «target tutti da 25 ms» e' falso (GPT, punto 1: concordo). R2 e R5 non esauriscono l'apprendimento di dinamica motoria: R2 media rappresentazioni dello stesso teacher, R5 allunga un kernel lineare | Nessuna variante nuova del JEPA (accordo con GPT) |
| H5 | **Il bersaglio del teacher (decoder a query, b) e' peggiore dell'encoder locale (a)** | R1 contro R0 (pilot) | P1 -0,76 [-1,25; -0,28]; P2 -1,27 [-2,63; +0,15]; DB2 -0,96 [-1,65; -0,31], EPN-612 -0,97 [-1,47; -0,50] | Il bersaglio a sta sotto R0 in tutti e due i compiti («regola non superata»; su P1 intervallo intero sotto lo zero) | Un seme | Nessuna |
| H6 | **Due epoche sono poche** | A: 1 epoca (C1 s0) 18,21; 2 epoche (R0) 18,84; 4 epoche (W1ARM A) 18,31-18,34 su P1 (run diversi: codice, seme, schedule) | Differenze dentro 0,6 punti fra run non appaiati | Non si vede crescita con le epoche; la pendenza 1 -> 2 per braccio (B2) e il controllo R0-e1 contro C1 s0 arrivano con i descrittivi | **Aperto fino a B2**; poi: se un braccio nuovo ha pendenza piu' ripida di R0, due epoche lo sottostimano | B2 (gia' pagato) |
| H7 | **Il corpus e' dominato dal polso Meta (emg2pose + emg2qwerty = 73,5% delle ore) e la rappresentazione si specializza** | Nessun esperimento diretto; indizi per dataset: EPN-612 (polso, Myo) e DB2 (avambraccio) reagiscono in modo opposto a molte varianti | Indizio, non prova | Non testato | Prova minima: la stessa sonda su un pretraining con quote cambiate costa un training (~40 GPU-ora): non e' «minima». Alternativa a costo zero: lettura per dataset di tutto quello che c'e' (fatta nei README) | Da decidere con GPT; non prioritaria rispetto a R3 |
| H8 | **La misura (sonda lineare cross-soggetto) e' vicina al tetto per questo corpus** | Riferimenti: Hudgins 14,30 / 67,48; encoder casuale (Step 1, su emg2qwerty) 16,0 / 73,1; A 18,3-18,8 / 81-82; DB3 (amputati) 0,03 = caso | A batte Hudgins di +4,01 [+2,97; +5,03] / +13,86 [+11,00; +17,03] e il casuale di +1,9 / +6,7 (Step 1, non trasferibile ai modelli nuovi: GPT, punto 7, concordo) | Il pretraining aggiunge informazione lineare; ma nessuna variante sposta P1 di piu' di 1 punto | Il tetto della sonda non e' misurato; il fine-tuning e' un'altra quantita' | Adattamento supervisionato a pari budget su un sottoinsieme fissato, pretrained contro da zero (GPT, passo 5): ~5-10 GPU-ora; da pianificare dopo R3 |
| H9 | **Le medie P1/P2 nascondono effetti opposti fra dataset** | Per dataset in tutti i confronti (README del pilot, JSON) | R5: EPN-612 -2,45 [-2,97; -1,92] contro UCI +3,15 [-1,78; +7,80]; ancore-only: DB2 +0,72 contro EPN-612 -0,63 | Vero: gli aggregati vanno sempre letti coi dataset (GPT, punto 6: concordo) | Le letture per dataset restano descrittive, senza correzione | Nessuna: e' una regola di lettura, gia' applicata |

*Interpretazione complessiva (Claude, marcata):* il quadro resta quello della sintesi del 24/10 nella sostanza, con tre correzioni di
formulazione dovute a GPT: (i) non «il readout non e' la causa» ma «due readout lineari per canale che mediano il tempo non lo sono»; (ii) non
«JEPA peso morto» ma «nessun beneficio robusto»; (iii) non «target da 25 ms» ma «target che chiedono statistiche del segnale, non struttura del
movimento». L'informazione lineare che la sonda legge viene da bersagli esterni; questo e' compatibile sia con «serve un bersaglio alla scala
del movimento» (R3) sia con «serve una misura diversa» (H1, H8). R3 con l'emendamento distingue le due cose.

## 2. I sette rilievi di GPT, verificati

| Rilievo di GPT | Verifica di Claude | Esito |
|---|---|---|
| Le ancore osservano gia' scale fino a 500 ms | `src/wearusfm/model/anchor_targets.py`: RMS 25 ms, bande 200 ms, inviluppo 500 ms, multi-scala 50 e 500 ms (`MS_SCALES`) | **Concordo.** La diagnosi di Claude non diceva «tutti da 25 ms» ma «statistiche locali del segnale»: la formulazione di GPT («struttura del movimento, non finestra») e' piu' precisa e la adotto |
| R2 e R5 verificano ricette specifiche | Protocollo del pilot §2: R2 media le uscite del teacher per regione, R5 allunga il kernel del front-end | **Concordo.** Lo diceva anche il §8 del pilot («token da 100-200 ms» come mossa diversa) |
| I readout per canale cambiano anche la rappresentazione letta e mediano il tempo | `harness/channel_readout.py`: `local` = encoder locale, `dec` = decoder a query; tutti mediati sul tempo | **Concordo.** Corretto nella tabella H1: non escludono un limite del readout in generale |
| Mancato superamento della soglia non equivale a equivalenza | Regola dei bracci: soglia +1 punto; ancore-only ha +0,51 [+0,16; +0,88] su P1 | **Concordo.** «Ancore-only = A» va letto «nessuna differenza oltre la soglia» |
| Rango e assenza di collasso non dimostrano utilita' | R1: rango teacher 163 contro 130, S peggiore | **Concordo**, ed e' gia' nel README del pilot come interpretazione |
| Le medie P1/P2 possono nascondere effetti opposti | R5 su P2: EPN-612 -2,45 e UCI +3,15 | **Concordo**; i per-dataset sono nei README e in questa tabella |
| Il confronto casuale dello Step 1 non va attribuito ai modelli successivi | Step 1: 4.000 passi su emg2qwerty, sonde storiche | **Concordo.** Nella diagnosi del 23/10 (F1) e' un fatto storico, non un numero dei modelli attuali |

## 3. Correzioni e precisazioni alla revisione di GPT

1. **CPEP e «RandInit».** Il fact-checker (26/10, arXiv 2509.04699v3, Sez. 3.3 e Tabella 2) conferma la correzione di GPT: «RandInit» e' l'encoder
   EMG inizializzato a caso e addestrato con la loss contrastiva, non un encoder casuale congelato; l'encoder di posa e' pre-addestrato (MAE) e
   congelato; i baseline a regressione diretta sono confrontati come encoder per la classificazione dei gesti, non sull'errore angolare. La
   rassegna locale va corretta (`docs/letteratura_fm_serie_temporali_20261023.md`): segnalato, non ancora fatto.
2. **«La loss esclude gli istanti senza alcun canale visibile».** Era vero nel codice di 71c0a93 e contraddiceva il §2 (ii) firmato; il
   `critico` lo ha misurato (12,5% delle patch di emg2pose con gli slab) e la correzione e' in d8650c6: il termine vale su tutte le patch
   esistenti. GPT aveva ragione; il testo del protocollo ora coincide col codice.
3. **«PoseHead e' LayerNorm + Linear, non una funzione lineare delle feature grezze».** Vero: la testa legge `LayerNorm(media sui latenti)` per
   istante, la sonda legge la media sul tempo dei latenti grezzi piu' la media dell'encoder locale. Dichiarato nel protocollo («Stato del
   codice»); non e' un difetto ma un allineamento imperfetto fra testa e sonda, da tenere a mente nella lettura di (b).
4. **«La loss di posa non viene ridotta al 23%».** Corretto: la riduzione e' per finestra, sulle finestre con posa; il peso 1,0 vale sulle finestre
   di emg2pose. Quello che e' «un quarto» e' la frazione delle finestre che portano il gradiente della posa a ogni passo.
5. **«R3 rischia di dare un altro risultato non diagnostico»**: concordo, ed e' la ragione dell'emendamento (`docs/emendamento_pilot_posa_v1.md`).
6. **Tabella dei sei punti**: lo stato di GPT e' corretto al 22/10; l'aggiornamento al 26/10 sta nel registro (testata) con questa chiusura.
7. **Dove non concordo del tutto:** GPT propone come «verifica piccola» `pool` contro `[pool; dec_per_canale]` sulle cache. E' fattibile e la
   metto in H1 come prova minima, ma con due cautele: la larghezza di `[pool; dec]` e' 768 + 16 x 384 per un montaggio a 16 canali, quindi la PCA nel
   fold e la regolarizzazione decidono il risultato; e un guadagno della concatenazione non distinguerebbe «piu' informazione» da «piu'
   capacita' della sonda». Va letta come descrittiva, non come criterio.

## 4. Fatti e interpretazioni, in breve

**Fatti:** nessun braccio del pilot passa; ancore-only e A non differiscono oltre la soglia; JEPA-only collassa; i readout `local`/`dec` non
battono `pool` con la regola; il rango non cresce con la taglia; A batte Hudgins; i dataset reagiscono in modo diverso (EPN-612 e DB2 spesso in
direzioni opposte). **Interpretazioni:** l'informazione lineare viene dai bersagli esterni; il limite puo' stare nell'obiettivo (nessun
bersaglio di movimento), nella misura (sonda lineare che media il tempo) o nel dominio (polso Meta dominante): R3 con l'emendamento separa la
prima dalla seconda e dalla terza. **Non verificato:** tutto cio' che riguarda R3 (nessun job).
