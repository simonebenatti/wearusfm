# TCN ispirata a TEMPONet con ramo lineare: lettore della posa a encoder congelato

Revisione documentale del **18/11/2026**, della proposta `c15aac9`. ID **pose-temponet-frozen-encoder-readout-20261117-v2**. **REVIEW_REQUIRED; runtime assente; nessun codice, merge, deploy o job autorizzato oggi.** Simone: «procedi con la revisione documentale, committa e pusha tutto [...] per ora niente codice o lanci».

Ramo `codex/pose-temponet-proposal-20261117`, base di lavoro `cfa6462`. Risposta a Claude `32e5452` e seguito `8d34f9c`; [disposizione puntuale dei rilievi](reviews/pose_temponet_response_20261118.md). Il livello A qui significa lettore a encoder congelato, non il protocollo A sull'efficienza dei dati, che resta rinviato. Nessuna conclusione firmata delle altre campagne è riscritta.

<a id="p1"></a>
## 1. Domanda e perimetro

**Una TCN con percorso affine diretto legge meglio la dinamica della posa dai latenti già disponibili?** Teacher T100-G 30M a **49.046 passi**, stesso checkpoint in tutti i bracci, nessun aggiornamento del FM o nuova estrazione. Nuove teste sui 40 utenti train della cache, confronto finale sui 15 utenti di sviluppo già consultati.

La posa era già supervisionata nel pretraining: questa prova non dimostra trasferimento a un compito nuovo o valore generale del FM. Un esito nullo riguarda questa ricetta e questo lettore; non dimostra assenza di informazione nei latenti. Backbone bidirezionale su quattro secondi: il contesto temporale non nasce nella testa e il sistema non diventa causale. Nessuna nuova loss, dose, architettura del FM, 300M o ricerca di ritardo.

<a id="p2"></a>
## 2. Provenienza e adattamento della TEMPONet

Fonte: `src/wearusfm/model/temponet.py` a `8917e2a911287d8939e44d210adf566e4c8e8561`, introdotto in `60b35bd`; modello e `scripts/temponet_emg2pose.py` identici al runtime raw `0c19c4bc32a96e3b72d15a09770d204a747fa13b`. Prove Git in [raw_source_identity.json](../results/infra/pose_temponet_proposal_20261117/raw_source_identity.json), ora con tutte le chiavi dei report, senza troncamento. Fonte bibliografica riportata dal codice: [COINS 2021, DOI 10.1109/COINS51742.2021.9524188](https://doi.org/10.1109/COINS51742.2021.9524188), [postprint](https://hdl.handle.net/11585/841320). Nessuna attribuzione di risultati del paper alla nuova testa.

TEMPONet raw restituisce una posa per finestra e usa stride, pooling, BatchNorm e flatten. Non si trapianta invariata su 160 patch a 40 Hz: i 128 ms raw a 2 kHz non sono 256 latenti. `receptive_field()` dà 378 passi per elemento finale; a T=160 il flatten ne combina due distanti 64, unione nominale 442. Secondo la verifica Claude, a T=160 la dipendenza effettiva in eval copre 0–159 non causale e 0–123 causale. Non è una promessa sul supporto fisiologico della testa nuova.

Si riprendono convoluzioni dilatate e ReLU in una **TCN compatta ispirata a TEMPONet**, senza riuso dei pesi raw né dei blocchi privati invariati, che impongono BN/pooling. Ramo Claude e runtime raw restano intatti. I loro test non vengono presentati come test della nuova testa.

<a id="p3"></a>
## 3. Architetture, inizializzazione e riferimenti

Input `(B,160,384)`, media dei 64 slot, nessuna media temporale. Cache FP16 convertita in FP32. Tutte le teste iniziano con **LayerNorm affine384, eps=1e-5, gamma=1, beta=0**. Si definisce `h=LN(x)` e, per MLP/TCN, `yhat=W h+b+g(h)`: il ramo affine384→20 usa **la stessa LN**, senza dropout, e aggiunge 7.700 parametri. Con `g=0` la funzione è una L0; non è garantita la stessa traiettoria di AdamW.

TCN, ramo `g`: `Conv1x1(384,64)–ReLU–Conv k3,d1(64,64)–ReLU–Conv k3,d2(64,64)–ReLU–Conv k3,d3(64,64)–ReLU–Dropout0.1–Conv1x1(64,32)–ReLU–Dropout0.1–Conv1x1(32,20)`. Bias in ogni strato, stride1, padding1/2/3, nessun pooling/BN/normalizzazione temporale/filtro d'uscita. **nn.Dropout per elemento**, non Dropout1d, disattivato in eval, solo dentro `g`.

MLP: `384→96→96→96→64→32→20`, cinque ReLU, dropout dopo i hidden64 e32, più lo stesso ramo affine. Campo della TCN e di L13: 13 patch contigue, offset −6…+6, ±150 ms fra centri (300 ms centro-centro, 325 ms di bin). È una scelta di contesto compatibile con crop6, non una replica del campo raw né una conseguenza del precedente lettore crop4 sul B student.

| Braccio | Vista | Lettore | Parametri |
|---|---|---|---:|
| L0 | lat | LN→Linear384,20, PoseHead rifittata | 8.468 |
| L13 | lat | LN→Conv384,20,k13,pad6 | 100.628 |
| MLP | lat | MLP istantanea più ramo affine | 73.000 |
| TCN-post | lat | TCN temporale più ramo affine | 72.904 |
| TCN-pre | pre | identica TCN, **compreso ramo affine** | 72.904 |

MLP e TCN: 71.808 pesi nelle trasformazioni, 11.489.280 MAC per finestra160, esclusi bias/LN/attivazioni. Differenza96 bias. Si mantengono i cinque bracci, senza ricerca di larghezza, dropout o RF.

**Inizializzazione nominata:** hidden di `g` Kaiming normal, fan_in, ReLU; bias hidden zero; ultimo strato di `g`, ramo affine e intera uscita di L0/L13 a zero. Nessun gate scalare a zero davanti al ramo già azzerato. Al primo backward imparano uscita e ramo affine, poi possono imparare gli interni. Questo non certifica convergenza. Le reti originali potevano già rappresentare L0 sul dominio limitato dalla LN mantenendo le ReLU attive con bias opportuni; il nuovo ramo rende la via affine diretta (prova nella risposta A-1).

Ordine di creazione: LN, strati di `g` nell'ordine sopra, ramo affine; quest'ultimo sotto fork_rng per non consumare il flusso del ramo `g`. L0/L13 deterministiche all'inizio. Hash dello stato iniziale prima di ogni traiettoria; stessa inizializzazione per testa/seme/fold fra LR. Generatori separati: init `100000+s`, dropout `200000+100*f+s`, shuffle `300000+100*f+s` (f=0/1, f=2 rifit). I seed di mondo sintetico non cambiano l'ordine dei batch. Tre semi L0/L13 cambiano solo shuffle; per le altre teste anche init/dropout. Nessuna uguaglianza delle traiettorie dedotta dal seme.

Riferimenti storici: ridge post49k **in forma chiusa sui medesimi40 utenti**, non “miglior lineare possibile”; testa teacher allenata sui158 utenti, finestre1–4s, mascheramento e EMA. La ridge storica pesa gli elementi validi in aggregato; qui si pesa ogni finestra ugualmente. Testa storica da latenti FP32, teste nuove da cache FP16; predizioni storiche sul crop6 già disponibili. Non si copia una testa fittata sui40 dentro i fold. Ridge pre49k (E_dyn circa1,032128) è un'ancora descrittiva già nel report storico, non un sesto contrasto primario.

<a id="p4"></a>
## 4. Supporto, selezione finita e controlli prima dei fit reali

### 4.1 Dati e loss

1.907 finestre/40 utenti train, 18.548/15 utenti sviluppo. Tutte hanno160 patch. Pin in [proposal.json](../results/infra/pose_temponet_proposal_20261117/proposal.json). Due cache:4,720 GiB; con coefficienti e due predizioni storiche già pinnati: **5,328 GiB**. Nessuna duplicazione persistente. Pesi dei target train158 e standardizzazione restano quelli della cache.

Supporto: giunto centrale valido, `6<=t` e `t+6<N_i`, contesto realmente presente da lunghezze/key_time_valid. Le maschere delle etichette vicine non entrano nelle feature. Crop6 → centri6…153,148 patch. Finestre senza centri supervisionati escluse da tutti i bracci e contate prima; nessuna esclusione dalle predizioni.

Loss: MSE standardizzata media sugli elementi validi della singola finestra, poi media delle finestre supportate, **tutte**, non solo dinamiche. Ritardo zero, nessun termine nuovo. Due fold: ID train normalizzati e ordinati, indici alterni; ognuno20 utenti, liste/hash e conteggi fissati nel manifest prima dei fit. Encoder/statistiche hanno già visto questi utenti: è CV del lettore condizionata al FM. I15 sviluppo sono già stati consultati in altre prove; test ufficiale escluso e nessuna conferma indipendente pretesa.

Per trigger e S-2 si usa una soglia **solo train**: per finestra media delle varianze population per giunto sui tempi validi dell'intera finestra160 (almeno2 punti), poi mediana sulle finestre del fitting della direzione del fold, limite inferiore1e-12. Soglia fissata prima delle curve e applicata al rispettivo held interno. Nei controlli i bersagli sono sintetici. La soglia storica0,2574231028556824 resta solo per il report finale, non entra nella selezione/estensione.

### 4.2 Ricetta e ricerca limitata

AdamW beta0,9/0,999, eps1e-8, weight decay0,01 solo matrici/kernel, no bias/LN; clip norma1, batch32, ultimo batch conservato. FP32, autocast/TF32 off; `torch.use_deterministic_algorithms(True)`, cuDNN deterministic=True, benchmark=False, CUBLAS_WORKSPACE_CONFIG=:4096:8; worker loader0; OMP/MKL/OPENBLAS/torch intra-op8, inter-op1. Versioni e contesto numerico pinnati nel pacchetto futuro e identici fra benchmark e fit. Errore di determinismo → arresto, non cambio silenzioso dei flag. LR costante, nessun warmup.

1. LR iniziali **{1e-4,3e-4,1e-3,3e-3}**, tutti a150 epoche, semeCV0. Fine ogni epoca: eval/no_grad FP32, MSE per finestra e accumulatori E_dyn, train loss. Checkpoint aggiuntivi a1/4,1/2,3/4 della prima epoca, passi ceil(q*S_fold), duplicati deduplicati. Passo0 solo diagnostico, mai selezionabile.
2. I due fold di ciascun LR avanzano a checkpoint sincronizzati, con stato/RNG distinto per traiettoria. A ogni nuovo minimo MSE pooled si salva la **coppia** di stati dei fold, non due migliori epoche separate; si conserva anche la coppia corrente per la continuazione. Non occorre salvare tutti i checkpoint. Per ogni LR, curve aggregate sui due held interni: MSE=sum(per-window loss)/numero finestre, E_dyn=rapporto delle somme, non media dei rapporti. Minimi/tie: checkpoint più precoce. Se minimo MSE **oppure** minimo E_dyn è nelle epoche146–150, continuare **entrambi i fold a300**, preservando optimizer/RNG; una sola estensione, non ripartenza.
3. Selezione unica del minimo MSE fra tutti i checkpoint ammessi; pareggi esatti: durata minore, poi LR minore. E_dyn non sceglie un secondo vincitore.
4. Se vince il bordo inferiore/superiore della griglia attiva, aggiungere rispettivamente **3e-5/1e-2**, una volta per lato, con la stessa politica150→300; massimo6 LR. Dopo ogni aggiunta si ricalcola la selezione; nessun altro LR o ricerca.
5. Sul **LR selezionato**, minimo MSE o E_dyn ancora nelle ultime5 epoche dell'orizzonte massimo, oppure LR ancora al bordo dopo l'estensione di quel lato, oppure checkpoint selezionato al primo quarto di epoca, dà **ottimizzazione non risolta entro il budget**. Un LR scartato ancora al bordo si registra, ma non invalida da solo il vincitore. Non si dimostra convergenza globale quando questi allarmi sono assenti.
6. Rifit reali sui40, semi0/1/2, da zero al LR e durata scelti (non300 automatiche); stessa frazione nella prima epoca tramite ceil(q*S_full). A pari epoche si fanno circa60 passi invece di30: la durata è esposizione, non identico numero di update. Nessuna decisione sul15 sviluppo. Prima dell'inferenza finale si bloccano manifest di selezione, pesi e hash di tutti i rifit.

S-2: per il LR selezionato, **distanza >5 epoche e perdita relativa E_dyn >=2%** rispetto al suo minimo interno → `objective_tradeoff`. Se E_dyn_min=0 e quello scelto>0 la perdita è infinita. Si conservano MSE e dinamica di entrambe le epoche e le curve intere; nessun cambio di selettore o seconda inferenza. È un compromesso livello/dinamica, non automaticamente non convergenza. Se riguarda L13, il contrasto3 resta una prestazione di questa selezione MSE, non una prova di impossibilità della classe temporale lineare. Se riguarda TCN, un nullo resta relativo a questa ricetta, non alla dinamica potenzialmente accessibile.

### 4.3 Riferimenti lineari descrittivi e calibrazione

L0/L13 sono affini di `z=LN(x,gamma=1,beta=0)`: la LN affine apprendibile si assorbe nei coefficienti. Per ogni fold calcolare un riferimento **minimi quadrati pesati non regolarizzati**, stessi crop/mask/target: peso dell'elemento valido `m_itj/q_i`, con q_i numero totale elementi validi della finestra, non per giunto. Intercetta libera; float64, **QR stabile a blocchi/TSQR e SVD di R**, soluzione minimum-norm con soglia relativa sul valore singolare del design **1e-8**. Non determinare il rango dagli autovalori della Gram, che quadrano il condizionamento. Verifica residuo normale relativo<=1e-6 e finitezza, rango e scarto numerico riportati; mancata verifica → riferimento non verificato e arresto prima dei fit reali. Nessuna inversa ingenua; blocchi e maschere per giunto conservati.

Questo quantifica il divario dalla MSE train minima della classe **entro tolleranza numerica**. Non è l'ottimo di AdamW, della ridge regolarizzata o della E_dyn held. Divario alto può derivare da regolarizzazione/arresto, non è da solo gate di non convergenza. Per L0/L13 pubblicare MSE train e MSE/E_dyn interne a fronte della forma chiusa. La MSE train è ricalcolata sull'intero fitting con il checkpoint selezionato, eval/no_grad e dropout off: non è la loss accumulata con pesi in evoluzione durante l'epoca. Lo stesso forward diagnostico finale è previsto per tutte le traiettorie selezionate e i rifit, massimo37 valutazioni del fitting aggiuntive. Non riusare invariata la SSE pooled storica né confondere alpha>0 con un lower bound.

Due disegni L0/L13 per due fold outer, tre RHS-target (reale, controllo lineare, temporale):12 blocchi di soluzioni, riusando4 disegni. Calibrazione sotto: altri4 disegni L0 (due inner per outer), RHS-base reale,L,T,rumore. **Ogni disegno si moltiplica per i pattern distinti di pesi/maschere dei giunti**. L13 ha4992 feature; un fattore R4993² float64 circa190 MiB per pattern. Accumulo a blocchi e fattorizzazioni sequenziali entro24GiB; costo misurato, non assunto economico. Nessuna nuova predizione sui15 sviluppo per queste forme chiuse.

### 4.4 Due controlli positivi sui latenti train

Controlli di qualità della ricetta, **non certificati universali di convergenza**. Tre teste ciascuno: L0, MLP+affine, TCN-post+affine; due fold outer, stessa selezione MSE e ricerca finita, nessun rifit sui40 o accesso ai15 sviluppo. L13 reale ha il controllo descrittivo precedente; TCN-pre condivide l'architettura qualificata ma ha le proprie curve e allarmi reali.

Per direzione outer, generatori noti osservano solo z=LN(lat). `L(z)` è un'affine384→20; `T(z)` una TCN congelata della stessa struttura/RF con tutti i hidden larghi8 (inseribile nella TCN più larga), ReLU e **senza dropout**, uscita teacher non nulla. Pesi Gaussiani Kaiming negli hidden e Gaussiani con varianza1/fan_in nell'uscita/L, bias zero. Seed `20261118+1000*f` per L, +1 per T, +2 per rumore; mondo lineare a=0, temporale a>0. Si salvano stati/hash prima di qualificare le teste. Per giunto si centrano/normalizzano L e T usando solo il fitting outer e peso per finestra; deviazione<=1e-8 → controllo non calibrabile.

Bersaglio pulito noto `y*=L+a*T`; osservato `y=y*+sigma*epsilon`. Epsilon AR(1) stazionario rho=exp(-1/4), epsilon0~N(0,1) ed epsilon_t=rho*epsilon_(t-1)+sqrt(1-rho²)*eta_t, eta~N(0,1), indipendente per finestra/giunto. Per il rumore NumPy PCG64, seed64 ricavato dai primi8byte little-endian di SHA256 del JSON canonico [seed, metadata_finestra, giunto]; mai hash() di Python né valori delle feature/etichette. Il generatore teacher usa Torch CPU pinnato, stati salvati; sigma è la deviazione standard stazionaria. Stesse maschere/crop. L'oracolo è y* **funzione delle feature disponibili**, non fattori nascosti del generatore. Non è un bound campionario garantito sul noisy held.

Calibrazione prima di qualsiasi fit neurale: nei20 utenti fitting outer, due sottofold alterni10/10; riferimenti L0 pesati. Stima interna E_real sui target reali e curve sintetiche usando solo questi20. Griglia temporale finita a={0.2,0.4,0.6,0.8,1}, sigma={1,1.5,2,2.5,3}; lineare a=0 con la stessa griglia sigma. Le soluzioni sui RHS-base si combinano linearmente, senza rifattorizzare25 disegni. I punteggi inner si aggregano come rapporto delle somme SSE/targetSS, non media dei rapporti; denominatore nullo/non finito → controllo non diagnostico. Soglie dinamiche per coppia derivate solo dal fitting inner; mai dalla soglia storica val.

Una coppia temporale è ammissibile se `abs(E_LS,synth−E_real)<=0.05` e guadagno `(E_LS,synth−E_oracle)/E_LS,synth` in **[0.06,0.10]** sulle predizioni inner appaiate. Scegliere distanza minima da0.08, poi daE_real, poi a e sigma crescenti. Lineare: stessa tolleranza0.05, scegliere errore più vicino aE_real, poi sigma minore. Nessun candidato ammissibile → **controllo non calibrato**, nessun nuovo seed/griglia automatica. Parametri distinti per direzione outer; i20 held outer non li scelgono. Le soglie dinamiche dei gate outer sono mediana del fitting outer sul bersaglio sintetico scelto.

Qualificazione, sui held outer, per fold e sul pooled (tutte devono passare):
- **Lineare:** per L0, MLP e TCN recupero >=90% del margine costante→oracolo sia su MSE che su E_dyn: `(E_const−E_head)/(E_const−E_oracle)>=0.90`. Costante media per giunto pesata dal fitting outer; su E_dyn la previsione costante ha errore1. Margini non positivi/non finiti → controllo non diagnostico.
- **Temporale:** il riferimento LS-L0 fittato sul fitting outer deve ancora dare margine relativo oracolo in[6%,10%] su E_dyn; fuori intervallo → controllo non diagnostico, non fallimento della TCN. La TCN deve recuperarne **>=50%**, `(E_LS−E_TCN)/(E_LS−E_oracle)>=0.50`. Si riportano anche L0 neurale/MLP, numeratore e denominatore; MLP non deve vincere il temporale. Non si usa il +21,3% della ridge con alpha ottimizzato sullo stesso held della review.

Se un controllo calibrato/diagnostico non passa: **ricetta non ammessa dal controllo positivo**, causa non attribuita automaticamente a ottimizzazione. Se un allarme di §4.2 resta aperto: **ottimizzazione non risolta entro il budget**. In questi casi, e anche per controllo non calibrato o non diagnostico, stop prima dei fit reali, artefatti conservati, eventuale specifica nuova da firmare; nessuna etichetta scientifica negativa o ricerca finché si vince. Un successo non garantisce adeguatezza sui target veri. Tolleranze e griglie sono proposte ingegneristiche non ancora misurate.

### 4.5 Matrice e arresti

CV reale40 traiettorie iniziali, massimo60; controlli48, massimo72;15 rifit reali. **103 traiettorie iniziali, massimo147**, continuazioni non contate due volte. Con30 passi/fold e60/full:531.000 aggiornamenti iniziali massimi,1.458.000 con tutte le estensioni. Dai soli1907 input: somma passi fold<=61, full<=60; limiti prudenti **537.600 / 1.477.800**. Formula `11*L*E*(S0+S1)+15*E*Sfull`. Massimo40.128 valutazioni held di fold incluse150/300,3 quarti e passo0, più37 forward diagnostici finali sul fitting (22 selezioni di fold +15 rifit). Gli step reali si contano nel benchmark, non si assumono metà finestre per metà utenti.

NaN/Inf, divergenza non finita, OOM, fit incompleto/TIMEOUT, supporti o pin diversi fermano la fase; nessun retry cieco, seme sostitutivo o riduzione della matrice. Fit reali con allarmi aperti vengono conservati senza etichette inferenziali sui contrasti che dipendono da quella testa. Se L0 o TCN-post non è ammissibile, niente promozione primaria. Nessuna vittoria costruita su un comparatore inadeguatamente allenato.

<a id="p5"></a>
## 5. Lettura finale e limiti

Primario: **TCN-post ispirata a TEMPONet con ramo lineare vs L0**, riduzione relativa E_dyn dinamiche. Soglia finale storica `pose_var>=0.2574231028556824`, sull'intera finestra, invariata. E_dyn centra pred e target sui medesimi tempi validi per finestra/giunto (>=2); rapporto somme SSE/targetSS, non media dei rapporti. Prima si mediano gli accumulatori per utente dei tre semi: prestazione media, **non ensemble**.

Famiglia fissa, `gain=100*(E_ref−E_TCN)/E_ref`:
1. L0 (primario); 2. MLP (contesto rispetto alla sola non linearità istantanea); 3. L13 (contro questa lettura temporale lineare fittata); 4. ridge post storica; 5. testa storica teacher.

Bootstrap utenti appaiato2000, seed20261102; IC percentile99 ciascuno, Bonferroni nominale95 sulla famiglia. Condizionato a un encoder e ai tre semi di testa. Con15 utenti la copertura non è garantita; il96–97%/85–91% delle simulazioni Claude non è la copertura “vera” di questi dati e99,8% non è un rimedio universale. Manteniamo metodo/repliche proposti. Estremo dell'IC entro0,1 punti percentuali da0 o2: flag `al_bordo_MC`, etichetta invariata, nessun nuovo bootstrap/seme per spostarla.

Prima i gate di qualità, poi **prima condizione vera** fra:
1. `beneficio_pratico_risolto`: gain>=2 e lower99>0;
2. `deterioramento_pratico`: gain<=−2 e upper99<0;
3. `beneficio_2pct_non_sostenuto`: upper99<2;
4. `inconclusivo`: altrimenti.

“Passa” significa esattamente1, non il solo IC sopra zero. Non riusare `practical_assessment` del lungo: regole diverse. **Necessario e sufficiente per candidata alla migliore lettura della dinamica centrata**, condizionatamente ai gate comuni e all'ammissibilità di L0/MLP/L13/TCN-post: passano **1,4,5 tutti** e né2 né3 ha etichetta `deterioramento_pratico`. Se MLP o L13 sono risolti migliori, la TCN non è candidata preferita, pur conservando gli eventuali miglioramenti sui riferimenti storici; il mancato accesso alla candidatura non trasforma un IC inconclusivo in un verdetto negativo. TCN/MLP passa → vantaggio pratico del contesto in questo controllo; il contrario favorisce l'istantaneo in questa ricetta; inconclusivo non prova irrilevanza del tempo. TCN/L13 passa → vantaggio sulla ricetta lineare temporale, con cautela S-2. Nessuna attribuzione causale “pura” al ramo g, che può imparare componenti affini.

Il2% sotto la testa storica richiede point E_dyn<=**0.7630564046643697**, circa2,91% sotto ridge, oltre ai requisiti sugli IC e L0. Non basta arrivare al2% sotto la sola ridge. Il successo non autorizza adozione operativa o livello B. MAE/livello: riportare delta e IC95 descrittivo; delta positivo è peggioramento stimato, lower95>0 peggioramento risolto, nessuna nuova soglia di non inferiorità. I compromessi vanno discussi.

Descrittivi predefiniti, IC95 appaiati, nessun gate aggiunto: L0 vs ridge e testa storiche (sanità/ricetta, non prova di ottimo lineare); MLP vsL0; L13 vsL0; TCN-post vsTCN-pre; TCN-pre e ancora ridge pre dal report storico. L0 inferiore alle storiche invita a leggere le differenze di dose/loss/FP16 e i controlli train, non a scegliere un nuovo LR sui15. Post>pre indica accessibilità con questa testa, non origine/distruzione dell'informazione; nullo non rende equivalenti le viste.

E_dyn all/static, correlazione centrata, varianza pred/target, livello, MAE rad/gradi per seme e media aritmetica dei valori; SD campionaria fra3semi separata dall'IC utenti, non somma di errori. Contrasti1–3 per seme s contro s; nove incroci solo descrittivi. Riferimenti definitori: `consolidation_long_pose._metrics`, `tokenizer_pose_readout.summarize`, `emg2pose_pose.JOINTS`. Nessuna selezione di giunti/utenti; nessun bootstrap gerarchico aggiunto. Il “no” resta condizionato ai semi osservati, il2% non è un pavimento del rumore.

<a id="p6"></a>
## 6. Confronto raw e controlli rinviati

Encoder random, raw da zero, altre dosi e nuova inferenza raw non sono autorizzati qui. Servono a domande più ampie sul pretraining. In un eventuale confronto futuro: stessi40 utenti/esposizione/criteri, dichiarando sempre la supervisione FM su158.

**Rettifica del metodo raw di c15aac9:** l'uscita TEMPONet emg2pose stima già la media di un blocco di10 campioni. Per riprodurre i target FM occorre **un endpoint per blocco valido che finisca a10j+9**, finestra raw `[q−255,q]`, quindi aggregare gli stessi blocchi/maschere per patch, standardizzazione e crop6. Predire ogni campione e poi mediare farebbe una media triangolare spostata di4,5 campioni. Circa13,7 milioni di endpoint sui15 sviluppo, ~1,1 volte la lettura ufficiale (stima Claude, non benchmark nuovo), non dieci volte. Conservare uscita grezza senza EMA: alpha raw scelto sugli stessi15. Da verificare indici assoluti e bordo prima di una prova separata; nessuna esecuzione implicita.

<a id="p7"></a>
## 7. Livello B separato

Nessun automatismo. Se si propone la testa nel pretraining servono protocollo/firma separati: inizializzazione appaiata del resto del FM, dati/ordine, maschere, dose/schedule/EMA, controllo PoseHead, lettori comuni sui due encoder oltre alle teste proprie. Più capacità può assorbire il compito o migliorare i gradienti: questa prova non decide quale. Affrontare v10§6.5; non basta una loss inferiore.

Ordine storico33,88GPUh vale per **due training a19.618**, non49.046; a49.046 circa81–84GPUh prima delle letture/overhead, non preventivo. Nessun riuso automatico del C0 di altre linee o cambi ai loro contrasti. La testa TCN72.904 è circa0,24% del30M e8,6 volte L0: nessuna innocuità scientifica dedotta dalla percentuale.

<a id="p8"></a>
## 8. Implementazione futura, verifiche e responsabilità

Oggi solo documenti/JSON descrittivi. Dopo eventuale firma §10: GPT prepara codice e piano eseguibile, Claude fa **una review del pacchetto operativo completo** e resta unico operatore proposto di questa attività, submitter/ledger comuni. Dopo convergenza e firma, le fasi ammesse dai gate non richiedono nuove autorizzazioni ripetute. Un esito negativo dei gate torna alla decisione pertinente, non autorizza mutazioni. Nessun cambiamento alle altre linee.

Base futura: questo ramo derivato da cfa6462, file sperimentali solo in **scripts/pose_temponet/** e test nuovi separati; nessun .py nuovo al primo livello scripts/, che invaliderebbe la guardia storica. Nessun merge del ramo raw o cambi a src/, FMConfig/PoseHead/config storici. Manifest delle sorgenti originali prima/dopo. La versione attuale non implementa alcun modulo.

GateR-1, nel codice futuro **prima dei contrasti**, senza influire sulle scelte: dai file storici pinnati riprodurre ridge **0.7859020004638595** e testa **0.7786289843513977**, tolleranza assoluta1e-12;8.327 finestre dinamiche,20.119.920 elementi. Ritaglio nuove uscite160: `[6:154]`, confrontato con gli indici/array storici148. La `pred` storica nella cache160, ritagliata, deve essere uguale al payload trained_head148 sui validi (stessi dtype/valori, non nuova inferenza). Conteggi e hash del supporto da soli non bastano: un offset di6 può conservarli. La lettura dei soli riferimenti storici non è una lettura anticipata delle predizioni nuove.

Test futuri necessari: parametri/MAC/RF; identità L0 col ramo g nullo; gradienti dopoavvio senza gate morto; dropout e LN; crop/allineamento e mask; per-window WLS e rango; generatori riproducibili e isolamento outer/inner; shuffle identico; selettore/pareggi/bordi/estensioni/rifit; entrambe le curve e allarmeS-2; quattro etichette/candidata; gate globale prima dei fit; equivalenza fixtureCPU e impostazioni numeriche benchmark/fit. Nessun test runtime dichiarato eseguito oggi.

Claude conserva ricevute, log, manifest e report nel proprio futuro ramo di esecuzione, `results/infra/pose_temponet_proposal_20261117/execution/`; GPT ne rivede copie versionate senza riscrivere gli esiti. Lo stato REVIEW_REQUIRED passa solo dopo la review effettiva, mai in anticipo.

<a id="p9"></a>
## 9. Risorse, dipendenze e gate meccanico proposti

**Nuovo tetto proposto:24GPUh +80coreh CPU-only =272coreh allocate**, 272ore locali solo con billing8 per GPU e billing=CPU sulla seriale. Sostituisce il vecchio4/12/44, non è budget firmato. **Nessun tempo A100 misurato per questa combinazione**.

| Fase | GPU/CPU | RAM host GiB | Ore cumulative inclusive | GPUh | coreh CPU-only | ore locali condizionate |
|---|---|---:|---:|---:|---:|---:|
| Checkout/preflight | 0/2 |24|0,25|0|0,5|0,5|
| Benchmark CPU |0/8|24|0,5|0|4|4|
| Riferimenti WLS e calibrazione |0/8|24|8 (2 job)|0|64|64|
| Benchmark GPU |1/8|32|0,5|0,5|0|4|
| Controlli + selezioni + rifit |1/8|32|20|20|0|160|
| Inferenza finale bloccata |1/8|32|1|1|0|8|
| Metriche/bootstrap |0/8|24|1|0|8|8|
| Raccolta |0/2|24|0,25|0|0,5|0,5|
| Riserva totale |stessi profili|24/32|—|2,5|3|23|
| **Totale** | | | |**24**|**80**|**272**|

Ogni tetto di fase include la somma **(walltime richiesto+180s)×risorse**; non si aggiunge l'overhead una seconda volta. Seriali mai oltre4h inclusive: per i2 jobWLS richiesta massima03:57:00, 8CPU. Piano eseguibile: ID/job/output unici, packing dei candidati basato sul benchmark, massimo40 jobGPU complessivi e7CPU-only; se non entra, stop prima dei fit, nessuna frammentazione per aggirare tetti. La riserva copre avvii/sforamenti contabilizzati, non retry. Una nuova diagnosi/ripartenza richiede decisione registrata.

Ordini a priori, non prestazioni misurate:537.600 update iniziali massimi a4,1–18,5ms/update →0,61–2,76GPUh di soli update;1.477.800 massimi →1,68–7,59GPUh; col fattore2 del gate massimo3,37–15,19GPUh **prima di validation/I/O**. Tempi analoghi raw di Claude, non un intervallo statistico della nuova TCN. Il costo CPUWLS non si stima dal vecchio384: è limitato a64coreh e misurato. Non si presenta una stima puntuale totale non verificata.

Saldo storico **09/11/2026 12:21**, `fccf88d`, `remote_listing.txt`:2.000.000−1.308.137=**691.863 ore locali**. Imputazione proposta: nuovo sottobudget “lettore posa TCN v2”;272 sarebbero0,03931% e lascerebbero691.591 **solo rispetto a quella fotografia**, non il residuo attuale. Quota933,1G/1T del16/11 è pure storica.

**Benchmark train-only** con stessi loader/file placement, cache cast, flag numerici, thread e hardware dei fit. Per ogni testa, fold/full: warmup1epoca esclusa,10epoche temporizzate, statistiche mediana e p95; coefficiente di proiezione **p95** con interpolazione lineare esplicita. Misurare train, eval, decompressione, salvataggio pesi/stato, avvio; cuda.synchronize ai limiti. Il benchmark di costo non sceglie LR/epoca/generatore. Una sola versione della politica; pin delle ricevute prima dei fit.

GateGPU: `2*(somma_epoche_train*p95_train + somma_eval*p95_eval + I/O + avvii + salvataggi)` per la **matrice massima**6LR300,15rifit300,132CV,40.128eval held e37eval del fitting selezionato;<=20GPUh e packing entro lo stesso tetto. Benchmark sulsolo train può misurare poche epoche, ma la proiezione include i controlli. Inferenza:2×proiezione15modelli×18.548finestre più scrittura3,32GiB e lettura riferimenti<=1GPUh. I/O scala con max(rapporto byte,val/train, rapporto finestre); tempi di inferenza per batch e ultimo batch reali.

GateCPU: misura separata formazione statisticheWLS, fattorizzazione per pattern, RHS/predizione, calibrazione25combinazioni, sintesi/bootstrap e serializzazione. Proietta tutti i pattern/disegni §4.3 con fattore2 più avvio/I/O; WLS/calibrazione<=64coreh e ciascun job<=4h inclusive, report<=8coreh. Se un gate non passa **nessun fit neurale**, inclusi controlli; benchmark può fare calcolo di misura ma non produce un candidato scientifico selezionato. Dopo i gate risorse si eseguono WLS/calibrazione e qualificazione; i fit reali richiedono anche gate positivi e riferimenti validi.

Memoria: CPUhost24GiB/GPUhost32GiB, VRAM gate48GiB su A10064. Misurare piccoRSS/VRAM e proiezione `1.25*(parte_fissa+byte_buffer_di_una_vista_val_e_un_batch)` usando un surrogato train ripetuto con le dimensioni dichiarate della val, senza leggerne etichette; include solver, copie e compressione. Niente accumuli per tutti gli utenti o20Gram se oltre il cap. Outputprocessati a flusso, max8GiB persistenti (15pred FP32=3,31664GiB); no pred interne di tutte le epoche, solo curve/checkpoint necessari. Conservare massimo gli stati correnti/seletti per candidato; previsione spazio misurata prima. Hashinput si possono leggere senza ispezionare nuovi punteggi.

**SCRATCH: dipendenza esplicita.** Le cache/predizioni principali sono suWORK, ma pose_stats.json e i riferimenti B19618 train_subset.npz(257.675.076byte)/val.npz(2.485.295.825byte), aperti dai validatori storici, sono suSCRATCH. Questa v2 sceglie di **mantenerli come prerequisiti di lettura**, non di rimuovere i controlli o copiarli senza autorizzazione. Path/pin nei manifest storici; hash prima del preflight. Se assenti/scaduti stop, nessun download/ricostruzione automatica. Il rinfresco12/11 riportato da Claude implica scadenze da22/12 solo se realmente applicato; rinfresco03/12 non promesso da questo piano. Nessuna dichiarazione di indipendenza daSCRATCH.

Preflight operatore: quotaWORK>=**12GiB+scritture pendenti delle altre linee** (8output+4margine), spazio temporaneo/VRAM, saldo e billing/AllocTRES primojob, prenotazioni, corsia seriale condivisa (non presumere libera), integritàinput/sorgenti e budget residuo. Fermo23/11 08:00–25/11 08:00 da registri, da rileggere prima dell'esecuzione: non promettere completamento entro una finestra di coda. Nessun SSH/controllo cluster effettuato in questa revisione.

<a id="p10"></a>
## 10. Tabella per la firma di Simone, dopo la review

| Decisione | Proposta da firmare, non già autorizzata |
|---|---|
| Oggetto | TCN compatta **ispirata a TEMPONet con ramo lineare**, non classe raw invariata né suoi pesi;72.904parametri |
| Livello | Solo lettori di cache teacher T100-G30M49.046 congelato;40train/15sviluppo |
| Bracci | L0, MLP+affine, L13, TCN-post+affine, TCN-pre+affine; due riferimenti storici e descrittivi dichiarati |
| Ricetta | §3–4:4LR iniziali/max6,150→300, MSE,3semi rifit; due controlli positivi e WLS descrittiva; estensioni finite |
| Criteri | §5:5IC99,2000bootstrap, soglia dinamica storica finale, quattro esiti ordinati; gate qualità precedenti |
| Esito positivo | Candidata al migliore lettore della **dinamica centrata** iff1/4/5 passano, nessun deterioramento pratico2/3 e gateammissione; nessuna adozione o livelloB automatici |
| Esito negativo/irrisolto | Chiusura di questa prova secondo il suo esito; nessuna conclusione di assenza d'informazione nelFM; nessuna ricerca automatica |
| Tetto | **24GPUh,80coreh CPU-only,272coreh/orelocali condizionate,8GiBoutput**; benchmark dell'intero massimo prima dei fit |
| Ruoli | GPT autore scientifico/codice/piano eseguibile; Claude revisore e unico operatore **di questa attività** |
| Portata futura della firma | Autorizza implementazione e, dopo una review favorevole del pacchetto operativo, benchmark e catena prevista entro tetti/gate, senza nuovi consensi per ciascun invio. Non autorizza un'estensione scientifica o retry. |

**Oggi la firma non c'è.** L'istruzione del18/11 autorizza solo la presente revisione documentale, commit/push e prompt all'ultima review. Dopo questa rilettura Simone decide sulla tabella. Il successivo pacchetto di codice/operatività sarà una consegna distinta, non un'altra revisione del disegno già convergente salvo modifiche sostanziali.
