# Valutazione scientifica dopo la tranche R3c/E3/J2

28/10/2026, GPT. Richiesta di Simone: valutare il punto raggiunto con il materiale disponibile e stabilire se servano nuove esecuzioni.
Fonti principali: consegne Claude `6aba4a7` (sonde/E3) ed `e2e525e` (audit); letti anche `fa07e8b` e `63acf94` per lo stato successivo.
Nessun nuovo job, training, estrazione o modifica al codice Claude. Ricalcolo locale dei JSON con sola libreria standard, review scientifica
indipendente di audit e sonde, snapshot remoto di sola lettura tramite leonardo-ops.

**Il materiale basta per una valutazione dello stato attuale. Non serve lanciare altro per questa lettura.** La costruzione dei bersagli
supera i controlli campionati, R3c e' un controllo utilizzabile senza vantaggio ampio e uniforme su R0, la ridge sui latenti offre un modesto
vantaggio sul costante, e R3 non ha ancora dimostrato apprendimento utile della posa. La causa del dbg resta aperta. B e' una prova
pertinente per vedere se il training impara con una dose maggiore, non una condizione necessaria per formulare questa valutazione.

La consegna notturna contiene errori di lettura dei risultati, distinti dalla review del codice `aecff1a`, i cui rilievi sostanziali
restano chiusi. In particolare non sono sostenuti «allineamento 10/10», «posa assente dai latenti» e «q debole misurato».

## 1. La catena: rassicurante sul campione, non certificazione universale

20 registrazioni, 10 utenti, 221 finestre: **100/100 controlli aggregati verificati**, nessuno fallito o non verificato. Sono cinque tipi
di controllo per registrazione (identita'/sidecar, raw EMG/processato, letture diretta/da blocco, ricostruzione posa, oracle delle patch),
non 100 esperimenti indipendenti. Il risultato riduce molto la plausibilita' di un errore sistematico della catena campionata.

Formulazione sostenuta: **nessun difetto rilevato nei controlli sui 20 casi esaminati**. Non equivale a escludere ogni difetto del corpus,
ogni percorso del training o una questione di definizione del target. La precedente review di codice e test fornisce ulteriori controlli,
ma nessuna di queste evidenze identifica da sola la causa del collaudo fallito.

Supporto delle patch: **92,523% piene, 7,158% vuote, 0,320% parziali**. Le patch con supporto parziale sono marginali in questo campione.
La frase della consegna sulle righe a zero va chiarita: **16/20 registrazioni hanno righe raw a zero**, fino al 16,69%; la quota invalida
raggiunge il 26,84%. Le righe non finite sono invece zero. Se si intendeva «nessuna riga nulla trattata erroneamente come valida», va
scritto cosi'; il JSON non dice che gli angoli grezzi siano sempre non nulli.

## 2. Allineamento e ridge RMS: rettifiche necessarie

`disallineato.stato="verificato"` indica che il confronto e' stato eseguito; **non significa che la predizione allineata sia migliore**.
Ricalcolando `MAE_disallineata - MAE_allineata`, il gap e' positivo in **7/10 utenti**, negativo in tre, non positivo in tutti e dieci.
I tre gap negativi sono -0,000548, -0,002591 e -0,008099; gap relativo mediano **+1,959%**. I dati aggregati del rapporto non permettono
di stimare un intervallo appaiato per finestra. Dunque «allineamento dimostrato 10/10» e' una lettura errata, non un nuovo bug dell'audit.

La ridge RMS batte il costante in **8/10 utenti**, non sette. Mediana dei rapporti MAE **0,849446**, dinamiche 0,855108, range 0,288–5,474;
cinque utenti sotto0,9. Ci sono 133 finestre fit,21 gap,67 test e9.845 patch test; disallineato su59 coppie/8.059 patch.

Questi numeri **non stimano una proprieta' uniforme del segnale**:

- In otto utenti, ultima finestra fit e prima test sono nello stesso recording; il fit ha gia' osservato quel contesto.
- I due peggiori casi (1,579 e5,474) passano invece a una mano/fase diversa e a test unprompted, senza finestre del recording di test
  nel fit. La dispersione confonde cambio di distribuzione, mano/fase, baseline e accessibilita'.
- **7/10 predittori battono ancora il costante sui target disallineati.** Il miglior rapporto allineato,0,288, appartiene a un caso
  unprompted che ha anche gap negativo; sul disallineato il rapporto e' circa0,286. Una forte riduzione rispetto alla media del fit
  puo' quindi riflettere postura/distribuzione del contesto, senza dimostrare capacita' di seguire il movimento specifico.
- Sei alpha su dieci sono al massimo della griglia,1000: questa piccola ridge non e' una stima del migliore decoder possibile.

E' corretto parlare di **predizione variabile nelle condizioni provate**; non di q misurato piccolo/moderato, segnale intrinsecamente
debole o limite della predicibilita' della posa. Il rapporto MAE non e' la frazione di varianza spiegata del sintetico.

La scansione dei lag resta esplorativa. Righe delle feature comuni, contesto temporale e bersagli/baseline che variano col lag non
permettono di chiamarla misura del ritardo elettromeccanico. Non cambierei lag/target ne' richiederei altri audit generici come
prerequisito obbligatorio di B sulla base di queste curve.

## 3. E3 su R3c: sotto criterio, non assenza dimostrata di informazione

| Supporto | Ratio MAE ridge/costante | Riduzione MAE [IC95 utenti] | Utenti sotto1 / sotto0,9 |
|---|---|---|---|
| Dinamiche,8.329 finestre | 0,949279 | 5,07% [3,72%;6,82%] | 15/15 / 1/15 |
| Tutte valide,16.684 finestre | 0,925558 | 7,44% [5,50%;9,55%] | 15/15 / 3/15 |

La regola firmata chiede limite superiore del ratio e mediana utente sotto0,9 sulle dinamiche: **non superata**. Tradurre
`accessible=false` con «non raggiunge il criterio operativo del10%», non con «posa non accessibile linearmente» in senso assoluto.

La ridge ha intercetta stimata sul sottoinsieme train, mentre il costante e' la media globale del train (zero standardizzato).
Il rapporto non separa la correzione marginale dell'intercetta dall'informazione dinamica delle feature; per R3c senza testa il report
non fornisce E2. Evidenza stretta: **questa ridge batte questo costante**. Non quantifica da sola la dipendenza dal movimento.

Inoltre la sonda vede il teacher R3c postdecay dopo49.046 passi, senza maschera; R3 dbg allenava lo student con maschere, LayerNorm nella
testa, ancore e rappresentazione in evoluzione per1.300 passi. Il risultato E3 non misura i latenti iniziali di R3, non ne stabilisce il
limite raggiungibile e non separa H1 da H2. I campioni sono utenti distinti (40train,15val); resta un checkpoint e una famiglia di sonde.

Eventuali letture aggiuntive come costante con intercetta stimata sul medesimo subset, MSE standardizzata o ridge su LayerNorm si
potrebbero progettare sulle cache esistenti, senza nuova estrazione GPU. Non sono necessarie per le conclusioni attuali e non sono
state eseguite o autorizzate come nuovi job in questa valutazione.

## 4. R3c contro R0: nessun vantaggio ampio e uniforme delle sole ancore

| Due epoche | Delta punti percentuali [IC95 appaiato] |
|---|---|
| P1 pool | -0,19 [-0,73;+0,33] |
| P1 decoder | +0,87 [+0,48;+1,29] |
| P2 pool | +0,99 [-0,53;+2,32] |
| P2 decoder | +0,36 [-0,56;+1,32] |

Nessuno degli otto confronti a una/due epoche supera insieme +1 punto e limite inferiore positivo. P1 decoder mostra un guadagno piccolo
a entrambe le epoche; P1 pool a una epoca peggiora di0,55 punti [−1,02;−0,10]. Non e' prova di equivalenza o di assenza generale di danno.
Un seme per modello, otto letture senza correzione multipla, pendenze fra epoche senza intervallo appaiato: mantenere la natura descrittiva.
R3c e' utile come riferimento per attribuire l'eventuale contributo della posa in R3; le sue sonde non dimostrano l'esito della posa.

## 5. Che cosa resta da distinguere e che cosa fare

| Ipotesi/domanda | Valutazione attuale |
|---|---|
| H3, errore sistematico della catena | Molto meno plausibile dopo audit e test; nessun difetto concreto dimostrato, non esclusione universale |
| H1, dose dbg insufficiente | Plausibile:42 aggiornamenti al picco, dose LR671, pannello originario piccolo; sintetico illustrativo, non previsione |
| H2, rappresentazione/obiettivo non rendono apprendibile la posa | Ancora aperta; E3 R3c sotto criterio non dimostra incapacita' di R3 |
| R3 apprende con ottimizzazione piu' lunga? | Non osservato: richiede un tratto nuovo di training |
| Posa generalizza e trasferisce? | Non dimostrato: servono poi E1/E2 e confronti di trasferimento sul candidato effettivo |

**Decisione raccomandata ora:** accettare la tranche come diagnosi tecnica utile, correggere le sovrainterpretazioni, lasciare fermo il
training completo e finalizzare la preparazione di B. Non alzare peso, non cambiare target/lag o architettura sulla premessa non misurata
di q debole. Non occorre un nuovo prompt o un nuovo job per ottenere questa valutazione: e' il lavoro svolto qui.

**B e' il prossimo test diretto di apprendibilita' a dose maggiore**, se Simone ne firma criterio e lancio: ricetta vera R3, un solo
collaudo da5.000 passi, pannello fisso verificato, confronto appaiato con passo0, baseline/varianza delle predizioni, M280/peso invariati,
traccia del training e budget definitivo prima del via (stima precedente5GPUh). La dose LR fino5000 e'4019,5 contro671 del dbg; non
sono passi Adam equivalenti. La bozza di criterio precedente resta una proposta da rivedere sul codice/pannello definitivo.

B non e' una prova causale pura di H1: piu' aggiornamenti cambiano anche l'encoder, e H1/H2 possono coesistere. Se migliora, mostra
apprendimento nella ricetta estesa sul pannello, non da solo generalizzazione o utilita' del foundation model. Se non migliora, indebolisce
la spiegazione «basta questa estensione», senza provare impossibilita' della posa. **E3 R0 e J2 bis non sono prerequisiti obbligatori**;
nessun via automatico a R3 completo deriva dall'esito di un controllo sul train.

## 6. Stato operativo verificato, distinto dai risultati scientifici

Il checkout Claude osservato e' `63acf94` con modifiche NON committate a trainer/loader/script/test del pannello B. Non sono state
modificate o sottoposte qui a review: occorre uno SHA stabile per valutarne la consegna. La documentazione registra una successiva
autorizzazione a E3 R0/J2 bis; lo stato live e' pero' quello sotto, non un completamento di quei risultati.

Snapshot **28/10/2026 00:08 CET**, clock UTC coerente, sola lettura tramite leonardo-ops:

| Job | Esito |
|---|---|
| 62478383, R0 val | CANCELLED alle00:03:21,6min41s, Slurm UID127686 |
| 62478387, R0 train | COMPLETED,3min04s; cache1.907 finestre |
| 62478392, J2 bis | CANCELLED alle00:03:21,6min55s, Slurm UID127686 |

Coda vuota. Nessun rapporto E3 R0 o J2 bis disponibile; nessuna attribuzione della cancellazione a una persona. B non compare fra i lanci.
Non rilanciare automaticamente. Lo stato precedente «tutto consegnato» vale per R3c/E3/J2 del27/10, non per questi tentativi successivi.
Le note con orari01:00 sono successive allo snapshot secondo il loro testo: per la fotografia operativa prevalgono sacct/squeue/log.

## Riproducibilita'

`docs/reviews/analyze_pose_delivery_20261028.py` ricalcola aggregati, gap, conteggi, riduzioni MAE e confronti dalle consegne originali.
Output `docs/reviews/pose_delivery_recalculation_20261028.json`, con hash SHA256 degli input. Non carica modelli, raw wearable o cache
di feature, non usa CUDA e non esegue training. Il giudizio e' stato confrontato con due letture scientifiche indipendenti.

```sh
python3 docs/reviews/analyze_pose_delivery_20261028.py \
  /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine \
  --out docs/reviews/pose_delivery_recalculation_20261028.json
```
