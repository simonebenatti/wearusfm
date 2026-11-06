# Tokenizer: 25/100 ms e non linearità — protocollo proposto del 02/11/2026

**Aggiornamento03/11 dopo la review Claude:** questo testo conserva la proposta iniziale.
Per il seguito vale l'[emendamento e risposta punto per punto](reviews/risposta_claude_tokenizer_20261103.md):
raggio0 esplicito nei tre bracci con JEPA0, EPN-612 come domanda principale distinta dalla posa,
gain scalare conservato e nessun pavimento statistico da una sola replica. Usare le
[configurazioni aggiornate](../results/infra/tokenizer_review_20261103/comparison_configurations.json),
non i JSON del02/11 a raggio2. Nessun nuovo training è stato lanciato.

**Stato: bozza scientifica concreta, prima dei nuovi risultati; non è un piano di sottomissione.**
Simone ha autorizzato di procedere con 100 ms e passo 25 ms e di provare successivamente un tokenizer
temporale non lineare, possibilmente apprendibile. Questa bozza traduce la direzione autorizzata in
confronti controllati; non attribuisce a Simone una firma delle nuove soglie numeriche o un budget
non ancora riconciliato. Nessun job è lanciato da questo documento. Codice, configurazioni complete,
hash, review, tempi misurati e disponibilità di budget devono entrare nel piano operativo prima dei job.

Riferimenti: [configurazione 100/25 già disponibile](../results/infra/frontend_window100_20261102/recipe.json),
[tre configurazioni complete verificate dalla CLI](../results/infra/frontend_window100_20261102/comparison_configurations.json),
[risultati B19618](reviews/b19618_results_20261029.md), [stato scientifico](piano_scientifico_20261102.md).
Le soglie indicate come **PROPOSTA NUOVA** sono parametri di questa bozza, non criteri storici firmati.
La versione esecutiva dovrà fissarle prima di leggere risultati dei nuovi modelli.

## Domande e tre bracci

1. Integrare 100 ms nominali prima dell'aggregazione dei canali produce una rappresentazione più utile
   rispetto a 25 ms, mantenendo passo, compito, dati e supporto della supervisione?
2. A 100 ms, una non linearità sulle risposte dei filtri temporali migliora rispetto alla stessa banca lineare?

| Nome | Finestra nominale | Passo / overlap nominale | Risposte del banco prima della proiezione | Raggio comune delle loss mascherate |
|---|---:|---:|---|---:|
| T25-L | 25 ms (`window_ms=null`) | 25 / 0 ms | Identity | 2 |
| T100-L | 100 ms | 25 / 75 ms | Identity | 2 |
| T100-G | 100 ms | 25 / 75 ms | GELU | 2 |

Contrasti principali: **T100-L meno T25-L**, poi **T100-G meno T100-L**.
T100-G meno T25-L è il confronto pratico secondario: serve anche quando i due passaggi hanno effetti opposti.
T100-G si prepara subito dopo la verifica tecnica del percorso a 100 ms: non richiede che T100-L abbia
già superato un criterio scientifico. Nessuna ricerca di finestre, attivazioni o larghezze sulla val.

Tutti sono nuovi FM, addestrati integralmente. Non si sostituisce il tokenizer di un checkpoint da 25 ms
durante l'inferenza e non si usa `resume`/`init-from` del B storico. Il nuovo T25-L è necessario perché
il B storico aveva raggio 0: una sua comparazione diretta con T100-L confonderebbe finestra e target allenati.

## Che cosa cambia nel tokenizer

Il percorso attuale è una banca di kernel continui **già apprendibili**: Fourier, spline e MLP che genera
kernel in funzione del tempo. A pesi fissati, l'integrale kernel×EMG è lineare rispetto al segnale;
la MLP generatrice del kernel non rende non lineare questa mappa ingresso→token.

Con le 64 risposte del banco `e(x)`, il confronto a 100 ms è:

- T100-L: `token = Linear(fe_gain * e(x))`;
- T100-G: `token = Linear(GELU(fe_gain * e(x)))`.

La GELU non aggiunge parametri né contesto temporale. I kernel e la proiezione continuano a ricevere
gradienti dalle stesse loss: è una prima prova di tokenizzazione temporale non lineare apprendibile,
senza CNN aggiuntiva, nuovo readout o aumento della larghezza. Non esaurisce gli encoder temporali non lineari.
Si conserva la calibrazione sintetica esistente di `fe_gain` prima dell'attivazione; non si introduce
una nuova normalizzazione dopo GELU. La diversa distribuzione delle risposte è parte dell'intervento:
media, deviazione, rango e gradienti iniziali sono registrati, non corretti mediante tuning sulla val.

T100-L e T100-G hanno esattamente gli stessi parametri. T25-L e T100-L mantengono 64 uscite e D384,
ma a 100 ms le basi scalano come nell'implementazione esistente: 104 armoniche, 80 nodi, omega 24,
contro 26/20/6. Sono 5.184 parametri aggiuntivi del frontend. Il primo contrasto riguarda la finestra
con questa regola delle basi, non un effetto della durata a numero di coefficienti rigorosamente identico.

L'anti-aliasing e la frequenza nativa rimangono attivi. Il contesto numerico è 100 ms per lato oltre
la finestra nominale: circa 225 ms complessivi per T25-L e 300 ms per i due T100. A 200 Hz la finestra
nominale contiene rispettivamente 5 e 20 campioni; il passo resta 5 campioni e l'uscita 40 token/s.
Le code anti-alias del banco corto non equivalgono a una finestra nominale lunga. Il modello resta
bidirezionale; questo esperimento non dimostra funzionamento causale o latenza online di 25 ms.

Il precedente [R5 a 150 ms](../results/pilot_obiettivi_20261024/README.md) non ha prodotto un vantaggio
aggregato chiaro: a due epoche P1 −0,53 pp, P2 +0,35 pp rispetto a R0. Aveva JEPA attivo, nessuna posa
e quote delle maschere diverse. È evidenza contraria alla promessa generica «più lungo basta», ma
non è questo confronto a 100 ms con obiettivo B, supporto comune e non linearità separata.

## Obiettivo, dati e appaiamento

Base: [configurazione B del 28/10](../results/infra/pilot_posa/config_b_atteso_20261028.json).
Per tutti: ancore 0,2, posa 1, JEPA 0; medesime definizioni delle ancore multi-scala e della posa,
testa di posa inizializzata a zero, riduzione per finestra, mano non sollecitata inclusa.
RVQ resta **spento**, come in B (`rvq_codes` e `rvq_codes_root` null); niente rami spettrali,
keep-readout, loss di velocità o altri termini nuovi. La risposta è condizionata a questa ricetta:
non dimostra indipendenza dall'obiettivo né impone di ripetere ora tutti i bracci anche con R3c.

Si mantengono corpus completo e manifest nested=1 di B, split e normalizzazioni train-only,
montaggi virtuali, frequenze native, finestre di training 1–4 s e filtro del loader.
Tutti: preset 30M/D384/K64, batch globale 32, seed 0, LR 1e-4, warmup 1.962 passi, weight decay 0,05,
gradient clipping 1, EMA 0,996. La ricetta WSD di riferimento resta `total_steps=49046`, decay_fraction=0,2:
5.000 e 19.618 sono snapshot prima del decay, non finali post-decay né autorizzazione a completare 49.046 passi.

Stesso stream, origine e ordine delle finestre, maschere, montaggi, target e normalizzazione.
Stesso seme non basta: il preflight confronta gli hash dei batch consegnati e dei target effettivi.
I pesi iniziali comuni devono coincidere; i frontend T100-L/G devono coincidere prima di applicare
l'attivazione. Il diverso numero di coefficienti in T25-L non deve cambiare l'estrazione dei pesi
dei moduli condivisi o lo stream dei dati. Registrare separatamente hash dei moduli comuni e del frontend.
Le configurazioni pubblicate conservano checkpoint a 5.000 e 19.618 e accendono il log dell'origine
dei batch in tutti i bracci. La CLI ricostruisce esattamente B storico dopo l'aggiunta dei soli default
compatibili; le differenze dei tre candidati sono verificate contro una lista esplicita.

## Supporto delle loss: controllo comune indispensabile

Le maschere raw restano quelle storiche: ratio 0,5, quote corta/media/lunga/spaziale
0,15/0,25/0,20/0,40, medesime geometrie. L'override esplicito `JEPAConfig.masked_loss_radius=2`
si applica a **tutti e tre**; deve essere almeno il `core_radius` richiesto dal tokenizer.
`None` conserva il comportamento storico per gli altri run.

La selezione delle query e i conteggi delle loss usano lo stesso override. Vale anche con JEPA=0,
perché le ancore leggono dalle query mascherate. La posa mantiene il proprio supporto per istante
e la propria maschera annotata: non viene erosa come le query delle ancore. Il raggio 2 protegge
il supporto nominale da 100 ms, non le code anti-alias né il contesto globale del teacher.
I campioni occultati sono azzerati prima del tokenizer; le chiavi nascoste restano escluse da local
encoder e Perceiver. La garanzia riguarda il segnale ricevuto dal modello, già filtrato dal loader.

Prima del training misurare, su batch train identici, i conteggi sopravvissuti per tipo di maschera,
dataset, finestra e componente di loss; verificare identità tra bracci e corrispondenza con le riduzioni DDP.
I buchi isolati da 1–2 patch non producono query core, quelli da 4–12 ne conservano al massimo 0–8;
sovrapposizioni e bordi possono cambiare i conteggi. Non scambiare la quota raw con la quota effettiva della loss.
RVQ non richiede un gate di copertura perché è spento. Se il supporto delle ancore attive risulta
inadeguato o vuoto, ridefinire una ricetta comune **prima** del training, non correggere solo il candidato.

## Dose, fasi e condizioni di avanzamento

1. **Collaudo tecnico comune:** geometria, anti-alias tra frequenze, integrità delle maschere,
   invarianza dello student alle perturbazioni dei campioni occultati, conteggi delle loss,
   gradienti finiti e non nulli nei kernel apprendibili, throughput e memoria. Eventuali 1.300 passi
   sono un collaudo numerico, non una misura sufficiente di utilità scientifica.
2. **Screening a 5.000 passi per tutti:** pannelli fissi di training e posa fuori dal train;
   lettura congelata della regressione descritta sotto. Non scegliere checkpoint secondo la val.
   P1/P2 completi sono previsti all'endpoint, per contenere costo e ripetizioni della validazione.
3. **Endpoint scientifico proposto a 19.618 per tutti:** stessi passi e batch consegnati, configurazione
   invariata e ripresa dal checkpoint proprio di ciascun braccio. Richiede copertura del budget totale
   verificata nel piano; **non è una continuazione automatica** del solo screening.

La dose è motivata dai risultati B: tra 5.000 e 19.618 passi l'errore dinamico della testa passa
da 0,927 a 0,867 e la varianza predetta da 6,3% a 15,0%. A 5.000 può emergere un segnale positivo,
ma un delta nullo/incerto non autorizza la conclusione «100 ms/GELU non funziona».
Se il budget consente solo 5.000, si consegna uno screening incompleto rispetto alla domanda finale.
Si ferma per errore tecnico o instabilità confermata, non perché un singolo checkpoint intermedio
non raggiunge una soglia statistica. Non proseguire soltanto il vincitore e confrontarlo con controlli più corti.

## Valutazione congelata prima dei nuovi risultati

**Regressione della rappresentazione.** Sorgente proposta: teacher per tutti i bracci, anche nelle
sonde di classificazione. Lettore istantaneo sulla media K64, D384, LayerNorm stateless FP32 eps 1e-5;
niente PCA, concatenazione di lag o selezione di slot. Ridge FP64 con intercetta non penalizzata,
stessa griglia E3, selezione train-only sui due fold di utenti alternati ordinati e refit sul train.
Griglia iniziale 10^-2…10^4, fino a tre estensioni per lato, pareggio al più piccolo alpha;
minimo ancora al bordo: problema di selezione da risolvere prima della lettura val, non tuning sulla val.

Stesse identità di finestre della precedente campagna: 40 utenti train, 200 registrazioni/1.907 finestre;
15 utenti val disgiunti, 1.950 registrazioni/18.548 finestre. Riutilizzare CSV, manifest, scale,
pose_stats e indici originali delle registrazioni; verificare hash e identità dei target/mask/std/pose_var.
Sono utenti già osservati nelle analisi precedenti: è validazione di sviluppo, non test intatto.
Le nuove feature vanno estratte dai nuovi checkpoint; quelle di B/R3c storici non sono sostituti.

**Supporto temporale comune nuovo:** `6 <= t` e `t+6 < n_patches`, per training della ridge e scoring
di tutti i bracci. Protegge i bordi anche per il supporto numerico da circa 300 ms dei T100.
Il bersaglio resta la posa della patch centrale da 25 ms, senza media su 100 ms o offset temporali.
Maschera annotata centrale invariata; per la metrica centrata una cella finestra/giunto deve avere
almeno due istanti validi. Dinamiche: `pose_var` originale della finestra intera >= **0,2574231028556824**;
non ricalcolare soglia o appartenenza dopo il taglio. Congelare e riportare i nuovi conteggi comuni.

Primario: `E_dyn = sum((pred-centered - target-centered)^2) / sum(target-centered^2)` sulle dinamiche,
con centratura temporale separata per finestra/giunto sul medesimo supporto valido.
Secondari: MAE in radianti, correlazione centrata, varianza predetta/target, distribuzioni per utente;
tutte le finestre e statiche restano secondarie. La varianza predetta da sola non basta: rumore o
riscalatura possono farla salire senza migliorare il movimento. La testa student resta diagnostica
separata, per leggere la traiettoria di apprendimento; non seleziona il teacher né gli alpha.
Non confrontare direttamente i nuovi numeri teacher/bordi6 con le vecchie metriche student/bordi4.

**Classificazione della rappresentazione.** Teacher allo stesso passo; pool768 senza PCA, stessa
ricetta convergente delle sonde e stesse identità OOF. P1 = 8/13 DB2 + 3/13 DB3 + 2/13 DB6;
P2 = 1/2 EPN612 + 1/2 UCI. Balanced accuracy; delta in punti percentuali. Fit e selezione soltanto
sui fold train previsti. Verificare convergenza e hash di righe/etichette/soggetti/fold prima di confrontare.
Non aggiungere MLP, decoder o selezione del readout secondo i nuovi risultati.

Per i tre contrasti registrati: 2.000 bootstrap appaiati per utente, seed 20261102, stessi ricampionamenti
tra bracci; nella classificazione campionare entro dataset prima dei pesi aggregati congelati.
Intervalli percentile 95%, statistiche e denominatori per utente salvati per replica indipendente.
Gli IC sono esplorativi e condizionati al seme e alle sonde; non includono variabilità del pretraining
né correzione globale per tutti gli endpoint. Nessuna affermazione di adozione basata sul miglior IC trovato.

## Criteri di utilità proposti, non soglie storiche

Da fissare nella versione esecutiva prima dei risultati:

| Parametro | Valore proposto | Uso e motivazione |
|---|---:|---|
| `min_relative_dynamic_error_gain` | **0,02 — PROPOSTA NUOVA** | Almeno 2% relativo di riduzione di E_dyn; evita di promuovere effetti minuscoli come lo 0,0123% del controllo slot o lo 0,49% del lettore temporale B. Non è una soglia clinica. |
| `classification_noninferiority_margin_pp` | **1,0 — PROPOSTA NUOVA** | Massima perdita considerata tollerabile per ciascuno di P1/P2; usa la scala pratica di un punto già discussa, ma il criterio di non inferiorità qui è nuovo. |
| `paired_ci_level` | **0,95** | Stessa convenzione descrittiva dei controlli precedenti; nessuna precisione aggiuntiva inventata. |

Per ciascun confronto principale si dichiara **segnale utile alla regressione** se la riduzione relativa
puntuale `(E_control - E_candidate)/E_control` è almeno 0,02 e il limite superiore dell'IC del delta
assoluto di E_dyn è <0. La soglia sulla stima puntuale non equivale a provare che tutto l'effetto superi 2%.

Si dichiara **classificazione conservata entro il margine proposto** solo se il limite inferiore dell'IC
del delta è >−1 pp **sia in P1 sia in P2**. Un IC che attraversa il margine non prova conservazione,
ma non è da solo prova di danno. Eventuali miglioramenti di MAE/ampiezza sono riportati separatamente;
la soddisfazione del criterio centrato non autorizza a chiamare migliore ogni aspetto della posa.

| Esito all'endpoint | Lettura consentita |
|---|---|
| Regressione utile e P1/P2 entro il margine | Candidato utile su queste misure; merita conferma, non adozione generale automatica. |
| Regressione utile ma perdita di classificazione | Beneficio di posa con compromesso/specializzazione; non soluzione generale al trasferimento. |
| Regressione utile, conservazione P1/P2 non dimostrata | Risultato promettente ma incompleto sulla non inferiorità. |
| IC di regressione comprende zero | Effetto non risolto alla dose e con il seme osservati, non equivalenza. |
| Migliora solo la testa student | Apprendimento del compito supervisionato; non dimostra una rappresentazione più utile alla sonda congelata. |

La posa è supervisionata nel pretraining: il test su nuovi utenti misura generalizzazione della posa,
non trasferimento a una regressione nuova. P1/P2 sono classificazioni interne, non il test ufficiale
su dati intatti. Per promuovere un miglioramento generale del FM servono una regressione distinta dalla
posa supervisionata, scelta prima dei risultati, e una conferma indipendente, almeno del contrasto migliore.
Queste verifiche ulteriori non sono nuove campagne autorizzate implicitamente da questa bozza.

## Costi, budget e piano ancora da completare

Riferimento osservato: [accounting B5k](../results/infra/pilot_posa/collaudo_b_20261028/accounting.json),
3.814 s su 4 GPU/32 CPU = **4,2378 GPU-ore e 33,9022 core-ore** per 5.000 passi.
È una base di scala, non un timing del nuovo tokenizer.

| Dose dei tre bracci | Stima GPU-ore prima dell'overhead | Stima core-ore training, a 32 CPU/4 GPU |
|---|---:|---:|
| Tutti a 5.000 | **12,7** | **101,7** |
| Tutti a 19.618, totale dall'inizio | **49,9** | **399,1** |

Le due righe non si sommano: la seconda include i primi 5.000 passi. Escludono collaudo,
checkout, estrazioni, fit CPU, conservazione dei checkpoint e conferme indipendenti.
Misurare il rapporto di secondi/passo e il picco di memoria per ogni braccio sullo stesso carico train;
applicare il fattore misurato al preventivo e registrare tetti di allocazione distinti dalle attese.
La parità scientifica è in passi/batch esposti, non ottenuta dando meno dati al modello più lento.

**Residuo della campagna dall'audit locale del 02/11: 63,292500 GPU-ore e circa 318,82 core-ore CPU-only.**
Il riferimento storico del [protocollo precedente](pose_temporal_pca_protocol_20261102.md) è superato
da questa riconciliazione, tracciata nell'[audit con fonti e hash](../results/infra/frontend_window100_20261102/budget_audit.json);
questi sono saldi del budget locale di campagna, non una nuova interrogazione del saldo CINECA.
La campagna separa il tetto CPU-only dai core allocati insieme alle GPU: le circa 399,1 core-ore
del training vanno riportate, ma **non sottratte dalle 318,82 CPU-only**. Estrazioni GPU e sonde solo CPU
andranno attribuite alle rispettive voci, senza contare due volte costi già sostenuti.

Sulla stima arrotondata di 49,9 GPU-ore, un margine preliminare del 15% porta il training a 57,385
GPU-ore e lascia 5,9075 GPU-ore a preflight/estrazioni: capienza provvisoria, non tetto esecutivo verificato.
Il timing dei nuovi frontend e il preventivo delle valutazioni possono consumare quel margine.
La preparazione scientifica/codice prosegue; nessun costo di una fase non coperta viene assunto implicitamente.

Prima della sottomissione: versionare configurazioni complete dei tre bracci e il supporto comune,
registrare SHA dei dati/codice, verifica di integrità e review indipendente, preventivo con margini
basato sul collaudo e fase effettivamente coperta. Usare submitter e ledger comuni e le autorizzazioni
operative applicabili; nessun job remoto, allocazione o modifica del budget è effettuato da questo documento.
