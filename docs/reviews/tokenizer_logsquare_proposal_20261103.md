# Proposta di feature log-quadratiche nel tokenizer — 03/11/2026

Stato: **valutazione di una proposta di Claude riportata da Simone**, non nuovo protocollo
approvato o lancio. Nessuna modifica ai tre training o ai lettori in corso; monitor disattivato.

## Giudizio

Plausibile affiancare alle64 risposte lineari `x` le64 trasformazioni `log(epsilon+x²)`:
si conserva accesso ai coefficienti con segno e si rende esplicita una trasformazione non
lineare prima del Perceiver. Con D384 il costo della proiezione aggiuntiva è24.576 pesi,
senza nuovo bias se si mantiene un'unica proiezione128→384. Filtri e proiezioni restano
apprendibili. La concatenazione richiede dimensioni/config/checkpoint dedicati; il gancio
delle attivazioni è utile, ma non basta aggiungere un nome alle attivazioni64→64 esistenti.

Non è un recupero di informazione già perduta: è una trasformazione deterministica di x.
Nel nostro frontend `x_j = integral(k_j(t)*EMG(t) dt)`. Il suo quadrato non coincide con
la media temporale dell'energia del segnale o di una banda. Le64 uscite Fourier sono
combinazioni apprendibili di basi, non coppie complesse in quadratura garantite. Il ramo
lineare conserva i coefficienti con segno, non garantisce tutta la fase del segnale grezzo.
Il log naturale non è, da solo, una misura in dB.

## Scale, pavimento e maschere

Il log comprime gli squilibri ma non standardizza. Per una scala a:
`log(epsilon+a²*x²)=2*log(abs(a))+log(x²+epsilon/a²)`.
Soltanto quando il pavimento è trascurabile la scala diventa un semplice offset; il bias
può assorbire un offset fisso, non cambiamenti di distribuzione con utente/dataset/masking.
Le scale1,2–2 e0,07–0,3 citate provengono da prove sintetiche. Il report reale già raccolto
`results/infra/tokenizer_execution_20261103/tokenizer_preflight_20261103/report.json`,
T100-L, `initial/emg2pose/2000/unmasked_same_visible`,8804token, riporta mediane della
std per famiglia8,4104/2,2032/1,0160. È un campione iniziale, non la distribuzione a5k/19618.

Per Z gaussiano standard e epsilon0, E[ln(Z²)]=-1,27036 e std=2,22144: i numeri citati
sono corretti in quel caso limite, non per tutti i kernel reali o epsilon>0. Con epsilon1e-3,
una gaussiana di std0,07 ha circa34,9% delle risposte con x²<epsilon; a std1,6 circa1,58%.
È un esempio matematico, non una stima sui dati reali. Il pavimento e l'inizializzazione
vanno verificati sul train prima di fissarli, includendo code e gradienti per famiglia/fs.

L'azzeramento dei campioni nascosti precede il tokenizer; visibilità/QC escludono i token
nascosti dall'encoder. Con100ms sovrapposti e code anti-alias una patch nascosta non implica
risposta zero: può includere campioni visibili vicini. Log(epsilon) è costante soltanto
quando tutto il supporto della risposta è nullo. Servono controlli di invarianza ai valori
occultati e del comportamento delle finestre parzialmente visibili; nessun difetto del
masking corrente è dimostrato da questa osservazione.

## Esperimento minimo eventualmente successivo

Finire il confronto25-L/100-L/100-G già avviato. Considerare poi100-L più il ramo
log-quadratico come singola ipotesi separata, stessa dose/dati/loss/supporti. Primo controllo
tecnico su train: scale, pavimento, finitezza/gradienti e maschere, senza tuning sulla val.
Proposta di inizializzazione: stessi pesi per tutta la parte condivisa (senza spostare lo
stato RNG dei moduli successivi quando si amplia la proiezione), matrice del nuovo
ramo a zero, bias condiviso invariato. L'uscita iniziale replica100-L, i nuovi pesi ricevono
subito gradiente; il contributo del ramo ai filtri si attiva quando quei pesi si muovono.
Un aumento di24.576pesi/capacità fa parte dell'intervento; non chiamarlo effetto isolato
della sola potenza. Non serve aprire ora una griglia di attivazioni o loss.

Un nullo significa assenza di vantaggio dimostrato per questa trasformazione e ricetta,
non inutilità della potenza, né assenza di informazione nei segnali. Non subordinare
logicamente un vero estrattore energetico al successo di questo controllo: sono ipotesi
diverse. La domanda fisica su energia/inviluppo resta testabile con filtraggio, modulo
quadrato e aggregazione locale, anche con parametri apprendibili.

## Fonti primarie controllate

- [emg2qwerty, paper NeurIPS2024](https://papers.neurips.cc/paper_files/paper/2024/file/a64d53074d011e49af1dfc72c332fe4b-Paper-Datasets_and_Benchmarks_Track.pdf), §5.2: feature spettrali log,32ms/passo8ms a2kHz e BatchNorm prima dei moduli successivi.
- [Codice ufficiale LogSpectrogram](https://github.com/facebookresearch/emg2qwerty/blob/main/emg2qwerty/transforms.py): spettrogramma, log10(spec+1e-6). Non è il log del quadrato di un singolo coefficiente reale arbitrario.
- [LEAF, codice ufficiale](https://github.com/google-research/leaf-audio/blob/master/leaf_audio/frontend.py): filtri complessi apprendibili, modulo quadrato, pooling temporale e compressione/normalizzazionePCEN. Precedente metodologico in audio, non evidenza di efficacia WearUsFM.

Review indipendente del critico concorde su queste rettifiche e sul confronto minimo
con ramo nuovo a zero. Nessun difetto o modifica richiesta ai run esistenti.

## Convergenza dopo il secondo riscontro di Claude

Registrata il 2026-11-03T15:13:38+01:00. In questa discussione GPT non ha consultato i nuovi
score della ridge63405791 né le prestazioni dei tre endpoint tokenizer. Si fissano
le seguenti scelte della **proposta** prima della loro lettura; non è ancora un
protocollo esecutivo completo e non dispone un nuovo lancio.

- Ramo scelto:64 risposte log-quadratiche individuali affiancate alle64 lineari.
  Nessuna somma/pooling fra kernel prima del log in questa candidata. L'estrazione
  di energie di banda resta un'ipotesi distinta.
- Base fissa:T100-L, finestra100ms/passo25ms, anche se T25-L risulterà migliore.
  Primario: candidato contro T100-L, separatamente per dinamica della posa ed EPN-612,
  mantenendo misure e limiti inferenziali del confronto corrente.
- Secondari: riportare anche il confronto con T25-L; il migliore dei due lineari
  si identifica per la metrica principale di ciascun endpoint, senza costruire un
  vincitore unico aggregando posa ed EPN. L'evidenza rispetto al migliore scelto
  sugli stessi risultati è esplorativa, non una validazione indipendente.
- Nessun training di controllo aggiuntivo soltanto per pareggiare24.576pesi
  (circa0,08% di30milioni). L'aumento di capacità resta dichiarato; la piccola quota
  sul totale non dimostra che il suo effetto sia nullo.
- Riusare il risultato del T100-L già avviato è sensato se la candidata parte dalla
  **stessa inizializzazione non addestrata**, con parità della parte condivisa/RNG,
  gain, stream, schedule, dose, maschere e valutatori. W_log inizialezero preserva la
  funzione iniziale, da verificare numericamente. Non inserire il nuovo ramo in un
  checkpoint5k/19618 e confrontarlo col controllo senza la stessa dose aggiuntiva:
  sarebbe un esperimento diverso, con relativo controllo di continuazione.
- Prima del lancio fissare epsilon e inizializzazione mediante diagnostica train,
  includendo i dati nativi200Hz. Il campione iniziale T100-L NinaProDB5/200Hz del
  preflight già raccolto ha2.455token unmasked_same_visible e mediane della std
  Fourier15,2936/spline4,3657/MLP2,2248. Come per emg2pose, queste mediane non misurano
  la frazione di risposte prossime a zero. Non calibrare su EPN-heldout o score val.

### Due precisazioni scientifiche ancora necessarie

La somma dei quadrati di coefficienti arbitrari è una forma quadratica positiva
nel segnale, non automaticamente l'energia di una banda fisica. Quadratura,
normalizzazione e struttura del banco determinano l'interpretazione; un pooling a
pesi positivi da solo non la garantisce. Nemmeno la frase «meno di due gradi di
libertà di potenza a25ms/200Hz» discende dal solo numero di campioni: dipende dallo
stimatore, banda e correlazioni. Il supporto del nostro banco include inoltre code
anti-alias. Non usare questa cifra non dimostrata come premessa della scelta100ms.
[Documentazione primaria sulla variabilità e i gradi di libertà degli stimatori spettrali](https://www.mathworks.com/help/signal/ug/bias-and-variability-in-the-periodogram.html).

### Costo: stima distinta dal preventivo esecutivo

Per T100-L5000 il consuntivo parent è3.926s su4GPU. Estrapolare linearmente fino a
19.618 dà **17,12GPUh**, quindi15–19GPUh è un ordine di grandezza plausibile del solo
training. Non è il costo totale né una garanzia: nuovo ramo, diagnostiche e I/O vanno
misurati. La politica corrente riserva per un braccio **22GPUh** di training
(4GPU×90min più4GPU×240min) e **2,75GPUh** di estrazioni per la matrice completa,
oltre a checkout/analisiCPU e all'eventuale collaudo nuovo. Questi sono riferimenti
per costruire il preventivo della candidata, non nuovi job prenotati in questa nota.

Review indipendente concorde sulla convergenza, sul riuso condizionato del controllo
iniziale e sulla lettura secondaria esplorativa. Codice, job e monitor invariati.

## Sequenza concordata e opzione energetica successiva

Dopo l'ultimo riscontro riportato da Simone, la sequenza è: completare il confronto
T25-L/T100-L/T100-G e le valutazioni previste, poi preparare l'unica candidata
T100-L con64 risposte log-quadratiche affiancate, nuovi pesi a zero e identica
inizializzazione non addestrata della parte condivisa. Controlli train anche a200Hz,
preventivo separato training (stima/tetto), estrazioni e CPU.

L'opzione «quarta», energie per banda esplicite in stile LEAF, resta condizionale:
filtraggio strutturato/normalizzato, modulo quadrato, aggregazione temporale e
compressione. Può seguire un beneficio della terza per approfondirne il meccanismo,
oppure un suo nullo se si decide di affrontare direttamente la domanda energetica.
Non viene implementata o lanciata automaticamente insieme alla terza. Le bande,
le componenti apprendibili e i confronti richiedono un disegno proprio.

La quarta risponde più direttamente alla domanda sulla potenza, ma un risultato
non la chiude in senso universale: riguarda banco, dose, dati e valutazione provati.
Anche l'aumento4× dei gradi di libertà passando25→100ms è una regola indicativa
a banda e ipotesi fissate, non una proprietà di qualunque stimatore. Il conto
2BT conB=70Hz dà3,5 e14; non sostituisce una stima dei gradi di libertà effettivi.
Nessun nuovo job o ripristino del monitor automatico in questa registrazione.
