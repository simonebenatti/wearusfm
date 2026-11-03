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
