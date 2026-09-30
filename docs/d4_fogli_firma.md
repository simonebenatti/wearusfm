# D4 — Fogli di firma (preparati il 30/09/2026)

**Come si usa.** Per ogni fatto: cosa affermiamo, dove guardare, cosa dice la fonte, cosa ha controllato AG e cosa cambierebbe se fosse
sbagliato. Tu apri la fonte, leggi la frase e dici «firmo N» oppure «N con correzione: ...» oppure «N no». Io scrivo `verificato da SB` nel
registro (`docs/fatti_da_verificare.md`) con la data. **Firmare non e' un'operazione mia: il controllo AG qui sotto NON e' la firma.**

**Ordine:** dal piu' importante per le scelte di progetto al meno importante. Cominciare dai primi 5 basta.

**Le citazioni qui sotto sono quelle LETTE DA AG OGGI (30/09/2026) sulle pagine**, con `curl` e estrazione del testo, non quelle del registro. Vedi la
sezione finale per il perche'. Le frasi sono copiate come estratte; dove l'estrazione ha un difetto lo dico.

---

## 1. Fatto 20 — Bracciale Meta sEMG-RD (emg2pose, emg2qwerty, Kaifosh)

**Cosa affermiamo.** 16 canali differenziali (bipolari) al polso; 2 kHz; ADC a 12 bit; ampiezza massima 6,6 mV; rumore 2,46 µVrms; filtro analogico
passa-banda 20–850 Hz (−3 dB); 20 mm fra i due elettrodi di una coppia; il dataset Kaifosh e' in piu' filtrato passa-alto a 40 Hz.

**Dove guardare.**
- arXiv 2410.20081 (emg2qwerty), sezione «Hardware»: https://arxiv.org/html/2410.20081
- Articolo sul dispositivo: https://pmc.ncbi.nlm.nih.gov/articles/PMC12818089/
- README di Kaifosh: https://github.com/facebookresearch/generic-neuromotor-interface

**Cosa dicono.**
- arXiv: «Each sEMG-RD has 16 differential electrode pairs utilizing dry gold-plated electrodes. Signals are sampled at 2 kHz with a bit depth of 12 bits
  and a maximum signal amplitude of 6.6 mV. Measurements are bandpass filtered with -3 dB cutoffs at 20 Hz and 850 Hz before digitization.»
- PMC: «The sEMG-RD uses 48 pogo-pin circular electrodes to provide good comfort and contact quality. The 48 channels are configured as 16 bipolar
  channels arranged along the proximal and distal regions, with the remaining electrodes used for shielding or grounding.» · «For each differential
  sensing channel, the center-to-center spacing between paired sensing electrodes is 20 mm.» · «The sEMG-RD features low-noise analog sensors with an
  input-referred RMS noise of 2.46 μVrms.»
- README: «sEMG is recorded at 2 kHz and is high pass filtered at 40 Hz.»

**Controllo di AG.** Tutte le frasi sopra compaiono nelle pagine. **Correzione al registro:** diceva «48 elettrodi in 16 coppie»; la pagina dice 48
elettrodi, di cui 16 canali bipolari, e gli altri per schermatura o massa. **Non trovati nelle fonti:** ordine dei 16 canali, orientamento sul polso,
frequenza di rete, unita' fisiche dei valori nei file HDF5. Per emg2pose (arXiv 2412.02725) ho trovato «2 kHz» ma non ho estratto la frase.

**Se fosse sbagliato:** cambia la banda effettiva di tutti i dataset Meta, quindi la vista canonica del tokenizer.

**Tua decisione:** [ ] firmo · [x] firmo con correzione · [ ] non firmo — **FIRMATO da Simone il 30/09/2026 senza i dettagli** («firmo 20 senza dettaglio»): firmati 16 canali differenziali, 2 kHz, 12 bit, 6,6 mV, 20-850 Hz, passa-alto 40 Hz di Kaifosh; non firmati 48 elettrodi, 20 mm, 2,46 µVrms.

---

## 2. Fatto 19 — NinaPro DB5

**Cosa affermiamo.** 16 canali = due bracciali Myo da 8; il secondo ruotato di 22,5° in senso orario; 10 soggetti sani; 52 movimenti piu' riposo;
6 ripetizioni; 200 Hz.

**Dove guardare.** https://ninapro.hevs.ch/instructions/DB5.html

**Cosa dice.**
- «olumns 1-8 are the electrodes equally spaced around the forearm at the height of the radio humeral joint. Columns 9-16 represent the second Myo,
  tilted by 22.5 degrees clockwise.» **Nel testo estratto manca la «C» iniziale della prima parola: guarda la pagina a occhio** (refuso della pagina o
  difetto dell'estrazione, non lo so).
- «This Ninapro dataset includes sEMG and kinematic data from 10 intact subjects while repeating 52 hand movements plus the rest position.»
- «The sEMG data are acquired using two Thalmic Myo Armbands The sEMG signals are sampled at a rate of 200 Hz.»
- «The dataset includes 6 repetitions of 52 different movements.»

**Controllo di AG.** Tutto presente. Il registro citava «10 intact participants» e «Six repetitions», che sulla pagina non sono scritti cosi'
(«10 intact subjects», «6 repetitions of 52 different movements»): stessa sostanza. **Non trovati:** significato di `stimulus`/`restimulus`, banda
hardware del Myo, frequenza di rete.

**Se fosse sbagliato:** gli anelli e la rotazione di DB5 nello schema dei montaggi.

**Tua decisione:** [x] firmo · [ ] firmo con correzione · [ ] non firmo — **FIRMATO da Simone il 30/09/2026** («firmo 19»).

---

## 3. Fatto 21 — NinaPro DB2, DB3, DB4, DB6, DB7

**Cosa affermiamo.** Tutti a 2 kHz. DB2: 40 sani, 12 Delsys Trigno, 49 movimenti x 10 ripetizioni. DB3: 11 amputati transradiali, fino a 12 Delsys
Trigno, fino a 49 movimenti. DB4: 10 sani, 12 elettrodi wireless. DB6: 10 sani, 14 Delsys Trigno, 7 prese x 12 ripetizioni, 5 giorni x 2 sessioni
(D = giorno, T = 1 mattina / 2 pomeriggio). DB7: 20 sani + 2 amputati, 12 Delsys Trigno IM. Posizionamento: DB2/3/4/7 con 8 elettrodi equispaziati +
2 su flessore/estensore + 2 su bicipite/tricipite; DB6 con 8 al livello radio-omerale e 6 piu' distali.

**Dove guardare.** https://ninapro.hevs.ch/instructions/DB2.html (poi DB3, DB4, DB6, DB7 con lo stesso indirizzo) · DB4 anche https://zenodo.org/records/1000138

**Cosa dicono.**
- DB2: «Subjects 40 intact subjects. Acquisition Setup The sEMG data are acquired using 12 Delsys Trigno electrodes» · «The dataset includes 10 repetitions
  of 49 hand movements.» · «The sEMG signals are sampled at a rate of 2 kHz.»
- DB3: «Subjects 11 trans-radial amputees. Acquisition Setup The sEMG data are acquired using up to 12 Delsys Trigno electrodes» · «The dataset
  includes up to 10 repetitions of 49 hand movements.» · «sampled at a rate of 2 kHz».
- DB4 (Zenodo): «The muscular activity is gathered using 12 active single–differential wireless electrodes from Cometa. The electrodes are positioned as
  shown in the figure: eight electrodes are equally spaced around the forearm in correspondence to the radio humeral joint; two electrodes are placed on
  the main activity spots of the flexor digitorum and of the extensor digitorum; two electrodes are placed on the main activity spots of the biceps and
  of the triceps.» DB4 (pagina HEVS): «Subjects 10 intact subjects. Acquisition Setup The sEMG data are acquired using 12 Cometa electrodes. The sEMG
  signals are sampled at a rate of 2 kHz.»
- DB6: «The data are recorded from 10 intact subjects repeating 7 grasps 12 times, twice a day for 5 days.» · nomi «S1_D1_T1.mat» «Subject: 1 Day: 1
  Time of the day: 1 (morning, while 2 is afternoon)» · «14 Delsys Trigno double differential sEMG Wireless electrodes» · «The first eight at the height of
  the radio humeral joint, the remaining 6 below.» · «The sEMG signals are sampled at a rate of 2 kHz.»
- DB7: «Subjects 20 intact subjects, 2 amputees. Acquisition Setup 12 active double differential wireless electrodes from a Delsys Trigno IM Wireless EMG
  system.» · «The sEMG signals are sampled at a rate of 2 kHz.»

**Controllo di AG.** Presenti. **Correzione al registro:** il registro diceva che «Cometa» per DB4 non era nella citazione; invece la pagina Zenodo e la
pagina HEVS lo scrivono. La disposizione «8 + 2 + 2» per DB2, DB3 e DB7 e' stata riletta il 30/09 (dopo la prima stesura di questo foglio): stessa frase di DB4 su tutte e tre le pagine («eight electrodes are equally spaced around the forearm in correspondence to the radio humeral joint; two electrodes are placed on the main activity spots of the flexor digitorum and of the extensor digitorum ...; two electrodes are placed on the main activity spots of the biceps and of the triceps»; su DB7 «sensors» al posto di «electrodes»). La frequenza di rete
(50 Hz) resta **non verificata**. Le licenze sono gia' decise.

**Se fosse sbagliato:** i montaggi misti (12 o 14 elettrodi) nello schema.

**Tua decisione:** [x] firmo · [ ] firmo con correzione · [ ] non firmo — **FIRMATO da Simone il 30/09/2026** («firmo 21»), per le frasi mostrategli in chat; i quattro dettagli prima esclusi (ripetizioni di DB3 e DB7, disposizione di DB4, 2 kHz di DB6) sono stati firmati col fatto 22 («+ recupero»).

---

## 4. Fatto 22 — DB7: amputati 21 e 22 · DB6: 2 colonne vuote

**Cosa affermiamo.** DB7: gli amputati sono i soggetti 21 e 22 (mano destra); il 21 non ha fatto gli ultimi due movimenti funzionali dell'esercizio 2
(38 classi). DB6: il file EMG ha 16 colonne, 2 vuote; 14 elettrodi.

**Dove guardare.** https://ninapro.hevs.ch/instructions/DB7.html · https://ninapro.hevs.ch/instructions/DB6.html

**Cosa dicono.**
- DB7, tabella dei soggetti: «21 Right Hand Amputated Right 28 180 76 50 6 Accident 0 6 0 s21.zip» e «22 Right Hand Amputated Right 54 182 90 50 18 Cancer
  (epithelioid sarcoma) 0 17 2 s22.zip» (riga come estratta: le intestazioni delle colonne sono sulla pagina, guardale per capire cosa sia il «50»).
- DB7: «One of the amputee subjects (Subject 21) did not perform the final two (functional) movements of Exercise 2; therefore, the number of classes
  associated with this subject is 38.»
- DB6: «emg (16 columns): sEMG signal of the 14 electrodes (2 columns are empty)».

**Controllo di AG.** La sostanza e' confermata. Il registro citava «Amputee Subjects: 2 participants (Subjects 21 and 22)» e «The EMG data contains 16 columns
(2 empty)»: **non sono frasi presenti cosi' sulla pagina**. Il «50% dell'avambraccio residuo» del registro va letto dall'intestazione della tabella.
**Non trovato nelle fonti:** quali colonne di DB6 siano le due vuote (nei nostri dati sono gli indici 8 e 9, con deviazione standard zero: e' un dato
nostro, non della fonte).

**Se fosse sbagliato:** anatomia nominale degli amputati e colonne di DB6.

**Tua decisione:** [x] firmo · [ ] firmo con correzione · [ ] non firmo — **FIRMATO da Simone il 30/09/2026** («firmo 22 + recupero»); il 50 e' nella colonna «Remaining Forearm (%)». Col recupero entrano nella firma anche i quattro dettagli rimasti fuori dal fatto 21.

---

## 5. Fatto 9 — `flash_attn_varlen_func` funziona in bf16 su A100

**Cosa affermiamo.** Nella forma d'uso prevista per il Perceiver (64 latenti fissi, lunghezze delle chiavi variabili) forward e backward funzionano in bf16.

**Dove guardare.** Nel repo: `results/step0/flash_attn_varlen_check_58134850.json` (10 righe) e lo script `bench/verify_flash_attn_varlen.py` (righe 57 e 74,
dove si costruiscono `cu_seqlens_q` e `cu_seqlens_k`).

**Cosa dice il file.** `"status": "OK"` · `"flash_attn_version": "2.5.5"` · `"forward_output_shape": [448, 8, 64]` · `"forward_output_dtype": "torch.bfloat16"` ·
`"backward_grads_finite": true` · `"gpu_name": "NVIDIA A100-SXM-64GB"` · `"k_latents": 64` · `"c_per_sample": [8, 12, 16, 32, 64, 128, 256]`.

**Controllo di AG.** Il file e' quello. 448 = 7 campioni x 64 latenti: coerente con le 7 lunghezze. **Non ricontrollabile ora:** l'esecuzione del job 58134850
(sta sul cluster, oggi in manutenzione). Questo fatto non e' una frase su una pagina: la verifica e' leggere il JSON e lo script.

**Se fosse sbagliato:** il Perceiver del piano.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## 6. Fatto 11 — Hyser

**Cosa affermiamo.** 256 canali a 2048 Hz, 20 soggetti, due sessioni in giorni diversi, 4 griglie 8x8 (ED, EP, FD, FP), licenza ODC-By 1.0.

**Dove guardare.** https://physionet.org/files/hd-semg/2.0.0/readme.txt · https://physionet.org/content/hd-semg/2.0.0/

**Cosa dicono.**
- readme: «Our dataset consisted of data from 20 subjects (8 female and 12 male volunteers).» · «The 256-channel HD-sEMG were acquired by four 8*8 electrode
  arrays. The electrode arrays were named as "ED" (extensor-distal), "EP" (extensor-proximal), "FD" (flexor-distal) and "FP" (flexor-proximal)» · «License:
  Open Data Commons Attribution License v1.0».
- pagina del progetto: «HD-sEMG signals were acquired at 2048 Hz sampling rate.» · «data of 20 subjects acquired in 2 experiment sessions (on 2 separate days)
  are stored in 40 folders, named "subject i _session j "».

**Controllo di AG.** Tutto presente. **Completamento del registro:** la frequenza di 2048 Hz veniva dall'header WFDB e le «due sessioni in giorni diversi»
risultavano non documentate (il readme non le dice); la pagina del progetto le dichiara entrambe. **Novita', non nel registro:** la pagina dice che nella fase di preprocessing i segnali sono stati
«filtered with a 10--500 Hz 8-order Butterworth bandpass filter» (poi c'e' un notch); non dice se i file rilasciati (che si chiamano «raw») siano gia' filtrati.
Da chiarire. **Non dicono:** se i canali sono monopolari o differenziali, l'orientamento delle griglie.

**Se fosse sbagliato:** le griglie HD nello schema.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## 7. Fatto 16 — GRABMyo

**Cosa affermiamo.** Gli header dichiarano 32 canali ma 4 (U1–U4) non sono EMG: marcano solo la posizione fra i 4 anelli. I 28 canali EMG sono in 4 anelli:
F1–F8, F9–F16 (avambraccio), W1–W6, W7–W12 (polso).

**Dove guardare.** https://physionet.org/content/grabmyo/1.1.0/

**Cosa dice.** Ho trovato sulla pagina, parola per parola: «4 channels remain unused» · «U1-U4» · «to distinguish the rings of electrode setup».

**Controllo di AG.** Le tre frasi ci sono. **Non ricontrollato:** l'assegnazione dei nomi F1–F8, F9–F16, W1–W6, W7–W12 ai quattro anelli (guardala tu sulla pagina).

**Se fosse sbagliato:** lo schema a 4 gruppi di GRABMyo.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## 8. Fatto 12 — putEMG (montaggio)

**Cosa affermiamo.** 24 elettrodi in 3 fasce da 8, a 45°, primo elettrodo di ogni fascia sull'ulna, numerazione oraria; 5120 Hz; registrazione monopolare;
licenza CC BY-NC 4.0.

**Dove guardare.** https://biolab.put.poznan.pl/putemg-dataset/

**Cosa dice.**
- «Signals are recorded from 24 electrodes fixed around subject right forearm using 3 elastic bands, creating a 3×8 matrix. Electrodes were evenly spaced (45°)
  around participant's forearm.»
- «First electrode of each band was placed over the ulna bone and numbered clockwise respectively. Resulting in following numbering pattern: elbow band [1-8],
  middle band [9-16], wrist band [17-24].»
- «The data was sampled at 5120 Hz, with 12-bit A/D conversion using 3 Hz high-pass and 900 Hz low-pass filter.»
- «Signals were recorded in monopolar mode with DRL-IN and Patient-REF electrodes placed in close proximity of the wrist of examined arm.»
- «Unless stated otherwise, all putEMG datasets elements are licensed under a Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)».

**Controllo di AG.** Tutto presente. **Attenzione:** nel registro questa riga non aveva URL ne' citazione («vedi dataloader_bench_spec.md §13»): era incompleta, l'ho
completata ora. Questo testo copre la numerazione degli **elettrodi**; il legame con le colonne `EMG_1…EMG_24` dei file resta il fatto 18, ancora aperto.

**Se fosse sbagliato:** il montaggio di putEMG nello schema.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## 9. Fatto 7b — Kaifosh «Discrete Gestures»

**Cosa affermiamo.** Accesso, dimensioni e licenza del dataset: 100 partecipanti (80/10/10), 51,4 + 6,2 + 6,4 ore, 2 kHz, CC-BY-NC-4.0. **La parte sulla licenza l'hai
gia' decisa: puoi saltarla.**

**Dove guardare.** https://github.com/facebookresearch/generic-neuromotor-interface (README)

**Cosa dice.** «The dataset and the code are CC-BY-NC-4.0 licensed» · «The dataset contains sEMG recordings from 100 participants in each of the three tasks
described in the paper: `discrete_gestures`, `handwriting`, and `wrist`.» · tabella «Number of Users 80 10 10» e «Hours 51.4 6.2 6.4» (Discrete Gestures, train/val/test) ·
comando: `python -m generic_neuromotor_interface.scripts.download_data --task $TASK_NAME --output-dir ~/emg_data`, con «`$TASK_NAME` is one of {`discrete_gestures,
handwriting, wrist`}» e «`--small-subset` downloads only 3 users per task».

**Controllo di AG.** Presente. Il registro scriveva `--task discrete_gestures` gia' specializzato: nel README c'e' la variabile. **Non verificato qui:** il formato .hdf5.

**Se fosse sbagliato:** solo l'uso (non commerciale) e le dimensioni.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## 10. Fatto 17 — putEMG: mappa dei gesti

**Cosa affermiamo.** `TRAJ_GT`: 0=Idle, 1=Fist, 2=Flexion, 3=Extension, 6=Pinch index, 7=Pinch middle, 8=Pinch ring, 9=Pinch small; -1 = pausa/transizione, non un gesto.

**Dove guardare.** https://github.com/biolab-put/putemg_examples, file `shallow_learn.py` riga 142 (e `biolab_utilities.py`, funzione `filter_transitions`, per il -1).

**Cosa dice.** `gestures = { 0: "Idle", 1: "Fist", 2: "Flexion", 3: "Extension", 6: "Pinch index", 7: "Pinch middle", 8: "Pinch ring", 9: "Pinch small" }`,
preceduto dal commento «# defines gestures to be used in shallow learn».

**Controllo di AG.** Coincide, riga 142. Nel registro gli spazi dentro le graffe sono diversi: irrilevante. **Non ricontrollato:** il significato di -1.

**Se fosse sbagliato:** solo le etichette dell'harness.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## 11. Fatto 15 — UCI-EMG a 1 kHz

**Cosa affermiamo.** Frequenza di campionamento nativa 1 kHz.

**Dove guardare.** Lobov et al. 2018, Sensors 18(4):1122: https://pmc.ncbi.nlm.nih.gov/articles/PMC5948532/

**Cosa dice.** Ho trovato parola per parola: «200 ms overlapping time windows at a 100 ms step» e «the sampling rate of 1 kHz».

**Controllo di AG.** Presente. Solo per l'harness.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## Cosa ho scoperto controllando il registro

Per preparare questi fogli ho riletto le pagine con un controllo di testo. Il registro dichiarava «citazione esatta», e in piu' righe non lo era:

1. **Righe 19, 20, 21, 22:** frasi rimaneggiate fra virgolette (per esempio «participants» al posto di «subjects», «Six repetitions» al posto di «6 repetitions of 52 different
   movements», frasi di DB6 e DB7 riassunte, frasi di fonti diverse messe insieme). **La sostanza dei fatti risulta confermata** nei casi controllati, ma le frasi virgolettate non erano
   quelle delle pagine.
2. **Due precisazioni del registro erano sbagliate:** «Cometa» per DB4 e' scritto sulle pagine; 2048 Hz e sessioni di Hyser sono sulla pagina del progetto.
3. **Riga 20:** «48 elettrodi in 16 coppie» non e' cio' che dice la pagina.
4. **Riga 12:** senza URL e senza citazione.
5. **Novita' sulle licenze (non le riapro, le hai decise):** sulle pagine HEVS di DB4, DB5, DB6 e DB7 la dicitura «Creative Commons Attribution No Derivatives 4.0 International» compare solo
   dentro un commento HTML, quindi non visibile; su DB2 e DB3 non compare. Non la considero una fonte.
