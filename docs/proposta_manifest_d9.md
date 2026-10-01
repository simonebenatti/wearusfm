# Proposta — manifest del pretraining (D9, passo 4): BOZZA da firmare entro il 25/10/2026

**Stato: bozza di AG del 01/10/2026, NON firmata.** Il manifest è **irreversibile lungo tutta la ladder** (piano, passo 4: «firma TU»). Riferimenti:
piano operativo §Passo 4 (le cinque scelte), v10 §2.8 (quote, classi del montaggio), v10 §10.3 (D è un manifest; `D_t`, `D_c`). I numeri vengono da
`scripts/manifest_table.py` sui riepiloghi di ingest in `results/passo2/` e si rigenerano col comando in fondo; **emg2pose e DB10 sono ancora in
ingest**: le loro ore qui sono ipotesi (400 h e 56 h), e le quote proposte sono frazioni, non ore, quindi reggono anche se quelle cambiano.

## Cosa contiene il corpus oggi

Patch di 25 ms (default di lavoro di D10, non deciso). Canali validi = stima (canali dell'array meno gli scartati dal QC). Kaifosh è nella tabella ma
**fuori dai totali**, perché la proposta (b) lo tiene come benchmark.

| Dataset | Classe | Ore | Soggetti | Canali validi (stima) | D_t (milioni) | D_c (miliardi) | Ruolo proposto |
|---|---|---|---|---|---|---|---|
| NinaPro DB2 | A | 28,8 | 40 | 12 | 4,2 | 0,05 | pretraining |
| NinaPro DB3 (amputati) | A | 7,6 | 11 | 11,6 | 1,1 | 0,01 | pretraining |
| NinaPro DB4 | A | 7,6 | 10 | 12 | 1,1 | 0,01 | pretraining |
| NinaPro DB6 | A | 19,7 | 10 | 14 | 2,8 | 0,04 | pretraining |
| NinaPro DB7 | A | 13,4 | 22 | 12 | 1,9 | 0,02 | pretraining |
| Camargo 2021 | A | 20,1 | 22 | 11 | 2,9 | 0,03 | pretraining |
| Zhang 2026, anatomical | A | 12,6 | 62 | 8 | 1,8 | 0,01 | pretraining |
| Zhang 2026, random | B | 12,3 | 62 | 8 | 1,8 | 0,01 | pretraining |
| Kaifosh (Discrete Gestures) | B | 63,9 | 100 | 16 | 9,2 | 0,15 | **benchmark (proposta b)** |
| emg2qwerty | B | 346,1 | 108 | 32 | 49,8 | 1,59 | pretraining |
| emg2pose | B | *~400 (ipotesi)* | 193 | 16 | *57,6* | *0,92* | pretraining |
| GRABMyo | B | 21,3 | 43 | 28 | 3,1 | 0,09 | pretraining |
| putEMG | B | 21,4 | 44 | 24 | 3,1 | 0,07 | pretraining |
| NinaPro DB5 (200 Hz) | B | 8,8 | 10 | 16 | 1,3 | 0,02 | pretraining |
| NinaPro DB8 | B | 8,5 | 12 | 15,4 | 1,2 | 0,02 | pretraining |
| NinaPro DB10 (MDS1) | B | *~56 (ipotesi)* | 45 | 12 | *8,1* | *0,10* | pretraining |
| CapgMyo-DBa | C | 0,4 | 18 | 128 | 0,1 | 0,01 | pretraining |
| CSL-hdemg | C | 6,0 | 5 | 168 | 0,9 | 0,15 | pretraining |
| Hyser | C | 20,2 | 20 | 256 | 2,9 | 0,75 | pretraining |

**Pretraining (senza Kaifosh): ~1.011 h; `D_t` ≈ 146 milioni di time-patch; `D_c` ≈ 3,9 miliardi di source-channel-patch.** Per classe:
**A 109,8 h (10,9%) · B ~874 h (86,5%) · C 26,7 h (2,6%).** (La tabella di v10 §10.3, ~9·10⁹ channel-patch, era prima delle esclusioni.)

Fuori dal corpus, con decisione registrata: NinaPro DB1 (inviluppo a 100 Hz), DB9 (solo cinematica), DB10 MDS2/MDS4 (inviluppo RMS a 100 Hz), due
soggetti di Zhang (esportazione diversa), EPN-612 e UCI-EMG (solo harness).

## (a) Quote per topologia

**Il problema è la ripetizione.** I dati della classe A sono l'11% delle ore. Con 4 epoche di consumo totale (v10 §10.3), una quota garantita `q`
per una classe di `H_c` ore su `H` totali dà `q · 4 · H / H_c` passaggi sui suoi dati:

| Quota garantita | 5% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| passaggi sulla classe A (110 h) | 1,8 | 3,7 | 7,4 | 11,0 | 14,7 |
| passaggi sulla classe C (27 h) | 7,6 | 15,2 | 30,3 | 45,5 | 60,7 |

Il 40% del piano porterebbe la classe A a ~15 passaggi: vicino ai ~16 che v10 §10.3 indica come il punto (nel testo) in cui la ripetizione smette di
rendere. La classe C è così piccola che anche il 10% la ripeterebbe 15 volte.

**Origine o presentata?** Il piano chiede di decidere se la quota si conta sulla topologia di origine o su quella presentata al modello (i
montaggi bipolari virtuali dalle griglie HD sono «tipo sparso»). Qui conta poco: le griglie HD sono il 2,6% delle ore, quindi riempire la quota A con
montaggi virtuali ripeterebbe i dati C, che sono ancora più scarsi di quelli A.

**Proposta di AG:** quote garantite **sulla topologia di origine: A 20%, B 75%, C 5%** (A e C circa il doppio della loro quota naturale), con un
**tetto di 8 passaggi per dataset** (il doppio delle 4 epoche che v10 §10.3 prende come prior di ripetizione innocua). Con questi numeri nessun
dataset supera 8 passaggi (tabella sotto). *Alternative:* A 30% con tetto 12; oppure quote naturali (nessuna garanzia, A 11%).

## (a-bis) Classe del montaggio e pesi dentro la classe

**Classi (v10 §2.8, la classe è del montaggio):** A = radi, anatomici e misti (NinaPro anello + mirati: DB2, DB3, DB4, DB6, DB7; Camargo; Zhang
anatomical); B = anelli e fasce (bracciali Meta, GRABMyo, putEMG, DB5, DB8, DB10, Zhang random); C = griglie HD (CapgMyo, CSL-hdemg, Hyser).
*Da confermare:* **gli amputati di DB8 e DB10** sono dichiarati sparsi nei metadati (fasce muscolari irregolari), ma fisicamente sono anelli sul
moncone: qui stanno nella classe del loro dataset (B). **Zhang random** è sparso nei metadati (ordine dei sensori ignoto per metà dei soggetti), ma è
un anello attorno all'avambraccio: classe B.

**Pesi dentro la classe.** Per ore (α = 1), emg2pose ed emg2qwerty prendono l'86% della classe B; uniforme per dataset (α = 0) ripete i dataset
piccoli decine di volte. **Proposta di AG: pesi ∝ ore^0,5 con il tetto di 8 passaggi** (l'eccesso di un dataset al tetto passa agli altri della
stessa classe). Risultato:

| Dataset | Classe | Quota dei campioni | Passaggi |
|---|---|---|---|
| NinaPro DB2 | A | 4,5% | 6,2 |
| NinaPro DB3 (amputati) | A | 1,5% | 8,0 |
| NinaPro DB4 | A | 1,5% | 8,0 |
| NinaPro DB6 | A | 3,7% | 7,6 |
| NinaPro DB7 | A | 2,6% | 8,0 |
| Camargo 2021 | A | 3,7% | 7,5 |
| Zhang 2026, anatomical | A | 2,5% | 8,0 |
| Zhang 2026, random | B | 2,4% | 8,0 |
| emg2qwerty | B | 24,5% | 2,9 |
| emg2pose | B | 26,3% | 2,7 |
| GRABMyo | B | 4,2% | 8,0 |
| putEMG | B | 4,2% | 8,0 |
| NinaPro DB5 (200 Hz) | B | 1,7% | 8,0 |
| NinaPro DB8 | B | 1,7% | 8,0 |
| NinaPro DB10 (MDS1) | B | 9,9% | 7,1 |
| CapgMyo-DBa | C | 0,1% | 8,0 |
| CSL-hdemg | C | 1,2% | 8,0 |
| Hyser | C | 3,7% | 7,4 |

Conseguenza da vedere: con il tetto, la classe A al 20% è quasi al massimo che può dare (8 × 110 / 4.044 ≈ 21,7%). Se la quota A deve salire, deve
salire anche il tetto.

## (b) Ruolo di Kaifosh (D3b)

**Proposta di AG: tutto benchmark, nessun soggetto nel pretraining.** Kaifosh aggiunge soggetti, non topologie (piano), alla classe B che è già
l'87% delle ore; è lo stesso dispositivo di emg2pose ed emg2qwerty. Come benchmark mai visto vale di più: 100 utenti nuovi, split ufficiale
80/10/10, un risultato pubblicato con cui confrontarsi. Licenza CC-BY-NC in entrambi i casi (D3a). *Alternativa:* gli 80 utenti di train nel
pretraining, i 20 di validazione e test come benchmark.

## (c) Esito di D6a

**Già chiuso** (21/09/2026): montaggi **al volo**, layout **packing** (D6a, D6b). Nessun montaggio precomputato: `D_c` conta le sorgenti, come nella
tabella.

## (d) Split dei soggetti

**Proposta di AG:**
- **soggetti di test per dataset**, tenuti fuori dal pretraining: gli split ufficiali dove esistono (emg2pose: colonne `held_out_user` e `split` del CSV dei
  metadati; Kaifosh: 80/10/10; emg2qwerty: gli 8 utenti di test di `config/user/user0-7.yaml`, fatto 10d); altrove il **20% dei soggetti, almeno 2, arrotondato per eccesso**, stratificato fra amputati e normodotati (DB7, DB8, DB10), con
  **seed 0**; in Zhang lo stesso soggetto ha i due modi: si divide per soggetto;
- **manifest sottocampionati per soggetti** (asse di v10 §10.6, D16): sottoinsiemi annidati del 12,5, 25 e 50% dei soggetti di pretraining, per
  dataset, stesso seed;
- **due sovrapposizioni note, da dichiarare:** (1) il tokenizer NeuroRVQ è stato addestrato su emg2pose ed emg2qwerty (fatto n. 4, dal paper): i
  loro soggetti di test sono stati visti dal tokenizer, e la valutazione su quei soggetti va segnata; (2) la sovrapposizione di persone fra i DB
  NinaPro (fatto n. 2, raccolto il 01/10): DB4 e DB5 hanno un soggetto in comune (ID non dichiarato); DB8 riusa probabilmente 5 normodotati e i 2
  amputati di DB7 (ID non dichiarati); gli ID sono locali a ogni DB. **Proposta:** i soggetti di test di NinaPro si prendono solo dai DB senza
  sovrapposizioni dichiarate (DB2, DB3, DB6, DB10); **DB4, DB5, DB7 e DB8 restano interi nel pretraining**, cosi' una persona presente in due DB non
  puo' finire nel test di uno e nel pretraining dell'altro. Gli amputati di test vengono da DB3 e DB10.

**Split proposti, generati** (`scripts/make_splits_draft.py`, seme 0; file `splits/draft/splits_draft.json`, split ufficiali con provenienza in
`splits/official/`; da rivedere, non congelati):

| Dataset | Soggetti | Pretraining | Test | 12,5% | 25% | 50% | Regola |
|---|---|---|---|---|---|---|---|
| camargo2021 | 22 | 17 | 5 | 3 | 5 | 9 | 20%, almeno 2 |
| capgmyo | 18 | 14 | 4 | 2 | 4 | 7 | 20%, almeno 2 |
| csl_hdemg | 5 | 3 | **2** | 1 | 1 | 2 | 20%, almeno 2 (= 40%: vedi sotto) |
| emg2pose | 193 | 158 | 35 | 20 | 40 | 79 | ufficiale (`held_out_user`) |
| emg2qwerty | 108 | 100 | 8 | 13 | 25 | 50 | ufficiale (user0-7) |
| grabmyo | 43 | 34 | 9 | 5 | 9 | 17 | 20%, almeno 2 |
| hyser | 20 | 16 | 4 | 2 | 4 | 8 | 20%, almeno 2 |
| kaifosh | 100 | 0 | 0 | - | - | - | benchmark (split ufficiale 80/10/10 conservato) |
| ninapro_db10 | 45 | 36 | 9 (3 amputati) | 5 | 9 | 18 | 20% per gruppo |
| ninapro_db2 | 40 | 32 | 8 | 4 | 8 | 16 | 20%, almeno 2 |
| ninapro_db3 | 11 | 8 | 3 | 1 | 2 | 4 | 20%, almeno 2 |
| ninapro_db4, db5, db7, db8 | 10, 10, 22, 12 | tutti | 0 | | | | sovrapposizione con altri DB (fatto n. 2) |
| ninapro_db6 | 10 | 8 | 2 | 1 | 2 | 4 | 20%, almeno 2 |
| putemg | 44 | 35 | 9 | 5 | 9 | 18 | 20%, almeno 2 |
| zhang2026 | 62 | 49 | 13 | 7 | 13 | 25 | 20%, almeno 2 (i due modi insieme) |

Da decidere: **CSL-hdemg** (5 soggetti) con «almeno 2» manda in test il 40% dei soggetti: alternative 1 solo soggetto di test, o tutto nel pretraining
(pesa 6 ore). **emg2qwerty:** il modello generico ufficiale usa 96 utenti, non i 100 del paper (4 utenti non sono in nessuna configurazione ufficiale):
qui vanno nel pretraining. Gli ID degli 8 utenti di test sono a 8 cifre (lo YAML ufficiale perde lo zero iniziale di 05775561).

## (e) Target RVQ

Secondo D5b (firmato): ancora RVQ sul **livello 0 del ramo 0**, accesa su camargo2021, capgmyo, emg2pose, emg2qwerty, grabmyo, hyser, kaifosh (se
resta nel pretraining), ninapro_db2/3/4/6/7; spenta su putEMG, CSL-hdemg, DB5. **I dataset nuovi (DB8, DB10, Zhang) vanno misurati** con la
regola già firmata (V2 ≤ 2 e conferma del ramo, opzione b): GPU, stima < 0,5 GPU-ora, da confermare col costo prima del lancio.

## Prima della firma (cosa manca)

1. Ingest completo di **emg2pose** e **DB10**, e la tabella rigenerata con le ore vere.
2. **Buchi** (tratti costanti) segnati su emg2pose, emg2qwerty, Zhang e DB10 (il campionamento li esclude: cambiano di poco le ore utili).
3. **V2 e conferma del ramo RVQ** su DB8, DB10 e Zhang (punto e).
4. Firma dei fatti n. 2 (sovrapposizione NinaPro) e 10d (split di emg2qwerty), raccolti il 01/10.
5. **Dove stanno i dati:** quasi tutto il processato è su `$SCRATCH` (purge a 40 giorni). Un manifest congelato deve puntare a dati che restano: serve
   una regola di rinfresco per tutta la durata della ladder, non solo il 25/10.
6. D10 (lunghezza della patch) cambia `D_t` e `D_c` ma non la composizione del manifest.

**Il manifest come file:** `data/_manifests/manifest-v1.json` con una riga per sessione (dataset, soggetto, sessione, percorso, hash del sidecar,
split, classe, peso), le quote e il tetto; tag git `manifest-v1`, hash del file nel registro, `D_t`, `D_c` e consumo realizzato per epoca
ricalcolati dallo stesso script (piano: «Chiuso quando»).

## Cosa si firma

- [ ] (a) quote garantite per classe sulla topologia di origine: A 20%, B 75%, C 5%, e tetto di 8 passaggi per dataset
- [ ] (a-bis) classi del montaggio come sopra (amputati di DB8/DB10 e Zhang random in B); pesi ∝ ore^0,5 dentro la classe
- [ ] (b) Kaifosh tutto benchmark
- [ ] (d) split: ufficiali dove esistono, altrimenti 20% (almeno 2), stratificati, seed 0; manifest annidati al 12,5/25/50%
- [ ] (e) ancora RVQ secondo D5b, estesa ai dataset nuovi solo se passano V2 e la conferma del ramo

Rigenerare i numeri:

    python3 scripts/manifest_table.py --extra emg2pose=400 --extra ninapro_db10=56 --without-kaifosh --quota A=0.2,B=0.75,C=0.05 --alpha 0.5 --max-passes 8
