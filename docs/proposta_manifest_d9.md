# Proposta — manifest del pretraining (D9, passo 4): FIRMATA il 03/10/2026

> **FIRMATA da Simone il 03/10/2026** («firmo 1-11, 13 e 14» sui fogli `docs/fogli_firma_d9_d10.md`, decisioni 1-9 = punti a, a-bis, b, d, e,
> f e rinfresco di `$SCRATCH`; registrazione in `docs/decisioni.md`). **Revisione del 02/10/2026 applicata.** La review indipendente dei documenti aveva trovato, e AG aveva
> verificato, gli 8 punti qui sotto; lo stato di ciascuno e' fra quadre. Le tabelle sono quelle del costruttore sui dati veri (job 59204297).
> 1. le ore, i passaggi e le quote qui sotto contano anche i soggetti di test (~1.011 h), mentre il manifest pesa solo il pretraining (~870 h): con il
>    tetto di 8 passaggi la classe C non arriva piu' al 5%. Si rifa' tutto coi numeri del costruttore (`scripts/build_manifest.py`) sui dati veri; **[risolto: tabelle del costruttore, solo pretraining, job 59204297; la classe C arriva al 5%]**
> 2. «Kaifosh: 100 utenti nuovi» non e' verificato: Kaifosh, emg2pose ed emg2qwerty sono dello stesso produttore e gli ID non sono confrontabili
>    (sovrapposizione ignota, da dichiarare come per NinaPro); idem per gli amputati di DB3 e DB10; **[riscritto in (b) e (d); fatto 23 firmato]**
> 3. emg2pose: le 3.539 registrazioni del test per fasi nuove escono dal pretraining (Simone, 02/10/2026: gia' negli split); **[riscritto in (d)]**
> 4. il tetto di 8 passaggi e' definito per 4 epoche: nei gradini a 2D i passaggi raddoppiano (16, il «forte decadimento» di v10 §10.3); **[riscritto in (a)]**
> 5. «origine» contro «presentata»: il piano propone la topologia presentata; la scelta va argomentata anche sul volume D_c (classe C ~23% di D_c; 22,1% sul solo pretraining, review del 03/10); **[riscritto in (a)]**
> 6. interpretazioni scritte come fatti (anelli degli amputati, Zhang random «e' un anello», soggetti di test «visti» dal tokenizer): vanno marcate; **[marcate in (a-bis) e (d)]**
> 7. i salti dell'asse dei tempi (emg2pose 1.668 registrazioni, emg2qwerty 76, Kaifosh) non spezzano i segmenti: una finestra puo' attraversarli. Va
>    deciso come trattarli prima del manifest; **[opzioni e numeri in (f); decisione di Simone]**
> 8. elenco dei fatti da firmare incompleto (servono anche 1 DB8, 4, 5, 10b, 10c, 10d); piccoli numeri da correggere. Aggiornamento
>    02/10/2026: 1 (DB8), 2, 10b, 10c, 10d, poi 5 e 23 firmati da Simone; resta il 4 (parziale). **[elenco in «Prima della firma»; numeri corretti
>    con le tabelle]**


**Stato: bozza di AG del 01/10/2026, testo rivisto e tabelle rifatte il 02/10/2026, firmata il 03/10/2026 (vedi in testa; congelata come
`manifest-v1`, job 59253155).** Il manifest è **irreversibile lungo tutta la
ladder** (piano, passo 4: «firma TU»). Riferimenti: piano operativo §Passo 4 (le cinque scelte), v10 §2.8 (quote, classi del montaggio), v10 §10.3
(D è un manifest; `D_t`, `D_c`). **I numeri vengono dal costruttore** (`scripts/build_manifest.py`, job 59204297 del 02/10, bozza con i parametri
proposti qui; manifest in `~/wearusfm_local/reports/passo4/manifest_59204297.json.gz`, hash del contenuto 59e40dbc0831b158, non nel repo) sui
sidecar veri, **solo pretraining**: i soggetti e le sessioni di test e il benchmark non contano nelle ore, nelle quote e nei passaggi.

## Cosa contiene il corpus oggi

Patch di 25 ms (D10, firmata il 03/10/2026). Kaifosh è nella tabella ma **fuori dai totali**, perché la proposta (b) lo tiene come benchmark.

| Dataset | Classe | RVQ | Ore pretraining | Ore test | Soggetti | D_t (milioni) | D_c (miliardi) | Ruolo |
|---|---|---|---|---|---|---|---|---|
| NinaPro DB2 | A | accesa | 23,1 | 5,8 | 40 | 3,3 | 0,04 | pretraining |
| NinaPro DB3 (amputati) | A | accesa | 5,5 | 2,1 | 11 | 0,8 | 0,01 | pretraining |
| NinaPro DB4 | A | accesa | 7,6 | 0,0 | 10 | 1,1 | 0,01 | pretraining |
| NinaPro DB6 | A | accesa | 15,7 | 4,0 | 10 | 2,3 | 0,03 | pretraining |
| NinaPro DB7 | A | accesa | 13,4 | 0,0 | 22 | 1,9 | 0,02 | pretraining |
| Camargo 2021 | A | accesa | 15,5 | 4,6 | 22 | 2,2 | 0,02 | pretraining |
| Zhang 2026, anatomical | A | accesa | 10,0 | 2,6 | 62 | 1,4 | 0,01 | pretraining |
| Zhang 2026, random | B | accesa | 9,7 | 2,6 | 62 | 1,4 | 0,01 | pretraining |
| Kaifosh (Discrete Gestures) | B | accesa | 63,9 (benchmark) | 0,0 | 100 | 0,0 | 0,00 | **benchmark (proposta b)** |
| emg2qwerty | B | accesa | 317,1 | 29,0 | 108 | 45,7 | 1,46 | pretraining |
| emg2pose | B | accesa | 287,8 | 134,9 | 193 | 41,4 | 0,66 | pretraining |
| GRABMyo | B | accesa | 16,9 | 4,5 | 43 | 2,4 | 0,07 | pretraining |
| putEMG | B | spenta | 17,0 | 4,4 | 44 | 2,5 | 0,06 | pretraining |
| NinaPro DB5 (200 Hz) | B | spenta | 8,8 | 0,0 | 10 | 1,3 | 0,02 | pretraining |
| NinaPro DB8 | B | accesa | 8,5 | 0,0 | 12 | 1,2 | 0,02 | pretraining |
| NinaPro DB10 (MDS1) | B | accesa | 45,1 | 11,1 | 45 | 6,5 | 0,08 | pretraining |
| CapgMyo-DBa | C | accesa | 0,3 | 0,1 | 18 | 0,0 | 0,01 | pretraining |
| CSL-hdemg | C | spenta | 4,8 | 1,2 | 5 | 0,7 | 0,12 | pretraining |
| Hyser | C | accesa | 16,2 | 4,0 | 20 | 2,3 | 0,60 | pretraining |

**Pretraining: 822,9 h in 19.034 sessioni; `D_t` = 118,5 milioni di time-patch; `D_c` = 3,25 miliardi di source-channel-patch.** Test: 210,9 h
(emg2pose 134,9 h: i 35 utenti tenuti fuori e le 3.539 sessioni del test per fasi nuove). Benchmark: Kaifosh 63,9 h. Per classe, pretraining:
**A 90,7 h (11,0%) · B 710,8 h (86,4%) · C 21,3 h (2,6%)**. Nessun soggetto degli split senza sessioni, nessuna sessione di test mancante. (La
tabella di v10 §10.3, ~9·10⁹ channel-patch, era prima delle esclusioni.)

Fuori dal corpus, con decisione registrata: NinaPro DB1 (inviluppo a 100 Hz), DB9 (solo cinematica), DB10 MDS2/MDS4 (inviluppo RMS a 100 Hz), due
soggetti di Zhang (esportazione diversa), EPN-612 e UCI-EMG (solo harness).

## (a) Quote per topologia

**Il problema è la ripetizione.** I dati della classe A sono l'11% delle ore di pretraining. Con 4 epoche di consumo totale (v10 §10.3), una quota garantita `q`
per una classe di `H_c` ore su `H` totali dà `q · 4 · H / H_c` passaggi sui suoi dati:

| Quota garantita | 5% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| passaggi sulla classe A (90,7 h) | 1,8 | 3,6 | 7,3 | 10,9 | 14,5 |
| passaggi sulla classe C (21,3 h) | 7,7 | 15,4 | 30,9 | 46,3 | 61,7 |

Il 40% del piano porterebbe la classe A a 14,5 passaggi: vicino alle ~16 epoche che v10 §10.3 riporta come punto stimato di forte decadimento
(Muennighoff et al., **su testo**). La classe C è così piccola, in ore, che anche il 10% la ripeterebbe 15 volte.

**Origine o presentata?** Il piano propone la quota **sulla topologia presentata**, «con un tetto esplicito alla ripetizione dei dati sparsi veri,
e lettura del pilot per topologia»: i montaggi bipolari virtuali ricavati dalle griglie HD sono campioni «tipo sparso» e riempiono parte della quota
senza ripetere i dati sparsi veri. La bozza del 01/10 proponeva invece l'origine, con un argomento solo sulle ore (le griglie sono poche ore, quindi
riempire la quota A con montaggi virtuali ripeterebbe i dati C). L'argomento era incompleto:
- **sul tempo** la classe C è scarsa (21,3 h, il 2,6% del pretraining): ogni campione virtuale ripete un tratto di tempo di una griglia;
- **sul volume dei canali** non lo è: con 128-256 canali per griglia, la classe C è il 22,1% di `D_c` (0,72 su 3,25 miliardi).
  Con i montaggi al volo (D6a), due passaggi sullo stesso tratto possono presentare coppie di elettrodi diverse: si ripete il tempo, non l'ingresso.

Conseguenza che vale con qualunque numero: se il tetto si conta sul **tempo di origine**, la classe C può dare al massimo `8 · H_C / (4 · H)` dei
campioni a E = 4 (`H_C` ore della classe C, `H` ore del pretraining), sommando campioni HD e montaggi virtuali. La scelta fra origine e presentata
sposta solo questa fetta: **8 × 21,3 / (4 × 822,9) = 5,2%** dei campioni. *Da dichiarare:* che per il modello un montaggio virtuale da una griglia equivalga a un
montaggio sparso vero non è misurato. È la premessa del piano, e la lettura del pilot per topologia serve anche a controllarla.

**Il tetto e la durata dell'addestramento.** Il tetto di 8 passaggi è definito a E = 4. Il manifest fissa i pesi, non E: con gli stessi pesi i
passaggi crescono con E. Nel pilot sulle epoche (E ∈ {1, 2, 4, 8}, v10 §10.3) e nel controllo 2D sul rung più grande, i dataset al tetto arrivano a
**16 passaggi**, cioè alle ~16 epoche di forte decadimento (su testo). Due letture:
- (i) **tetto a E = 4** (bozza del 01/10): un manifest solo per tutta la ladder; a E = 8 e a 2D i dataset al tetto si ripetono 16 volte, e il pilot
  lo misura. È proprio la curva di riuso dei dati che v10 §10.3 indica come risultato interessante; il pilot va letto anche per i dataset al tetto,
  non solo in aggregato;
- (ii) **tetto al massimo E previsto** (8 passaggi a E = 8, cioè 4 a E = 4): nessun dataset supera 8 passaggi in nessun gradino. Però una classe al
  tetto dà al massimo `4 · H_c / (4 · H) = H_c / H` dei campioni, cioè la sua quota naturale: **le quote garantite di A e C non potrebbero superare
  le proporzioni del corpus**, e la garanzia di v10 §2.8 resterebbe vuota.

**Proposta di AG:** quote garantite **A 20%, B 75%, C 5%** (A e C circa il doppio della loro quota naturale, 11,0% e 2,6%), **dichiarate sulla
topologia presentata** come nel piano. **Tetto di 8 passaggi per dataset a E = 4, contato sul tempo di origine**,
lettura (i). Il campionatore registra per ogni campione sia la classe di origine sia quella presentata, così il pilot si legge in tutti e due i modi.
Con questi numeri le quote si realizzano esattamente: A 20,0%, B 75,0%, C 5,0%, nessuna
quota inassegnabile (la classe C al tetto potrebbe dare al massimo il 5,2%). *Alternative:* quota sull'origine (bozza del 01/10); A 30% con tetto
12; quote naturali (nessuna garanzia).

## (a-bis) Classe del montaggio e pesi dentro la classe

**Classi (v10 §2.8, la classe è del montaggio):** A = radi, anatomici e misti (NinaPro anello + mirati: DB2, DB3, DB4, DB6, DB7; Camargo; Zhang
anatomical); B = anelli e fasce (bracciali Meta, GRABMyo, putEMG, DB5, DB8, DB10, Zhang random); C = griglie HD (CapgMyo, CSL-hdemg, Hyser).
*Da confermare (fatti firmati, poi interpretazioni di AG):*
- **amputati di DB8 e DB10.** Fatti 1 e 10c, firmati il 02/10: in DB8 i due amputati hanno «a similar configuration» con 13 e 12 sensori, per lo
  spazio limitato sul moncone; in DB10 i due array stanno «around the right forearm or residual limb», e S108 non ha il secondo array. Nei metadati
  sono **sparsi** (Simone, 01/10/2026: fasce muscolari irregolari). *Interpretazione di AG:* fisicamente restano anelli sul moncone, quindi stanno
  nella classe del loro dataset (B). Alternativa: classe A, coerente con i metadati;
- **Zhang random.** Fatto 10b, firmato il 02/10: per il README i sensori sono «equally spaced on the circumference of the forearm», per l'abstract
  «placed randomly around the circumference»; l'ordine dei sensori è ignoto per 32 partecipanti. Nei metadati è sparso. *Interpretazione di AG:* un
  anello con rotazione (e, per 32 soggetti, ordine) sconosciuti, quindi classe B.

**Pesi dentro la classe.** Per ore (α = 1), emg2pose ed emg2qwerty prendono l'86% della classe B (85,1% sul solo pretraining; 86% contava anche i soggetti di test, review
del 03/10); uniforme per dataset (α = 0) ripete i dataset
piccoli decine di volte. **Proposta di AG: pesi ∝ ore^0,5 con il tetto di 8 passaggi** (l'eccesso di un dataset al tetto passa agli altri della
stessa classe). Risultato:

| Dataset | Classe | Quota dei campioni | Passaggi |
|---|---|---|---|
| NinaPro DB2 | A | 4,2% | 6,0 |
| NinaPro DB3 (amputati) | A | 1,3% | 8,0 |
| NinaPro DB4 | A | 1,8% | 8,0 |
| NinaPro DB6 | A | 3,5% | 7,3 |
| NinaPro DB7 | A | 3,2% | 7,9 |
| Camargo 2021 | A | 3,5% | 7,4 |
| Zhang 2026, anatomical | A | 2,4% | 8,0 |
| Zhang 2026, random | B | 2,4% | 8,0 |
| emg2qwerty | B | 25,8% | 2,7 |
| emg2pose | B | 24,6% | 2,8 |
| GRABMyo | B | 4,1% | 8,0 |
| putEMG | B | 4,1% | 8,0 |
| NinaPro DB5 (200 Hz) | B | 2,1% | 8,0 |
| NinaPro DB8 | B | 2,1% | 8,0 |
| NinaPro DB10 (MDS1) | B | 9,7% | 7,1 |
| CapgMyo-DBa | C | 0,1% | 8,0 |
| CSL-hdemg | C | 1,2% | 8,0 |
| Hyser | C | 3,8% | 7,6 |

**10 unità su 18 sono al tetto di 8 passaggi** e altre 5 fra 7,1 e 7,9; emg2qwerty ed emg2pose, che fanno metà dei campioni, restano sotto i 3
passaggi, NinaPro DB2 a 6,0.
Conseguenza: con il tetto, la classe A al 20% è quasi al massimo che può dare (8 × 90,7 / (4 × 822,9) = 22,0%). Se la quota A deve salire, deve
salire anche il tetto.

## (b) Ruolo di Kaifosh (D3b)

**Proposta di AG: tutto benchmark, nessun soggetto nel pretraining.** Kaifosh aggiunge soggetti, non topologie (piano), alla classe B che ha già
la gran parte delle ore; usa lo stesso bracciale di emg2pose ed emg2qwerty (fatto 20, firmato). Come benchmark ha uno split ufficiale 80/10/10
(fatto 7b, firmato) e un risultato pubblicato con cui confrontarsi. *Pero'* (fatto 23, firmato il 02/10): i 300 partecipanti pubblici sono stati scelti a caso
fra gli utenti di training del paper, quindi i numeri pubblicati, misurati sui partecipanti tenuti fuori dal paper, non sono sullo stesso test
dello split pubblico (interpretazione di AG). Licenza CC-BY-NC in entrambi i casi (D3a). *Alternativa:* gli 80 utenti di
train nel pretraining, i 20 di validazione e test come benchmark.

**Correzione (review del 02/10):** la bozza del 01/10 diceva «100 utenti nuovi». Non è verificato. I tre dataset Meta vengono dallo stesso
produttore, nessuna fonte del registro dice se abbiano partecipanti in comune, e gli ID non sono confrontabili fra i dataset: un utente di test di
Kaifosh potrebbe essere nel pretraining attraverso emg2pose o emg2qwerty. Il **fatto 23** (firmato il 02/10) conferma che **nessuna fonte lo dichiara**, in un senso o nell'altro, e che gli ID hanno formati diversi
nei tre dataset. Quindi il benchmark Kaifosh si dichiara «utenti forse visti nel pretraining (stesso produttore, sovrapposizione ignota)», come per NinaPro; lo
stesso vale per i soggetti di test di emg2pose ed emg2qwerty rispetto agli altri due dataset.

## (c) Esito di D6a

**Già chiuso** (21/09/2026): montaggi **al volo**, layout **packing** (D6a, D6b). Nessun montaggio precomputato: `D_c` conta le sorgenti, come nella
tabella.

## (d) Split dei soggetti

**Proposta di AG:**
- **soggetti di test per dataset**, tenuti fuori dal pretraining: gli split ufficiali dove esistono (emg2pose: colonne `held_out_user` e `split` del CSV dei
  metadati; Kaifosh: 80/10/10; emg2qwerty: gli 8 utenti di test di `config/user/user0-7.yaml`, fatto 10d); altrove il **20% dei soggetti, almeno 2, arrotondato per eccesso**, stratificato fra amputati e normodotati (DB7, DB8, DB10), con
  **seed 0**; in Zhang lo stesso soggetto ha i due modi: si divide per soggetto;
- **emg2pose, anche per sessione** (Simone, 02/10/2026): le 3.539 registrazioni (58,6 h di campioni; 58,8 h dai tempi del CSV) del test ufficiale per fasi nuove (`split = test`,
  `held_out_stage = True`, di utenti non tenuti fuori) escono dal pretraining e restano per la valutazione (`test_sessions` negli split);
- **manifest sottocampionati per soggetti** (asse di v10 §10.6, D16): sottoinsiemi annidati del 12,5, 25 e 50% dei soggetti di pretraining, per
  dataset, stesso seed;
- **sovrapposizioni da dichiarare** (nessuna si risolve con gli ID):
  (1) **tokenizer NeuroRVQ.** Dal paper (fatto 5, firmato il 02/10: appendice E, Tabella 7, e §5.1) il tokenizer è
  pre-addestrato su emg2pose ed emg2qwerty. Quali soggetti o split abbiano usato **non è dichiarato da nessuna fonte** (paper, repo, scheda HF):
  la valutazione sui soggetti di test di emg2pose ed emg2qwerty si segna quindi «soggetti forse visti dal tokenizer». La bozza del 01/10 li dava
  per «visti», ma era un'interpretazione;
  (2) **NinaPro** (fatto 2, firmato il 02/10): DB4 e DB5 hanno un soggetto in comune (ID non dichiarato); per DB8 la sovrapposizione con DB7 è
  probabile (5 normodotati su 10 e i due amputati avevano già partecipato agli esperimenti di Krasoulis et al. 2017, il paper di DB7; ID non
  dichiarati); DB1, DB2 e DB3 sono disgiunti per aritmetica; gli ID sono locali a ogni DB; **per DB6, DB7 (verso gli altri) e DB10 nessuna fonte
  dichiara nulla**. **Proposta:** i soggetti di test di NinaPro si prendono solo dai DB senza sovrapposizioni *dichiarate* (DB2, DB3, DB6, DB10):
  «non dichiarate» non vuol dire «assenti». **DB4, DB5, DB7 e DB8 restano interi nel pretraining**, così una persona presente in due di questi DB non
  può finire nel test di uno e nel pretraining dell'altro. Gli amputati di test vengono da DB3 e DB10: **se DB3 e DB10 abbiano amputati in comune non
  è dichiarato**, ed è un rischio da dichiarare come per i dataset Meta;
  (3) **dataset Meta**: vedi (b), fatto 23.

**Split proposti, generati** (`scripts/make_splits_draft.py`, seme 0; file `splits/draft/splits_draft.json`, split ufficiali con provenienza in
`splits/official/`; congelati il 03/10/2026 come `splits/v1/splits_v1.json`, decisione 6):

| Dataset | Soggetti | Pretraining | Test | 12,5% | 25% | 50% | Regola |
|---|---|---|---|---|---|---|---|
| camargo2021 | 22 | 17 | 5 | 3 | 5 | 9 | 20%, almeno 2 |
| capgmyo | 18 | 14 | 4 | 2 | 4 | 7 | 20%, almeno 2 |
| csl_hdemg | 5 | 4 | 1 | 1 | 1 | 2 | 1 soggetto di test (Simone, 01/10/2026) |
| emg2pose | 193 | 158 | 35 | 20 | 40 | 79 | ufficiale (`held_out_user`); in più 3.539 sessioni di test per fasi nuove |
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

**CSL-hdemg** (5 soggetti): **un solo soggetto di test** (Simone, 01/10/2026: «CSL-hdemg: un solo soggetto di test»; con «almeno 2» sarebbe stato il
40%). **emg2qwerty:** il modello generico ufficiale usa 96 utenti (conteggio di AG su `splits/official/emg2qwerty_users.json`, annotato nella riga
del fatto 10d), non i 100 del paper (4 utenti non sono in nessuna configurazione ufficiale):
qui vanno nel pretraining. Gli ID degli 8 utenti di test sono a 8 cifre (lo YAML ufficiale perde lo zero iniziale di 05775561).

## (e) Target RVQ

Secondo D5b (firmato): ancora RVQ sul **livello 0 del ramo 0**, accesa su camargo2021, capgmyo, emg2pose, emg2qwerty, grabmyo, hyser, kaifosh (se
resta nel pretraining), ninapro_db2/3/4/6/7 **ninapro_db8** (V2 1,46, run 59104658) e **zhang2026** (V2 1,55, run 59108493), ramo 0 confermato per entrambi, e **ninapro_db10** (V2 1,84, run 59183099, ramo 0 confermato il 02/10); spenta su putEMG,
CSL-hdemg, DB5.

## (f) Salti dell'asse dei tempi (punto 7 della revisione)

**Cosa sono.** Un salto è un passo fra due timestamp consecutivi più lungo di 1,5 volte il periodo nominale, cioè almeno un campione mancante. Lo
calcolano gli ingest di emg2qwerty, emg2pose e Kaifosh: `time_axis` nei sidecar, con il numero dei salti e le posizioni dei primi 10.000. Non sono i
«buchi» del QC, che sono tratti di valore costante. **Numeri del costruttore** (tutte le sessioni, job 59204297):

| Dataset | Sessioni con salti | Salti | Tempo mancante | Salto massimo | Elenco troncato |
|---|---|---|---|---|---|
| emg2pose | 1.668 su 25.253 | 222.477 | 3.432 s | 4,1 s | 0 sessioni |
| emg2qwerty | 76 su 1.135 | 27.017 | 562 s | **73,3 s** | 0 |
| Kaifosh (benchmark) | 46 su 100 | 5.098 | 63 s | 0,1 s | 0 |

Nessuna posizione di salto fuori dall'array. DB10 è
già spezzato alle pause dei `ts` in fase di ingest, con la stessa soglia (1,5 volte il passo mediano), e le pause diventano `trials`. Il campionamento
attuale non spezza i segmenti ai salti: una finestra può attraversarne uno, e il modello vede come contigui due tratti che non lo sono.
**[Stato al 03/10: firmata (i), decisione 8. Il dataloader spezza i segmenti ai salti; il costruttore del manifest no, e `manifest-v1` e' congelato
cosi'. Le sue ore e i suoi pesi contano anche i tratti piu' corti della finestra minima (1 s): ~2,4 h della classe B su 710,8, che il loader non
campiona. Effetto sulle quote: ~0,3% della classe B.]**

Opzioni:
- (i) **spezzare i segmenti a ogni salto elencato nel sidecar**, come i trial: nessuna finestra attraversa un salto. Si perde il tempo nei tratti
  piu' corti della finestra: con finestre da 4 s l'**1,2% di emg2pose e lo 0,2% di emg2qwerty** (in tutto 4,1 h della classe B su 710,8; 1 s: 2,4 h;
  8 s: 6,1 h). È lo scopo per cui le posizioni sono state salvate (commento nell'ingest di Kaifosh: servono «per non
  fare finestre a cavallo di un buco») ed è quello che si fa già per DB10. Nessuna sessione ha l'elenco troncato (oltre 10.000 salti), quindi le
  posizioni ci sono tutte;
- (ii) **spezzare solo i salti lunghi**: serve una soglia nuova, da congelare in `decisioni.md` prima di guardare la distribuzione delle durate;
- (iii) **ignorarli**: nessuna perdita, ma ingressi con discontinuità nascoste.

**Proposta di AG:** (i), senza soglie nuove. Il costo e' piccolo (sotto l'1,2% dei due dataset coinvolti), e senza spezzare una finestra di
emg2qwerty potrebbe incollare due tratti distanti 73 s. Se si firma (i), dataloader e costruttore spezzano i segmenti ai salti (modifica piccola,
con test; il costruttore gia' conta questo caso nella tabella di sensibilita'). **[03/10: fatto solo nel dataloader; il costruttore non e' stato
cambiato prima del congelamento. Vedi la nota sopra; review del 03/10.]**

## Prima della firma (cosa manca)

1. ~~Ingest completo di **emg2pose** e **DB10**~~ (finiti il 01-02/10); ~~tabelle dal costruttore~~ (job 59204297, 02/10).
2. ~~**Buchi**~~ (tratti costanti; il campionamento li esclude): scritti su DB8 (01/10), emg2qwerty (189, 02/10) ed emg2pose (2.786 in 576 sessioni,
   02/10, job 59204293); Zhang e DB10 ne hanno 0.
3. ~~**V2 e conferma del ramo RVQ**~~: DB8, Zhang e DB10 entrano (punto e; DB10 il 02/10, run 59183099).
4. **Fatti:** 1 (DB8), 2, 10b, 10c, 10d, 5 (NeuroRVQ: dataset del tokenizer, soggetti non dichiarati, licenza dei pesi) e 23 (sovrapposizione fra
   i dataset Meta, non dichiarata) firmati il 02/10/2026. **Resta** il 4, parziale (banda del pretraining del tokenizer non trovata): non blocca D9.
5. **Dove stanno i dati:** quasi tutto il processato è su `$SCRATCH` (purge a 40 giorni). Un manifest congelato deve puntare a dati che restano.
   Rinfresco fatto il 01/10 (job 59105021); la proposta in `decisioni.md` è un rinfresco **ogni 21 giorni per tutta la ladder** (prossimo il
   22/10): **da firmare con D9**, come regola e non come giro singolo.
6. **D10** (patch, contesto, masking): la patch cambia `D_t` e `D_c` ma non la composizione del manifest; il contesto decide quanto tempo resta
   campionabile: **con 4 s fissi la classe C perde 6,9 h su 21,3 e la sua quota del 5% diventa irraggiungibile** (al tetto darebbe al massimo il
   3,5%); con il contesto variabile da 1 a 4 s proposto in D10 non si perde nulla (`docs/proposta_d10.md`). D9 e D10 vanno firmati insieme.
7. **Salti dell'asse dei tempi:** scelta in (f).

**Congelato il 03/10/2026** come `manifest_59253155.json.gz` (tag git `manifest-v1`, hash in `docs/decisioni.md`). **Il manifest come
file** (proposta originale: `data/_manifests/manifest-v1.json`; e' invece fuori dal repo, nei risultati del job) con una riga per sessione (dataset, soggetto, sessione, percorso, hash del sidecar,
split, classe, peso), le quote e il tetto; tag git `manifest-v1`, hash del file nel registro, `D_t`, `D_c` e consumo realizzato per epoca
ricalcolati dallo stesso script (piano: «Chiuso quando»).

## Cosa si firma

- [x] (a) quote garantite per classe A 20%, B 75%, C 5% (realizzate esattamente con i dati veri), dichiarate sulla topologia presentata; tetto di 8 passaggi per dataset a E = 4,
      contato sul tempo di origine, lettura (i) per E = 8 e 2D
- [x] (a-bis) classi del montaggio come sopra (amputati di DB8/DB10 e Zhang random in B: interpretazioni di AG); pesi ∝ ore^0,5 dentro la classe
- [x] (b) Kaifosh tutto benchmark, con la sovrapposizione fra i dataset Meta dichiarata
- [x] (d) split: ufficiali dove esistono (emg2pose anche per sessione), altrimenti 20% (almeno 2), stratificati, seed 0; manifest annidati al
      12,5/25/50%; sovrapposizioni dichiarate (tokenizer, NinaPro, DB3/DB10, dataset Meta)
- [x] (e) ancora RVQ secondo D5b, estesa ai dataset nuovi solo se passano V2 e la conferma del ramo
- [x] (f) salti dell'asse dei tempi: segmenti spezzati a ogni salto del sidecar
- [x] rinfresco di `$SCRATCH` ogni 21 giorni per tutta la ladder

Rigenerare i numeri (CPU seriale, 0 GPU-ora, ~16 minuti; uscita in `$WORK/wearusfm_runs/results/passo4/`, fuori dal repo):

    sbatch scripts/slurm/build_manifest.sbatch                    # bozza: split di splits/draft, parametri di questa proposta
    sbatch scripts/slurm/build_manifest.sbatch --version manifest-v1 --splits <split firmati>   # SOLO dopo la firma
