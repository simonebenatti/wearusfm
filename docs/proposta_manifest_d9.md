# Proposta — manifest del pretraining (D9, passo 4): BOZZA da firmare entro il 25/10/2026

> **IN REVISIONE (02/10/2026) — non firmare questa versione.** La review indipendente dei documenti ha trovato, e AG ha verificato, gli 8 punti
> qui sotto. **Stato al 02/10 (fra quadre):** il testo e' riscritto per i punti 2-6 e 8; le tabelle (punto 1) e la scelta sui salti (punto 7)
> aspettano i numeri del costruttore.
> 1. le ore, i passaggi e le quote qui sotto contano anche i soggetti di test (~1.011 h), mentre il manifest pesa solo il pretraining (~870 h): con il
>    tetto di 8 passaggi la classe C non arriva piu' al 5%. Si rifa' tutto coi numeri del costruttore (`scripts/build_manifest.py`) sui dati veri; **[aperto: tabelle ancora del 01/10]**
> 2. «Kaifosh: 100 utenti nuovi» non e' verificato: Kaifosh, emg2pose ed emg2qwerty sono dello stesso produttore e gli ID non sono confrontabili
>    (sovrapposizione ignota, da dichiarare come per NinaPro); idem per gli amputati di DB3 e DB10; **[riscritto in (b) e (d); fatto 23 da raccogliere]**
> 3. emg2pose: le 3.539 registrazioni del test per fasi nuove escono dal pretraining (Simone, 02/10/2026: gia' negli split); **[riscritto in (d)]**
> 4. il tetto di 8 passaggi e' definito per 4 epoche: nei gradini a 2D i passaggi raddoppiano (16, il «forte decadimento» di v10 §10.3); **[riscritto in (a)]**
> 5. «origine» contro «presentata»: il piano propone la topologia presentata; la scelta va argomentata anche sul volume D_c (classe C ~23% di D_c); **[riscritto in (a)]**
> 6. interpretazioni scritte come fatti (anelli degli amputati, Zhang random «e' un anello», soggetti di test «visti» dal tokenizer): vanno marcate; **[marcate in (a-bis) e (d)]**
> 7. i salti dell'asse dei tempi (emg2pose 1.668 registrazioni, emg2qwerty 76, Kaifosh) non spezzano i segmenti: una finestra puo' attraversarli. Va
>    deciso come trattarli prima del manifest; **[opzioni in (f); decisione di Simone, coi numeri del costruttore]**
> 8. elenco dei fatti da firmare incompleto (servono anche 1 DB8, 4, 5, 10b, 10c, 10d); piccoli numeri da correggere. Aggiornamento
>    02/10/2026: 1 (DB8), 2, 10b, 10c, 10d firmati da Simone; restano 4 e 5, piu' il nuovo 23. **[elenco in «Prima della firma»; i piccoli numeri
>    si correggono con le tabelle]**


**Stato: bozza di AG del 01/10/2026, testo rivisto il 02/10/2026, NON firmata.** Il manifest è **irreversibile lungo tutta la ladder** (piano,
passo 4: «firma TU»). Riferimenti: piano operativo §Passo 4 (le cinque scelte), v10 §2.8 (quote, classi del montaggio), v10 §10.3 (D è un manifest;
`D_t`, `D_c`). **I numeri delle tabelle sono ancora quelli del 01/10** (`scripts/manifest_table.py` sui riepiloghi di ingest in `results/passo2/`):
contano anche i soggetti di test e usano ore ipotetiche per emg2pose (400 h) e DB10 (56 h), i cui ingest sono poi finiti il 01-02/10. **Non vanno
letti come proposta:** si sostituiscono con l'uscita di `scripts/build_manifest.py` sui dati veri, solo pretraining.

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

Il 40% del piano porterebbe la classe A a ~15 passaggi: vicino alle ~16 epoche che v10 §10.3 riporta come punto stimato di forte decadimento
(Muennighoff et al., **su testo**). La classe C è così piccola, in ore, che anche il 10% la ripeterebbe 15 volte.

**Origine o presentata?** Il piano propone la quota **sulla topologia presentata**, «con un tetto esplicito alla ripetizione dei dati sparsi veri,
e lettura del pilot per topologia»: i montaggi bipolari virtuali ricavati dalle griglie HD sono campioni «tipo sparso» e riempiono parte della quota
senza ripetere i dati sparsi veri. La bozza del 01/10 proponeva invece l'origine, con un argomento solo sulle ore (le griglie sono poche ore, quindi
riempire la quota A con montaggi virtuali ripeterebbe i dati C). L'argomento era incompleto:
- **sul tempo** la classe C è scarsa (tabella del 01/10: 26,7 h, da rifare): ogni campione virtuale ripete un tratto di tempo di una griglia;
- **sul volume dei canali** non lo è: con 128-256 canali per griglia, nella tabella del 01/10 la classe C è ~23% di `D_c` (0,91 su 3,9 miliardi).
  Con i montaggi al volo (D6a), due passaggi sullo stesso tratto possono presentare coppie di elettrodi diverse: si ripete il tempo, non l'ingresso.

Conseguenza che vale con qualunque numero: se il tetto si conta sul **tempo di origine**, la classe C può dare al massimo `8 · H_C / (4 · H)` dei
campioni a E = 4 (`H_C` ore della classe C, `H` ore del pretraining), sommando campioni HD e montaggi virtuali. La scelta fra origine e presentata
sposta solo questa fetta; quanto vale lo dirà il costruttore. *Da dichiarare:* che per il modello un montaggio virtuale da una griglia equivalga a un
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

**Proposta di AG (quote da ricalcolare coi numeri del costruttore):** quote garantite **A 20%, B 75%, C 5%** (A e C circa il doppio della loro quota
naturale), **dichiarate sulla topologia presentata** come nel piano. **Tetto di 8 passaggi per dataset a E = 4, contato sul tempo di origine**,
lettura (i). Il campionatore registra per ogni campione sia la classe di origine sia quella presentata, così il pilot si legge in tutti e due i modi.
Con il tetto la classe C potrebbe non arrivare al 5% (punto 1 della revisione). *Alternative:* quota sull'origine (bozza del 01/10); A 30% con tetto
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

**Proposta di AG: tutto benchmark, nessun soggetto nel pretraining.** Kaifosh aggiunge soggetti, non topologie (piano), alla classe B che ha già
la gran parte delle ore; usa lo stesso bracciale di emg2pose ed emg2qwerty (fatto 20, firmato). Come benchmark ha uno split ufficiale 80/10/10
(fatto 7b, firmato) e un risultato pubblicato con cui confrontarsi. Licenza CC-BY-NC in entrambi i casi (D3a). *Alternativa:* gli 80 utenti di
train nel pretraining, i 20 di validazione e test come benchmark.

**Correzione (review del 02/10):** la bozza del 01/10 diceva «100 utenti nuovi». Non è verificato. I tre dataset Meta vengono dallo stesso
produttore, nessuna fonte del registro dice se abbiano partecipanti in comune, e gli ID non sono confrontabili fra i dataset: un utente di test di
Kaifosh potrebbe essere nel pretraining attraverso emg2pose o emg2qwerty. Il compito è registrato come **fatto 23 (da raccogliere)**. Finché resta
aperto, il benchmark Kaifosh si dichiara «utenti forse visti nel pretraining (stesso produttore, sovrapposizione ignota)», come per NinaPro; lo
stesso vale per i soggetti di test di emg2pose ed emg2qwerty rispetto agli altri due dataset.

## (c) Esito di D6a

**Già chiuso** (21/09/2026): montaggi **al volo**, layout **packing** (D6a, D6b). Nessun montaggio precomputato: `D_c` conta le sorgenti, come nella
tabella.

## (d) Split dei soggetti

**Proposta di AG:**
- **soggetti di test per dataset**, tenuti fuori dal pretraining: gli split ufficiali dove esistono (emg2pose: colonne `held_out_user` e `split` del CSV dei
  metadati; Kaifosh: 80/10/10; emg2qwerty: gli 8 utenti di test di `config/user/user0-7.yaml`, fatto 10d); altrove il **20% dei soggetti, almeno 2, arrotondato per eccesso**, stratificato fra amputati e normodotati (DB7, DB8, DB10), con
  **seed 0**; in Zhang lo stesso soggetto ha i due modi: si divide per soggetto;
- **emg2pose, anche per sessione** (Simone, 02/10/2026): le 3.539 registrazioni (58,8 h) del test ufficiale per fasi nuove (`split = test`,
  `held_out_stage = True`, di utenti non tenuti fuori) escono dal pretraining e restano per la valutazione (`test_sessions` negli split);
- **manifest sottocampionati per soggetti** (asse di v10 §10.6, D16): sottoinsiemi annidati del 12,5, 25 e 50% dei soggetti di pretraining, per
  dataset, stesso seed;
- **sovrapposizioni da dichiarare** (nessuna si risolve con gli ID):
  (1) **tokenizer NeuroRVQ.** Dal paper (fatto 5, raccolto il 02/10, **da firmare**: appendice E, Tabella 7, e §5.1) il tokenizer è
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
`splits/official/`; da rivedere, non congelati):

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
40%). **emg2qwerty:** il modello generico ufficiale usa 96 utenti, non i 100 del paper (4 utenti non sono in nessuna configurazione ufficiale):
qui vanno nel pretraining. Gli ID degli 8 utenti di test sono a 8 cifre (lo YAML ufficiale perde lo zero iniziale di 05775561).

## (e) Target RVQ

Secondo D5b (firmato): ancora RVQ sul **livello 0 del ramo 0**, accesa su camargo2021, capgmyo, emg2pose, emg2qwerty, grabmyo, hyser, kaifosh (se
resta nel pretraining), ninapro_db2/3/4/6/7 **ninapro_db8** (V2 1,46, run 59104658) e **zhang2026** (V2 1,55, run 59108493), ramo 0 confermato per entrambi; spenta su putEMG, CSL-hdemg, DB5. **Resta da misurare DB10** con la
regola già firmata (V2 ≤ 2 e conferma del ramo, opzione b): GPU, stima < 0,5 GPU-ora, da confermare col costo prima del lancio.

## (f) Salti dell'asse dei tempi (punto 7 della revisione)

**Cosa sono.** Un salto è un passo fra due timestamp consecutivi più lungo di 1,5 volte il periodo nominale, cioè almeno un campione mancante. Lo
calcolano gli ingest di emg2qwerty, emg2pose e Kaifosh: `time_axis` nei sidecar, con il numero dei salti e le posizioni dei primi 10.000. Non sono i
«buchi» del QC, che sono tratti di valore costante. Registrazioni con salti: emg2pose 1.668, emg2qwerty 76; Kaifosh ne ha, ma è benchmark. DB10 è
già spezzato alle pause dei `ts` in fase di ingest, con la stessa soglia (1,5 volte il passo mediano), e le pause diventano `trials`. Il campionamento
attuale non spezza i segmenti ai salti: una finestra può attraversarne uno, e il modello vede come contigui due tratti che non lo sono.

Opzioni:
- (i) **spezzare i segmenti a ogni salto elencato nel sidecar**, come i trial: nessuna finestra attraversa un salto. Si perdono solo le finestre a
  cavallo (tempo da contare col costruttore). È lo scopo per cui le posizioni sono state salvate (commento nell'ingest di Kaifosh: servono «per non
  fare finestre a cavallo di un buco») ed è quello che si fa già per DB10. Le sessioni con più di 10.000 salti (`gaps_truncated`) non hanno tutte le
  posizioni nel sidecar: si escludono o si ricalcolano;
- (ii) **spezzare solo i salti lunghi**: serve una soglia nuova, da congelare in `decisioni.md` prima di guardare la distribuzione delle durate;
- (iii) **ignorarli**: nessuna perdita, ma ingressi con discontinuità nascoste.

**Proposta di AG:** (i), senza soglie nuove. Il costruttore conta già, per sessione e per unità, i salti, il tempo mancante e il salto massimo. Il
numero da guardare prima della firma è il tempo perso con le finestre a cavallo, più le sessioni troncate. Se si firma (i), costruttore e dataloader
spezzano i segmenti ai salti (modifica piccola, con test).

## Prima della firma (cosa manca)

1. ~~Ingest completo di **emg2pose** e **DB10**~~ (finiti il 01-02/10). **Tabelle da rigenerare col costruttore**, solo pretraining (punto 1 della
   revisione).
2. **Buchi** (tratti costanti; il campionamento li esclude): scritti su DB8 (01/10) ed emg2qwerty (189, 02/10); Zhang ne ha 0. **DB10 ed emg2pose
   in misura**, in sola lettura: un'eventuale scrittura si chiede a parte.
3. **V2 e conferma del ramo RVQ:** DB8 e Zhang entrano (punto e); **DB10 dopo la misura dei suoi buchi**.
4. **Fatti:** 1 (DB8), 2, 10b, 10c, 10d firmati il 02/10/2026. **Restano:** 4 e 5 (NeuroRVQ: dataset del tokenizer e sovrapposizione coi soggetti
   di test), e il nuovo **23** (sovrapposizione fra i dataset Meta, da raccogliere).
5. **Dove stanno i dati:** quasi tutto il processato è su `$SCRATCH` (purge a 40 giorni). Un manifest congelato deve puntare a dati che restano.
   Rinfresco fatto il 01/10 (job 59105021); la proposta in `decisioni.md` è un rinfresco **ogni 21 giorni per tutta la ladder** (prossimo il
   22/10): **da firmare con D9**, come regola e non come giro singolo.
6. D10 (lunghezza della patch) cambia `D_t` e `D_c` ma non la composizione del manifest.
7. **Salti dell'asse dei tempi:** scelta in (f).

**Il manifest come file:** `data/_manifests/manifest-v1.json` con una riga per sessione (dataset, soggetto, sessione, percorso, hash del sidecar,
split, classe, peso), le quote e il tetto; tag git `manifest-v1`, hash del file nel registro, `D_t`, `D_c` e consumo realizzato per epoca
ricalcolati dallo stesso script (piano: «Chiuso quando»).

## Cosa si firma

- [ ] (a) quote garantite per classe (numeri dal costruttore), dichiarate sulla topologia presentata; tetto di 8 passaggi per dataset a E = 4,
      contato sul tempo di origine, lettura (i) per E = 8 e 2D
- [ ] (a-bis) classi del montaggio come sopra (amputati di DB8/DB10 e Zhang random in B: interpretazioni di AG); pesi ∝ ore^0,5 dentro la classe
- [ ] (b) Kaifosh tutto benchmark, con la sovrapposizione fra i dataset Meta dichiarata
- [ ] (d) split: ufficiali dove esistono (emg2pose anche per sessione), altrimenti 20% (almeno 2), stratificati, seed 0; manifest annidati al
      12,5/25/50%; sovrapposizioni dichiarate (tokenizer, NinaPro, DB3/DB10, dataset Meta)
- [ ] (e) ancora RVQ secondo D5b, estesa ai dataset nuovi solo se passano V2 e la conferma del ramo
- [ ] (f) salti dell'asse dei tempi: segmenti spezzati a ogni salto del sidecar
- [ ] rinfresco di `$SCRATCH` ogni 21 giorni per tutta la ladder

Rigenerare i numeri del 01/10 (da sostituire con l'uscita di `scripts/slurm/build_manifest.sbatch`):

    python3 scripts/manifest_table.py --extra emg2pose=400 --extra ninapro_db10=56 --without-kaifosh --quota A=0.2,B=0.75,C=0.05 --alpha 0.5 --max-passes 8
