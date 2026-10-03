# D9, D10 e sanity JEPA — fogli di firma (preparati il 03/10/2026)

**Come si usa.** Per ogni decisione: cosa si decide, i numeri, le opzioni, la proposta di AG e cosa costa sbagliare. Tu rispondi per numero:
«firmo 1-9», «4 con correzione: ...», «12 no». Io registro la firma in `docs/decisioni.md` con la tua frase e la data. **Firmare non e' un'operazione
mia: le proposte qui sotto NON sono decisioni.** I dettagli stanno nelle bozze: [`proposta_manifest_d9.md`](proposta_manifest_d9.md) (D9) e
[`proposta_d10.md`](proposta_d10.md) (D10); i numeri vengono dalla bozza del manifest del 02/10 (job 59204297).

**Gruppo 1** (1-14) sblocca il manifest congelato e il primo run. **Gruppo 2** (15-20) serve prima del sanity JEPA, il primo run su GPU (meta'
ottobre). Il manifest e' **irreversibile lungo tutta la ladder** (piano, passo 4).

---

## Gruppo 1 — manifest (D9), D10, filtraggio

### 1. Quote per classe — D9 (a)

**Si decide:** quanta parte dei campioni va a ogni classe di montaggio (A radi/anatomici, B anelli e fasce, C griglie HD), e se la quota si conta
sulla topologia di origine o su quella presentata al modello.
**Numeri:** ore di pretraining A 90,7 h (11,0%), B 710,8 h (86,4%), C 21,3 h (2,6%). Con A 20 / B 75 / C 5 le quote si realizzano esattamente; al
tetto la classe C potrebbe dare al massimo il 5,2%, la A il 22,0%.
**Opzioni:** (a) A 20 / B 75 / C 5 sulla topologia presentata (il piano); (b) le stesse sull'origine; (c) A 30% con tetto 12; (d) proporzioni
naturali, nessuna garanzia.
**Proposta di AG:** (a), con il campionatore che registra per ogni campione sia l'origine sia la topologia presentata.
**Se sbagliata:** si rifa' il manifest e si rifanno i run della ladder.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 2. Tetto di ripetizione — D9 (a)

**Si decide:** quante volte al massimo si ripassa su un dataset.
**Numeri:** con il tetto di 8 passaggi a E = 4, 10 unita' su 18 sono al tetto, 5 fra 7,1 e 7,9; emg2pose 2,8, emg2qwerty 2,7. Nel pilot a E = 8 e
nel controllo 2D i dataset al tetto arrivano a 16 passaggi (le ~16 epoche di «forte decadimento» di v10 §10.3, su testo).
**Opzioni:** (i) tetto 8 a E = 4, contato sul tempo di origine, e il pilot misura cosa succede a 16; (ii) tetto 8 al massimo E (4 a E = 4): ma allora
nessuna classe puo' superare la sua quota naturale e le quote garantite non servono.
**Proposta di AG:** (i).
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 3. Classe degli amputati di DB8/DB10 e di Zhang random — D9 (a-bis)

**Si decide:** in che classe stanno montaggi che nei metadati sono «sparsi» ma fisicamente sono anelli.
**Fatti firmati:** amputati di DB8 «a similar configuration» con 13 e 12 sensori (fatto 1); DB10, array «around the right forearm or residual limb»
(fatto 10c); Zhang random, sensori sulla circonferenza dell'avambraccio, ordine ignoto per 32 partecipanti (fatto 10b).
**Proposta di AG (interpretazione):** classe B per tutti e tre (anelli con rotazione o ordine ignoti). **Alternativa:** classe A, coerente con i
metadati «sparsi».
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 4. Pesi dentro la classe — D9 (a-bis)

**Proposta di AG:** pesi proporzionali a ore^0,5, con l'eccedenza di un dataset al tetto ridistribuita agli altri della classe (per ore, emg2pose ed
emg2qwerty prenderebbero l'86% della classe B; uniforme ripeterebbe i piccoli decine di volte). Risultato nella tabella di D9.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 5. Kaifosh — D9 (b), D3b

**Proposta di AG:** tutto benchmark, nessun soggetto nel pretraining. **Da dichiarare** (fatto 23, firmato): nessuna fonte dice se i tre dataset Meta
abbiano persone in comune; i 300 partecipanti pubblici di Kaifosh erano utenti di training del paper, quindi i numeri pubblicati non sono sullo
stesso test dello split pubblico. **Alternativa:** gli 80 utenti di train nel pretraining.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 6. Split dei soggetti — D9 (d)

**Proposta di AG:** split ufficiali dove esistono (emg2pose `held_out_user` piu' le 3.539 sessioni del test per fasi nuove, gia' deciso il 02/10;
emg2qwerty gli 8 utenti user0-7; Kaifosh 80/10/10); altrove 20% dei soggetti, almeno 2, stratificato amputati/normodotati, seme 0; CSL-hdemg un
solo soggetto di test (gia' deciso il 01/10); **DB4, DB5, DB7, DB8 interi nel pretraining** (sovrapposizioni dichiarate fra DB NinaPro, fatto 2);
manifest annidati al 12,5 / 25 / 50% dei soggetti. Sovrapposizioni da dichiarare nei risultati: tokenizer NeuroRVQ (fatto 5), NinaPro, DB3/DB10,
dataset Meta. Test: 210,9 h.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 7. Ancora RVQ nel manifest — D9 (e)

**Proposta di AG:** secondo D5b (gia' firmata): accesa su 15 dataset (DB8, Zhang e DB10 entrati il 01-02/10), spenta su putEMG, CSL-hdemg e DB5.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 8. Salti dell'asse dei tempi — D9 (f)

**Numeri:** emg2pose 222.477 salti in 1.668 sessioni (massimo 4,1 s), emg2qwerty 27.017 in 76 (massimo **73 s**). Oggi una finestra puo' attraversarli
e incollare tratti lontani.
**Opzioni:** (i) spezzare a ogni salto: costa l'1,2% di emg2pose e lo 0,2% di emg2qwerty a 4 s; (ii) solo i salti lunghi, con una soglia nuova;
(iii) ignorarli.
**Proposta di AG:** (i). E' gia' implementato nel dataloader (`split_at_gaps`).
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 9. Rinfresco di `$SCRATCH` — D9, «Prima della firma»

**Si decide:** come tenere vivi i dati per tutta la ladder. Quasi tutto il processato sta su `$SCRATCH` (purge a 40 giorni).
**Proposta di AG:** rinfresco delle date ogni 21 giorni per tutta la ladder (22/10, 12/11, 03/12, 24/12), col job gia' collaudato (CPU, minuti,
0 GPU-ora), ogni volta autorizzato da te.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 10. Patch — D10 (1)

**Proposta di AG:** 25 ms (piano). Divide la griglia da 200 ms del tokenizer, sta nella scala corta di masking, e gate D8 e passo 0 sono gia'
misurati li'. Cambiarla vorrebbe dire rifare il gate.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 11. Contesto — D10 (2)

**Numeri:** con finestre fisse da 4 s la classe C perde 6,9 h su 21,3 (CSL-hdemg e CapgMyo per intero, Hyser l'11%) e la quota C del 5% della
decisione 1 diventa irraggiungibile (al tetto: 3,5%). Con un minimo di 1 s non si perde nulla.
**Opzioni:** (i) fisso 4 s (piano e D2); (ii) variabile 1-4 s; (iii) fisso 2 s.
**Proposta di AG:** (ii), 1-4 s, inizio su multipli di 200 ms dall'inizio della prova. Il minimo di 1 s e' fissato dal masking (una maschera lunga da
500 ms con altrettanto contesto), non dai dati. Gia' implementato nel dataloader.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 12. Masking — D10 (3)

**Proposta di AG:** quote **attese** sui token nascosti: temporale corta 15%, media 25%, lunga 20%, spaziale 40%; meta' del temporale medio-lungo
come slab RVQ allineati (dove l'ancora e' accesa); spaziale al piu' meta' dei canali validi. Misurato sul generatore: nasconde 0,50-0,55 per un
budget di 0,5; quote realizzate (50 campioni per montaggio) corta 15-16%, media 12-14%, lunga 8-11%, slab 17-22%, spaziale 42-43%.
**Da scegliere:** una maschera temporale (non slab) nasconde (a) **un canale solo**, oggi nel codice, oppure (b) **un gruppo spaziale** (arco
d'anello, rettangolo di griglia, compartimento): sulle griglie dense (a) e' un compito facile, perche' i vicini sono visibili. **Proposta di AG:**
(b), «tubi» come in V-JEPA; sui montaggi sparsi il gruppo puo' essere un canale solo. Modifica piccola, con test.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 13. Filtraggio — nuova (v10 §4.3)

**Si decide:** dove filtrare. **I dati processati non sono filtrati all'ingest**; v10 §4.3 vuole passa-banda 20-450 Hz e notch a 50 e 60 Hz
«offline, mai on-the-fly», per risparmiare CPU.
**Numeri (indicativi, Mac, un processo):** 100 ms per batch da 8 senza filtro, 125 ms con il filtro; il passo 0 stimava ~68 finestre/s per GPU a
30M con 8 processi per GPU, contro 65-80 finestre/s **per processo** qui.
**Opzioni:** (a) filtro nel dataloader (gia' implementato); (b) copia filtrata su disco (spazio su `$SCRATCH` e un job CPU, da stimare).
**Proposta di AG:** (a), **registrata come deviazione da v10**, confermata da una misura del ritmo su Leonardo con sessioni vere (CPU e
`boost_qos_dbg`, stima < 1 GPU-ora, costo esatto prima del lancio).
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 14. Congelamento di `manifest-v1`

**Si fa, dopo le firme 1-9:** gli split in bozza diventano gli split firmati (`splits/v1/`), il costruttore gira con `--version manifest-v1` (CPU
seriale, ~16 minuti, 0 GPU-ora: rifiuta di partire con split in bozza o soggetti mancanti), tag git `manifest-v1`, hash del file nel registro,
`D_t` e `D_c` ricalcolati (piano, «Chiuso quando»). Serve la tua autorizzazione al job.
**Tua decisione:** [ ] autorizzo · [ ] non ancora

---

## Gruppo 2 — prima del sanity JEPA

Valori che la regola del progetto vuole **congelati prima di guardare i risultati**. Il sanity JEPA (piano, passo 6, «Chiuso quando»: un solo dataset
omogeneo, «non collassa») e' il primo run su GPU: `boost_qos_dbg`, qualche run da 30 minuti, stima 1-3 GPU-ora sui ≤ 100 del passo 6, costo esatto
prima del lancio.

### 15. Frazione nascosta

**Proposta di AG:** 0,5, fissa per il sanity e per la finestra 1, senza sceglierla sui risultati (v10 non da' un numero; non propongo un valore
dalla letteratura senza averlo verificato). Un'ablation sul masking non e' nel piano; se il sanity collassa, la si riapre come decisione nuova.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 16. Soglie d'allarme delle diagnostiche (v10 §7.1)

**Proposta di AG:** valutate ogni 500 passi su un batch fisso di validazione:
- **collasso da query:** allarme se il rapporto (varianza fra campioni a query fissa / varianza fra query) e' sotto **0,05** per 3 valutazioni di
  fila;
- **rango effettivo** delle uscite del backbone (medie per istante): allarme se scende sotto **il 10% della dimensione** del modello per 3
  valutazioni di fila.
Le soglie sono mie, non da v10: l'idea e' fermarsi su un collasso evidente, non su un calo. Un allarme ferma il run e si guarda prima di
continuare.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 17. Dataset del sanity

**Proposta di AG:** emg2qwerty (317 h di pretraining, un solo dispositivo, due anelli da 16 a 2 kHz, ancora RVQ accesa). **Alternativa:** emg2pose
(un anello, 288 h).
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 18. Peso delle ancore e momento EMA

**Proposta di AG:** peso complessivo delle ancore 0,2 (centro dell'intervallo 0,1-0,3 di v10 §6.2); momento EMA del teacher 0,996 costante per il
sanity (valore di lavoro; lo schedule vero e' D14, prima della finestra 1).
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 19. Dropout del livello muscolo (D15, anticipata)

**Proposta:** 0,4, la proposta del piano (§12), cosi' il sanity usa gia' il valore della finestra 1. Su emg2qwerty nessun canale ha il muscolo
noto, quindi per il sanity non cambia nulla; conta dalla finestra 1.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

### 20. Target JEPA per il sanity (D11)

**Proposta:** (b), uscita del decoder a query del teacher (default di lavoro del piano), con la diagnostica della decisione 16 attiva dal primo
passo. D11 vera si chiude prima del passo 7: (a) entra comunque fra le ablation della finestra 1.
**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

**Dopo le firme:** registro tutto in `docs/decisioni.md`, aggiorno le bozze, congelo `manifest-v1` (se autorizzi la 14) e preparo il sanity JEPA.
Restano per la finestra 1 (entro fine ottobre): D12 (soglia del pilot), D14 (schedule WSD), D16 (frazioni dei soggetti).
