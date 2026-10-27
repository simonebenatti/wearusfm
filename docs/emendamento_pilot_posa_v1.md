# Emendamento 1 al protocollo `pilot_posa_v1` (R3 / R3c): rendere la seconda ondata diagnostica

**Stato: FIRMATO da Simone il 26/10/2026 («emendamento firmato, finisci i test»); passato anche a GPT per la lettura scientifica (`docs/prompt_gpt_emendamento_r3_20261026.md`). Proposta di Claude del 26/10/2026, su richiesta di Simone** («Prima della campagna completa, prepara un emendamento esplicito al protocollo
gia' firmato, senza modificarne silenziosamente le regole. La questione principale e' distinguere: a) la posa viene imparata fuori campione?
b) l'informazione e' accessibile nella rappresentazione? c) trasferisce agli altri dataset?»). Dal 26/10 la responsabilita' scientifica e' di
GPT: l'emendamento va a GPT e a Simone per la firma; Claude lo esegue. **Nessuna regola firmata cambia**: il criterio §5 (R3 contro R0, +1
punto con IC95 sopra zero su P1 e P2), le deroghe, le soglie del collaudo e la scelta sulla mano non sollecitata restano quelle del 26/10. Si
AGGIUNGONO letture e controlli, con le loro regole congelate prima di guardare. Nessun job senza il via di Simone; nessuna nuova variante del
modello.

## E0. Le tre domande e che cosa risponde a ciascuna

| Domanda | Confronto | Che cosa misura | Stato nel protocollo firmato |
|---|---|---|---|
| (a) La posa viene **imparata** fuori campione? | R3 su utenti di emg2pose esclusi dal training, contro il predittore costante e contro il controllo disallineato | Se la testa stima il movimento dal segnale, su persone mai viste | **Mancava** (il protocollo legge la posa solo sul pannello fisso del training): lo aggiunge E1-E2 |
| (b) L'informazione e' **accessibile** nella rappresentazione? | Sonda lineare per istante dai latenti congelati alla posa, su R3, R3c e R0 | Se i latenti del backbone portano la posa anche senza la testa (R3c, R0) e quanto la testa la rende piu' leggibile (R3) | **Mancava**: lo aggiunge E3 |
| (c) **Trasferisce** agli altri dataset? | Criterio §5: R3 contro R0 su P1 e P2 (adozione); R3 contro R3c (attribuzione alla posa); R3c contro R0 (togliere il JEPA) | Il candidato contro il default; il merito della posa; per dataset | **C'e'** (criterio firmato + descrittivi): E5 lo precisa |

Piu' E4 (gradienti della posa e delle ancore nei moduli condivisi) e E6 (albero delle decisioni congelato).

## E1. Valutazione cinematica fuori campione (domanda a)

- **Insieme di valutazione:** lo split ufficiale `val` di emg2pose: 1.950 registrazioni, 32,6 h, **15 utenti, tutti assenti dal training**
  (il pretraining usa lo split `train`: 158 utenti; `val` e `test` non entrano ne' nel manifest ne' in `pose_stats.json`), 57 sessioni, fasi
  miste (unconstrained 468, freestyle a una e due mani, fasi guidate). Il `test` ufficiale resta intatto. Le statistiche di standardizzazione sono
  quelle del train (`pose_stats.json`, gia' cosi').
- **Finestre:** per registrazione 10 finestre da 4 s a inizio fisso (griglia a 200 ms, come il loader: `align_ms`), seme 20261026, senza
  sovrapposizione fra finestre della stessa registrazione; stessa pipeline di preprocessing dell'estrazione del readout (`fm_features`: filtro,
  scala per soggetto); encoder senza maschera (come la sonda). ~19.500 finestre.
- **Predizione:** la testa di posa di R3 sui latenti per istante, de-standardizzata con le statistiche del train. Per ogni finestra e
  istante: errore assoluto per giunto in radianti (o nell'unita' misurata dal collaudo dell'ingest).
- **Riferimenti:** (i) **predittore costante** = media del train per giunto (MAE_costante); (ii) **controllo disallineato** (E2).
- **Metriche:** MAE per giunto e media sui 20 giunti; **per utente** (15 valori) e complessiva; **rapporto** MAE_R3 / MAE_costante; separatamente
  per finestre **dinamiche** e **poco dinamiche** (varianza della posa nella finestra sopra o sotto la mediana di `val`); per fase.
- **Regola di lettura congelata («posa appresa fuori campione»):** il rapporto MAE_R3 / MAE_costante, mediato sui 20 giunti e calcolato sulle
  finestre dinamiche di `val`, ha **limite superiore dell'IC95 (bootstrap sui 15 utenti, 1.000 ricampionamenti, seme 20261009) < 0,9** e **mediana
  per utente < 0,9**. Cioe': almeno il 10% meglio del costante, con l'incertezza fra utenti. Sotto 0,8 complessivo si annota «appresa in modo
  sostanziale» (descrittivo). Letta all'endpoint a 2 epoche; a 1 epoca descrittiva.
- Per R3c e R0 non c'e' testa: (a) si misura solo su R3; la loro posa si legge in E3.

## E2. Controllo disallineato in valutazione (non un training con etichette sbagliate)

- Lo **stesso modello R3** valutato sulle finestre di `val` con il **target di posa di un'altra finestra dello stesso utente e della stessa fase**,
  a distanza di almeno 8 s nella stessa registrazione (o, se la registrazione e' corta, della stessa fase in un'altra sessione dello stesso
  utente), appaiamento fisso col seme 20261026: distribuzioni marginali e metadati uguali, movimento diverso. Coppie con posa quasi identica
  (distanza standardizzata fra le due finestre sotto il 10° percentile) si scartano e si ritirano.
- **Regola congelata:** il rapporto MAE_disallineato / MAE_costante ha **limite inferiore dell'IC95 > 0,95** sulle stesse finestre dinamiche
  (cioe' il controllo non batte il costante). Se lo batte, la testa legge qualcosa che non e' il movimento della finestra (metadati, posizione
  di riposo dell'utente, deriva lenta): (a) **non e' interpretabile** e si registra prima di leggere il criterio §5.
- Si riporta anche il **gap** MAE_disallineato - MAE_R3 per utente: la parte di errore che dipende davvero dall'allineamento EMG-posa.
- **Precisazione (27/10/2026, dalla review di GPT `docs/reviews/r3_updates_review_20261027.md`, scritta PRIMA di qualunque risultato E2; voce
  in `decisioni.md`):** le coppie hanno anche lo **stesso lato** (stesso utente, fase e lato); il fallback e' SOLO un'altra sessione dello stesso
  utente, mai un'altra registrazione della stessa sessione (che puo' essere la mano opposta allo stesso istante: nessun disallineamento, e con
  la standardizzazione per lato coordinate diverse); senza candidati la finestra si esclude e si conta, i vincoli non si rilassano. Il **gap** si
  calcola sullo **stesso supporto** del disallineato: stesse finestre dinamiche con coppia conservata, intersezione delle maschere valide della
  finestra e della partner, stessi pesi, stessa aggregazione per utente; il rapporto riporta MAE allineata e disallineata su quel supporto, il
  gap e i conteggi (coppie tenute, stessa registrazione / altra sessione, escluse, scartate come quasi identiche). La E1 e il gate di E2
  (rapporto col costante e soglia 0,95) restano invariati.

## E3. Accessibilita' nella rappresentazione (domanda b)

- Per **R3, R3c e R0** (endpoint a 2 epoche; a 1 epoca descrittivo), encoder congelato senza maschera, latenti del backbone **per istante**
  (`media sui K latenti di z[s, p]`, la parte del backbone di `pool_p1p2` prima della media sul tempo): **regressione ridge lineare** dai
  latenti alla posa standardizzata, allenata su un sottoinsieme fisso del `train` (200 registrazioni, 40 utenti, 10 finestre l'una, 8 istanti per
  finestra, seme 20261026; iperparametro di ridge scelto per validazione interna fra utenti del train), valutata su `val` con le stesse
  metriche di E1 (rapporto col costante, per utente, dinamiche / poco dinamiche).
- **Letture (descrittive, congelate nella forma):** (b1) R3c e R0 sotto 0,9: i latenti delle sole ancore (o di A) portano gia' la posa in modo
  lineare -> il bersaglio di posa aggiunge poco alla rappresentazione, al piu' alla testa; (b2) R3 sotto 0,9 e R3c/R0 sopra: la posa e' entrata
  nella rappresentazione grazie al termine; (b3) tutti sopra 0,9 ma la testa di R3 (E1) sotto: l'informazione e' accessibile alla testa (LayerNorm
  per istante) ma non alla sonda lineare sui latenti grezzi: allineamento testa-sonda da rivedere prima di trarre conclusioni.
- Costo: nessun training. Latenti per istante di 6 modelli-endpoint su `val` e sul sottoinsieme del train: estrazione GPU (vedi E7).

## E4. Gradienti della posa e delle ancore nei moduli condivisi

- Nel collaudo dbg di R3 e poi ogni 500 passi del training, **senza aggiornamenti in piu'** (come `keep_grad_every`): norma del gradiente del
  termine di posa pesato e di quello delle ancore pesato sui parametri condivisi (tokenizer, identita' di canale, encoder locale, pooling,
  backbone), e **coseno** fra i due gradienti. Diagnostica (`metrics.jsonl`: `shared_grad_pose`, `shared_grad_anchors`, `shared_grad_cos`),
  senza criterio di passaggio.
- **Lettura congelata (descrittiva, registrata):** se nei passi 500-1.300 del collaudo la norma del gradiente della posa sui moduli condivisi e'
  **sotto l'1%** di quella delle ancore, si registra «la posa non muove l'encoder» prima del lancio e il peso si ridiscute con GPT e Simone (il
  cancello del §2 (iii) resta quello firmato; questa e' un'informazione in piu', non una soglia nuova).
- Codice: `--pose-grad-every N` nel trainer, sul modello di `keep_grad_every` (autograd.grad con `retain_graph`), coperto dal test del default
  invariato (spento = nessuna differenza). Da scrivere solo dopo la firma dell'emendamento.

## E5. Attribuzione e adozione, separate (domanda c)

- **Adozione:** criterio §5 firmato, R3 contro R0, su P1 **e** P2. Invariato.
- **Attribuzione alla posa:** R3 contro R3c, stessa chiamata (`compare --single-seed --models`), stessi endpoint, letta con la stessa soglia ma
  **descrittiva** (non decide l'adozione). **R3c contro R0** dice che cosa fa togliere il JEPA a 2 epoche.
- **Conferma a due semi:** se R3 passa il criterio e si vuole affermare che il beneficio viene dalla posa, la conferma col seme 1 include **R3c
  seme 1** oltre a R3 seme 1 e R0 seme 1 (GPT, passo 3): +47 / 60 GPU-ora rispetto al preventivo del 26/10.
- **Per dataset:** tutte le letture anche per dataset (DB2, DB3, DB6, EPN-612, UCI), descrittive: un guadagno solo sui bracciali e' coerente con
  una specializzazione al polso, non con la mano.

## E6. Albero delle decisioni, congelato prima di leggere

| Esito | Lettura | Mossa successiva |
|---|---|---|
| E2 fallisce (il controllo disallineato batte il costante) | (a) non interpretabile | Si ferma la lettura di (a); si cerca la perdita (metadati, deriva) prima di qualunque altra cosa |
| (a) NO: posa non appresa fuori campione | Dati / target / ottimizzazione | Breve prova supervisionata con una piccola rete temporale sugli stessi ingressi e target (GPT, passo 2): se impara, il limite e' nel nostro tokenizer/backbone; se no, nel target o nella pipeline |
| (a) SI', (b) R3 accessibile, (c) R3 non passa §5 e R3 non batte R3c | La posa e' appresa e leggibile ma non trasferisce | Esposizione, montaggio, dominio: lettura per dataset; allineamento a traiettorie di posa (GPT, passo 4) come variante successiva, una sola |
| (a) SI', (b) R3 NON accessibile ai latenti grezzi (b3) | Limite di readout / allineamento | Intervenire sulla lettura (testa senza LayerNorm o sonda per istante), non dichiarare assente l'informazione |
| (a) SI', (c) R3 passa §5 solo su P2 | Specializzazione ai bracciali | Non adozione; esposizione e montaggio; DB3 letto a parte |
| (a) SI', (c) R3 passa §5 su P1 e P2, R3 batte R3c | Candidato | Conferma col seme 1 (R3, R3c, R0); solo dopo, scaling |
| (c) R3c batte R0 e R3 non batte R3c | Il guadagno viene dal togliere il JEPA, non dalla posa | Nessuna adozione di R3; R3c va letto con la regola dei bracci (conferma col seme 1 di R3c) prima di qualunque default |

## E7. Durata, criteri e costo della verifica aggiuntiva

- **Durata:** la prova scientifica e' il tronco a 2 epoche (49.046 passi), non il collaudo da 1.300 passi (controllo tecnico, GPT concorda). E1-E3
  si leggono all'endpoint a 2 epoche (criteri) e a 1 epoca (descrittivi).
- **Costo aggiuntivo stimato** (basi: estrazione del readout P1, 82.339 finestre in 12-14 min su 1 A100 = ~100 finestre al secondo):
  latenti per istante di `val` (~19.500 finestre) e del sottoinsieme del train (~2.000 finestre) per 6 modelli-endpoint (R3, R3c, R0 x 2 epoche +
  R3/R3c a 1 epoca dove serve): **~1,0 GPU-ora attesa, tetto 4,0** (6 job da 40 min); CPU: predizioni della testa, disallineamento, ridge e
  bootstrap: **~12 core-ore attese, tetto 64** (4 job da 2 h a 8 core); E4 nel collaudo e nel training: costo trascurabile (un `autograd.grad` ogni
  500 passi). Codice: `scripts/pose_eval.py` (estrazione dei latenti per istante + predizioni + controllo disallineato + ridge + bootstrap),
  launcher, test: ~1 giorno, solo dopo la firma.
- **Totale della seconda ondata con l'emendamento:** 98,1 / 132,0 GPU-ora del preventivo del 26/10 + ~1,0 / 4,0 = **~99 / 136 GPU-ora**, ~161 /
  595 core-ore; conferma, se passa e con R3c, 141 / 180 GPU-ora. Calendario: +1 giorno sui 5-6 stimati.
- **Che cosa NON fa l'emendamento:** nessuna nuova variante del modello o del bersaglio (l'allineamento a traiettorie di GPT resta condizionale
  all'albero E6), nessuna sonda non lineare per P1/P2 (H1 della diagnosi: prova a parte, su CPU, se GPT la vuole), nessun cambio di soglie firmate.

## E8. Il prossimo esperimento prioritario, in una riga per Simone

**R3 + R3c con questo emendamento**: gia' firmati e preparati, con in piu' (a) la posa misurata su 15 utenti mai visti contro il costante e contro
il disallineato, (b) la sonda per istante dai latenti su R3, R3c e R0, (c) adozione e attribuzione separate per dataset, i gradienti nel collaudo,
e l'albero E6 congelato. Costo ~99 / 136 GPU-ora. Se il risultato e' negativo, l'albero dice gia' quale delle tre cause e' esclusa e quale no.

## E9. Stato del codice dell'emendamento (26/10/2026, sera; dopo la firma)

Fatto, test CPU verdi, nessun job: **E4** `--pose-grad-every N` nel trainer (`RunConfig.pose_grad_every`; `shared_grad_pose`, `shared_grad_anchors`,
`shared_grad_cos` in `metrics.jsonl` ai passi multipli di N; nel contratto DDP sul micro-batch finale del processo 0, gradiente locale; a un
processo come `keep_grad_every`; il training non cambia bit per bit con o senza), lettura E4 in `scripts/arms_collaudo.py` (`gradienti_condivisi`,
descrittivo, `flag_sotto_1pct` sui passi 500-1.300). **E1-E3** `src/wearusfm/harness/pose_eval.py` + `scripts/pose_eval.py` (`--stage extract`:
finestre fisse 10 x 4 s sulla griglia a 200 ms del loader, seme 20261026, preprocessing del training con la scala di sessione, encoder senza
maschera, latenti per istante in float16, testa, target e maschere di `PoseStore`; `--stage report`: E1 con bootstrap sui 15 utenti e regola
congelata, E2 appaiamento per utente e fase a >= 8 s con scarto delle coppie quasi identiche, E3 ridge con alpha scelto fra gli utenti del train)
e launcher `scripts/slurm/pose_eval_extract.sbatch` (1 A100, 40 min) e `pose_eval_report.sbatch` (seriale, 2 h). Test: `tests/cpu/test_pose_eval.py`
(6) e `test_pose_training.py` (6, con E4); il rapporto applica solo le regole scritte sopra. Due dettagli: la «finestra dinamica» usa la varianza
della posa standardizzata sulle patch valide, mediana calcolata su `val` nel rapporto stesso; la ridge e' con intercetta, su istanti campionati
(8 per finestra nel train), alpha fra 0,1 / 1 / 10 / 100 per errore medio su due meta' di utenti del train.
