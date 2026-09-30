# Proposta — ancora RVQ ristretta: dove si accende e quali codici si predicono

**Stato: proposta di AG del 30/09/2026, FIRMATA da Simone il 30/09/2026 (le tre firme in fondo), senza modifiche. Regola CONGELATA col commit
che registra la firma; nessuna delle misure descritte era stata lanciata prima.** Non si ritocca dopo aver visto un risultato: se si rivela sbagliata
si apre una nuova decisione. Risponde alle due
domande aperte di D5b (`docs/decisioni.md`, «D5b — l'ancora RVQ NON si scarta»): (1) su quali dataset, (2) quali codici.
Decisione gia' presa da Simone il 30/09: l'ancora si **restringe** ai dataset sotto la soglia di V2.

## In breve

- **Dove:** accesa dove il tokenizer ricostruisce bene (V2 <= 2) o dove e' in-distribuzione; spenta dove V2 non passa; per i dataset **mai misurati**
  resta spenta finche' V2 non e' misurato.
- **Cosa si predice:** solo il **livello 0**. Quali dei 4 rami, lo decide una regola scritta ora, prima di misurare.
- **Come si giudica:** l'ablation run 4 contro run 5 (v10 §10.1). Le misure qui sotto decidono cosa entra nella run 5, non se l'ancora funziona.

## 1. Dove si accende

| Dataset | Stato | Perche' |
|---|---|---|
| CapgMyo, GRABMyo, Camargo | **acceso** | V2 misurato: 1,34 / 1,35 / 1,75 (soglia 2) |
| emg2pose, emg2qwerty | **acceso** | sono i dati su cui il tokenizer e' stato addestrato (V2 non si applica: e' il riferimento) |
| putEMG, CSL-hdemg | **spento** | V2 non passa: 2,24 e 2,16. Circa 27 h su 210 oggi ingerite |
| NinaPro DB5 | **spento** | 200 Hz: gia' deciso (mascherata, nessun upsampling) |
| Kaifosh, NinaPro DB2/3/4/6/7, Hyser | **spento fino a V2** | mai misurati. Si accende solo se V2 <= 2, con lo stesso riferimento (emg2pose multi-utente) e la stessa scala |

**Regola generale:** l'ancora e' accesa su un dataset solo se il dataset e' del tokenizer o se V2 e' misurato e <= 2. Vale anche per i dataset futuri
(Zhang, DB8, DB10, ...).
**Costo di misurare V2 sui 7 dataset nuovi** (stima di AG, non un impegno): 1 GPU, meno di 0,5 GPU-ora (< 4 ore locali) sui 30 del passo 1-bis, di cui
finora ne abbiamo usati meno di 1. Richiede il cluster. Aggiungere dataset al confronto V2 e' una modifica dell'insieme di D5a: la firma di questa
proposta la autorizza, la soglia X = 2 non cambia.

## 2. Quali codici si predicono

### Cosa sappiamo (dai dati del run 59048369; **gia' visti**, quindi non sono una prova indipendente)

Frazione di codici invariati al **livello 0** sotto il rumore di V4 (soglia 0,75, mai raggiunta), sui dataset accesi:

| Dataset | ramo 0 | ramo 1 | ramo 2 | ramo 3 |
|---|---|---|---|---|
| emg2pose | 0,46 | 0,31 | 0,66 | 0,66 |
| CapgMyo | 0,33 | 0,16 | 0,60 | 0,58 |
| Camargo | 0,36 | 0,19 | 0,54 | 0,54 |
| GRABMyo | 0,11 | 0,06 | 0,30 | 0,35 |

Ai livelli profondi la frazione scende quasi a zero (emg2pose, ramo 0: livello 15 cambia nel 100% dei casi). Identita' del dataset (6 classi,
incluse putEMG e CSL: da rifare sulle classi accese): 5 bande 0,655; **livello 0, 4 rami 0,881**; livelli 0-3 0,917; **livelli 8-15 0,988**; solo ramo 0
(16 livelli) 0,690; solo ramo 3 (16 livelli) 0,952.

### Proposta

1. **Solo il livello 0.** Motivi: in un RVQ il livello 0 e' la descrizione piu' grossolana (i livelli successivi raffinano il residuo); e' il piu'
   stabile; e l'identita' del dataset sta soprattutto nei livelli profondi.
2. **Quali rami: regola congelata prima di misurare.** Per ciascuno dei 4 rami si prende il solo livello 0 e si calcola, sulle classi accese
   (emg2pose, CapgMyo, GRABMyo, Camargo, poi le altre se si accendono), l'accuratezza della sonda dataset-ID (stessa sonda, stesso split per
   soggetto di V3) dagli istogrammi di quei codici, e la si confronta con quella delle 5 potenze di banda **sulle stesse classi**.
   **Un ramo e' idoneo se la differenza e' <= 10 punti (la stessa Y di D5a).** L'ancora predice il livello 0 di **tutti i rami idonei**, ciascuno con
   la propria testa. **Se nessun ramo e' idoneo**, l'ancora RVQ non parte e si torna da Simone (la strada gia' registrata come candidata e' il target
   discreto proprio, k-means sulle feature fisiche).
3. **La stabilita' non e' un criterio di selezione**, perche' nessun livello raggiunge il 75% e ogni soglia nuova sarebbe scelta guardando i
   numeri. Si **riporta** per ramo e per dataset e si accetta come rumore di etichetta, dichiarato. E' la deroga di D5b, non una novita'.

### Cosa e' pre-registrato e cosa no

L'identita' del solo livello 0 **per singolo ramo** non e' mai stata calcolata: quella parte della regola e' scritta prima di vederla. La stabilita' e'
gia' vista (tabella sopra). La regola vale sui dati del run; sui dataset nuovi (Kaifosh, NinaPro, Hyser), non ancora passati dal tokenizer, la si
**riapplica** come conferma.

## Rischi e cose non incluse

- **Etichette rumorose:** al livello 0 i codici invariati vanno dal 6% al 66%. GRABMyo e' il caso peggiore (0,06-0,35) pur passando V2.
- **Stime larghe:** la sonda dataset-ID ha pochi campioni (nel run a 6 classi l'intervallo al 95% era largo circa 6-7 punti su 84 campioni di test);
  con 4 classi sara' simile o piu' largo. La regola usa la stima puntuale, come V3; un ramo a ridosso della soglia va letto con questa cautela.
- **Non incluse, si possono misurare se vuoi:** un target piu' grossolano (raggruppare gli 8192 codici del livello 0 in poche centinaia, per
  fondere i codici vicini che si scambiano sotto rumore) e un target morbido (la distribuzione sui codici piu' vicini invece del codice unico).
  Sono idee di AG, non del piano ne' verificate in letteratura; il margine mediano fra codici e' 0,04, ma i rami con margine alto cambiano lo
  stesso (ramo 0, livello 0: 32%), quindi il guadagno e' incerto.
- **Sovrapposizione soggetti** con quelli visti dal tokenizer (v10 §6.3): emg2pose e emg2qwerty sono suoi dati di addestramento; il limite resta
  da dichiarare se non e' ricostruibile.

## Cosa si firma

- [x] **Dove:** tabella del §1 e regola «acceso solo se del tokenizer o V2 misurato <= 2».
- [x] **Cosa:** solo il livello 0.
- [x] **Quali rami:** la regola dell'idoneita' (differenza <= 10 punti sulla sonda dataset-ID, un ramo alla volta).

**Ordine dopo la firma** (tutto richiede il cluster per copiare gli array o per la GPU): (1) copiare `arrays_59048369`; (2) sonda per ramo sulle classi
accese (CPU, minuti); (3) V2 sui 7 dataset nuovi (GPU, < 0,5 GPU-ora), poi rifare (2) sulle classi che si accendono. Scadenza: D9, 25/10.
