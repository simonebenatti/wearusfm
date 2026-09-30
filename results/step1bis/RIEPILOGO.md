# Passo 1-bis — risultati del run vero di V1-V4 (job 59048369, 30/09/2026)

**Questi sono DATI, non un verdetto.** D5b (adottare, restringere o scartare l'ancora RVQ) resta una
decisione di Simone. Soglie e definizioni: `docs/decisioni.md`, «D5a» e «D5a - Definizioni operative»
(congelate il 29/09/2026 prima di ogni esecuzione).

File: `step1bis_59048369.json` (report completo, hash sha256 verificato fra cluster e Mac),
`emg2pose_reference_files.txt` (i 48 file di riferimento, 16 utenti x 3 registrazioni, seed 0).
Gli array (codici e token per dataset, ~103 MB) NON sono nel repo: stanno su Leonardo in
`$WORK/wearusfm_runs/results/step1bis/arrays_59048369/`.

## Provenienza e costo
- Checkpoint `NeuroRVQ_EMG_tokenizer_v1.pt`, sha256 `0d255bcc9f1c75ccc374cba06eab476f15bf5fb2a87115d6dc8d2dc0adadce49`;
  codice NeuroRVQ commit `926e770d9d16b6aa308404280fa0cc0211a6f9fb`.
- Codice wearusfm: il campo `provenance.wearusfm_commit` del JSON (`f9f5ab2`) e' letto a **fine** run; il codice
  caricato all'avvio era `db9a9ce` o `2404d68`, che differiscono solo per il report parziale, non per V1-V4.
- Durata 609 s su 1 GPU (~0,17 GPU-ora, ~1,4 ore locali). Con i due collaudi il passo 1-bis ha usato ben
  meno di 1 GPU-ora sui 30 di budget.
- Parametri: 5 dataset ingeriti piu' il riferimento emg2pose, 78 gruppi per dataset (= 19.968 token), 80 finestre
  per V1, al piu' 2 sessioni per soggetto, 3 semi di rumore, seed 0. Nessun dataset escluso.

## Scala di arrivo (calibrata solo su emg2pose)
Errore mediano di ricostruzione su emg2pose: varianza unitaria 0,0606; scala nativa di emg2pose 0,0352.
Vince la scala nativa, fattore **21,72** (mediana della deviazione standard di sessione di emg2pose).

## V1 — un canale alla volta: PASSA
Errore mediano multi-canale 0,0289; canale-per-volta 0,0355; rapporto **1,23** (soglia 1,5), su 20.480 token.

## V2 — errore di ricostruzione per dataset
Riferimento emg2pose (16 soggetti): errore mediano 0,0352. Soglia congelata: rapporto <= 2.

| Dataset | Soggetti | Errore mediano | Rapporto | Esito |
|---|---|---|---|---|
| CapgMyo | 18 | 0,0471 | 1,34 | passa |
| GRABMyo | 43 | 0,0476 | 1,35 | passa |
| Camargo 2021 | 22 | 0,0618 | 1,75 | passa |
| CSL-hdemg | 5 | 0,0761 | 2,16 | **non passa** |
| putEMG | 44 | 0,0790 | 2,24 | **non passa** |

## V3 — dataset-ID dai codici contro le 5 bande: NON PASSA
6 classi (5 dataset + emg2pose), 468 campioni da 256 token, split per soggetto, 84 campioni di test, caso 0,167.
- Codici: accuratezza bilanciata **0,964** (IC 95% 0,912-1,000).
- 5 potenze di banda: **0,655** (IC 95% 0,586-0,720).
- Differenza **0,310** (soglia congelata 0,10). Anche prendendo l'estremo sfavorevole degli intervalli
  (0,912 - 0,720 = 0,19) la differenza resta oltre soglia.

## V4 — stabilita' per livello RVQ: NON PASSA
Al rumore congelato (bianco, deviazione standard = noise floor del dataset) **nessun livello e' stabile**, nemmeno
il livello 0, su nessun dataset (soglia: >= 75% dei codici invariati in tutti e 4 i rami). Frazione di codici
invariati al livello 0, per ramo (0-3):

| Dataset | Rumore = soglia (congelato) | Rumore = 0,1 x soglia (controllo) |
|---|---|---|
| emg2pose | 0,46 / 0,31 / 0,66 / 0,66 | 0,89 / 0,83 / 0,94 / 0,93 |
| CapgMyo | 0,33 / 0,16 / 0,60 / 0,58 | 0,86 / 0,71 / 0,93 / 0,93 |
| GRABMyo | 0,11 / 0,06 / 0,30 / 0,35 | 0,78 / 0,66 / 0,88 / 0,89 |
| putEMG | 0,07 / 0,08 / 0,16 / 0,36 | 0,68 / 0,62 / 0,80 / 0,88 |
| CSL-hdemg | 0,43 / 0,37 / 0,55 / 0,66 | 0,89 / 0,84 / 0,93 / 0,94 |
| Camargo 2021 | 0,36 / 0,19 / 0,54 / 0,54 | 0,88 / 0,74 / 0,93 / 0,91 |

Controllo di determinismo: a rumore zero la frazione e' 1,0 su tutti i dataset. La frazione scende in modo
monotono con il livello (dai livelli profondi in su e' quasi zero) e sale col rumore piu' basso: il
comportamento e' coerente, non e' un difetto dell'esecuzione. Anche su emg2pose, che il tokenizer ha visto
in addestramento, i codici sono fragili.

## Nota sul caricatore (aggiunta successiva)
Il run e' stato eseguito con il caricatore delle sessioni di allora, che per i dataset salvati come UNA registrazione
continua concatenata (Camargo) trattava l'intera sessione come un solo segmento: il filtro passabanda attraversava le
giunture fra prove. Dal commit che introduce l'ingest di Hyser il caricatore segmenta per prova quando il sidecar ha
`trials` con `offset` e `n_samples` (come dice il testo congelato di D5a: "le prove si filtrano e ricampionano una per
una"). L'effetto sui numeri sopra non e' stato misurato; riguarda solo Camargo (le giunture sono poche rispetto ai
token) e non cambia le soglie.

## Lettura meccanica delle regole congelate (D5a)
V2: l'ancora si restringe ai dataset sotto soglia (CapgMyo, GRABMyo, Camargo, piu' emg2pose); V3: **scartata**;
V4: **scartata**. La firma di D5b e' di Simone.

## Analisi DESCRITTIVE, NON CONGELATE (non cambiano nessun verdetto)
Fatte dopo aver visto il risultato, sui soli array salvati (`scripts/step1bis_v3_extras.py`), stesso split e stessa
sonda di V3. La riproduzione dei due numeri congelati (0,655 e 0,964) coincide esattamente.

| Sonda (accuratezza bilanciata, 6 classi) | Valore | IC 95% |
|---|---|---|
| [congelata] 5 bande | 0,655 | 0,586-0,720 |
| [congelata] codici, tutti rami e livelli | 0,964 | 0,912-1,000 |
| spettro fine (76 bin da 5 Hz, 20-400 Hz) | 0,929 | 0,899-0,940 |
| spettro fine + statistiche di ampiezza | 0,905 | 0,875-0,917 |
| codici, solo livello 0 (4 rami) | 0,881 | 0,794-0,946 |
| codici, livelli 0-3 | 0,917 | 0,854-0,964 |
| codici, livelli 8-15 | 0,988 | 0,988-0,988 |
| codici, solo ramo 0 (>20 Hz) | 0,690 | 0,631-0,761 |
| codici, solo ramo 3 (>250 Hz) | 0,952 | 0,900-0,987 |

Letture, con le cautele del caso:
1. **L'esito di V3 dipende molto dalla forza della baseline.** Con uno spettro fine al posto delle 5 bande il
   divario scende a 0,035 (0,964 - 0,929). L'identita' del dataset e' quasi tutta gia' nello spettro (differenze
   di dispositivo e di elettrodi). Le bande spettrali sono pero' le ancore continue GIA' PREVISTE dal piano (v10 §6.3: RMS,
   forma spettrale, spaziali), accanto alle quali l'ancora RVQ doveva essere la terza: non sono un'alternativa all'RVQ,
   sono la base su cui l'RVQ si aggiungeva (correzione del 30/09/2026: prima era scritto «alternative»).
   Rafforzare la baseline dopo aver visto il risultato e' asimmetrico (puo' solo ridurre il divario): usarla per
   D5b richiede una nuova decisione registrata, non una lettura silenziosa.
2. L'identita' sta nei **livelli profondi** (8-15: 0,988) e nel **ramo alto in frequenza** (ramo 3: 0,952);
   il ramo 0 da solo e' vicino alla baseline (0,690). Coerente con l'ipotesi di v10 §6.3.
3. Campioni: 78 per dataset, 84 di test; CSL-hdemg ha solo 5 soggetti.
