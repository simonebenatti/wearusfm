# Cosa e' successo al pretraining: sintesi dopo il readout per canale (Claude, 24/10/2026)

Chiude il passo 1 della diagnosi del 23/10 (`docs/diagnosi_pretraining_20261023.md`; Simone: «finire P2, capire cosa e' successo, poi lavorare
sulle tue soluzioni»). Tutti i numeri sono S della validazione interna (P1: NinaPro DB2/DB3/DB6 cross-soggetto; P2: EPN-612 e UCI-EMG), in
punti percentuali, con la regola dei bracci congelata (delta >= +1 punto, limite inferiore dell'intervallo al 95% > 0, delta > 0 nei due semi).

## 1. Le tre domande del passo 1 e le risposte

| Domanda | Prova | Risposta |
|---|---|---|
| L'encoder batte le feature classiche? | Baseline classica (4 feature di Hudgins per canale, stessa sonda lineare, stessi fold) contro il readout medio di A | **Si'**: P1 +4,01 [+2,97; +5,03], P2 +13,86 [+11,00; +17,03], «migliora» in entrambi. *Interpretazione:* con lo stesso readout l'encoder contiene piu' informazione lineare delle 4 feature di Hudgins |
| La media sui canali nasconde informazione? (causa 3 della diagnosi) | Readout per canale (`local` e `dec` a 768 componenti) contro il readout medio, due semi, P1 e P2 | **No, con la regola congelata**: P1 local -2,00 «peggiora», dec -0,42 regola non superata; P2 local -2,49 «peggiora», dec +1,33 [-0,93; +3,54] regola non superata. Nessuno strato «migliora» in un compito |
| Il JEPA aggiunge qualcosa alle ancore che la media non mostra? | A contro ancore-only con lo stesso readout per canale, **solo P1** (le sonde P2 di ancore-only sono in coda) | **Sotto la soglia di +1 punto**: local +0,20 [+0,01; +0,39] (l'intervallo esclude lo zero: una differenza piccola c'e'), dec -0,78 [-1,12; -0,45]. «Ancore-only = A» regge anche per canale, nel senso della regola |

Controlli: `pool` riestratto = storico (0,18337, zero predizioni diverse); PCA a 768 spiega il 99,9-100% della varianza delle colonne per
canale; UCI-EMG (364 finestre) di fatto a larghezza piena.

## 2. Cosa resta in piedi della diagnosi

- **Causa 3 (readout) cade come causa principale.** Con la regola congelata, a livello di compito, il readout medio non sottostima l'encoder
  in modo misurabile con una sonda lineare (dentro un dataset si': EPN-612 con `dec`, +1,41 [+1,03; +1,83]). Resta un dettaglio utile: il decoder per canale guadagna 1,4 punti su EPN-612 (montaggio Myo uguale per tutti) e perde su DB2/DB6 (montaggi
  per soggetto): la media sui canali e' un'invarianza al posizionamento, utile fra montaggi diversi, superflua a montaggio condiviso. Indizio,
  non prova: si registra e basta.
- **Cause 1 e 2 (scala del target e target (b)) restano le candidate.** Sulla P1, con `local` per canale, l'encoder di A si legge 0,2 punti
  meglio di quello di sole ancore (intervallo [+0,01; +0,39]): una differenza c'e', ma e' un quinto della soglia; sul decoder la rete di sole
  ancore si legge meglio di 0,8 punti. *Interpretazione (mia):* il decoder di ancore-only e' il predittore delle ancore fisiche, quello di A
  porta anche la predizione dell'embedding del teacher; il termine JEPA, con target per patch da 25 ms letto dal decoder a query, aggiunge poca
  informazione lineare. E' coerente con F3-F5 della diagnosi (collasso senza ancore, rango che non cresce con la taglia, quota di contenuto che
  scende lungo il training). La P2 di ancore-only (in coda) completa il quadro.
- *Interpretazione (mia):* **il readout non e' la causa principale; la taglia non lo e' finche' l'obiettivo e' questo** (F4: rango uguale a
  30M, 100M e 300M; la C4 e' sospesa, non smentita); LR e stabilita' sono regolari (C1-C3); il contenuto spettrale del banco a 25 ms c'e'
  (F6, F7). Il collo di bottiglia piu' probabile e' nell'obiettivo. R5 non prova la risoluzione spettrale ma la finestra d'integrazione del
  kernel: un'altra cosa.

## 3. Conseguenze operative

1. **Readout per il pilot degli obiettivi:** criterio sul `pool` (il riferimento storico, confrontabile con tutto il passato); `dec` per canale
   a 768 accanto come descrittivo, perche' e' l'unico posto dove un cambiamento a montaggio condiviso si vedrebbe prima.
2. **Il pilot degli obiettivi e' il passo successivo giusto** (`docs/pilot_obiettivi_v1.md`): R1 target (a), R2 target per regione, R5 finestra
   150 ms, contro R0 = A rifatto a 2 epoche. R2 e R5 attaccano la causa 1 (unita' alla scala dell'inviluppo), R1 la causa 2.
3. **La C4 a 300M** misura un modello piu' grande dello stesso obiettivo: la diagnosi non e' cambiata, resta in attesa (decisione di Simone).
4. **Costi del passo 1:** 2,74 GPU-ora di estrazioni (tetto 10) e ~35 core-ore di sonde finora (21 sonde; le 8 descrittive P2 in coda).

## 4. Limiti di questa lettura

Validazione interna, due semi, sonda lineare: una lettura non lineare per canale (MLP) o un fine-tuning potrebbero mostrare differenze che la
sonda lineare non vede; non entrano nel criterio del pilot ma restano un'ablation possibile dopo. Quattro letture del confronto 1 senza
correzione (dichiarato nel protocollo). P1 cross-soggetto con DB3 (amputati) al caso in tutti i casi.
