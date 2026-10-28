# Proposta di Claude: continuare il run di B fino al passo 19.618 e decidere li' (28/10/2026)

**Stato: PROPOSTA, da discutere con GPT e da firmare da Simone. Non autorizza nessun job.** Richiesta di Simone: «la tua idea o proposta
quale sarebbe?» e «strutturamela meglio». Tutto cio' che segue e' un'opinione tecnica con numeri; le scelte scientifiche restano di GPT,
le firme di Simone.

## 1. Da dove partiamo

| Evidenza | Valore | Fonte |
|---|---|---|
| B, pannello di train a 5.000 passi | MSE 1,160 -> 1,024 (rapporto 0,883), IC95 [0,839; 0,932], 23 utenti su 32 in calo; curva non satura fra 4.500 e 5.000 | `results/infra/pilot_posa/collaudo_b_20261028/verdetto.json` (GPT), lettura informale di Claude |
| E1 sui 15 utenti di val, finestre dinamiche | MAE testa / costante 0,938, IC95 [0,917; 0,952], mediana per utente 0,947, 15 su 15 sotto 1 | `B5000_student_E1E2_20261028.json` (GPT) |
| E2, bersagli allineati contro disallineati | 0,334 contro 0,352, gap +0,018, positivo in 14 utenti su 14 | idem |
| Ritmo e costo misurati | 0,708 s/passo mediano, 4 A100; B 4,24 GPU-ora | summary e sacct del job 62523672 |
| Budget della seconda ondata | circa 80 GPU-ora e 420 core-ore dopo B ed E1/E2 | decisioni.md |

Cosa dicono: la testa impara davvero e segue qualcosa del movimento (il vantaggio sparisce spostando i bersagli nel tempo), ma poco:
6% su utenti nuovi contro il 10% del criterio dell'endpoint. Cosa NON dicono: se la rappresentazione dell'EMG, che e' l'oggetto del
pilot, ne guadagna, resta indifferente o ne soffre. Quella domanda si misura solo col readout P1/P2 confrontato con un controllo.

## 2. La proposta in una frase

Riprendere il run di B dal suo `checkpoint.pt` (stesso config, stesso codice 24139ab) fino al passo **19.618**, cioe' il checkpoint che la
ricetta di R3 tiene gia' (`--keep-checkpoints 19618`, punto di partenza del ramo a 1 epoca), e li' fare due letture pre-registrate: la testa
(E1/E2 su val) e la rappresentazione (readout P1/P2 appaiato contro R3c allo stesso passo). Poi decidere se completare R3 o fermarsi.

Perche' 19.618 e non 49.046 subito: a 19.618 si e' speso circa un quarto del costo del tronco e, se si continua, nulla e' buttato (il run
E' il tronco di R3). Perche' non fermarsi a 5.000: la curva del pannello scende ancora e un solo punto a 5.000 non dice se il 6% diventa
10% o si ferma li'.

## 3. Letture pre-registrate a 19.618 (soglie PROPOSTE, da fissare con GPT prima del lancio)

### 3a. La testa impara ancora? (E1/E2 sullo student, val senza maschera, stesso codice e stessa val di oggi)

Attesa a due lati, scritta prima della misura:

| Esito | Definizione proposta | Lettura |
|---|---|---|
| In progresso | E1 sulle dinamiche <= 0,90 con limite superiore dell'IC95 < 0,917 (l'inferiore di oggi) E gap mediano di E2 > 0,018 (il valore di oggi) | la dose conta: proseguire e' sensato |
| Fermo | E1 sulle dinamiche >= 0,92 (dentro l'IC di oggi) | la testa ha trovato il suo livello; a 49.046 il 10% e' improbabile |
| Ambiguo | ogni altro caso | decisione di GPT e Simone, con i numeri |

E1/E2 restano descrittivi: le soglie firmate (0,9 e 0,95) valgono per l'endpoint a 49.046, non qui.

### 3b. La rappresentazione ne soffre? (readout P1 e P2, `pool`, R3 a 19.618 contro R3c a 19.618)

Confronto appaiato con la chiamata del pilot (`channel_readout.py --stage compare --single-seed --models`, soggetti ricampionati insieme,
1.000 ricampionamenti, seme 20261009), riferimento R3c, candidato R3. R3c e' il controllo ideale: stessa ricetta, stesso seme, stesso stream
di batch, senza il termine di posa (da verificare sul config di R3c che il loader non cambi il campionamento).

| Esito | Definizione proposta (punti di S) | Lettura |
|---|---|---|
| Danno | delta <= -1 con limite superiore dell'IC95 < 0 su P1 o su P2 | la posa toglie alla rappresentazione: fermarsi |
| Nessun danno rilevato | nessun compito nel caso «danno» | si puo' proseguire |
| Beneficio (informativo) | delta >= +1 con limite inferiore > 0 | non richiesto per proseguire; si registra |

Quattro letture (due compiti x `pool`; `dec` descrittivo se si vuole, a costo di sonde in piu'), un solo seme, senza correzione per confronti
multipli: come nel pilot, dichiarato.

### 3c. Tabella di decisione

| 3a \ 3b | Nessun danno | Danno |
|---|---|---|
| In progresso | continuare a 49.046 + ramo a 1 epoca (decisione separata, preventivo a parte) | fermarsi; R3 non passa per via della rappresentazione |
| Fermo | fermarsi; con GPT valutare se vale (C) peso piu' alto o chiudere la posa | fermarsi |
| Ambiguo | decisione di GPT e Simone | fermarsi |

Un esito «continuare» non e' un via: apre il preventivo del tronco restante (circa 26 GPU-ora) e del ramo (9 GPU-ora).

## 4. Costi (stime dai tempi misurati; i tetti sono proposte)

| Voce | Stima | Tetto proposto |
|---|---|---|
| Continuazione 5.000 -> 19.618: 14.618 passi x 0,708 s = 2 h 52 su 4 A100 + avvio, ripresa dello stream e 29 pannelli (~2 min) | 11,5 GPU-ora, 92 core-ore | **3 h 30 di SLURM (TIME_LIMIT_S 12.000): 14 GPU-ora, 112 core-ore** |
| E1/E2 a 19.618: estrazione della val (26 min su 1 A100) + rapporto (1 min, 8 core) | 0,45 GPU-ora, 4 core-ore | 0,67 GPU-ora, 8 core-ore |
| Readout: 4 estrazioni GPU (R3 e R3c a 19.618, P1 e P2) | 1,4 GPU-ora, 11 core-ore | 5 GPU-ora, 40 core-ore (come il preventivo di R3c) |
| Sonde `pool` P1 e P2 per i due checkpoint (4 job seriali, 8 core) | 22 core-ore | 64 core-ore |
| **Totale fino alla decisione** | **~13,4 GPU-ora, ~130 core-ore** | **~20 GPU-ora, ~225 core-ore** |

Residuo atteso dopo la decisione: circa 66 GPU-ora, abbastanza per il tronco restante e il ramo (circa 35 GPU-ora) se si prosegue.
Tempo di parete: circa 3 h di run piu' la coda, poi 1 h di estrazioni e circa 3 h di sonde seriali: una decisione entro un giorno.

## 5. Procedura tecnica (tutto esistente, nessun codice nuovo)

- **Ripresa**: stesso checkout di campagna `pilot-posa-b-20261028` (24139ab), `train_ddp.sbatch` sulla cartella di B
  `.../runs/pilot_posa/PILOT-POSA-R3-B-30M-s0-20261028` con quarto argomento `19618`, **senza** `REQUIRE_NEW_RUN` (la guardia serve a partire
  da zero; qui si riprende), stesso `EXTRA_ARGS` del comando di B (identita' del config: il trainer rifiuta un config diverso, e il pannello
  deve coincidere con `pose_panel_set.json`). Lo stream riprende dallo stato nel checkpoint; `metrics.jsonl` e `pose_panel.jsonl`
  continuano in coda ai file di B (il verdetto di B e' gia' al sicuro nelle copie di GPT). Il checkpoint `checkpoint_step0019618.pt` lo scrive
  la ricetta da sola; `max_steps 19618` chiude il job con `summary.json`.
- **Un solo job**, nessun job di scorta: un'interruzione si tratta come una decisione nuova, non con un rilancio automatico.
- **Letture**: `pose_eval.sbatch` con `--which student` (come oggi) sul checkpoint 19.618; `channel_readout_extract/probe` dal checkout
  usato per il pilot e per R3c, su `checkpoint_step0019618.pt` di R3 e sullo stesso file di R3c (esiste: da li' e' partito il suo ramo e1);
  confronto sul Mac con la chiamata del pilot. Prima di leggere: controllo di provenienza come nel pilot (SHA dei checkpoint, codice, OOF).
- **Chi fa cosa**: GPT scrive comandi e voce di criterio congelato, Claude li rivede; sottomissione con la procedura comune dopo la firma;
  lettura congiunta.

## 6. Cosa questa proposta NON cambia e NON autorizza

- Non cambia pesi, target, lag, M = 280, soglie di E1/E2 all'endpoint ne' il criterio §5 del pilot; non riapre il collaudo dbg.
- Non autorizza R3 completo, semi in piu', retry, E3 su R0, J2 bis.
- Il pannello di train e le letture a 19.618 sono descrittivi e intermedi: il criterio di R3 resta quello dell'endpoint a 49.046 piu' ramo.

## 7. Alternative, per confronto

| Alternativa | Costo fino alla risposta | Pro | Contro |
|---|---|---|---|
| **A. Questa proposta: fermata a 19.618** | ~13 GPU-ora | decisione a un quarto del costo; il run resta il tronco di R3; confronto pulito con R3c allo stesso passo | una fermata in piu' (circa un giorno) |
| B. Dritti a 49.046 + ramo e1 | ~45 GPU-ora + valutazione | nessuna fermata; risposta finale col criterio firmato | si spende tutto prima di sapere se la rappresentazione ne soffre |
| C. Fermarsi a 5.000 | 0 | budget salvo per altro | la domanda del pilot su R3 resta senza risposta; il 6% su utenti nuovi non basta a chiudere in nessun senso |

Raccomandazione di Claude: **A**. Se GPT preferisce B, l'unica cosa che chiederei di tenere e' il confronto appaiato con R3c a 19.618
(costa 1,4 GPU-ora e 22 core-ore in piu' e si fa mentre il tronco prosegue), perche' e' l'unico punto in cui R3 e R3c si confrontano allo
stesso passo e con gli stessi batch.
