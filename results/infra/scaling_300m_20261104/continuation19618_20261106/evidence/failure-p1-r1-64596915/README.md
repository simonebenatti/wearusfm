# Fallimento del rilancio unico di p1, 64596915 (16/11/2026)

Job `SCALING-300M-eval19618-p1-r1-20261116` = 64596915 su lrdn0707: `FAILED 1:0` dopo 29 min 53 s (15:31:46-16:01:39), dallo
stesso checkout `df2de35` del job sostituito 64572000 (`../../../continuation19618_20261116_p1_relaunch/README.md`).
Il sottoprocesso d'estrazione ha scritto le tre cache ed e' uscito con codice 0 («estrazione finita in 1723 s»). Il job si e'
fermato nello stesso controllo d'identita' delle uscite del primo tentativo, ma su un altro file:
`ValueError: input changed while hashing: .../T100-G-300M/p1-r1/ninapro_db3.npz` (`slurm_log_64596915.txt`).
`ninapro_db2` ha superato controllo e verifica (provenance alle 16:01:36); `ninapro_db6` non e' stato controllato;
`receipt.json` manca. La cartella `p1-r1` resta com'e'. Letture del 16/11 alle 16:06 in `reading_20261116_1606.txt`:
`ls`, `p1.log`, `stat`, `lfs getstripe -c` delle cache a 19.618, `sacct`.

| Tentativo | Cache | Scritta | Controllata | Esito |
|---|---|---|---|---|
| 64572000 | `ninapro_db2` | 10:05:15 | 10:20:59-10:21:07 | superato |
| 64572000 | `ninapro_db3` | 10:08:33 | 10:21:07-10:21:08 | superato |
| 64572000 | `ninapro_db6` (2,19 GB) | 10:20:57-10:20:59 | 10:21:08-10:21:11 | **fallito** |
| 64596915 | `ninapro_db2` | 15:46:01-15:46:02 | 16:01:29-16:01:36 | superato |
| 64596915 | `ninapro_db3` (0,49 GB) | 15:49:19-15:49:20 | 16:01:36-16:01:39 | **fallito** |
| 64596915 | `ninapro_db6` (2,19 GB) | 16:01:28-16:01:29 | non raggiunto | — |

Correzione della prima lettura. Il README di `failure_64572000_diagnosis_20261116/` indicava una corsa fra lo scarico differito
del file appena scritto e il suo controllo. Il secondo caso non la conferma: `ninapro_db3` era stato scritto 12 min 16 s prima del
suo controllo. Quel README non si riscrive, perche' e' gia' fra i documenti del pacchetto; la correzione e' qui.

Fattore comune dei due fallimenti (osservazione): entrambi cadono 7-12 s dopo la fine dell'estrazione, cioe' dopo la scrittura
di `ninapro_db6`, 2,19 GB in 1-2 s. Il primo e' su db6 stesso, il secondo su db3 mentre quel volume appena scritto veniva scaricato.
Gli altri sette controlli a 19.618 sono passati, e non distinguono una finestra netta:
- tre cadevano nei 10 s dopo la scrittura di db6: i due di db2 (0-8 s e 0-7 s) e quello di db3 del primo tentativo (8-9 s);
- quattro, pose e p2, non avevano una scrittura cosi' grande e veloce a ridosso. In p2 `epn612` e' stato controllato ~25 s dopo
  la sua scrittura; in pose `val` era stato scritto lentamente in 3 min.
Il fallimento appare dunque probabilistico entro una decina di secondi dalla grande scrittura. Al 5.000 (05/11, prima delle
manutenzioni del 10-14/11) non e' fallito nessun controllo, nemmeno con tempi simili.

Lettura (interpretazione, non verificata): sul nodo che ha appena scritto molti dati, il client Lustre cambia per qualche secondo
gli attributi riportati da `stat` (mtime o ctime, granularita' di un secondo) anche di file vecchi della stessa cartella. Il
controllo `tokenizer_preflight.file_identity` confronta inode, dimensione, mtime e ctime prima e dopo lo sha256 e non distingue
questo da un cambiamento del contenuto. Non sappiamo quale campo sia cambiato: il messaggio non lo dice, e le date lette dopo il
job sono quelle della scrittura. Un rilancio identico ripete gli stessi tempi.

Conseguenze. La catena resta ferma, il rilancio unico approvato e' consumato e nessun altro rilancio parte. pose 64571999 e p2
64572001 restano validi (`COMPLETED 0:0` con ricevuta), non raccolti. `lrd_all_serial` e i lettori non scrivono file grandi, ma
`check_unchanged` dei lettori usa lo stesso confronto di attributi sulle cache: se l'instabilita' non dipende dalla scrittura,
riguarda anche loro. Al 5.000 i lettori non l'hanno mai incontrata.

Consumo:

| Voce | GPU-s | GPU-ore | core-ore |
|---|---|---|---|
| Rilancio 64596915 | 1.793 | 0,498056 | 3,984444 |
| Stadio, fin qui | 16.972 | 4,714444 | 37,715556 |
| Riga p1, fin qui | 3.675 | 1,020833 | 8,166667 |

Residuo dello stadio: 2,285556 GPU-ore su 7. Un altro tentativo di p1 al suo limite piu' il margine porterebbe lo stadio a
6,764444 GPU-ore su 7 e 54,115556 core-ore su 56. La riga p1 arriverebbe a 3,070833 GPU-ore sui suoi 2.
