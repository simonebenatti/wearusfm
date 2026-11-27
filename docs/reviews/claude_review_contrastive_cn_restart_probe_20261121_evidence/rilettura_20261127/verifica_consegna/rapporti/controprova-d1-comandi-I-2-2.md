# Controprova del rilievo I-2

Esito: il rilievo **regge**, gravità **importante** (confermata).

## Tentativo di confutazione
- Attenuante: la tabella «Parametri fissati dal codice» (sequenza, righe 53-54) dà i percorsi letterali di `CODE_DIR`
  e della cartella di uscita; il passo 5 mostra il modello corretto `ssh leonardo 'bash -s'`; il passo 6 usa il percorso letterale.
  Un operatore attento può ricostruire i comandi.
- Attenuante: i passi 3, 4 (verifica finale) e 6 (pre-controllo) sono letture; un loro errore non brucia un ID,
  e il brief (riga 26) dice di fermarsi su un errore inatteso.

## Perché non basta
- Sul Mac `WORK` e `CODE_DIR` sono vuote (verificato: `WORK=[] CODE_DIR=[]`). Le righe 90 e 111 scrivono `"$WORK"` e
  `"$CODE_DIR"` senza indicare dove vanno espanse; `CODE_DIR` è definita solo dentro l'heredoc del passo 5, dopo il passo 4.
  I passi 3 e 6 (righe 89, 92-93, 147) non danno alcun comando.
- La quota non è un controllo di forma: in `leonardo/stato_pre_lancio_20261127.out.txt` (righe 42-45) il progetto
  20147512 usa 991G su 1T, cioè circa 9 GB liberi contro la soglia di 4 GB. Un `ssh leonardo lfs project -d "$WORK"`
  scritto senza apici arriva come `lfs project -d` senza percorso, legge il progetto della cartella corrente (home) e
  porta a una quota sbagliata: informazione sbagliata proprio sul controllo che ha poco margine. Se invece il comando
  fallisce, l'operatore si ferma e il lancio si blocca senza motivo.
- Non lo porto a «bloccante»: seguendo i testi alla lettera nessun ID viene bruciato e nessuna regola viene violata.

## Correzione
Come proposto dal revisore: comandi pronti con apici singoli e percorsi letterali (per esempio
`ssh leonardo 'bash -s' < stato_pre_lancio.sh` per il passo 3 e la rilettura del passo 6, più
`ssh leonardo 'git -C /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121 rev-parse HEAD; git -C ... status --porcelain'`).
