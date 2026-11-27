# Controprova del rilievo I-2

Esito: il rilievo regge. Gravita' corretta: importante.

## Verifiche fatte
- Sequenza, righe 87-93: il passo 3 elenca i controlli a parole e non da' nessun comando ssh. Il solo frammento di comando e'
  `lfs project -d "$WORK"` / `lfs quota -hp <ID> "$WORK"` (riga 90), senza `ssh leonardo` e senza apici singoli.
- Riga 111: `git -C "$CODE_DIR" rev-parse HEAD`. `CODE_DIR` viene definita solo dopo, dentro l'heredoc del passo 5 (riga 125)
  e come valore di --export al passo 6 (riga 155). Al passo 4 non esiste ne' sul Mac ne' in una nuova sessione ssh.
- Riga 147 (passo 6): "rileggere la quota e verificare che la cartella di uscita manchi ancora", di nuovo senza comando.
- Brief, punto 3: solo l'elenco dei controlli, senza comandi.
- Sul Mac `env | grep -c '^WORK=\|^CODE_DIR='` restituisce 0. Con zsh, `print -r -- ssh leonardo git -C "$CODE_DIR" rev-parse HEAD`
  restituisce `ssh leonardo git -C  rev-parse HEAD`, e `lfs project -d "$WORK"` restituisce `lfs project -d `: argomento vuoto.

## Tentativo di confutazione
- I percorsi letterali ci sono: tabella alle righe 53-54 (CODE_DIR e cartella di uscita). Un operatore attento potrebbe ricostruire
  i comandi da solo. Questo attenua il rischio, ma non elimina il fatto: i testi non danno un comando pronto.
- Un controllo di assenza scritto con virgolette doppie (per esempio `test -e "$CODE_DIR"`) sul percorso vuoto risponderebbe
  "assente" anche se la cartella esistesse: e' un falso via libera. Il danno resta limitato perche' il driver rifiuta
  una cartella di uscita gia' esistente (riga 59). I controlli sono in sola lettura: un loro errore non consuma un ID,
  quindi il rilievo non e' bloccante.

## Correzione
Come quella proposta dal revisore: comandi pronti `ssh leonardo '...'` con percorsi letterali (/leonardo_work/IscrB_WearUsFM/...)
per i passi 3, 4 (rev-parse e status del checkout) e 6.
