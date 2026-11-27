# Controprova D3-4 (raccolta incompleta)

Esito: il fatto regge in parte, la gravita' scende a **minore**.

## Che cosa ho verificato
- Sequenza, righe 169-191: l'elenco della raccolta non cita il log `campaign_checkout_<jobid>.out`, le uscite in
  `~/wearusfm_local/cn_restart545_20261127` (righe 120-121), la quota o gli state.json del submitter. Non da' nemmeno un
  comando di copia. Su questo il rilievo e' vero.
- Dove stanno i file: la cartella di uscita (riga 54) e i log (riga 56) sono su `/leonardo_work/.../wearusfm_runs/`, cioe'
  su WORK, che non viene cancellato a 40 giorni. Le uscite della validazione e gli state.json restano sul Mac, in
  `~/wearusfm_local`. Nulla di quello che manca si perde se non lo si copia subito.
- La raccolta viene dopo la fine del job: un errore qui non brucia nessun ID e non tocca il lancio. Le istruzioni contro
  gli errori seri ci sono gia': i `.pt` restano su WORK, e prima del commit vanno controllati dimensioni e `git status`
  (righe 188-189).

## Perche' minore
Seguendo il testo alla lettera, l'operatore raccoglie i file che servono alla lettura scientifica (report, ricevute,
log del job GPU, accounting). Quelli che mancano si possono recuperare in ogni momento, senza costi e senza rischi.
Manca il comando di copia, ma copiare file dal cluster e' un'operazione normale del progetto. Il doppio criterio
(«tutti i non .pt» piu' l'elenco) porta al massimo a copiare qualche file in piu' o in meno, non a un esito diverso.
La correzione proposta resta utile per avere un archivio completo, ma non e' necessaria per il lancio.
