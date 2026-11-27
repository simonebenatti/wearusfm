# Controprova R1 (lente dei fatti)

Esito: regge in parte, gravità corretta **minore**.

Regge:
- La frase di Simone è solo «Sì, firmo entrambi», alla domanda «Firmi la proposta faa4a8d7 e il tetto di 18 GPU-ore?» (claude_GPT_shared su origin/worktree-claude-training-engine; decisioni.md:8314 del worktree di release). Le 169,6 core-ore compaiono nella riga «Lettura mia» del registro e nella trascrizione in decisioni.md:8320, non nella frase di Simone.
- Il tetto riguardava «17 allocazioni di preflight» (decisioni.md:8320; contrastive_targets_execution_20261119.md:76-87). «Per la coorte» è una semplificazione.
- Il conto incrementale sul residuo (13,26 / 122,2678, poi 12,26 / 114,0011) è scritto da GPT (decisioni.md:8409; protocollo riga 39).

Non regge (parte confutata):
- La registrazione della firma in decisioni.md:8320 riporta esplicitamente «Risorse firmate ... 18 GPU-ore e 169,6 core-ore richieste». Dire che le 169,6 ore sono state firmate corrisponde quindi a quanto registrato, anche se non alla frase letterale.
- La proposta firmata stessa prevede di localizzare il primo scarto «per rank, tensore e gruppo DDP, a partire da 545–546» (decisioni.md:8318), cioè questa prova.
- Quando ha scelto «Prova mirata» (e0ecbd4b), Simone aveva davanti la frase «circa 4,74 GPU-ore consumate delle 18 firmate» e il costo della prova (0,47-0,59 GPU-ore attese, tetto 1). Il conto sul residuo gli era quindi già stato presentato.

Gravità: l'operatore non usa questa frase per decidere l'invio. Il via viene dall'indicazione del 27/11 e dalla scelta del 21/11, e il costo (1 GPU-ora) sta comunque sotto la soglia autonoma di 5 GPU-ore. Imprecisione di attribuzione senza effetto sul lancio: minore. Basta scrivere «tetto di 18 GPU-ore (169,6 ore locali nella registrazione del 20/11) per le 17 allocazioni del preflight».
