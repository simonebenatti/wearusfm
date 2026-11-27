# Controprova di R1 (attribuzione del tetto a Simone)

Esito: regge solo in parte. Gravità corretta: minore (non importante).

## Che cosa confuta il rilievo
- docs/decisioni.md:8315-8320 (worktree di release b58a66f6), voce registrata del 20/11 «Firma di Simone»: «La firma copre la proposta scientifica faa4a8d7 e i seguenti limiti del preflight», poi «Risorse firmate per 17 allocazioni di preflight: 18 GPU-ore e 169,6 core-ore richieste». Le 169,6 ore locali compaiono quindi come limite firmato nella decisione registrata, non solo nella «Lettura mia» del registro.
- Nella stessa voce, la proposta firmata dice: «Localizzare al primo passo, per rank, tensore e gruppo DDP, a partire da 545–546». È proprio lo scopo di questa prova. Il legame fra la firma e la prova quindi c'è.
- Conseguenze per l'operatore: nessuna. Il via viene dal messaggio di Simone del 27/11 (citato alla lettera in testa ai due testi) e dalla scelta «Prova mirata» del 21/11. Il job da 1,0 GPU-ora (1,4 con i margini) sta anche sotto la soglia autonoma delle 5 GPU-ore del 28/10. L'operatore non usa il residuo per decidere nulla: comandi, risorse e ID restano gli stessi.

## Che cosa regge
- La firma registrata copre «17 allocazioni di preflight». Questa prova aggiunge due allocazioni. Il calcolo che le pone sul residuo è di GPT (decisioni.md:8409, «Preventivo incrementale»), non di Simone. «Restano ... sul tetto firmato» e «per la coorte» dicono quindi un po' più di quanto Simone ha firmato.

## Correzione minima
Aggiungere una frase: le 2 allocazioni della prova sono contate da GPT sul residuo del tetto firmato per 17 allocazioni del preflight (decisioni.md, 21/11), e il via di questa prova è quello del 27/11.
