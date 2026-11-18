# Review interna dell'adeguamento tempi FT — 04/11/2026

Revisore `ft_walltime_critic`, sola lettura del diff da5b3ac37; nessun file
modificato o job sottomesso. Nessun rilievo bloccante.

Ricetta identificativa invariata, hash2ba1b7f; piano, controller, ricevute e
launcher coerenti con4h/14100s frozen e7h/24900s FT/scratch. Il vecchio benchmark
non passa la guardia dei sorgenti; quello nuovo deve confermare policy, memoria
e tempi. Entrambi gli stati FT awaiting_claude_walltime_review e ricevuta nuova
null impediscono l'esecuzione anticipata. Continuazione17dd455 invariata.

Tetti verificati144GPUh/1152coreh per24job, ipotesi alternativa tutta joint72GPUh.
Rieseguiti44test tutti passati in4,83s, bash-n, diff-check e ricalcolo hash/costi/gate.
Autore: stessi44test passati in5,19s. Sono test mirati, non l'intera suite.

Limiti:7h è un'allocazione compatibile con la proiezione prudenziale, non una
garanzia di completamento; il nuovo benchmark è ancora necessario. Matrice
lunga e aggregatoreOOF/IC restano separati e incompleti. Questa review non
sostituisce la rilettura Claude richiesta da Simone.
