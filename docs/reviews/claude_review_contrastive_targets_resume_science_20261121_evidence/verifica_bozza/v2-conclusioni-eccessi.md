# V2 conclusioni ed eccessi - rapporto di lavoro
Fonti primarie lette: bozza; clone 918245b6 (scripts/contrastive_target_resume.py, training/run.py 700-830 e 1436-1500,
contrastive_objectives.py 415-439, pairing.py, decisioni.md 8276-8330, proposta faa4a8d7 riga 49, lettura e prompt GPT);
metrics.jsonl e trace-*.ddp.jsonl di CN/CR/CN_A/CN_B/CR_A; preflight_readout.json; leonardo/sola_lettura; output degli script.
Problemi principali: (1) limiti firmati in decisioni.md 8319 chiedono localizzazione per rank, tensore e gruppo DDP e
dichiarano l'ordine di riduzione "un'ipotesi da verificare": la bozza non lo cita e dice che gli artefatti bastano;
(2) norme fuori da DDP coprono 218/270 tensori (SHARED_PREFIXES), non solo "non il proiettore": mancano anche decoder (34),
anchor_heads (10), pose_head (4); sono norme di vettori gia' all_reduce, non per rank; (3) "solo" stati di processo fuori
dalle impronte e' troppo forte; (4) 2.4 conseguenza bf16 presentata come fatto (nel rapporto C e' [D]); (5) i quattro
tensori del proiettore sono un gruppo separato (salto 0,71% -> 1,03%; secondi momenti 0,050% -> 0,155%), non solo "coda";
(6) 3.2 "al 546 tutto identico" falso per i momenti; (7) intervento: identita' prova la sufficienza del riscaldamento,
non specificamente dell'ordine di riduzione; (8) "due fasi per braccio" non trovato nelle fonti.
