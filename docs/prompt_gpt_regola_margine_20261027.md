# Prompt per GPT (responsabile scientifico dal 26/10/2026): verifica della regola di validita' della posa (margine M)

Contesto. Seconda ondata del pilot degli obiettivi (`docs/pilot_posa_v1.md`, firmato; emendamento 1 firmato). I bersagli di posa di emg2pose hanno
righe tutte a zero dove la cinematica inversa fallisce (17% delle righe nel collaudo, 12,7% nel paper); attorno a ogni blocco di zeri si scartano M
campioni per lato. La regola per scegliere M dal collaudo dell'ingest e' in `docs/decisioni.md` (voci del 26 e 27/10/2026 sulla seconda ondata, in
particolare «REGOLA DI LETTURA ... CONGELATA», «COLLAUDO ... LETTO ... NON DETERMINATO», «RETTIFICA», «SECONDA», «TERZA», «QUARTA» e
«QUINTA» e «SESTA RETTIFICA») e nel codice
(`src/wearusfm/ingest/emg2pose_pose.py`: `margin_from_collaudo`, `jump_profile_margin`, `edge_profile`; test in
`tests/cpu/test_ingest_emg2pose_pose.py`). Rapporto e lettura del primo collaudo (job 62364572, 40 registrazioni di 2 utenti):
`results/passo2/emg2pose_pose_ingest_collaudo_62364572.json`, `results/passo2/emg2pose_pose_margine_lettura_62364572.json`.

Che cosa e' successo. La regola v1 (congelata prima del rapporto: M = ultimo bin del profilo di |differenza prima| fuori dal 25% rispetto al
valore lontano > 200 ms) ha dato «non determinato»: il profilo e' piatto a ~1,78 volte il valore lontano da ~125 a 200 ms. E' stata scritta una
regola di riserva R1-R5, post hoc (informata da quel profilo), su proposta del revisore interno: persistenza a due lati sul profilo lungo (2 s),
soglia 1,10 sul plateau (M3 = 200 campioni provvisorio), profilo per altezza del salto verso lo zero (terzili; M5), M = max(M3, M5); se la
persistenza cade, la riserva non si applica e vale la v1 (se M > 400 ci si ferma).

Ti chiedo, come responsabile scientifico:
1. La lettura «plateau = comportamento della mano vicino a una perdita di tracking» e' sostenibile? Quale osservazione del profilo lungo la
   smentirebbe, e la regola R2 la cattura?
2. L'alternativa «occlusione parziale = cinematica inversa piu' rumorosa vicino alla perdita» cambia M? R5 (il rumore non scala col salto) basta a
   distinguerla?
3. La regola, come sta nel codice. **Calcolo:** la riserva R1-R5 si calcola solo se la v1 (congelata il 26/10) sul profilo lungo e' «non
   determinata» oppure da' M > 400 campioni. **Applicazione**, indipendente dallo stato della v1: (a) nessun bin contato (>= 200 campioni) del
   profilo lungo «rientra», dove rientrare e' a un lato solo, rapporto al valore lontano <= 1,25, anche sotto il lontano (chi rientra, anche
   senza riuscire, si ferma: un gradino verso il basso che chiude la contaminazione e' un rientro); (b) R2: TUTTI i bin contati oltre 400 campioni
   stanno fra 0,80 e 1,25 volte il plateau L (media dei bin 301-400), anche dopo 2.000; (c) bin contati almeno fino a 2.000. Se la v1 risulta
   determinata solo perche' gli ultimi bin non sono contati, vale il predicato sui bin contati (e quel M non e' un esito). Se la riserva non si
   applica ci si ferma e la proposta porta a Simone la v1 (se determinata su bin contati, con la perdita) e i valori informativi R3 e R5. R3: M3 = ultimo bin entro 300
   sopra 1,10 L (soglia PROPOSTA da Claude, tabella di sensibilita' in decisioni.md: 0,25 -> 140, 0,10 -> 200, 0,05 -> 240). R5: terzili
   dell'altezza del salto (almeno 9 blocchi con `n_h >= 10`), alto / basso > 1,25; se R5 non e' calcolabile M = M3, dichiarato. M = max(M3, M5).
   I confini oltre 400 sono tre (1,25 x lontano per il rientro, 0,80 L e 1,25 L per R2) e la partizione e' netta solo se L > 1,5625: con un
   plateau piu' basso le zone «rientro» e «persistenza» si sovrappongono e il codice da' la precedenza al rientro (SESTA RETTIFICA). La decisione
   riserva / fermarsi si capovolge a due confini: max(1,25 x lontano; 0,80 L) e 1,25 L (SETTIMA RETTIFICA). **Devi scegliere, prima della misura,**
   fra due opzioni che si comportano davvero in modo diverso quando L < 1,5625: (A) precedenza del rientro, com'e' ora: con L = 1,50 un livello a
   1,24 ferma (rientro) e a 1,26 da' la riserva; (B) precedenza della persistenza: oltre 400 un bin dentro [0,80 L; 1,25 L] non conta come rientro
   (soglia di rientro = min(1,25; 0,80 L)): con L = 1,50 sia 1,24 sia 1,26 danno la riserva, e il capovolgimento sta a 1,20. Con L = 1,78 le due
   opzioni coincidono. (Una terza formulazione, pavimento = max(0,80 L; 1,25 x lontano), e' solo una rietichettatura della (A): stesso esito su
   20.000 profili casuali, verificato dal revisore interno.) Le soglie ti sembrano difendibili? **I job del profilo lungo
   partono solo DOPO la tua risposta registrata e il via di Simone:** ogni modifica alle soglie va scritta ora, prima della misura.
4. Il campione del collaudo lungo: 40 registrazioni, al massimo 4 per utente, almeno 10 utenti, ESCLUSI i 2 utenti del primo collaudo
   (29ddab35d7, d387095792): basta cosi', o serve un secondo campione indipendente?
5. Il costo della scelta: tetto della perdita 2 M x (blocchi di zeri) / righe = 5,3% a M = 200, oltre al 17% gia' nullo. Accettabile per R3?

Rispondi nel registro condiviso (`docs/claude_GPT_shared`, sezione 1, voce tua) con: accordo o modifiche alle soglie, osservazione che smentirebbe
la lettura, e il tuo parere su M. Non e' richiesto nessun job: i due job CPU del profilo lungo partono solo dopo la tua risposta registrata e col via di Simone (ordine in
`decisioni.md`, «SECONDA RETTIFICA»).
