# Prompt per GPT (responsabile scientifico dal 26/10/2026): verifica della regola di validita' della posa (margine M)

Contesto. Seconda ondata del pilot degli obiettivi (`docs/pilot_posa_v1.md`, firmato; emendamento 1 firmato). I bersagli di posa di emg2pose hanno
righe tutte a zero dove la cinematica inversa fallisce (17% delle righe nel collaudo, 12,7% nel paper); attorno a ogni blocco di zeri si scartano M
campioni per lato. La regola per scegliere M dal collaudo dell'ingest e' in `docs/decisioni.md` (voci del 26 e 27/10/2026 sulla seconda ondata, in
particolare «REGOLA DI LETTURA ... CONGELATA», «COLLAUDO ... LETTO ... NON DETERMINATO» e la «RETTIFICA») e nel codice
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
3. Le soglie (0,80-1,25 per R2, 1,10 per R3, 1,25 per R5, 2.000 campioni contati, 9 blocchi per i terzili) ti sembrano difendibili? Se cambi una
   soglia, scrivila prima che il collaudo lungo giri.
4. Il campione: 40 registrazioni con al massimo 4 per utente (almeno 10 utenti) bastano? Serve un secondo campione indipendente?
5. Il costo della scelta: tetto della perdita 2 M x (blocchi di zeri) / righe = 5,3% a M = 200, oltre al 17% gia' nullo. Accettabile per R3?

Rispondi nel registro condiviso (`docs/claude_GPT_shared`, sezione 1, voce tua) con: accordo o modifiche alle soglie, osservazione che smentirebbe
la lettura, e il tuo parere su M. Non e' richiesto nessun job: i due job CPU del profilo lungo partono solo col via di Simone.
