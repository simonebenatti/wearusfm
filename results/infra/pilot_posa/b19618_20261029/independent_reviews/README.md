# Verifiche indipendenti B19618

I JSON conservano verifiche effettuate sulla raccolta completa dei60file in
`/private/tmp/wearusfm-b19618-results-20261029/`, senza nuovi fit o job remoti.
`recompute_pose.py` e `audit_integrity.py` sono gli script originali, con i percorsi della sessione.
Per rieseguirli servono la raccolta completa e il checkout indicati nelle prime righe; i quattro file
voluminosi omessi dall'archivio Git sono elencati in `../final/archive_selection.json`, con gli hash
nel manifest. Il programma `../summarize.py` genera invece sintesi/grafico dai report versionati.

Pose:1.049confronti da somme aggregate, bootstrap indipendente; nessun rifit ridge o nuova esecuzioneM2.
Transfer:delta/pesi e identita' verificati; IC originali, OOF non ricampionati localmente.
Integrity:198controlli, inclusi60hash e accounting senza doppio conteggio del preflight.

I riepiloghi warning originali dell'operatore restano immutati: transfer/integrity aggiungono la categoria
classi assenti in DB3 (489messaggi perbraccio), che il primo riepilogo non censiva.
