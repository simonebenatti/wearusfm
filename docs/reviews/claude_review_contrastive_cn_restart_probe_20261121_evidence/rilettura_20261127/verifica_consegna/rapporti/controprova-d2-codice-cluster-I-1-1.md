# Controprova del rilievo I-1 (lente dei fatti)

Esito: il rilievo regge. Gravità corretta: importante.

Verifiche fatte:
- contrastive_cn_restart_compare.py: le righe 205-223 (branch_traces) richiedono che trace-{rank}.jsonl e numeric_profile.rank{rank}.json di reference abbiano lo stesso sha256 della copia in ARCHIVE (righe 18-19: reloads_execution_20261121/CN_A e benchmark_execution_20261121/CN). La riga 264 fa lo stesso per metrics.jsonl e la riga 356 per trace-{rank}.ddp.jsonl del continuo. Le righe 400-405: ogni ValueError/OSError diventa status INCONCLUSIVE, il report viene scritto e lo script esce con 0. Quindi il job finisce COMPLETED 0:0 dopo che le due gambe GPU sono già state eseguite.
- contrastive_cn_restart.py, commands() alle righe 105-137: controlla solo i pin fork, ordinary_reference e continuous_reference (checkpoint), più gli input del job. Il ramo --validate-only (righe 144-152) non chiama compare e non legge questi file.
- Sequenza, passi 3 e 5 (righe 87-143) e sezione «Che cosa non si può verificare prima» (righe 202-206): nessuna verifica di questi file. La sezione parla della gamba nuova che non riproduce CN_A, non dei file storici su WORK. Il brief non li cita (grep vuoto).
- stato_pre_lancio.sh e la sua uscita del 27/11: solo ls dei checkpoint. Il confronto confronto_corretto_veri legge solo checkpoint.pt. Nella review (parte nuova) non compaiono controlli su trace, metrics o profile.
- Le copie dell'archivio esistono nel commit di release (ls delle due cartelle: ci sono tutti i file citati).

Perché non è bloccante: le copie dell'archivio sono state raccolte da quei file il 21/11, quindi è probabile che siano identiche. Non ho potuto verificare che siano identiche senza accedere al cluster. Il controllo costa poco (sha256 di file piccoli sul nodo di login), mentre una differenza brucerebbe l'unico tentativo GPU. Per questo il rischio va trattato come importante.
