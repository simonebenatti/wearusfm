# Controprova del rilievo D3-1

Esito: il rilievo regge. Gravita' corretta: importante.

Fatti controllati sulla sequenza (docs/contrastive_cn_restart_probe_launch_20261127.md):
- Righe 87-93: il passo 3 e' solo un elenco. L'unica forma di comando e' `lfs project -d "$WORK"` / `lfs quota -hp <ID> "$WORK"` (riga 90). Non c'e' `ssh leonardo` e non sono indicati gli apici singoli. Le altre forme ssh compaiono solo alle righe 121 e 158.
- Riga 91: "Conservare l'uscita" non dice dove. La sola `mkdir -p ~/wearusfm_local/cn_restart545_20261127` e' alla riga 120 (passo 5). Prima non ce n'e' nessuna: grep di mkdir/wearusfm_local da' solo le righe 72 (ledger) e 120-121.
- Per coda e cartelle assenti non c'e' nessun comando. Il revisore li aveva usati (stato_pre_lancio_20261127.out.txt: ls delle due cartelle, project ID 20147512, squeue, scontrol show reservation, limiti della QoS dbg).
- Il brief al punto 3 ripete l'elenco senza comandi.

Due attenuanti, che non fanno cadere il rilievo:
- I percorsi delle due cartelle sono scritti per esteso nella tabella dei parametri (righe 53-54).
- `lfs` non esiste sul Mac (`which lfs` restituisce "lfs not found"). Se il comando venisse eseguito localmente con $WORK espanso, fallirebbe in modo visibile, non silenzioso. Sono controlli in sola lettura, quindi un errore qui blocca il lancio o lo rallenta, ma non brucia un ID di job. Per questo la gravita' resta "importante" e non sale a "bloccante".

Margine attuale: 991G usati su 1T (circa 33 G liberi), ben sopra i 4 GB richiesti.
