# Pubblicazione del primo checkout documentale

Controller rivisto e piani: 93486b66fa2e25f7dc69ee07c3acde1b43f890ce, su origin e SHA remoto verificato. Metadata plans: 60389a7157b0e7e7c2cd58c6fd9e0e387c5dd6dc, su origin e SHA remoto verificato; solo tre nuovi documenti rispetto a df2de35.

Gate checkout-plans superato alle 07:46:53 UTC: 9 controlli, 1 job seriale di altra linea, zero della linea scaling, WORK libero 90,55 GiB. Righe esatte affiancate. Il precedente verbale refused (07:45:42) documenta esclusivamente il DNS negato dal sandbox locale durante git ls-remote, prima di qualsiasi accesso riuscito o invio; la stessa lettura con accesso di rete ha superato il gate. Nessun retry di job e nessun cambiamento ai criteri.

Il trasporto git verso Leonardo usa, solo nel processo, GIT_SSH_COMMAND=ssh con ProxyCommand=none, ControlMaster=no, ControlPath=none, ConnectTimeout=15, coerentemente con le letture SSH e il submitter. Non si modifica la configurazione condivisa. Il riferimento deploy è nuovo e non aggiorna checkout attivi.
