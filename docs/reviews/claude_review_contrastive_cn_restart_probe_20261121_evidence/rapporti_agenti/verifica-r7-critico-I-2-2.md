# Verifica avversaria I-2 (verifica-r7-critico-I-2-2)

Esito: il rilievo regge nei fatti. La gravita' corretta e' **minore**, non importante. Ho fatto solo letture del codice e nessuna esecuzione.

## Fatti verificati (dal codice, clone a 9d34dc69)
- `contrastive_cn_restart.py:133-136`: compare = processo separato con `--report root/report.json`. `:171-177`: un fallimento scrive `execution_failure.json` (INCONCLUSIVE).
- `contrastive_cn_restart_compare.py:308-320`: le eccezioni del confronto (ValueError, OSError, ...) vengono catturate, quindi `status=INCONCLUSIVE` e `write(args.report)`. `write` = `contrastive_target_resume.py:51-52`, `open("x")`: un nuovo confronto ha bisogno di un altro percorso.
- Il confronto legge solo file: `observe.py:144` con map_location cpu, `compare.py:224` costruisce WearUsFM su CPU, e non tocca CUDA, ambiente o SLURM. I dati grezzi sono scritti nelle gambe (`observe.py:87,123`, `leg.py:116,160`, trace/metrics/checkpoint del runner) sotto `plan.output` su WORK, che e' persistente. Nessun codice della consegna li cancella.
- Nell'ultima voce di `docs/decisioni.md` e nel protocollo (righe 15, 21, 23) non c'e' alcuna regola per rieseguire il solo confronto. Ci sono solo "un errore di esecuzione prima del confronto viene raccolto separatamente" e l'alternativa continuo+ripresa "richiederebbe un'altra decisione".

## Perche' minore
1. Il danno vero (GPU consumate ed esito INCONCLUSIVE) lo causano B-1 e B-2, gia' bloccanti. La loro correzione e' comunque una condizione della release. I-2 non aggiunge alcun controesempio che renda il run INCONCLUSIVE.
2. Quando manca la regola, l'esito non si perde e non viene attribuito male. Il report INCONCLUSIVE dichiara la causa (`reason` con il tipo di eccezione). I dati grezzi restano su WORK e il confronto e' una funzione pura di quei file, rieseguibile su lrd_all_serial senza GPU. Per recuperare serve una decisione registrata a posteriori (una correzione del lettore, criteri invariati). Non servono nuove GPU e la prova unica non va ripetuta.
3. Il rischio residuo riguarda la procedura: una correzione del lettore dopo la misura potrebbe essere contestata come post hoc. Le soglie restano congelate (LIMIT 1e-6, classify, criteri in decisioni.md), quindi la correzione non tocca i criteri.

## Correzione minima proposta
Aggiungere a decisioni.md (o al protocollo) una frase prima del lancio: se il report e' INCONCLUSIVE per un errore del solo confronto e le gambe hanno completato, si riesegue su CPU (lrd_all_serial) il confronto corretto e rivisto da Claude, sugli stessi dati grezzi autenticati, con criteri e soglie invariati e un nuovo percorso per il report. Si conserva anche il primo report. L'opzione `--compare-only` nel package e' facoltativa: cambierebbe le impronte del package e richiederebbe un nuovo giro di review, mentre la frase ha lo stesso effetto procedurale.

## Verifiche positive
- Su WORK il confine fra dati grezzi (gambe) e lettura (confronto) e' netto e persistente.
- Grazie all'apertura 'x' un report esistente non si sovrascrive, quindi il primo esito resta tracciato.
