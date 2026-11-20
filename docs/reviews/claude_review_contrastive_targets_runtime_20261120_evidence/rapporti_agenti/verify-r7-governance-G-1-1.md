# verify-r7-governance-G-1-1 (scettico n. 1 su G-1)

Avvio: 20/11. Clone 0efa4d45 (sola lettura). Obiettivo: confutare G-1 se possibile.

## Riproduzione documentale (fatti)
- faa4a8d7 science.md r.3 «Decisione proposta a Simone», r.52 «proposta da discutere prima degli invii»; decisioni faa4a8d7 «Proposta per Simone, ... training e invii sospesi»; registro «Preventivo e release restano da approvare». CONFERMATO.
- Timeline: 94f50942 11:51, faa4a8d7 11:57, fbfde797 13:52 (19/11). Il mandato «Ti affido l'implementazione dell'emendamento RMS100/fast125 e la preparazione del pacchetto per 3-launcher» (fbfde797 decisioni:8278) e' quindi SUCCESSIVO alla proposta: lettura possibile come via all'implementazione, ma non nomina profilo, cancelli o tetto.
- 438c8d65 preflight r.190-191 required_before_first_submission[0] = «publish and present resource proposal to Simone»; fbfde797 r.315-319 sostituito da review codice / ledger-quota 3-launcher / audit only. CONFERMATO.
- fbfde797 plan.json r.317-322 scientific_decisions APPROVED_BY_SIMONE (D1-D3 del 16/11) «Simone authorizes launches after convergence»; r.323-329 target_windows DIRECTED_BY_SIMONE, execution «Prepare and present new costs before any launch». CONFERMATO.
- fbfde797 preflight resources: cap 18/169,6; historical_reserved_cap 14/129,33; historical_unspent_not_new_authorization 7,91/71,51. CONFERMATO.
- Review Claude 94f50942 R-3 titolo «Preventivo da completare prima della presentazione a Simone». Anche il revisore presupponeva la presentazione a Simone.

## Riproduzione codice (fatti)
- contrastive_run.py@fbfde797:160-161 `scientific_decision_pin`/`budget_decision_pin` solo nel ramo training e solo controllo di verita' (stringa non vuota); invariato da 438c8d65:157 (preesistente). :168 preparazione = APPROVED_FOR_PREPARATION. CONFERMATO.
- contrastive_release.py:95-97 sei chiavi esatte e status APPROVED; :236 make_job scrive APPROVED_FOR_PREFLIGHT dal binding. CONFERMATO.
- contrastive_job.py: audit, cuda e readout senza file di release (solo job APPROVED_FOR_PREFLIGHT + binding + accounting audit); calibration/full/resume con release APPROVED_FOR_PREPARATION (nessun builder: file dell'operatore). CONFERMATO.
- preflight_plan.json@fbfde797: executable=false, submission_authorized=false (solo dichiarativi: nessun codice li legge, solo test_contrastive_targets_plan.py:28).
- Voce decisioni di faa4a8d7 e voce di registro «Preventivo e release restano da approvare»: assenti in fbfde797/0efa4d45 (grep 0); faa4a8d7 non antenato di fbfde797; copia della science doc identica (blob 5bdb0d15) e contiene ancora «proposta da discutere prima degli invii».
- Deposito oggetti: commit fino alle 14:04:46 del 19/11; nessuna approvazione successiva visibile (niente fetch per regola).

## Ricerca di motivi per confutare
1. Autorizzazione permanente 16/11 (plan.json:322 «Simone authorizes launches after convergence»): data nel contesto del collaudo con tetto 14 GPUh/129,33; il 17/11 «per questo giro», «Tetti invariati: 14». Non copre 18 GPUh nuovi: il piano stesso marca il residuo come historical_unspent_not_new_authorization.
2. Precedente diagnostici 18/11 (nessuna citazione numerica di Simone, «Simone richiede il pacchetto e il prompt per 3-launcher»): rientravano nel tetto 14 (review Claude 0afd0864: «cumulativo 11,2844 su 14 GPU-ore»). Qui il cumulativo e' 24,09 > 14: il precedente non si applica.
3. Precedenti dello stesso giorno (TCN 455b3220, scaling 57650126): GPT registra un «lancia»/«fallo partire» esplicito di Simone e scrive «Non si attribuisce a Simone una citazione numerica mai pronunciata». Per la linea 3 non c'e' un «lancia»: solo «Ti affido l'implementazione ... e la preparazione del pacchetto per 3-launcher» e «Gli invii e i controlli su Leonardo spettano a 3-launcher» (ruoli).
4. Il codice non ha mai verificato firme di Simone (anche il training: pin solo non vuoti). La governance e' procedurale: e' vero, ma proprio per questo la rimozione del passo procedurale dal preflight (438c8d65:191) e' l'unico presidio tolto. plan.json (legato dall'hash della review) conserva «Prepare and present new costs before any launch»: incoerenza fra i due piani legati.
5. Simone fa da tramite fra le chat (6283b5bd: «il contatto con Claude passa esclusivamente da lui»; 18/11 «Simone ha consegnato direttamente il prompt a 3-launcher»): mitigazione pratica, non una «decisione registrata» (CLAUDE.md:267-269: «non una conferma al volo»).
6. Regola delle 5 GPU-ore (CLAUDE.md:149-152): ogni cella e' 2 GPUh, ma «rispettando i budget della campagna» e «Non frazionare artificialmente»: full 6 GPUh e reload 8 GPUh superano 5 per fase. Non copre.

## Prove eseguite da me
- Sonda review_payload (scratch v1g1/probe_block.py, PYTHONPATH sulla copia): blocco base ACCETTATO; con "conditions", con "scope" o status "APPROVED_CONDITIONAL" RIFIUTATI («review is not an approved target-cohort package review»). Il blocco non puo' portare condizioni.
- plan.json a 438c8d65 e fbfde797: identici authorization «Simone authorizes launches after convergence» e execution «Prepare and present new costs before any launch»; a 438c8d65 il preflight aggiungeva «publish and present resource proposal to Simone», tolto a fbfde797 senza spiegazione (la risposta R-3 non nomina Simone; la risposta non contiene «Simone»: grep 0).
- Tutti i passi documentati dopo la review (CLAUDE.md r.3, prontuario r.133/148/159, prompt r.58, decisioni 8302) vanno direttamente alla release audit e poi alle fasi GPU; nessun passo di Simone.
- decisioni 16-19/11: ogni rilascio della linea 3 «entro 14 / 129,333333»; nessuna approvazione oltre 14 GPUh; 18/11 «Il preflight proposto costa al massimo14GPUh ... non è rilasciato».

## Esito
NON CONFUTATO. Fatti e righe citate tutti riprodotti. Imprecisioni minori del rilievo, che non lo indeboliscono:
- CLAUDE.md:183 («nessun job senza il via di Simone», 26/10) e' in parte superata dalle regole del 28/10 (autonomia <5 GPUh, test approvati), che pero' non coprono 18 GPUh («rispettando i budget della campagna», «Non frazionare artificialmente»; full 6 e reload 8 GPUh per fase).
- La modalita' CUDA non richiede nemmeno un file di release (solo job da make_job + binding): l'esposizione e' maggiore di quanto scritto.
- Anche per il training i pin di decisione/budget sono solo stringhe non vuote (preesistente): la governance e' procedurale per costruzione; proprio per questo conta il passo procedurale tolto.
Gravita': BLOCCANTE per il rilascio (non per il codice): con il blocco, la procedura documentata porta a spendere fino a 18 GPUh (cumulato 24,09 contro tetto approvato 14, residuo 7,91 dichiarato non-autorizzazione) senza decisione registrata (CLAUDE.md:267-269). Rimedio senza nuovo codice: decisione di Simone in decisioni.md (non e' nei file legati dagli hash) e passo ripristinato nel prontuario/prompt per 3-launcher; se si ripristina in preflight_plan.json cambia l'hash e serve un nuovo delivery_pin.
Pulizia: rimossi v1g1/ e v1g1_commits.txt; copia git pulita (git status vuoto).
