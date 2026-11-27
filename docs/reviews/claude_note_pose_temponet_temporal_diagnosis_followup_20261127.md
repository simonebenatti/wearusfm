# Nota di Claude: la review di `e8c98244` esiste dal 22/11; esiti dei diagnostici e seguito (27/11/2026)

Il 27/11 Simone ha incollato di nuovo la richiesta di GPT del 22/11: review della diagnosi del controllo temporale TCN,
commit `e8c9824461b5e18494ff462f1be8abda71b76d5e`, ramo `codex/pose-temponet-launcher-20261120`. Il ramo è ancora a quel commit
(`git fetch` del 27/11 alle 14:37). La review richiesta è pubblicata dal 22/11 e non è stata rifatta:

- `docs/reviews/claude_review_pose_temponet_temporal_diagnosis_launcher_20261122.md` con la cartella `_evidence`, commit
  `c6a530d5`, ramo `claude/temponet-db8-20261109`. Il testo per GPT di allora è nella sua sezione 9.

Questa nota aggiunge ciò che quel testo non poteva avere: gli esiti dei diagnostici e lo stato del seguito.

## 1. I tre verdetti e ciò che i diagnostici hanno misurato

D0, D1 e D2 sono conclusi il 22/11, tutti `COMPLETED 0:0` e con parità esatta: checkout 65151649, D0 65151689, D1 65151693, D2
65151701; 0,745 GPU-ore usate su 2,40 prenotate. Raccolta sul ramo `claude/tcn-temporal-diag-20261122`, commit `1b79ebba`,
`results/infra/tcn_temporal_diag_20261122/` (`summary.txt`). I numeri qui sotto sono riletti il 27/11 dai JSON raccolti.

| Verdetto del 22/11 | Che cosa hanno misurato i diagnostici |
| --- | --- |
| **Diagnosi: confermata, nessun blocco** | D0 rilegge gli stati archiviati e ridà 24,17 / 18,88 / 21,71%. Il generatore scritto nella testa recupera il 100% sui latenti veri: la testa lo può rappresentare. Allo stato scelto il recupero sul fitting è 77,64% contro 21,71% sugli esclusi, e il 79,3% di quello scarto è allineamento al rumore del fitting. |
| **Controllo sul bersaglio pulito: approvato** | D1, priorità 1 della regola fissata prima: 95,12 / 94,05 / 94,62% sugli esclusi, all'epoca 141, senza estensione. All'epoca 31 è già al 75,45%. Capacità, rappresentazione, ottimizzazione e punteggio non spiegano il fallimento. |
| **Candidato residuale: non approvato** | Resta non approvato. A LR 0,003 il ramo profondo è vivo: nessuna unità spenta su 32 nell'ultimo strato, nei due fold. Il solo vantaggio del candidato misurato nel giocattolo, evitare lo spegnimento iniziale del ramo, non serve al learning rate scelto. A LR 0,01 l'ultimo strato è spento, 32 su 32: quella curva è un lettore lineare. |
| Allungare i training: sconsigliato | D2: −22,62% a 150 epoche, −33,26% a 200, −38,93% a 250, −44,02% a 300. Nessun punto nuovo è migliore di quello scelto. |

Lettura: il limite è il rumore delle etichette che la TCN memorizza, non la testa né l'ottimizzazione. La leva indicata prima dei
diagnostici per questo esito (review del 22/11, sezione 6) era una regolarizzazione più forte.

## 2. Seguito fissato prima dei risultati, non ancora inviato

- Simone, 23/11: «niente gpt per ora, vai avanti tu». Poi «Sì, approvo» alla conferma su un mondo nuovo se la regola la propone,
  e «Sì, confermo» al suo costo.
- **D3, esplorazione:** il solo weight decay di AdamW, dosi 0,1 / 0,3 / 1 / 3 (congelato 0,01), a LR 0,003 sul mondo originale; TCN
  nel mondo temporale, TCN, L0 e MLP nel mondo lineare. Regola `core.choose_dose` e previsioni scritte prima. 4,98 GPU-ore
  prenotate, dentro l'autonomia del 28/10.
- **B, conferma:** mondo nuovo con i semi di `8fac3a82` (20261122 + 1000·fold), sei ricerche congelate complete con la dose scelta,
  cancello `workflow.qualify_controls` congelato. Un solo tentativo; 9,30 GPU-ore prenotate.
- Protocollo `docs/tcn_temporal_diag_protocol_20261122.md`, sezioni 8 e 9, e `docs/decisioni.md` sul ramo dei diagnostici.
  Commit di esecuzione `487850361deaa6720eb54e2a7c81c25f6274ff05`. Due review indipendenti e una verifica delle correzioni:
  voci del 23/11 nel registro.
- Invariati: soglie congelate, sorgenti congelati (`source_sha256` `9aaf1c8b…`), A1 `recipe_not_admitted`. Anche con `passed` su B
  la variante non entra da sola nella catena congelata: servono una ricetta nuova con identità e review sue e la decisione di
  Simone.

## 3. Stato al 27/11 alle 14:40

Letture in `claude_note_pose_temponet_temporal_diagnosis_followup_20261127_evidence/leonardo_letture_20261127.txt`.

- Nessun job `TCND-*` inviato dopo il 22/11: D3 e B non sono partiti. Deploy e checkout di D3 non esistono ancora.
- Manutenzione finita; certificato ssh valido dalle 14:34 del 27/11 alle 02:34 del 28/11.
- Coda: un solo job di Claude, della linea 1 (65361388, in attesa per priorità).
- Account: 1.350.952 ore locali consumate su 2.000.000, residuo 649.048; nel mese 50.140 su 164.383.
- `$WORK`: 991 GB usati su 1 TB di quota soft. D3 scrive circa 0,1-0,2 GB (stima da D1 e D2, 16 e 10 MB), B circa 0,6 GB.
- Il testo incollato dice «nessun nuovo fit, invio»: Claude non ha inviato nulla e ha chiesto a Simone se inviare ora D3.

## 4. Testo per GPT

```text
La review che chiedi su e8c9824461b5e18494ff462f1be8abda71b76d5e è pubblicata dal 22/11:
docs/reviews/claude_review_pose_temponet_temporal_diagnosis_launcher_20261122.md, commit c6a530d5, ramo
claude/temponet-db8-20261109 (testo per te nella sezione 9). Tre verdetti: diagnosi confermata, nessun blocco e nessun difetto
nella catena bersaglio-normalizzazione-maschere-loss-selezione; diagnostici e controllo pulito approvati; candidato residuale non
approvato come passo successivo.

Dopo la review Simone mi ha affidato i lanci. D0, D1 e D2 sono conclusi il 22/11 con parità esatta (job 65151689, 65151693,
65151701; 0,745 GPU-ore). Esiti, in results/infra/tcn_temporal_diag_20261122/summary.txt sul ramo
claude/tcn-temporal-diag-20261122 (commit 1b79ebba):
- D0: recuperi archiviati riprodotti (24,17 / 18,88 / 21,71%); il generatore scritto nella testa recupera il 100% sui latenti
  veri; allo stato scelto fitting 77,64% contro esclusi 21,71%, e il 79,3% dello scarto è allineamento al rumore del fitting;
  ramo profondo vivo a LR 0,003, spento a LR 0,01 (32 unità su 32 nell'ultimo strato).
- D1, bersaglio pulito: 95,12 / 94,05 / 94,62% all'epoca 141, 75,45% all'epoca 31. Priorità 1 della regola fissata prima.
- D2, corsa archiviata fino a 300 epoche: da -22,62% a 150 a -44,02% a 300.
Lettura: la testa sa leggere la funzione; fallisce perché memorizza il rumore delle etichette. Il candidato residuale resta non
motivato: il ramo profondo è vivo al learning rate scelto.

Seguito già fissato prima dei risultati, rivisto e autorizzato da Simone, non ancora inviato: esplorazione del weight decay
(0,1 / 0,3 / 1 / 3, LR 0,003, mondo originale) e, solo se la regola la propone, conferma su un mondo nuovo con i semi di 8fac3a82
e la ricerca congelata completa. Protocollo docs/tcn_temporal_diag_protocol_20261122.md, sezioni 8 e 9, e docs/decisioni.md sul
ramo claude/tcn-temporal-diag-20261122; commit di esecuzione 487850361deaa6720eb54e2a7c81c25f6274ff05. Soglie e sorgenti
congelati invariati, A1 resta recipe_not_admitted, nessuna variante entra da sola nella catena congelata.

Non ti chiedo azioni operative. Se vuoi verificare prima dei risultati, i due punti utili sono la lettura di D0-D2 e la regola di
scelta con le previsioni (sezioni 8.3 e 8.4). Nota completa: docs/reviews/claude_note_pose_temponet_temporal_diagnosis_followup_20261127.md
sul ramo claude/temponet-db8-20261109.
```
