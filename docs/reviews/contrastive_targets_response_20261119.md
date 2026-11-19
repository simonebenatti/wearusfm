# Risposta alla review 94f50942 e attuazione della decisione faa4a8d7

19/11/2026. **REVIEW_REQUIRED.** Base `438c8d65`. Nessun invio o accesso operativo a Leonardo.
La chiusura qui indica l'implementazione proposta; non equivale a una misura CUDA riuscita
né alla nuova approvazione di Claude. Prontuario: `docs/contrastive_targets_execution_20261119.md`.

| Punto | Risposta, prova e limite |
|---|---|
| R-1, parità | Implementata negli stati vivi di C0 prima degli update 128 e 256: due backward per ciascuno dei tre bracci, batch globale 32, profilo bf16/math/TF32 spento. `contrastive_target_parity.py` verifica identità dei gradienti, stato e supporto. Test con backward CPU reali e mutazioni delle evidenze; le 256 loss e norme delle traiettorie indipendenti restano descrittive. |
| R-1, ripresa | Il continuo CN/CR salva a 544 e prosegue a 576; A/B ripartono dagli stessi byte. `contrastive_target_resume.py` e il runner verificano tutte le tre coppie con l'1% storico, buffer e contatori esatti, limiti per tensore. Distanze da 544 separate. Test con trainer CPU reale e due processi nuovi. |
| R-2 | `contrastive_release.py` lega la review originale Git al commit completo rivisto, ai sorgenti e ai due piani. Dispatcher e runtime rifiutano review storiche o di un altro pacchetto. File di release separati evitano hash circolari. |
| R-3 | 17 allocazioni: 18 GPUh / 169,6 coreh richieste. Sei checkout da 4 CPU / 2 GiB / 240 s; il sesto separa CUDA dai full, che richiedono la sua ricevuta. Consumo effettivo storico 6,086667 GPUh / 57,827778 coreh distinto dai massimi. Residuo aritmetico, saldo corrente ancora da acquisire, margine SLURM e manutenzione dichiarati. |
| R-4 | `validate_audit_accounting` lega audit, report, sorgenti, ID e TSV sacct: COMPLETED, 0:0, durata massima 7.200 s, 8 CPU, limite richiesto 120 min. Controllo prima della calibrazione e nella lettura finale; test negativi di TIMEOUT, exit code, durata, pin, ID e duplicati. |
| M-1 | Test negativi isolano i due controlli `config_sha256` in `validate_audit` e `read_preflight`, più la ricevuta in `authenticate_calibration`. Il test storico richiede il messaggio della specifica barriera di identità. |
| N-1 | C0−T100-G è descrittivo: cambiano impalcatura, RMS e spettro rapido; il profilo numerico è dichiarato separatamente. Nessuna attribuzione causale alla sola impalcatura. |
| N-2 | Intestazioni nei tre prontuari storici rimandano al pacchetto del 19/11; il testo e i risultati storici restano conservati. |
| N-3 | Punto, supporto o profilo richiesto mancante lascia inconcluso il confronto. Il readout verifica le 28 ricevute numeriche dei sette run prima dei contrasti; profili differenti bloccano. Nessuna riscrittura dei diagnostici storici. |
| N-4 | Test proteggono il puntatore `PLAN`, il testo descrittivo del readout e i campi target contro `build_review_configs`. Le prove negative isolano le barriere di secondo livello. |
| N-5 | `contrastive_local_validation.py` registra comandi completi, ambiente, hash dei test e dei log, codici di uscita e sorgenti. Pinna i log originali del 18/11 senza dichiararli rieseguiti. La nuova selezione esplicita sostituisce quella storica di 121 test non documentata. |
| N-6 | Limite dichiarato: `validate_recipe` può completare `target_version` mancante con v3. I config nuovi sono completi e pinnati lungo la catena; nessuna modifica indiscriminata del parser generico. |
| N-7 | Limite dichiarato: `run_frozen` verifica `8905e346`, non il nuovo runtime. La ricevuta locale registra il confronto Git degli artefatti storici rispetto alla base. |

I test locali non qualificano CUDA/NCCL, il tempo, la memoria o la nuova calibrazione.
Questi esiti restano da misurare da 3-launcher dopo la review. I training lunghi non sono rilasciati.
