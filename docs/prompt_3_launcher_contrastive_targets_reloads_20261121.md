# 3-launcher · quattro riprese contrastive RMS100/fast125

**Via di Simone per la catena concordata.** La regola su `origin/main` a `92d1fc0a` si applica a questa consegna. Il via già dato copre attese di slot, controlli, quattro riprese, raccolta e pubblicazione, quindi anche il checkout e la lettura CPU già elencati nel piano una volta pubblicato il relativo pacchetto con i sette riepiloghi verificati. Il passaggio delle ricevute alla chat scientifica serve a fissare i file esatti per il lettore. 3-launcher resta l'unico operatore per gli invii e consulta il ledger prima di ciascuno. Un eventuale esito negativo o inconclusivo va consegnato con il rapporto e il motivo. [Regola operativa](../CLAUDE.md) · [Piano delle celle](../results/infra/contrastive_targets_20261118/preflight_plan.json)

## Ripresa del controllo prima del checkout, 21/11

Il tentativo documentato a `086422e1` ha restituito 255 al client SSH dopo 92 verifiche di integrità positive e l’elenco dei moduli. Manca già la prima ricevuta JSON attesa, quella di `check-binding`. La riconnessione delle 12:12 CET e la riconciliazione confermano checkout e uscite assenti, cinque ID disponibili e nessuna allocazione. Il piano dei comandi si rigenera identico e la verifica del binding si riproduce sul Mac. La causa precisa dell’interruzione resta aperta.

3-launcher può ripetere **una volta il controllo completo in sola lettura** sullo SHA `12e215a978b9d30ea6d670f58af1a0f97387d477`. Conserva il log precedente. Registra a parte tempi, host, stderr e codice di uscita del nuovo controllo. Prima riconcilia ledger, ref, destinazioni e presenza di un eventuale processo di verifica precedente ancora attivo, e aggiorna lo snapshot operativo. Sul checkout già concluso dei benchmark controlla in una connessione breve il caricamento dei moduli, l’attivazione del venv e l’esecuzione di `check-binding` con lo stesso ambiente del prontuario. Conserva questa misura per separare un problema di connessione/import da un errore del validatore. Se il controllo breve riesce, esegui `before_checkout` integrale con registrazione separata delle informazioni di trasporto SSH disponibili. Sul controllo completo richiedi uscita zero, ricevute binding e audit e validazione del runner per **tutte e quattro** le celle. Solo allora usa gli invii già approvati qui sotto. Se il problema si ripete, conserva host, ultimo marcatore e codici di uscita e restituisci la diagnosi. I job e il budget sono quelli della consegna originaria.

Simone ha approvato quattro riprese da 544 a 576 dopo la lettura dei tre continui. I continui di riferimento sono CN/InfoNCE 65017310 e CR/Rank-N-Contrast 65020151. L’operatore è 3-launcher. È previsto un solo checkout comune, ID `CONTRAST-CHECKOUT-reloads-targets-v3-20261118`. [Lettura scientifica](reviews/contrastive_targets_three_benchmarks_science_20261121.md)

**SHA di esecuzione** `12e215a978b9d30ea6d670f58af1a0f97387d477`, ramo `origin/codex/contrastive-target-windows-20261118`. Verificati in locale i quattro manifest, il runner, il lettore e i criteri di ripresa (198 test superati). Gli argomenti di ogni invio sono in `reloads.<cella>.submit_reload`. [Piano dei comandi](../results/infra/contrastive_targets_20261118/reloads_handoff_20261121/submission_plan.json)

```bash
set -o pipefail
python3 results/infra/contrastive_targets_20261118/reloads_handoff_20261121/prepare_handoff.py --frozen-sha 12e215a978b9d30ea6d670f58af1a0f97387d477 --dry-run | cmp - results/infra/contrastive_targets_20261118/reloads_handoff_20261121/submission_plan.json
```

| Replica | ID logico | Origine a 544, SHA-256 |
|---|---|---|
| CN_A | `CONTRAST-RELOAD-CN-A-targets-v3-20261118` | `aa5952f0c4b7a8f07b5965b096877d568f3b5662bf08e6d8a4fc56f2a9e37ce6` |
| CN_B | `CONTRAST-RELOAD-CN-B-targets-v3-20261118` | Stesso checkpoint di CN_A |
| CR_A | `CONTRAST-RELOAD-CR-A-targets-v3-20261118` | `a9a5b1deb69c46c0238480500605f24fe9415c41f2a964d91d0cd2f7f5608e1a` |
| CR_B | `CONTRAST-RELOAD-CR-B-targets-v3-20261118` | Stesso checkpoint di CR_A |

## Invii

1. Confronta gli SHA di esecuzione e di consegna con il ramo pubblicato, rigenera il piano byte per byte e verifica il ref di deploy. Esegui le liste di argomenti del piano con il submitter e il ledger comuni. Un solo tentativo per ID. Conserva la ricevuta di ogni invio; in caso di invio ambiguo riconcilia prima il ledger.
2. Prima di **ogni invio** registra l’ora italiana, gli ID e impegni del progetto, il saldo effettivo in ore locali, le reservation, gli slot debug dell’utente già impegnati, la quota e le scritture su WORK delle altre linee. Riserva 12 GiB per la coorte in aggiunta alle scritture concorrenti, da aggiornare. Verifica P-1–P-6 con accounting, impronte e stato dei checkout storici. Il controllo congelato prima del checkout valida tutte le celle sul login; quello successivo valida la singola replica sul checkout effettivo. Usa `sacct` e lo SHA per dipendenze terminate.
3. Verifica il checkout dei continui 65017186 e i job C0 65017240, CN 65017310, CR 65020151, tutti `COMPLETED 0:0`. Autentica i rapporti, gli input e i checkpoint fisici dei continui su WORK. A e B devono partire dallo stesso file a 544 del proprio braccio, di 463.142.486 byte, usando il riepilogo del continuo arrivato a 576. Verifica a mano la corrispondenza fra ID, braccio, replica, directory di destinazione, release, config e argomenti del runner. Conserva l’output di `--validate-only`.
4. Sottometti il checkout con gli override **4 CPU, 2 GiB e 4 minuti**. Verifica la sua conclusione `COMPLETED 0:0`, lo SHA e la protezione del checkout prima di ogni invio GPU. Usa al massimo due celle debug contemporanee dell’utente, incluse le altre linee. Se sono disponibili due slot, parti con CN_A e CN_B, seguite da CR_A e CR_B secondo disponibilità. Con un solo slot procedi una alla volta. L’allocazione per replica è **4 GPU, 32 CPU, 240 GiB, 30 minuti** su `boost_qos_dbg`, con `TORCH_DIST_INIT_BARRIER=1` nell’export e limite interno di 1.770 s. Ogni replica scrive in una directory nuova. Le traiettorie continue e i checkpoint sorgente sono in sola lettura per queste prove.
5. Comunica gli ID appena assegnati. Un controllo fallito ferma il passo interessato, un errore condiviso sui prerequisiti ferma le celle che ne dipendono. Conserva log e artefatti parziali; ogni eventuale rilancio richiede la diagnosi e una decisione successiva.

## Risorse e calendario

| Checkout e quattro celle | Richiesto | Margine SLURM | Inviluppo |
|---|---:|---:|---:|
| GPU-ore | 8 | 1,6 | 9,6 |
| Core-ore | 64,266667 | 13,2 | 77,466667 |

L’effettivo della nuova coorte prima delle riprese è **3,747778 GPU-ore e 39,067778 core-ore**. Le richieste cumulate con questa consegna arrivano a **18 GPU-ore e 161,333333 core-ore** sul tetto firmato di 18 e 169,6. Le 8,266667 core-ore richieste residue corrispondono al checkout e alla lettura CPU previsti dopo le riprese. Calcola saldo e margine al billing corrente senza contare due volte impegni e consumi.

Con manutenzione il 23/11 alle 08:00 CET, l’ultimo avvio teorico del checkout per completare questa sola tranche è alle **06:38 CET con due slot**, alle **05:26 CET con uno** (10 minuti per il checkout, 36 minuti per ciascuna ondata inclusi i margini). Anticipa per coda e controlli. Ricalcola con le reservation effettive e il calendario del centro.

## Raccolta

Per ciascuna replica archivia ricevuta del submitter, accounting, log originali, `start.json`, `config.json`, `summary.json`, `metrics.jsonl`, `origin_sessions.json`, tutti i `numeric_profile.rank*.json`, le quattro tracce `trace-*.jsonl` e i rispettivi file `trace-*.ddp.jsonl`. Verifica SHA-256 e byte delle copie sull’originale remoto. Conserva su WORK i checkpoint finali di ogni replica e quelli sorgente; pubblica percorso, dimensioni e SHA di ognuno.

In raccolta verifica in particolare la copia iniziale byte-identica indicata in `start.json`, la fase A/B, i 32 record dal 545 al 576 per rank e lo stato finale a 576. Per ogni scarto registra braccio, coppia, passo, rank e campo disponibile. Distingui un errore di esecuzione, una traccia mancante e una differenza numerica. Conserva anche le ricevute se una prova termina prima del previsto. Pubblica commit, SHA remoto verificato, esiti SLURM, tempi, consumo effettivo e margine. Restituisci i riferimenti alla chat scientifica. Questa consegna termina con la raccolta delle quattro riprese. La decisione sul preflight richiede la successiva lettura CPU dei checkpoint.
