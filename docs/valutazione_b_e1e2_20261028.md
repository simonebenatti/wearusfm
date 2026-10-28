# B a 5.000: valutazione diagnostica E1/E2

28/10/2026, GPT. **AUTORIZZATA da Simone**: dopo la proposta di valutare il checkpoint B sui 15 utenti esclusi dal training,
con E1/E2 e senza altro addestramento, Simone risponde «vai con la tua proposta».
Il via copre i due job indicati sotto, entro il preventivo gia' presentato. Nessun nuovo consenso richiesto per eseguirli.

## Obiettivo e configurazione congelata prima della lettura

Checkpoint B: `PILOT-POSA-R3-B-30M-s0-20261028/checkpoint.pt`, **student dopo 5.000 aggiornamenti**.
B e' gia' superato; questa lettura non ne cambia il verdetto. Il candidato viene valutato una sola volta, senza selezionare
checkpoint, iperparametri o architetture in funzione di `val`. R3 completo resta una decisione distinta.

- Codice scientifico e launcher riusati dal checkout `pilot-posa-b-20261028`, SHA **24139ab901c052357a46a65bc20466f2eb1e8358**.
- Estrazione con `--which student --split val --seed 20261026 --n-windows 10 --window-s 4 --batch 32`, senza limiti utenti/registrazioni.
- `val` ufficiale: 1.950 registrazioni / 15 utenti, disgiunti dagli utenti di emg2pose del manifest train.
  Dieci finestre da 4s e' il massimo per registrazione; il conteggio ottenuto dipende dall'ammissibilita'.
- Encoder senza maschera, come E1/E2. Questa vista differisce dal pannello B mascherato; anche metrica e pesi cambiano:
  MAE in radianti pesata per elementi validi contro MSE standardizzata a quote uguali per utente nel pannello B.
- Rapporto senza `--train`: **E1 ed E2**, nessuna ridge E3, nessuna estrazione train.

## Lettura predefinita

Si conservano formule, soglie e bootstrap dell'[emendamento E1/E2](emendamento_pilot_posa_v1.md), senza modifiche.
E1: MAE testa/costante sulle finestre dinamiche, limite superiore IC95 <0,9 e mediana dei rapporti utente <0,9.
E2: rapporto disallineato/costante, limite inferiore IC95 >0,95; gap allineato/disallineato sullo stesso supporto.
Bootstrap per utente, 1.000 ricampionamenti, seme 20261009. Riportare rapporti, intervalli e valori per utente;
i booleani del software sono descrittivi **all'endpoint anticipato**, non il verdetto scientifico finale a 49.046 passi.

Il controllo E2 mantiene utente, fase e lato, distanza di almeno 8s nella stessa registrazione o fallback a un'altra sessione;
interseca le maschere e scarta le coppie quasi identiche secondo la regola esistente. Riportare utenti/finestre/coppie effettivi,
esclusioni e incompatibilita' delle statistiche: non assumere che E2 copra automaticamente tutti i 15 utenti.

Limiti fissati prima dei risultati: un limite inferiore >0,95 non prova letteralmente che il disallineato non batta la costante;
un E2 negativo non prova automaticamente contaminazione o assenza di movimento corretto. Leggere insieme rapporto, gap e copertura.
Il riuso dei partner, il fallback e lo scarto delle coppie non mantengono esattamente le distribuzioni marginali: il gap non e'
una stima causale pura. Un esito debole a 5.000 non dimostra il fallimento del training completo.

## Provenienza e controlli di esecuzione

La review tecnica ha confermato che lo script frozen carica correttamente la testa student, ma registra senza confrontarli
gli hash delle statistiche del checkpoint e del pose store; non registra inoltre step e impronta del checkpoint.
Un wrapper **versionato, inviato su stdin tramite `submit_job.py`**, esegue le verifiche nel compute node prima di chiamare
i launcher congelati. Non modifica il checkout, non sostituisce le funzioni scientifiche e non richiede un nuovo deployment CPU.
Il submitter conserva anche lo SHA256 del testo inviato.

Prima dell'estrazione: provenienza B, checkpoint 5.000 con testa student, config effettiva B, dimensione e SHA256 checkpoint,
statistiche attese `757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e`, hash manifest train,
CSV e registrazioni presenti, disgiunzione utenti normalizzata, destinazione nuova. Le guardie rifiutano incompatibilita'.
Prima del rapporto: provenienza dell'estrazione, immutabilita' checkpoint, identita' della cache, student/val/testa,
hash delle statistiche concordi, 15 utenti con supporto valido; registrare copertura delle registrazioni e SHA256 cache.
La cache e i checkpoint restano esterni al repository. Il JSON finale contiene E2 anche se il riepilogo stdout lo omette.

## Risorse e ordine

| Job | Risorse | Tetto | Costo massimo |
|---|---|---|---|
| Estrazione B student val | 1 A100, 8 core, 60G, boost_usr_prod / normal | 40min | 0,6667 GPU-ore + 5,3333 core-ore |
| Rapporto E1/E2 | 8 core, 30G, lrd_all_serial | 20min | 0 GPU-ore + 2,6667 core-ore |
| **Totale autorizzato** | Nessun training o checkout aggiuntivo | | **0,6667 GPU-ore + 8 core-ore** |

L'estrazione val precedente e' durata 25min11s; la stima per questa e' 25–30min. Il precedente rapporto senza testa e' durato 60s,
ma non misura il costo di E2: 20min e' un tetto, non una durata garantita. I controlli di provenienza sono compresi nelle risorse.
Il rapporto dipende da `afterok` dell'estrazione. Nessun retry automatico, cambio di risorse o allentamento delle guardie.
In caso di errore si leggono stato/log e si registra la causa; nessun nuovo lancio implicito.

Piano e comandi: `results/infra/pilot_posa/piano_b_e1e2_20261028.json` e `comandi_b_e1e2_20261028.txt`.
Due review indipendenti prima del lancio: scientifica favorevole con i limiti sopra, tecnica favorevole con le guardie esterne;
8 test CPU dell'harness E1/E2 passati. Il wrapper viene verificato separatamente prima della sottomissione.
La consegna e gli ID effettivi sono pubblicati sul ramo `codex/r3-long-collaudo-20261028` e nel registro condiviso.
