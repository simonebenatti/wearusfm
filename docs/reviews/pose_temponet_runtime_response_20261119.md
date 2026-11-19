# Risposta alla review operativa TCN 527f8da

19/11/2026. Base rivista: `2f982114719dc2abd723bdbf0147d7fef34a731c`.
Review originale: `527f8dae716398203f8f2d7413d93494f020e185`, copiata senza modifiche.
**REVIEW_REQUIRED**, nessun invio o accesso a Leonardo. Operatore: **2-TCN head launcher**.
Il protocollo scientifico v3, la proposta JSON, le teste, i target, la loss e le soglie
restano invariati. Questa consegna corregge il runtime prima di qualsiasi ricevuta reale.

| Rilievo | Correzione e verifica |
|---|---|
| B-1 | Il launcher antepone src/scripts a PYTHONPATH mantenendo i moduli del cluster. Test del contratto del launcher. |
| B-2 | Arrotondamento al minuto prima di costi e limiti; validate_graph, apply_training_pack e submit_command rifiutano secondi non interi di minuto. Test dal tempo misurato fino a runner._allocation. |
| B-3 | Submitter con `--agente gpt`, verificato contro le scelte del parser effettivo. |
| B-4 | `_progress` legge i punti intermedi, senza validare una curva completa. Log degli hash iniziali al passo zero e prima dei rifit; regressione con callback reale e catena completa. |
| I-1 | `GateNotPassed` conserva stato e gate; il gate combinato è un artefatto, distinto da un errore tecnico. Stati non calibrato/non diagnostico preservati, anche per teacher degenere. Controllo combinato prima di emettere i job GPU e di nuovo nel runner. |
| I-2 | Il benchmark usa `workflow._score_model` e il percorso di input/cast/crop della produzione. L'inferenza include ogni finestra nell'ordine fisico. |
| I-3 | Identità e snapshot riscaldati prima dei dieci campioni. Snapshot/restore in memoria separati da scrittura/lettura atomica degli artefatti; i costi su disco riguardano gli artefatti realmente prodotti, non ogni epoca. |
| I-4 | Per la quota di margine temporale, WLS L0/L13 fittate su y* e valutate su y. Statistiche, residui e nomi dei bersagli registrati; prova che il margine L0 sia zero. |
| I-5 | SHA del piano di 40 esadecimali richiesto prima dei comandi; wrapper verifica commit, percorso e hash prima di creare il checkout. Nessun riuso automatico di checkout esistenti. |
| I-6 | `resource-plan --operator-checks` richiede una misura successiva ai timestamp UTC dei due benchmark autenticati. Nuovo pin della misura nel piano prima del gate. |
| L1-2 | Hash delle inizializzazioni emessi prima degli update, nei controlli/selezioni e nei rifit. |
| L3-4 | Limite operativo documentato: verificare i modelli CPU dei due nodi seriali prima del rilascio; nessun nodo letto qui e nessun campo numerico eliminato. |
| L3-6 | La review di release deve nominare il commit completo del codice o il source_sha256. |
| L3-7 | Gli stadi successivi passano da un piccolo dispatcher che verifica EXPECTED_SHA ed esegue il launcher congelato nel checkout. |
| L3-8 | Documentato: nessun rinfresco SCRATCH durante uno stadio; fra stadi, contenuti invariati, è compatibile. |
| L3-9 | Limite documentato: ID/checkout fissi, ogni tentativo aggiuntivo richiede decisione e nuova consegna; nessun retry implicito introdotto. |
| L4-4 | Proiezione del report con I/O, due valutazioni R-1 e valutazione a batch 32. |
| L4-5 | Misurato il primo caricamento applicativo del runner. La page cache del sistema operativo non è controllata: non viene dichiarata una misura di disco freddo. |
| L5-2 | Soglia necessaria `E_dyn <= 0.7630564046643697` esplicita nella candidatura; valore, soglia e risultato salvati. Test anche sul double immediatamente sopra soglia. |
| L5-3 | Seconda parità R-1 salvata nel rapporto, nella ricevuta e in un JSON separato. |
| L5-5 | Test dei 24 stadi attraverso run_stage, con controlli, selezioni e rifit reali, poi inferenza, metriche e raccolta. Limiti della fixture sotto. |
| L5-4 | Nessuna modifica: accolto il rilievo già confutato dalla review sui double esterni alla banda inclusiva. |

## Portata del test della catena

`tests/cpu/test_pose_temponet_e2e_chain.py` usa il vero runner e le vere funzioni dei
sei controlli, cinque selezioni e quindici rifit, senza sostituire run_selection o
run_refits. WLS, calibrazioni e gate dei riferimenti sono realmente eseguiti su una
fixture a rango basso, con geometria 160×384, 40 utenti train e 15 utenti finali.

Sono simulati: allocazione SLURM, censimento numerico, CUDA tramite CPU, preflight,
benchmark, cache e riferimenti storici. La ricetta privata del test dura 1→2 epoche.
Il vero qualify_controls viene eseguito: il suo esito viene poi forzato solo per
consentire l'attraversamento della catena breve, con `fixture_forced_admission=True`
e `fixture_unforced_status` conservato. La parità R-1 usa valori/conteggi della fixture;
questo test non qualifica i dati veri. Nessuna conclusione scientifica viene ricavata.

La ricevuta locale finale conserva comandi, log, hash e l'eventuale fallimento storico
preesistente. Nessuna prova locale certifica ambiente, tempi, memoria o determinismo
A100: restano gate del benchmark dopo la nuova review favorevole.

Verifica finale: **263 test TCN passati in 157,84 s**, inclusa la catena dei 24 stadi;
**125 test storici passati e un fallimento preesistente**, identico alla baseline già
registrata e alla review Claude. Nessuna nuova regressione. Tre launcher con `bash -n`
valido; proposta scientifica, grafo dei 24 job e sorgenti FM/TEMPONet raw invariati.
Ricevuta e comandi riproducibili in `revision_runtime/validation/local_validation.json`.
