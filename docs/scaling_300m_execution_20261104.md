# Esecuzione del collaudo 30M/300M — 4 novembre 2026

Claude approva senza blocchi nel commit `befab4a`, review
`docs/reviews/claude_review_scaling_300m_preflight_20261104.md`. Simone trasmette
il verdetto e conferma che il collaudo può essere sottomesso così com'è.
L'approvazione include l'esecuzione; non serve un secondo consenso.

## Identità e risorse congelate

- Codice: `05d4bf0663481522a56d96c0b07847924cf5bd26`, già su origin.
- Deploy nuovo: `refs/heads/deploy/scaling_300m_preflight_20261104_05d4bf0`.
- Checkout detached nuovo:
  `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_preflight_20261104_05d4bf0`.
- Output nuovo, esterno al codice:
  `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/scaling_300m_20261104/preflight_05d4bf0`.
- Piano nel checkout: `results/infra/scaling_300m_20261104/preflight_plan.json`.
- Ledger unico: `~/wearusfm_local/jobs/`; unico submitter della linea: GPT in questa chat.

| ID logico | Risorse | Durata massima | Tetto |
|---|---|---|---|
| SCALING-300M-checkout-20261104 | 2 CPU, 4G, lrd_all_serial | 10 min | 0,333 core-ore |
| SCALING-300M-preflight-20261104 | 1 nodo, 4 A100, 32 CPU, 480G, boost_qos_dbg | 30 min | 2 GPU-ore + 16 core-ore allocate |

Sono esclusivamente checkout e collaudo: nessun checkpoint scientifico, training
a 5.000, estrazione o fit è incluso in questi due invii. Nessun monitor automatico.
Il codice, il launcher e il piano del collaudo rimangono identici a quelli rivisti.

## Sequenza operativa

1. Verificare assenza degli ID nel ledger; accesso remoto, ref corrente/configurazione
   e hook del repository di deploy; destinazioni nuove e directory dei log esistente.
   Una connessione muta non dimostra né assenza di job né coda vuota.
2. Pubblicare questo piano e le precisazioni del protocollo su origin e verificare
   lo SHA remoto. Pubblicare solo il nuovo ref deploy al codice completo indicato,
   dopo il controllo degli hook; nessun push o merge su main.
3. Sottomettere il checkout con `scripts/submit_job.py`, passando il launcher locale
   immutato su stdin, override espliciti 2 CPU/4G/10 min e SHA completo.
4. Leggere esito e log del checkout; richiedere `COMPLETED 0:0`, HEAD esatto,
   checkout pulito e messaggio di completamento. Non usare checkout parziali.
5. Sottomettere il collaudo con lo stesso submitter, `--dopo` l'ID logico del
   checkout (`afterok`), launcher dal checkout congelato, CODE_DIR/EXPECTED_SHA
   espliciti, piano e output sopra. Nessuna risottomissione se la risposta è ambigua:
   usare la riconciliazione del ledger prima di ogni altra azione.
6. Pubblicare ID, ricevute e stato osservato, informando il coordinamento. Alla fine
   raccogliere report/rank/log/risorse e verificare parità, completezza e gate.
   Nessun training scientifico parte automaticamente dal solo successo tecnico.

Gli argv concreti sono conservati in
`results/infra/scaling_300m_20261104/execution_plan.json` insieme agli hash degli
script e dei piani. Eventuali errori di rete, deploy o gate fermano la fase senza
retry di calcolo e senza cambiare il protocollo.

## Requisiti del seguito

Il protocollo aggiornato registra S1–S6 e precisa S7 della review: provenienza
del codice di training, conservazione del controllo 30M, margine di memoria del
percorso reale, ripiego a due microbatch con batch globale 32, controllo secondario
a dimensione pareggiata, limiti del LR fisso e bootstrap appaiato.
T25-L meno B19618 rimane contesto descrittivo, non una soglia di rumore tra run.
F1–F6 restano facoltativi differiti per preservare il codice già approvato; F7 è
precisato nel protocollo sui 112 update misurati per taglia.

La disponibilità di un piano pubblicato non equivale a una sottomissione: fanno
fede gli ID effettivi e le ricevute conservate successivamente.

## Stato operativo osservato alle 17:12 CET

Piano pubblicato nel commit `70cbd85`. **Nessuna sottomissione effettuata.**
Il master SSH risponde al controllo locale e accetta una sessione, ma non arriva
output da `hostname`; anche una connessione separata completa l'handshake senza
restituire il comando. Le sole sessioni di lettura avviate da questa chat sono
state interrotte. Nessun push deploy o invocazione del submitter, nessuna
prenotazione dei due ID nel ledger. Lo stato della coda remota non è noto.

Evidenza in `results/infra/scaling_300m_20261104/submission_status.json`.
Alla ripresa dell'accesso si procede dal piano approvato, senza un nuovo consenso.
Il codice non viene cambiato per questo problema di accesso e non si crea un monitor.
