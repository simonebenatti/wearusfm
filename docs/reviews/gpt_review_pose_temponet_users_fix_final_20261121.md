# TCN — conferma finale della rilettura GPT — 21/11/2026

**APPROVATO. Nessun blocco residuo di review per il nuovo preflight.**
La conferma finale di Claude è stata trasmessa da Simone il 21/11: quattro
revisori su quattro conclusi, nessun blocco, nessuna modifica dei sorgenti.
È quindi chiusa l'attesa riportata nella review GPT del 20/11 (`e5e8be6c`).
Il nuovo tentativo è già autorizzato dalla decisione L3-9: non occorre chiedere
un altro consenso né fare un altro giro di review dello stesso codice.

## Identità confermate

- Base del diff complessivo: `31877ad85c619ec69214cd5e9fb53df50e452148`.
- Punta finale di Claude: `5146ecc2ca7215f739bce54ed064d44aa48ad50f`.
- Codice per `--reviewed-code-sha`: `5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba`.
- Identità dei 246 sorgenti:
  `37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de`.
- Piano `results/infra/pose_temponet_execution_20261120/plan.json`, SHA256
  `1e8e2b46b046072c706738995e745f608f009d33406d970729a3eb22a73f563e`.

Il diff `0e548bf8 → 5146ecc2` su `src/` e `scripts/` è vuoto. Tutti i 246
sorgenti del checkout di review coincidono byte per byte con il codice indicato.
Il piano è invariato e rigenerabile esattamente. Le conclusioni tecniche della
review GPT del 20/11 rimangono valide; la correzione non cambia il confronto
scientifico. La punta finale di Claude è integrata nel ramo di review con merge,
conservando tutti gli SHA originali.

## Aggiunte rilette

Gli otto test aggiunti coprono, fra l'altro, il lettore vero negli stadi successivi,
il rifiuto delle liste alterate anche dopo il ricalcolo dell'identità del piano,
il controllo su val, le costanti e la tolleranza R-1 e la nuova campagna nella
riga del submitter. Nessuna richiesta di modificare il runtime.

Le prove del revisore dei dati reali sostengono l'identità dei percorsi numerici
su **array sintetici alla geometria reale di val**, con perturbazione massima
`4,441e-16`, sotto la tolleranza `1e-12`. I circa **9 GiB e 3 minuti** sono una
stima con riscontro in un job storico (8,86 GiB), non la misura del nuovo job.
Il preflight deve ancora verificare i digest completi e R-1 sui file veri.
Non sono stati eseguiti dal revisore, a dimensione reale, il benchmark GPU
completo, il diagnostico L13 della WLS, il fold 1 e gli stadi successivi alla WLS.
Questi limiti restano espliciti, senza anticipare il superamento dei gate.

Le due nuove note operative sono corrette e **non bloccano il preflight CPU**:

1. Prima degli invii GPU dipendenti, l'operatore confronta il profilo `gpu8` con
   evidenza di un secondo nodo: modello CPU, nome, capacità e memoria GPU.
   Un'eventuale differenza richiede la procedura per classe di nodo già prevista;
   non autorizza ad allentare il digest o a procedere nonostante il disaccordo.
2. Nessun rinfresco dei tre input su SCRATCH durante la loro lettura da un job TCN.
   Il rinfresco previsto per il 03/12 va coordinato con l'operatore: un `touch`
   cambia mtime/ctime e può far scattare la guardia anche a byte invariati.

## Evidenze e seguito operativo

Rieseguiti **279 test TCN, tutti passati in 176,88 s**; nei cinque file storici
**125 passati e il solo fallimento preesistente** sullo stato del vecchio piano
lungo, identico a quello già documentato. Nessuna nuova regressione. Non ho
rieseguito la suite completa del repository: i sorgenti sono invariati e la
copertura comprende tutti i test TCN, inclusi gli otto aggiunti.

Le verifiche locali di questa rilettura sono in
`results/infra/pose_temponet_execution_20261120/gpt_review/followup_20261121/`:
script `audit.py`, risultato `audit.json`, comandi, hash dei log e risultati dei
test in `validation/local_validation.json`. La review precedente e le relative
prove del 20/11 restano conservate nella loro forma storica.

**Solo 2-TCN head launcher** esegue la sequenza del prontuario del 20/11:
integrazione senza riscrittura dei commit, nuova evidenza operatore, `release-plan`
legato al codice sopra e a questa review, nuovo riferimento di deploy e checkout,
sei verifiche locali, poi **TCNH-preflight-20261120 una sola volta** tramite
submitter e ledger comuni. Le fasi successive restano subordinate alle ricevute
e ai gate già approvati. Checkout, ID e ricevuta fallita del primo tentativo non
si riutilizzano per la nuova esecuzione; gli otto input scientifici restano quelli
già pinnati nel piano.

Il piano sorgente resta deliberatamente `REVIEW_REQUIRED` e non rilasciato:
la preparazione del piano di esecuzione compete all'operatore. Questa chat ha
effettuato soltanto merge locale, review, test e pubblicazione su origin;
nessuna lettura di Leonardo, release-plan, deploy, sottomissione o monitor.
