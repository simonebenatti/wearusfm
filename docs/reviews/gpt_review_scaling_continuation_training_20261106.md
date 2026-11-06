# Review GPT: continuazione 300M 5.000 -> 19.618, stadio di training

**APPROVATO. Nessun rilievo bloccante.** Claude puo' eseguire le tre righe di
`training_submit_lines.txt` nell'ordine pubblicato: deploy, checkout e training,
con i controlli previsti, senza altre review fra questi passaggi e senza un
nuovo consenso di Simone. Nessuna modifica al codice richiesta prima del lancio.

Pacchetto: deploy `73524b63be534f81b3db4ade4b7fc7f47129a0f3`, comandi
`92320d39646fd06c7b3b9954a6ca5b7e48455c86`, ramo
`claude/scaling-300m-exec-20261106`; registro Claude `69f85b4`.
Piano SHA256 `eded0922bd1022c417b8d70523c0c66d1ddb1bd8ad273925ec6d53fda5957d5f`.

## Copertura e rilascio

Questa e' l'unica review dello stadio A richiesto da Simone il 06/11 alle 17:22.
Copre le tre righe pubblicate: riferimento di deploy, checkout
`SCALING-300M-cont19618-checkout-20261106`, training
`SCALING-300M-train19618-20261106`. L'operatore e' Claude. I controlli freschi
prima degli invii, l'attesa del checkout COMPLETED 0:0, la verifica del suo
HEAD/piano e i gate dentro il job sono esecuzione del piano, non nuove review
o richieste di consenso. Stop al primo errore o gate fallito, nessun retry cieco.

Audit di rilascio indipendente: **225 controlli passati**. Verificati 103 sorgenti
scientifici e quattro di esecuzione, pin dei file nel commit di deploy, 11 file
del run a 5.000, ricevuta precedente, 19 pin del controllo e letture grezze.
Tutti i comandi corrispondono al piano; `/dev/stdin` presente per il checkout,
nessun afterok che sostituisca l'accettazione del checkout.

Il trainer scientifico resta identico; cambia solo l'orchestrazione della ripresa.
Riletta la catena nel trainer: ripristino student/teacher/AdamW, contatori e RNG
per rank, stream dalla posizione corretta e generatore separato del dataloader.
Schedule 49.046/warmup 1.962; passi 5.001-19.618 al LR originale, nessun nuovo
warmup. Copia verificata dei quattro file, nuova cartella e inventari distinti.
L'endpoint e' accettato soltanto con 19.618, stop per max_steps, file attesi,
checkpoint coerenti, log completi, pannelli invariati e memoria riconciliata.

## Test indipendenti

**238 casi unici superati: 156 nuovi (117 wrapper +39 launcher), 82 del percorso
storico a 5.000.** Il launcher ha chiuso in 40,50 s; gli 82 storici in 35,06 s.
Gli oracoli coprono la continuita' CPU senza worker, con due worker, in DDP con
prefetch e con testa/pannello di posa e tutti e quattro i file copiati.

La prima esecuzione nel sandbox ha impedito due prove: memoria condivisa negata
da `torch_shm_manager` e Gloo incapace di risolvere l'indirizzo locale. Conservate
le diagnosi; i rimanenti 115 casi del wrapper erano passati. Rieseguite le sole
due prove fuori dal sandbox, senza modifiche al codice: **2/2 PASS in 129,50 s**.
Il dump diagnostico a 60 s di questa ultima esecuzione mostra normale lavoro
nel frontend durante la ripresa; non e' un timeout o fallimento del test.
La limitazione locale e' quindi risolta; resta il limite scientifico dichiarato
che non e' stata misurata uguaglianza bit per bit su GPU.

## Giudizio esplicito sulle scelte a-g

| Punto | Giudizio | Precisazione |
|---|---|---|
| a. Nuova cartella | Favorevole | Preserva gli artefatti a 5.000; la copia verificata viene ripresa con il trainer invariato. |
| b. time-limit 51.300 s | Favorevole | Margine ragionevole rispetto a ~42.875 s stimati per gli update. I 1.800 s residui sono condivisi da verifiche iniziali, avvio e chiusura; non sono una garanzia di durata assoluta. |
| c. Ricevute su scratch | Favorevole | Coerente con quota WORK e raccolta immediata sul Mac. Le evidenze non devono restare soltanto su scratch. |
| d. Memoria | Favorevole | Gate sull'allocato <=0,9 della GPU invariato; riservato per rank registrato e controllato fra allocato e memoria fisica, senza introdurre una nuova soglia arbitraria. |
| e. Prove CPU/GPU | Favorevole alla distinzione | Oracoli CPU del meccanismo di ripresa; nessuna affermazione di uguaglianza bit per bit GPU, non misurata. Non richiedo un nuovo collaudo GPU. |
| f. Lettura a 19.618 | Approvata integralmente | Primari, NI, bootstrap e matched invariati; nessuna scelta del checkpoint migliore, raccomandazione circolare o autorizzazione a dosi ulteriori. |
| g. Stadio di valutazione | Favorevole alla proposta | Un unico pacchetto di codice/piano della valutazione, poi benchmark, fit, aggregati e confronto con i gate previsti, senza review fra i singoli invii. Codice non consegnato oggi: questa review non lo attesta e non blocca il training in attesa che sia scritto. |

Per g la proposta sul successivo pacchetto e' distinta dal rilascio A attuale:
non reintroduce review prima di ciascun job. La scelta di Simone sulle modalita'
operative prevale sulle formulazioni storiche. Non vi e' alcun nuovo via di
Simone richiesto per deploy, checkout e training qui rivisti.

## Precisazioni non bloccanti

Non richiedono cambiare i byte del rilascio prima del lancio; si registrano nel
resoconto operativo, conservando gli SHA approvati.

1. `chain_plan.md` §6: un fallimento prima del training consuma anche il tempo
   GPU/CPU dei controlli nell'allocazione, non soltanto l'attesa in coda.
2. §5: le estrazioni restano entro **7 GPU-ore richieste +0,15 di tolleranza**.
   I 7,35 residui dopo le 59 nominali comprendono anche 0,20 di tolleranza del
   training; non sono un nuovo tetto trasferibile alle estrazioni.
3. §6: il controllo AdamW impedisce regressioni e avanzamenti superiori a
   14.618 per parametro; richiede che **il massimo** avanzi esattamente di
   14.618. Non afferma che tutti i parametri condizionali ricevano un update
   a ogni passo. La continuita' dei momenti e' verificata negli oracoli CPU.
4. §8: il guard `required_until > now + 53100 s`, con orizzonte
   `2026-11-22T00:00:00Z`, rifiuta partenze da circa **21/11 alle 10:15 CET**.
   Il riferimento alla manutenzione del 22/11 non estende la validita' del piano.
5. `check_unchanged` confronta anche inode/mtime/ctime, non solo SHA. Un rinfresco
   scratch mediante touch durante la tranche puo' causare rifiuto finale anche
   a contenuto invariato. Coordinare l'eventuale giro del 12/11 con il run e gli
   artefatti pinnati, senza allentare i gate o mettere a rischio la conservazione.
   Non ho evidenza di un conflitto gia' presente o di quel rinfresco gia' inviato.

La corsia seriale per A0 resta da coordinare operativamente con la linea 2;
la review favorevole non sposta o cancella job altrui. Lo stato corrente della
coda e i futuri controlli di runtime non sono attestati da un audit locale.

## Risorse e regola operativa su main

Training: 1 nodo, 4 GPU, 32 CPU, 480G, QoS normal, 14h45: **59 GPU-ore e
472 core-ore allocate**. Checkout: 2 CPU, 4G, 5 min. Il piano resta nel freeze v2;
controllo 30M con scadenza minima ricostruita all'08/12 01:27 CET, da riverificare
dal job; orizzonte del piano piu' breve come precisato sopra.

`CLAUDE.md` aggiornato e pubblicato su main in **e114a78**: approvazione della
continuazione acquisita, una review del pacchetto prima dei lanci inclusi,
nessuna review o consenso intermedi. Superate anche le frasi obsolete che
dicevano ancora «19.618 non autorizzata». Le linee 2 e 3 restano invariate.

Prove in `results/infra/scaling_300m_20261104/continuation19618_training_review_20261106/`.
GPT non ha eseguito SSH, deploy, invii o modifiche al codice di Claude.
Alla chiusura sono comparsi nel worktree Claude nuovi file di valutazione non
tracciati: esclusi da questa review e dal deploy73524b6. Riverificati invariati
i nove file della consegna dopo i test; vale lo SHA esplicito del rilascio.
