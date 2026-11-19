# Review unica — scaling 300M a 49.046, training e valutazione

19/11/2026. **APPROVATO**, nessun rilievo bloccante sul commit
`87da975bc0fd4824684e166659bb996e118ff107`, ramo
`codex/scaling-49046-launcher-20261118`. Review svolta su un checkout detached
pulito di quel commit, con tre verifiche indipendenti parallele e riesecuzione
dei test mirati. Non sono stati modificati codice, checkout o job del launcher.

L'approvazione copre il pacchetto completo di
`docs/scaling_49046_execution_20261119.md`: due segmenti di training, archivio,
sei estrazioni, validazione e benchmark, costruzione del budget, ridge e sonde,
aggregati e confronto. I documenti futuri vengono prodotti dai costruttori
rivisti usando le ricevute effettive; non richiedono altre review per invio.

Simone ha già autorizzato il seguito e chiesto di farlo partire dopo review
favorevole senza rinnovare il consenso. **La chat 1-30/300 launcher può quindi
procedere direttamente con deploy, checkout ed esecuzione**, usando i cancelli
del piano, il submitter e il ledger comuni. Resta l'unico operatore. Nessuna
prosecuzione oltre 49.046, nuova ricetta, nuovo seme o retry è implicito.

## Ripresa e continuità del training

I piani canonici e i mirror si rigenerano byte per byte. I 103 sorgenti
scientifici restano congelati; il diff di `src/`, `train_jepa.py` e del submitter
rispetto alla consegna scientifica è vuoto. Rimane la ricetta storica T100-G:
tokenizer 100 ms/GELU con passo 25 ms, target RMS 25 ms e spettro rapido 50 ms.
Le nuove finestre dei target presenti su main non entrano nella ripresa.

Verificati ripristino di student, teacher, guadagno, AdamW, RNG per rank, stream
e allarmi; ordine torch prima di NumPy; copia iniziale e immutabilità del run
sorgente. Il secondo segmento usa ricevuta, piano e inventario del primo già
raccolto e COMPLETED 0:0, non una dipendenza accodata in anticipo. La precedente
sovrascrittura del piano con l'output di sacct è corretta e coperta da regressione.

Lo schedule resta quello originale a 49.046. Ricalcolo indipendente del LR per
numero di update: 39.238 = 1e-4; 39.239 = 9,998980528086451e-5;
49.046 = 1,0194719135492303e-8. Inventario finale, contatori AdamW, intervallo dei
record e pannelli alle cadenze native sono verificati dal wrapper. La prova di
ripresa bit-identica attraverso il decay è su un piccolo training CPU reale;
non attesta bit-identità GPU.

Sono coerenti i pin del 300M a 19.618 e l'attestazione pubblicata del controllo
30M terminale: quest'ultima esegue realmente `checkpoint_identity`, oltre al
controllo dell'hash. Il wrapper ripete le verifiche previste sui file effettivi.
Gate memoria sull'allocato per rank entro il 90% della GPU; riservato registrato
separatamente. Nessuna memoria o durata futura è dichiarata già misurata.

## Valutazione e lettura scientifica

I cinque valutatori derivano esattamente dalle sorgenti accettate a 19.618,
secondo le sostituzioni dichiarate. Rimangono griglie, selezione sul training,
bootstrap appaiato per soggetto (2.000 repliche, seme 20261102), pesi P1/P2,
proiezione fissa 896→384 e primari EPN612/E_dyn sulla vista naturale. Conservati
il margine di non inferiorità di 1 pp e il miglioramento pratico di posa del 2%.

La nuova estrazione simmetrica dei due checkpoint finali è esplicita e compresa
nei costi. Non si riaddestra il 30M. Il riuso tramite adattatore delle vecchie
cache della linea 2 non è certificato da questa consegna: riestrarre entrambe
le taglie con la stessa procedura è accettabile e non dichiara invalide quelle
cache. Piano metadata, hash e guardie dei launcher ora ammettono realmente le
sei celle, senza dipendere da un piano inesistente nel checkout del codice.

La finalizzazione P1-r2 è collegata al percorso reale di estrazione per entrambe
le taglie, con audit completo e controlli sugli ingressi conservati. Verifica
indipendente su piccoli file per i tre dataset P1 di entrambi i bracci; ricevute,
provenienze e log della guardia coerenti. Pose/P2 mantengono le guardie precedenti.

L'interazione dose×taglia è **sempre descrittiva** in questo pacchetto:
`comparable=False`, nessun IC o verdetto inferenziale sull'interazione. Questo
chiude il rilievo sulla certificazione incompleta della parità inter-dose senza
modificare i primari a 49.046. Nessuna selezione del checkpoint migliore; restano
un solo seme di pretraining e utenti di sviluppo.

## Controlli operativi, conservazione e risorse

Le verifiche indipendenti confermano il percorso ordinario gate→lines, non solo
funzioni isolate: attestazione del codice, pin nei checkout, documenti, ledger,
prerequisiti raccolti, accounting dal sacct grezzo, quota, tranche e scadenza del
verbale. Il primo checkout passa con Git e pin reali in locale; undici
transizioni senza predecessori sono rifiutate. Uscite fallite o booleane,
coda piena, quattro job della linea e quota insufficiente non generano invii.

Archivio concreto prima delle estrazioni: tre checkpoint in una nuova cartella
persistente WORK, con hash/dimensioni, fsync e sorgenti immutabili. Le dimensioni
future derivano dalle ricevute, non dal peso storico a 19.618. La quota WORK
pubblicata è 45,95 GiB liberi; gate fresco minimo 30 GiB. Restano minimo 20 GiB
su SCRATCH e orizzonte 19/12, con controllo della retention dei pin.

| Componente nuova | Richiesta | Con margine scheduler |
|---|---:|---:|
| Due training, 4 GPU | 119 GPU-ore | 119,4 GPU-ore |
| Sei estrazioni, 1 GPU | 14 GPU-ore | 14,3 GPU-ore |
| Totale GPU | 133 GPU-ore | 133,7 GPU-ore |
| CPU associate ai job GPU | 1.064 core-ore | 1.069,6 core-ore |
| CPU-only, massimo delle fasi | — | 347,8 su 400 core-ore |

Il conto CPU-only include cinque checkout e archivio. I fit hanno limite di
240 core-ore richieste/246 con margine, al massimo 15 shard e 3 ore per shard;
il numero effettivo deriva dai benchmark. La riserva di 52,2 core-ore non estende
i limiti di fase né autorizza retry. Il vecchio consuntivo 94,732778/351 è chiuso
e separato. I 30 minuti lasciati da ciascun timeout del trainer al wrapper sono
coerenti con i walltime di 19h45 e 10h.

## Prove, limiti e rilascio

Rieseguito dalla chat scientifica il comando finale del pacchetto sui cinque
file di test mirati: **250 passati in 149,23 s**, exit 0; hash dei sorgenti
invariati prima/dopo. In aggiunta: 37 verifiche operative indipendenti,
ricalcolo dei piani/LR e verifica indipendente del finalizzatore P1. Evidenze in
`results/infra/scaling_49046_20261118/gpt_review_20261119/`.

Non si dichiara verde la suite CPU intera. Il launcher documenta una regressione
estesa con 881 passati e un fallimento storico legato al ledger vivo, riprodotto
anche sulla base accettata 2df4ca68; un giro locale più ampio interrotto rimane
incompleto. Questi esiti non sono usati come prova GPU o come successo del run.

**Rilievi bloccanti: nessuno.** Gli errori segnalati sulla bozza sono chiusi nei
byte del commit rivisto. Nessuna modifica facoltativa è richiesta prima del
lancio. Restano i controlli operativi già approvati: primo errore reale, TIMEOUT,
incongruenza o costo oltre i tetti arresta lo stadio senza retry automatico;
attendere un job già accettato non equivale a rilanciarlo.

Questa è la review unica richiesta, non un nuovo consenso di Simone. Il launcher
può eseguire la catena fino alla raccolta del confronto e riportarne l'esito alla
chat scientifica. I dati futuri restano da verificare con i cancelli già rivisti;
non costituiscono motivi per chiedere ulteriori approvazioni di routine.
