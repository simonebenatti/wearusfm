# Fine-tuning EPN del T100-G 30M — risultati a cinque epoche

Il pretraining dà un vantaggio a questa dose supervisionata: **fine-tuning
93,77%, scratch 92,36%, differenza +1,40 punti percentuali, IC95% [0,93; 1,84]**.
È un risultato favorevole sullo sviluppo EPN, con un seme e ottimizzazione breve.
La regola pre-registrata F-R2 scatta: scratch ha la migliore validation all'ultima
epoca in tre fold su quattro. Il risultato non dimostra un vantaggio a convergenza.

Tutti i 24 fit sono COMPLETED 0:0; l'ultimo termina il 05/11 alle 02:51:58 CET.
L'aggregatore è una consegna separata di sola lettura: 15 test CPU passati,
review algoritmica indipendente favorevole e ricalcolo indipendente concorde
entro 1e-12 su punteggi e intervalli. Nessuna inferenza held ripetuta,
nessuna modifica ai runtime o alla selezione e nessun nuovo job per l'analisi.

## Risultati e interpretazione

314 soggetti, 47.100 finestre, sei classi. Metrica: balanced accuracy, cioè
macro-media dei recall delle classi sulle predizioni OOF concatenate, non media
dei quattro punteggi di fold. IC descrittivi da 2.000 bootstrap appaiati per
soggetto, seed 20261102, stessi campionamenti nei tre regimi, zero scarti.

| Regime | Che cosa viene allenato | Balanced accuracy | IC95% |
|---|---|---:|---:|
| Frozen | Nuova testa LayerNorm + lineare, teacher 19.618 congelato | 70,575% | [69,176; 71,977] |
| Fine-tuning | Testa ed encoder inizializzato dal teacher 19.618 | 93,766% | [92,921; 94,601] |
| Scratch | Stessa architettura, inizializzazione casuale | 92,363% | [91,427; 93,270] |

| Contrasto predefinito | Differenza in punti percentuali | IC95% appaiato |
|---|---:|---:|
| Fine-tuning − frozen | +23,191 | [21,945; 24,395] |
| Fine-tuning − scratch | +1,403 | [0,930; 1,837] |

**Il confronto con scratch è l'evidenza pertinente al contributo del
pretraining.** A pari cinque epoche e con la selezione concordata, partire dal
teacher aiuta. Scratch raggiunge già un punteggio alto: non abbiamo dimostrato
che il pretraining sia indispensabile, né che il vantaggio persista a dose maggiore.
I costi di pretraining non entrano nella parità delle cinque epoche downstream:
questa non è una dimostrazione di maggiore efficienza di calcolo complessiva.

**Il +23,19 riguarda soltanto il nuovo controllo frozen.** La logistica storica
sul medesimo teacher e sulla stessa popolazione raggiungeva **82,577%**:
12,002 punti sopra la nuova testa. Le ricette di lettura/ottimizzazione differiscono;
non sostituiamo a posteriori il controllo primario. Tutti i frozen selezionati
hanno la migliore validation alla quinta epoca. Il 70,575% non è quindi un
limite dimostrato delle rappresentazioni congelate, e il contrasto di 23 punti
non quantifica il vantaggio rispetto al miglior lettore congelato disponibile.
La causa specifica del divario con la logistica non è isolata da questa prova.

## Selezione e regola sulla dose

Ogni regime/fold prova i due LR approvati per cinque epoche. LR ed epoca sono
scelti soltanto sulla validation; parità risolta con prima epoca e candidato
di LR inferiore. Il candidato C0 non predice held; C1 carica la sua ricevuta,
confronta le validation e predice held una volta. Tutti i 24 candidati
completano cinque epoche, senza esiti non finiti.

| Regime | Candidato scelto nei quattro fold | LR encoder / testa | Migliori epoche, fold 0–3 |
|---|---|---|---|
| Frozen | C1 in 4/4 | encoder fermo / 1e-2 | 5, 5, 5, 5 |
| Fine-tuning | C1 in 4/4 | 1e-4 / 1e-2 | 4, 4, 5, 5 |
| Scratch | C0 in 4/4 | 1e-4 / 1e-3 | 4, 5, 5, 5 |

F-R2/SCRATCH_RULE usa soltanto le validation dei quattro scratch selezionati:
tre migliori epoche su quattro sono la quinta, quindi **flag=true**. La mediana
validation(epoca5) − validation(epoca4) è +0,389 punti, inferiore alla soglia
alternativa di +0,5; basta la prima clausola della regola OR. Non è una prova
che ogni run stia ancora migliorando, né dell'entità del possibile seguito.

Il seguito previsto come direzione è un nuovo confronto FT/scratch a 15 epoche,
dalle stesse inizializzazioni originali e con schedule a 15 epoche, senza ripresa
dai modelli selezionati a cinque. **È una nuova proposta per Simone, con preventivo
e review prima di qualunque lancio.** Nessuna estensione eseguita in questa lettura.

## Provenienza e verifiche

Runtime fit `a2eeb755676a0ab780fc269f0679684eb41cd6f2`, sorgenti
`aa857fb321bae83fd4b6463092c89dd2dba27e248cbe063f25a30032bf529b98`, ricetta
`2ba1b7f523385342535d89e0132c34bd97bf26f9d900677872d7aac92301b13a`.
Checkpoint teacher `a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056`.
Le review Claude f70dc72 e 2d157cc restano le approvazioni del runtime;
questa analisi successiva ha la propria [review indipendente](finetune_results_critic_20261105.md).

Verificati 84 file contro hash e dimensioni remoti; 24 identità dei checkpoint
contro SHA256 calcolati sui file remoti, senza scaricare/caricare i pesi.
Verificate provenienze C0→C1, ricetta, supporto storico pinnato, ruoli disgiunti,
copertura OOF senza duplicati, classi testuali native e pairing esatto
di riga/target/soggetto/fold fra regimi. Le selezioni e i punteggi held
si ricostruiscono dai file persistiti. Tutte le curve sono nel JSON finale.

Evidenze in [final_20261105](../../results/infra/consolidation_20261104/final_20261105/README.md).
I file OOF e i pesi restano esterni al repository, persistenti sotto WORK.
Il report include statistiche aggregate per soggetto/classe, non segnali grezzi.

## Consuntivo

| Voce | Job | GPU-ore effettive | Core-ore allocate |
|---|---:|---:|---:|
| Frozen | 8 | 9,743611 | 77,948889 |
| Fine-tuning | 8 | 29,263889 | 234,111111 |
| Scratch | 8 | 29,264167 | 234,113333 |
| Totale fit | 24 | **68,271667** | **546,173333** |
| Benchmark iniziale + v2 | 2 | 0,251944 | 2,015556 |
| Tre checkout | 3 | 0 | 0,032222 |
| Totale delle 29 voci | 29 | **68,523611** | **548,221111** |

I soli fit erano proiettati a 122,465173 GPU-ore, con tetto 144 GPU-ore /
1.152 core-ore. Il consuntivo usa secondi e risorse effettivamente allocate da
SLURM, non le durate massime prenotate. Il checkout iniziale serve anche la
continuazione ed è contato qui una sola volta; non duplicarlo in un totale
della campagna. Il training della continuazione è escluso da questa tabella.

## Limiti e seguito del consolidamento

Sviluppo EPN interno P2, con soggetti separati fra fit/validation/held,
calibrazione prevista dal protocollo e nessun test ufficiale valutato.
Il bootstrap è condizionato a questi modelli/fold: non include riaddestramenti,
incertezza fra semi o una nuova popolazione indipendente. Un solo task di
classificazione non dimostra trasferimento nella regressione o utilità generale
del FM. Le soglie V-R1 di P1/P2/posa non sono applicate al solo fine-tuning EPN.

Il training lungo resta la linea già approvata; controller della matrice a due
momenti/profondità, pin e preventivi dei valutatori sono ancora una consegna
separata. Il nuovo aggregatore FT non li dichiara pronti. Nessun monitor
automatico, nuovo ramo scientifico o cambio dei runtime attivi.
