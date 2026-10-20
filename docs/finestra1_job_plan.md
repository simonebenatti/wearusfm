# Finestra 1 — proposta di default e lista dei job

> Archivio del 21/10/2026: bozza GPT del 12/10, recuperata fra i file non tracciati.
> Contenuto storico, in parte superato: per piano, budget e stato correnti usare
> `docs/decisioni.md`, i protocolli firmati e la tabella in `docs/claude_GPT_shared`.
> La pubblicazione di questa bozza non adotta i default o i job proposti qui sotto.

Prima lista GPT del 12/10/2026, **da completare con costi/fattibilita' Claude e firmare da Simone**.
Non sostituisce decisioni firmate, non congela ancora una campagna e non autorizza lanci.
Riferimenti: v10 §10.1/10.3/10.6, piano W7–W8 (1–3 novembre, esecuzione entro 15 novembre), D11/D12/D14/D15/D16.
Lista preliminare entro 14/10; firma/configurazioni e budget entro 25/10; prove dei job su dbg entro 31/10.

## Default di lavoro proposto

- Obiettivo **A**: JEPA target b + ancore fisiche attuali e multi-scala; RVQ spento; keep_weight=0.
  B keep=0,05 resta candidato, non scelto dal solo miglioramento spettrale. Prima baseline random e firma scientifica.
- Peso ancore 0,2 e media per finestra come configurazione attuale. **Riesaminare il peso** prima della firma:
  la multi-scala aggiunge termini, non equivalenti alle sole tre ancore al momento della prima scelta.
  Misurare norme dei gradienti pesati su encoder condiviso per JEPA/ancore/keep, non solo valori delle loss.
- Patch 25 ms, contesto 1–4 s, masking D10 con LONG corretto, K=64, identita' concatenata geometria/anatomia,
  dropout muscolo 0,4. Front-end attuale Fourier/spline/MLP 24/24/16, nessuna wavelet nel default.
- Preset **firmati**: 30M 28,3 M; 100M 104,8 M; 300M 292,0 M. Architettura esatta in LADDER/decisioni;
  non confondere nominali con parametri misurati. Perceiver attuale 1 cross + 1 latente, non due blocchi configurabili.
- **LR picco 3e-4 e batch globale 32 restano quelli del sanity fino alla calibrazione**, anche sui rung grandi.
  DDP su un nodo 4 GPU; accumulo se la memoria lo richiede, senza cambiare batch globale. FSDP/multinodo fuori finestra 1.
- WSD D14, D di riferimento 4 epoche comune ai rami; warmup 2% passi a D (minimo 1.000), stabile e decay 20%.
  Checkpoint pre/post-decay; ultimo update a lr_picco/n_decay, zero al confine dopo l'update (convenzione implementata).
  EMA 0,996, AdamW wd 0,05, clip 1,0. Passi per epoca da corpus/quote effettivi, non vecchia stima fissa.
- Manifest v1.1 e scale verificati, da congelare per campagna. Montaggi virtuali **condizionati** a attesa dati <=2%:
  misura storica 3,4–3,6%, dunque non ancora valida l'accensione automatica D15. Profilare per fasi e decidere prima dei run.
  CapgMyo resta escluso dai virtuali; orientamento anatomico/ordine canali/fatti aperti non diventano assunti.
- Zona di bordo 100 ms vs target lento 500 ms: questione di definizione ancora aperta, non correzione silenziosa nel dataloader.
  Se cambia, nuovi target/versione e confronti coerenti; non alterare in corsa i report gia' conclusi.

## Lista dei bracci, evitando run duplicate

Tutte le ablation: stesso corpus, endpoint D, sampler/masking, regime di valutazione e semi accoppiati.
Il riferimento condiviso non si rilancia per ogni tabella. **Due semi proposti** per i confronti a 30M,
da firmare con costo finale; non ridurli dopo aver osservato risultati. Pilot/asse soggetti: due semi gia' previsti.

| ID logico di famiglia | Bracci rispetto al riferimento | Stato prima del lancio |
|---|---|---|
| W1-OBJ | A riferimento; JEPA puro (anchor0); soli target espliciti; waveform MSE | A/puro disponibili; ultimi due da implementare e definire |
| W1-TARGET | target b vs a del teacher | Entrambi disponibili; riferimento riusato, diagnostica collasso query |
| W1-KEEP | A vs B keep0,05 | Candidato subordinato a baseline; training pilota emg2qwerty non prova sul corpus intero |
| W1-ID | concatenazione vs solo geometria vs gating topologia | Ultimi due da implementare; P1 sparso primario, Camargo separato |
| W1-K | K32/64/128 | Flag disponibile; sonda stratificata pre/post-bottleneck da scrivere |
| W1-LOCAL | locale2 vs4 nel riferimento 30M | Flag disponibile; leggere per topologia, non solo aggregato |
| W1-BOTTLENECK | 1 vs2 blocchi | Non configurabile oggi: contratto FMConfig e implementazione da concordare |
| W1-NEIGHBOR | raggio/vicinato piccolo vs default vs grande | Valori/protocollo da firmare, non inventati come gia' decisi |
| W1-LR-BATCH | calibrazione 30–100M di LR/batch | Griglia e regola selezione da congelare prima dei nuovi risultati |

**La vecchia run5 RVQ e' chiusa/esclusa**, non entra nel lotto. Waveform MSE richiede un decoder continuo
compatibile con fs native; "soli target espliciti" deve precisare quali target spaziali di v10 sono realizzati,
non limitarsi a JEPA_weight=0 e dichiarare l'intero braccio implementato. Nessun nuovo default scientifico implicito.

Secondarie, dopo priorita'/budget: locale off; muscolo dropout0; singole famiglie front-end;
front-end separati per fs. Wavelet e' proposta separata: non aggiunge automaticamente una run alla finestra.

## Pilot epoche e asse soggetti

- **W1-EPOCH-100M-s0/s1**: due trunk stabili fino a 6,4 epoche, poi decay fino a8; rami da0,8/1,6/3,2
  fino a1/2/4. Totale ~9,4 epoche per seed, salvataggi intermedi e stream esatto, checkpoint su scratch.
  D12 sceglie E4 oE8 con la regola firmata: non sostituirla con una nuova soglia dopo l'A/B.
- **W1-SUBJECT-{100M,300M}-{12.5,25,50,100}-s{0,1}**: manifest annidati per soggetto,
  stratificati per dataset/topologia, E4 nominali. Il 100M100% a4 condivide il ramo del pilot se OGNI
  configurazione/sampler/endpoint coincide. Nessuna condivisione se i default vengono cambiati dopo.
- **Controllo esposizione25% a stessi passi del100%** (v10 §10.6): 16 epoche nominali seE4.
  Taglia/e e semi di questo controllo vanno esplicitati nella firma e nel budget, non omessi per risparmiare.
- Risultati attesi disponibili entro15/11; se un default cade, run di conferma100M con scelte finali
  e possibile ripetizione del pilot, come da piano. Non cambiare insieme obiettivo, K e masking senza controllo.

## Cose da misurare prima di chiamare un job pronto

1. Equivalenza loss/gradienti/AdamW/EMA e stop/resume su CPU; conferma NCCL/bf16, memoria300M e throughput4GPU.
2. Dataloader: lettura/filtro/target/maschere/collate, attesa sul rank piu' lento, cache calda/fredda,
   stream replay, esposizioni/time-patch/source-channel-patch realmente consumati e quote realizzate.
3. P1/P2 con encoder casuale; dettaglio DB3/UCI e per seed; non mascherare classi difficili nella media.
4. Dataset-ID sugli embedding finali; rilevanza canale; sonda fisica e temporale (conduzione/coerenza),
   stratificata ai punti di §7.4. Misure spettrali non sostituiscono la dinamica temporale.
5. Bilanciamento gradienti e contribution delle famiglie del tokenizer, costo/FLOP/MFU, non solo loss totale.

TinyMyo/PhysioWave, CNN/TCN da zero, fine-tuning e Discrete Gestures restano debiti dell'harness v10;
questa prima lista non li dichiara completati. Stabilire quali debbano precedere l'interpretazione della finestra.

## Budget e consegna a Claude

Il totale ~5.500 GPUh e' una stima di pianificazione, **non un budget registrato/autorizzato**.
I fogli riportano ~210GPUh/100M a4 e ~640GPUh/300M a4: non misure DDP dei preset nuovi.
Da questi ordini di grandezza, solo pilot (~987GPUh) e asse soggetti (~2.768GPUh, escluso100M100%
condiviso) fanno ~3.755GPUh, prima di ablation, sonde e controllo esposizione. Ogni controllo25% a
stessi passi100%, due semi, costa circa2 volte il costo della taglia aD; farlo su entrambe aggiungerebbe
~1.700GPUh. **Non e' lecito chiudere la lista completa sotto5.500 senza ricalcolo e scelte esplicite.**

Claude aggiunge per ID: SHA/config esatta, stato implementazione, GPU/processi/accumulo, memoria,
step/s, passi aD, durata/tranche24h, walltime/margini, costo previsto massimo e spazio checkpoint.
GPT chiude protocollo scientifico e review; Simone firma default, perimetro e budget. Il lotto congelato
usa tag/SHA e CODE_DIR, main resta libero per sviluppo. Nessun launch da HEAD mobile o retry indistinto.
