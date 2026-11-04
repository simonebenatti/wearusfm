# Esito del collaudo 30M/300M — 4 novembre 2026

**Collaudo tecnico completato; nessun blocco tecnico osservato nei 128 passi per
taglia. Il training scientifico a 5.000 non è ancora pronto.** Job63526191:
`COMPLETED 0:0`, dalle18:35:59 alle18:45:05 CET, 546s su quattro A100.
Consuntivo0,606667GPU-ore e4,853333core-ore; checkout63525937 separato,
0,008889core-ore. Codice congelato05d4bf0, nessun nuovo checkpoint scientifico.

| Misura | 30M T100-G | 300M T100-G |
|---|---:|---:|
| Parametri student/decoder/teste, senza EMA |28.362.013|292.054.847|
| Aggiornamenti per rank |128|128|
| Finestre globali |4.096|4.096|
| Picco memoria allocata GPU |13,1121GiB|28,4057GiB|
| Picco memoria riservata |21,0703GiB|43,8984GiB|
| Tempo medio di aggiornamento, 112 passi misurati |0,70785s|2,76668s|
| Norma gradiente mediana nella fase al picco LR |4,6261|8,3264|
| Fattore di clipping mediano nella stessa fase |0,21617|0,12010|

Il 300M richiede3,9086 volte il tempo per aggiornamento misurato. La misura somma
i massimi fra rank dei tempi forward/backward e clip/AdamW/EMA; esclude i primi
otto passi di ogni fase. Non include tutto il tempo del loader, i pannelli,
il checkpointing o la diagnostica. Non va moltiplicata ingenuamente per5.000
per promettere la durata del training.

Il clipping interviene in ogni aggiornamento di entrambi i modelli. Nel300M è
più forte nella fase di stress, ma non si osservano NaN/Inf o gruppi diagnostici
senza aggiornamento. Questo non dimostra che LR1e-4/clip1 siano ottimali, né
impone di cambiarli. Le loss del prefisso tecnico, con passaggio anticipato al
picco LR e dati di train, non stabiliscono utilità downstream o convergenza.

## Integrità verificata

Raccolti report, quattro report di rank e otto stream JSONL; tutti13SHA256 uguali
agli originali remoti. Conservate copie compresse lossless, manifest e hashes in
`results/infra/scaling_300m_20261104/preflight_63526191/`.
Il verificatore locale `verify_scaling_preflight_63526191.py` non importa il
controller del collaudo e ricalcola da questi artefatti:

-1.024 record completi,128 passi per taglia e rank,64nativi+64stress;
-512 coppie rank/passo con stessi dati, maschere, target, supporti e schedule;
-identità iniziale del frontend/gain, dose e assenza di EPN/UCI nello stream;
-corrispondenza JSONL/report, LR esatto, finitezza e aggiornamenti nei gruppi
  monitorati, incluse QKV/MLP del primo e ultimo blocco;
-tempi, clipping, memoria e identità di codice/piano/input.

Il subagent revisore ha ricalcolato direttamente dai quattro rank medie, rapporto
dei tempi e memoria: lettura tecnica favorevole, nessuna obiezione aperta.
La verifica riesce sia sugli originali raccolti sia sulle copie compresse.

## Cosa manca prima dei 5.000

Il proxy prespecificato di memoria della produzione è
`28,405681 × max(1, 15,473959 / 13,112072) = 33,522417GiB`.
È sotto il limite90% della memoria effettiva, **57,080402GiB**. Non si attiva
quindi il ripiego accum2 sulla base del proxy. Questo non esercita il percorso
reale: il collaudo non include prefetch e diagnostiche periodiche della produzione.

La prima ripetizione di fixed eval/pannello posa con AdamW e prefetch residenti,
e la diagnostica del gradiente della posa, avvengono al passo500. La prova tecnica
S3 minima deve quindi arrivare a500 da zero con ricetta intatta; i soli passi0
o64/128 non coprono quel picco. Si prepara un wrapper separato senza cambiare
il trainer storico e senza riusare il checkpoint tecnico per la prova scientifica.
Preventivo e review del wrapper precedono l'invio; **nessun job S3 è ancora inviato**.

In parallelo si completano lettori posa/classificazione a D896/pool1792,
controllo secondario S5, confronto appaiato, regressioni e piano di valutazione.
Claude ha comunicato approvazione della preparazione offline con248test passati;
il fallimento emerso nella sua suite completa non è ancora identificato e non
viene classificato come regressione di questa linea senza confronto col ramo base.
La review scritta e il relativo SHA restano da ricevere. Nessun nuovo consenso
scientifico richiesto; restano lavoro implementativo e verifiche concrete.

### Aggiornamento dopo la consegna della review

Ricevuta e verificata su origin la review Claude `5e4eb8a02fb75f6756c5c2d1be53126ebd39ab37`,
copiata integralmente in `claude_review_scaling_300m_scientific_offline_20261104.md`.
Verdetto: **APPROVATO** sulla preparazione offline, nessun blocco. Suite completa:
2.275 passati e un fallimento preesistente del launcher B19618, riprodotto anche
alla base `6d80e8d`; non è una regressione introdotta dallo scaling. Il trainer
rimane identico ai 103 sorgenti congelati per il confronto. Questa approvazione
non copre automaticamente i lettori e il controllo S3 aggiunti successivamente.
