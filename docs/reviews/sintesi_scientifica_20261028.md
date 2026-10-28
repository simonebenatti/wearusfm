# Sintesi scientifica e prossimi passi — 28/10/2026

**La ricetta di R3 apprende la posa e una componente dinamica anche su utenti esclusi dal training. Il beneficio e' ancora
modesto, le predizioni temporali sono attenuate e l'utilita' come foundation model rimane da dimostrare.**

**Seguito del 28/10:** review Claude `986b008` letta e corretta; E3 B approvata da GPT con protocollo
[congelato](../e3_b_20261028.md). [Risposta punto per punto](risposta_review_claude_e3_b_20261028.md).
GPT implementa e lancia, Claude resta revisore per istruzione successiva di Simone. Le proposte sotto conservano il contesto
della sintesi iniziale; stato esecutivo aggiornato nel registro e nel piano E3.

## Che cosa abbiamo verificato

| Passaggio | Risultato sostenuto | Limite |
|---|---|---|
| Audit dei bersagli | 100/100 controlli aggregati sui 20 casi esaminati, nessun difetto concreto rilevato | Non certifica ogni registrazione o un allineamento fisiologico perfetto |
| Collaudo dbg, 1.300 passi | Il criterio non e' stato raggiunto; dose utile dello schedule molto breve | Non dimostrava che la ricetta fosse incapace di apprendere |
| Collaudo B, 5.000 passi | MSE train −11,713%, criterio B superato, 23/32 utenti migliorano | Train, un seme; non identifica causalmente la sola dose |
| E1 su 15 utenti esclusi dal training | MAE dinamiche −6,245%, IC95 [4,774%; 8,296%], tutti i 15 sotto la costante | Beneficio modesto; metrica/pesi/vista diversi da B |
| E2, stesso supporto | Gap allineato/disallineato +0,017654 rad, IC95 [0,013260; 0,023309], positivo in 14/14 utenti valutabili | Il cambio di finestra cambia anche la postura media |
| Separazione delle medie e della dinamica | Sulle dinamiche MSE centrata −7,252%, IC95 [6,083%; 8,752%], guadagno in 14/15 utenti | Centratura concede correzione perfetta del bias per finestra, non prestazione operativa |
| Finestre poco dinamiche | Errore centrato +5,945%, peggioramento in 15/15 utenti; livello medio migliora | Gruppo definito dalla mediana del target, non necessariamente riposo |

Sulle dinamiche la varianza predetta e' il 6,324% della varianza del target sullo stesso supporto, e la correlazione pooled
centrata e' 0,270. I guadagni MSE sono +0,044640 sul livello medio e +0,038692 sulle variazioni: non soltanto postura,
ma non ancora una ricostruzione forte del movimento. Non si deduce che basti amplificare le uscite.

Il ricalcolo proposto da Claude conferma quantitativamente quest'ultimo punto: sulle dinamiche la riscalatura globale
ideale a posteriori sarebbe 1,0734 e ridurrebbe il rapporto centrato solo da 0,927475 a 0,927134 (MSE assoluta 0,0001818).
Non prova calibrazione per giunto/utente e non viene applicata. La diagnosi Claude di zeri invalidi entrati nella ridge
precedente non e' confermata: la maschera attuale e' uguale sui 20 giunti e `_instants` esclude le patch interamente invalide.
E3 R3c mantiene il suo valore descrittivo; non la svalutiamo per quel presunto difetto.

L'audit originario non dimostrava «allineamento 10/10»: il confronto RMS aveva gap positivo in 7/10 utenti.
Analogamente, E3 R3c aveva circa il 5,07% di vantaggio sul costante: sotto il suo criterio operativo, non assenza di
informazione. Non confrontiamo quel numero direttamente con E1 B: teacher/student, durata e lettore sono diversi.

## Come cambia la diagnosi

- **Dose del dbg insufficiente (H1): sostenuta.** Con la ricetta estesa compare apprendimento e si vede un beneficio fuori
  dai soggetti train; B non isola la dose dall'evoluzione dell'encoder.
- **Catena dei bersagli difettosa: meno plausibile**, senza difetti dimostrati nel campione esaminato.
- **Assenza totale di informazione utile alla posa (H2 estrema): non coerente con le nuove letture.** Restano possibili
  limiti di rappresentazione, testa, ottimizzazione, obiettivo e dati.
- **Utilita' del foundation model: ancora aperta.** Una testa supervisionata che migliora la posa non prova trasferimento
  su altri dataset/task o un vantaggio delle rappresentazioni rispetto ai controlli.

## Direzione successiva

La tranche attuale e' conclusa: non serve ripetere audit generici o altri controlli per interpretarne i risultati.
Prima del training completo, il prossimo approfondimento pertinente e' **E3 sullo stesso checkpoint B student**:
encoder congelato, piccolo lettore ridge addestrato solo nel train, iperparametri selezionati esclusivamente fra utenti
train, e lettura val con protocollo definito prima. La cache val esiste; serve l'estrazione del sottoinsieme train.
Tempi e costi effettivi vanno scritti nel piano operativo, sfruttando la nuova autonomia per i piccoli job.

Per la domanda attuale non basta il solo MAE totale della ridge: va affiancata la stessa separazione media/dinamica, per
non scambiare un migliore livello medio per piu' informazione sul movimento. Un lettore che recupera piu' dinamica
indicherebbe informazione accessibile con quel lettore oltre la testa attuale; non dimostrerebbe da solo un unico collo
di bottiglia. Un esito simile manterrebbe aperti dose, rappresentazione e obiettivo, senza provare un limite assoluto.

R0 resta un riferimento utile per confrontare le ricette, a pari vista student/teacher, protocollo e durata. Non isola il
contributo della posa, perche' cambia anche JEPA: per tale attribuzione serve il confronto con **R3c abbinato**, come
previsto dal [protocollo del pilot](../pilot_posa_v1.md). I precedenti job interrotti non sono risultati completati.
J2 bis ha priorita' inferiore: non emerge una nuova prova di errore di allineamento che lo renda necessario ora.

Dopo questa diagnosi si decide se proseguire la ricetta invariata o verificare una lettura diversa. Eventuali scelte di
architettura/iperparametri si sviluppano su sottosplit train; val non diventa un banco di tuning ripetuto. R3 a 49.046 passi e i
confronti di trasferimento restano una decisione distinta con obiettivo, costo e letture fissati prima del lancio.
Nessuna di queste proposte successive e' stata sottomessa nella tranche chiusa qui.

## Stato operativo e autonomia

Ultimo job 62551730 COMPLETED in 13 s, 0 GPU-ore / 0,014444 core-ore. E1/E2: 0,435278 GPU-ore / 3,595556 core-ore.
Fonte e output verificati, test e review indipendenti completati, documenti e registro sul ramo GPT di origin.

Istruzioni aggiornate di Simone: job con costo massimo <5 GPU-ore autonomi; lettura log/commit/push autonomi;
approvazione esplicita di una proposta gia' sufficiente per sottometterla ed eseguirla anche oltre 5 GPU-ore, senza secondo consenso.

Riferimenti: [audit e R3c](pose_scientific_assessment_20261028.md), [B](collaudo_b_results_20261028.md),
[E1/E2 B](b_e1e2_results_20261028.md), [dinamica B](dinamica_b_results_20261028.md).
