# Sintesi scientifica e prossimi passi — 28/10/2026

**La ricetta di R3 apprende la posa e una componente dinamica anche su utenti esclusi dal training. Il beneficio e' ancora
modesto, le predizioni temporali sono attenuate e l'utilita' come foundation model rimane da dimostrare.**

**Seguito del 28/10:** review Claude `986b008` letta e corretta; E3 B approvata da GPT con protocollo
[congelato](../e3_b_20261028.md). [Risposta punto per punto](risposta_review_claude_e3_b_20261028.md).
GPT implementa e lancia, Claude resta revisore per istruzione successiva di Simone. Le proposte sotto conservano il contesto
della sintesi iniziale; stato esecutivo aggiornato nel registro e nel piano E3.
**E3 conclusa e letta:** estrazione 62561780 e rapporto 62561881 COMPLETED 0:0 (255 / 106 s), da `520158e`.
La ridge post-LayerNorm non recupera altra dinamica: rapporto 0,930307 contro 0,927475 della testa; delta appaiato
+0,002832, IC95 [+0,000898; +0,004602]. Migliora invece il livello medio e la MSE totale.
Consuntivo 0,070833 GPU-ore / 0,802222 core-ore; nessuna continuazione del training lanciata.
[Risultati, modifiche completate ed elenco dei punti aperti](e3_b_results_20261028.md).

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
| E3 B post-LayerNorm | Ridge leggermente peggiore sulla dinamica (delta +0,2832 punti percentuali di Vy), migliore sul livello medio | Non dimostra ottimalita' della testa o saturazione dell'encoder |

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

La tranche E3 e' conclusa. Il rifit lineare non ha recuperato dinamica aggiuntiva; il ricalcolo della scala globale
prevede un beneficio molto piccolo, senza aver applicato una riscalatura. Non serve ripetere la stessa sonda
o riottimizzarla sulle finestre dinamiche della val gia' osservata.

La prossima proposta raccomandata e' preparare un punto intermedio a **19.618 passi**, mantenendo la ricetta di B:
valutare l'evoluzione con differenze appaiate B19618-B5000 e confrontare rappresentazioni/trasferimento con R3c.
Prima servono verifica effettiva di stream/schedule/dose, copia immutabile del checkpoint5000, protocollo riformulato
e preventivo definitivo. P1/P2 e accessibilita' della posa richiedono confronti comuni; R3c non ha la LayerNorm
della testa di B, quindi questo E3 non si trasferisce automaticamente al confronto fra encoder.
E2 resta un controllo di associazione, non deve crescere obbligatoriamente; IC inconcludenti non provano equivalenza
o assenza di danno. La precedente proposta Claude di circa 13 GPU-ore e' una stima da verificare, non un lancio avvenuto.

R0 resta un riferimento utile per confrontare le ricette, a pari vista student/teacher, protocollo e durata. Non isola il
contributo della posa, perche' cambia anche JEPA: per tale attribuzione serve il confronto con **R3c abbinato**, come
previsto dal [protocollo del pilot](../pilot_posa_v1.md). I precedenti job interrotti non sono risultati completati.
J2 bis ha priorita' inferiore: non emerge una nuova prova di errore di allineamento che lo renda necessario ora.

Il punto intermedio proposto resta da preparare e rivedere; E3 non ne impone automaticamente l'esecuzione. Eventuali scelte di
architettura/iperparametri si sviluppano su sottosplit train; val non diventa un banco di tuning ripetuto. R3 a 49.046 passi e i
confronti di trasferimento restano una decisione distinta con obiettivo, costo e letture fissati prima del lancio.
Nessuna di queste proposte successive e' stata sottomessa nella tranche chiusa qui.

## Stato operativo e autonomia

Ultimi job E3 62561780/62561881 COMPLETED: 0,070833 GPU-ore / 0,802222 core-ore.
Dinamica 62551730: 0 GPU-ore / 0,014444 core-ore. E1/E2: 0,435278 GPU-ore / 3,595556 core-ore.
Fonte e output verificati, test e review indipendenti completati, documenti e registro sul ramo GPT di origin.

Istruzioni aggiornate di Simone: job con costo massimo <5 GPU-ore autonomi; lettura log/commit/push autonomi;
approvazione esplicita di una proposta gia' sufficiente per sottometterla ed eseguirla anche oltre 5 GPU-ore, senza secondo consenso.

Riferimenti: [audit e R3c](pose_scientific_assessment_20261028.md), [B](collaudo_b_results_20261028.md),
[E1/E2 B](b_e1e2_results_20261028.md), [dinamica B](dinamica_b_results_20261028.md).
