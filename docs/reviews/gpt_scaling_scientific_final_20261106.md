# Scaling 30M/300M a 5.000: lettura scientifica indipendente GPT

Prima lettura scritta dal report finale prima di aprire l'analisi di Claude.
Fonte: consegna `7fdb2e85945b23930de5347cc8999492b5d12bec`,
`final_stages_20261106/collected_outcome_20261106_1441/report.json`, SHA256
`4c10e7bb3007bdb46b5a1ce975b5531700ad9ac0aa0c413442de410c78b0c664`.
Confronto T100-G, teacher a 5.000, ricetta ed esposizione uguali, un seme di
pretraining. Sonde su encoder congelato; non fine-tuning dell'intero FM.

**Il 300M mostra un vantaggio sulla classificazione EPN612, sostenuto anche da
un piccolo guadagno su P1. La dinamica della posa resta sostanzialmente invariata.
Propongo di proseguire il confronto fino a 19.618, mantenendo la ricetta: la
motivazione e' verificare la persistenza del vantaggio classificativo con piu'
training, non un miglioramento della posa gia' dimostrato.**

## Risultati alle dimensioni naturali

Delta sempre 300M meno 30M. Classificazione in percentuale di balanced accuracy;
delta classificativi in punti percentuali. Intervalli appaiati al 95%.

| Misura | 30M | 300M | Delta [IC95] | Lettura |
|---|---:|---:|---:|---|
| EPN612, primario | 76,3142 | 78,2314 | +1,9172 [+1,3290; +2,5265] | Vantaggio |
| P1, media pesata dei tre NinaPro | 15,3592 | 15,9606 | +0,6014 [+0,1346; +1,0262] | Piccolo guadagno; non inferiorita' sostenuta |
| P2, media EPN612/UCI | 77,4710 | 78,7716 | +1,3006 [-0,3834; +2,7215] | Superiorita' irrisolta; non inferiorita' sostenuta |
| E_dyn posa, primario, minore meglio | 0,899816 | 0,899931 | +0,000115 [-0,003387; +0,003778] | Nessun vantaggio osservato |

Per dataset, i delta naturali sono +0,7213 pp su DB2, +0,4086 su DB3,
+0,4112 su DB6 e +0,6839 su UCI. Il solo IC di DB2 esclude zero fra questi
quattro secondari. P2 include EPN: non costituisce una replica indipendente
del risultato primario. P1 e' un indice di task diversi; il suo livello assoluto
non va confrontato con quello di P2 come se fossero lo stesso problema.
I pesi congelati di P1 sono 8/13 DB2, 3/13 DB3 e 2/13 DB6; P2 usa 1/2 EPN
e 1/2 UCI. Non abbiamo scelto questi pesi dopo i risultati.

S8 congelata: EPN `better`, posa `unresolved`; non inferiorita' P1/P2
`supported` rispetto al margine -1 pp. Esito `propose_19618_confirmation`.
La riduzione puntuale di E_dyn e' -0,01277%, senza il segnale pratico del 2%.
Non inferiorita' non significa equivalenza, e P2 non ha superiorita' dimostrata.

## Che cosa aggiunge il controllo a dimensione pareggiata

Per la classificazione, la vista naturale ha 768 ingressi nel 30M e 1.792
nel 300M. La vista matched proietta separatamente i due blocchi del 300M
da 896 a 384 ciascuno, con matrici casuali ortonormali fisse e indipendenti
dai dati, prima dello scaler. Non e' una compressione imparata o ottimizzata.

| Misura | 300M matched | Matched meno 30M [IC95, pp] |
|---|---:|---:|
| EPN612 | 76,1911 | -0,1231 [-0,6815; +0,4034] |
| P1 | 15,5409 | +0,1817 [-0,2427; +0,5901] |
| P2 | 78,4592 | +0,9882 [-0,2634; +2,1153] |

Su EPN il 300M naturale supera il proprio matched di 2,0403 pp
[1,6686; 2,4268]; su P1 di 0,4198 [0,1291; 0,6994]. Il vantaggio misurato
non sopravvive chiaramente a questa compressione. La disponibilita' dell'intera
rappresentazione conta per il risultato, ma il controllo **non dimostra che il
vantaggio sia solo una testa piu' grande**: proiettare elimina direzioni e puo'
cambiare informazione accessibile e regolarizzazione. Non separa causalmente
profondita', larghezza dell'encoder, informazione perduta e capacita' del lettore.
Il confronto primario resta quello naturale prespecificato.

## La domanda della posa rimane aperta

La lettura precedente della sola posa resta valida. E_dyn ~0,90 corrisponde a
circa il 10% di riduzione dell'errore centrato rispetto a una previsione senza
variazioni temporali. Il passaggio a 300M non la aumenta. Correlazione centrata
0,3213 -> 0,3194; varianza predetta/target 7,08% -> 7,58%, cioe' deviazione
standard circa 26,6% -> 27,5% del bersaglio. Previsioni un poco meno piatte,
senza migliore inseguimento del movimento. Nemmeno il controllo matched
porta un vantaggio apprezzabile su E_dyn.

L'IC del delta e' stretto rispetto al riferimento pratico del 2% (~0,018 di
E_dyn). A questa dose e con questi training, quindi, manca anche un guadagno
praticamente interessante, non soltanto una dichiarazione di significativita'.
Restano un solo seme e una ridge su medie degli slot: non e' una dimostrazione
che nessuna informazione dinamica esista nei latenti completi.

## Proposta scientifica e limiti

1. Completare la verifica tecnica del report e confrontare questa lettura con
   Claude. Nessun nuovo esperimento necessario per interpretare il risultato.
2. Proporre a Simone il seguito 300M da 5.000 a 19.618 con stesso schedule,
   obiettivi, dati e seme, contro il controllo 30M allo stesso passo gia' previsto.
   Tenere primari naturali e matched secondario. Il seguito chiede se il guadagno
   persiste o cresce con la dose; non e' una replica indipendente tra semi.
3. Non introdurre insieme una nuova loss o il ramo log-quadratico: altrimenti
   cambierebbe la domanda del confronto. Le altre linee mantengono il loro scopo.
4. Non promuovere ancora 300M a modello unico. Il confronto e' a esposizione
   uguale, non a costo uguale o a convergenza, e non dimostra efficienza superiore.
   Una conferma fra semi sarebbe una domanda successiva, non gia' autorizzata.

IC ottenuti con 2.000 repliche di soggetti, seme 20261102, condizionati a
checkpoint, training e sonde fissati. Soggetti di sviluppo; nessuna garanzia
congiunta al 95% o correzione per molteplicita'. Nessuna conclusione sulla
sufficienza del dataset per un FM, sul fine-tuning o su tutti gli usi del modello.

Il freeze v2 gia' pubblicato propone per il seguito 66 GPU-ore richieste
(59 training + 7 estrazioni; 66,35 incluse tolleranze), piu' un tetto di
350 core-ore CPU-only. Sono tetti condizionali, non consumi previsti certi o
autorizzazioni. Il piano concreto richiede controllo del 30M disponibile,
continuita' della ripresa e review. **Nessun seguito a 19.618 e' stato autorizzato
o sottomesso da questa review: la decisione resta a Simone.**

## Verifica numerica e confronto successivo con Claude

La lettura sopra e' stata salvata prima di aprire l'analisi Claude; hash del
prefisso e data in `final_outcome_review_20261106/independent_reading_freeze.json`.
In parallelo, un secondo revisore ha ricostruito da zero i punti e gli IC dai
conteggi pubblicati, senza importare le funzioni del progetto: **112 confronti
numerici superati**, massimo scarto 1,42e-14. Script riproducibile
`scaling_final_independent_20261106_check.py`, report compatto
`final_outcome_review_20261106/independent_numbers.json`. Nessuna rilettura dei
vettori OOF dal cluster; verifica aritmetica dei riepiloghi, distinta dalla
catena di provenienza attestata nella review tecnica.

Letta poi l'analisi Claude su `b5617a332ca30a372c52a558dc6060e436ece411`,
`docs/reviews/claude_analysis_scaling_300m_5000_20261106.md`.
**Convergiamo su tutti i punti che decidono il seguito:** EPN migliore, posa
invariata nella misura primaria, NI sostenuta su P1/P2, matched insufficiente
per separare le spiegazioni causali, un solo seme e sviluppo. S8 e' applicata
correttamente. Non serve cambiare criteri o rifare esperimenti per accettare
il risultato. Alla domanda di Claude se serva prima spiegare il controllo
matched, rispondo di no: il vantaggio primario naturale giustifica la proposta
di continuazione nella stessa ricetta; attribuirne la causa resta una domanda
diversa. Non userei il nullo dopo compressione come nuovo veto a posteriori.

Due precisazioni descrittive, senza conseguenze sui primari:

- **Rettifica dei conteggi per utente EPN nel §7 di Claude.** Da conteggi interi,
  naturale: **199 meglio, 100 peggio, 15 pari**, non 200/101/13. Matched:
  **148 meglio, 153 peggio, 13 pari**, non 148/155/11. Per tutti gli utenti
  EPN ci sono esattamente 25 finestre per ciascuna delle sei classi: basta
  confrontare i totali interi delle predizioni corrette. Il confronto diretto
  delle medie float con zero conta alcuni residui di ±1,11e-16 come differenze.
  La mediana +2 pp e gli score/IC restano invariati. Correzione da riportare
  nello script descrittivo e nel documento Claude, senza toccare il report
  congelato o rifare alcun job. Riproduzione:
  `scaling_epn_ties_20261106_check.py`, output `epn_exact_ties_audit.json`.
- DB3 resta vicino al livello casuale e incerto; preferisco questa formulazione
  a «nessuna taglia legge quei dati» o «il delta non si interpreta». Il delta
  resta stimabile ma non risolto. Sulle statiche E_dyn peggiora leggermente
  (+0,00541, IC [+0,00031; +0,01199]): e' un secondario centrato, non un nuovo
  criterio di arresto e non il confronto con una media globale predicibile.

La mia raccomandazione e' quindi di portare a Simone **una sola continuazione
300M fino a 19.618**, con la motivazione classificativa esplicita e la posa
come domanda ancora aperta. Il costo maggiore e il mancato guadagno sulla
dinamica impediscono di concludere gia' che 300M sia la scelta generale migliore.
