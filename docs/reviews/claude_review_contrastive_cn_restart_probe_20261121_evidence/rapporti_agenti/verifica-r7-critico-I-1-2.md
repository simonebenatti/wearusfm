# Verifica avversaria di I-1 (r7-critico): soglia 1e-6 e cancellazione fra rank

Esito: **il rilievo regge in parte, ridimensionato a minore.** Il meccanismo che il revisore ipotizza, cioè una forte cancellazione sistematica fra rank sui tensori vettoriali, non ha un'origine nel modello CN. Resta però un rischio piccolo ma reale sui **due parametri scalari** del modello: per ciascuno, la probabilità di superare la soglia è circa 0,6-1,3%.

## Che cosa ho eseguito (simulazione CPU dichiarata)
- `agents/verifica-r7-critico-I-1-2/ring_tail.py`, output in `ring_tail.out`. La simulazione modella l'anello NCCL come rotazione ciclica dell'ordine di somma fp32 di 4 contributi già divisi per 4. Le rotazioni diverse rappresentano il chunk diverso dopo il cambio degli offset. La regola applicata è quella di compare.py:86/289. Per rank i contributi sono `mu + rumore` indipendente, con SNR 0, 0,5, 1 e 3.
  - n=1: P(rel>1e-6) = 0,0127 (SNR 0), 0,0090, 0,0057, 0,0019; massimo 1,5e-2.
  - n=5..54: 0 casi su 200.000 prove per SNR; massimo 5,1e-7.
  - n=1024 e n=147456: mediana circa 5e-8, massimo 6,4e-8.
- Controesempio scalare concreto (`scalar_example.out`). Contributi locali/4 = [1,15e-5, -3,42e-4, 8,44e-5, 2,53e-4]. Le due rotazioni danno 6,633556e-6 e 6,633549e-6, quindi rel = 1,097e-6 > 1e-6. Lo scarto viene solo dall'ordine della somma, eppure classify (compare.py:115-116) restituirebbe HYPOTHESIS_UNSUPPORTED.
- Inventario dei 270 parametri dalla config CN_A (modello istanziato sul Mac): 252 con almeno 100 elementi; 6×54, 3×32, 3×24, 16, 5, 6, 20; **2 scalari**, `anchor_heads.rms.bias` e `anchor_heads.env.bias`, entrambi fp32. Le teste di ancoraggio sono attive (`jepa.anchor_weight 0.2`). I tensori float64 (8) hanno un'unità di arrotondamento di 1e-16 e restano fuori pericolo.

## Che cosa deduco dal codice
- Una cancellazione sistematica fra rank richiede una correlazione negativa fra i gradienti locali. Nel CN ogni rank ha un batch proprio, la loss è locale e c'è un all_gather differenziabile (`contrastive_memory.py:336-353`, il backward somma con all_reduce). Non ci sono SyncBN né vincoli globali a somma zero. Con rumore indipendente, ||Σ|g_r|||/||Σg_r|| vale circa 1,7 in norma, e per i vettori la soglia resta lontana di un ordine di grandezza. Lo scenario vettoriale con un rel di 3e-4 usato da r7 (contributi opposti costruiti a mano) non ha corrispondenza nel modello.
- Per uno scalare non c'è media sugli elementi: basta che la somma dei 4 contributi cada per caso vicino a zero (coda di |N(0,2σ)|). Con i bias di regressione dopo 544 passi, quindi vicini all'ottimo e con un gradiente per rank dominato dal rumore del batch, siamo nel caso SNR circa 0-1.
- Nessun altro controllo intercetta il caso: `excessive` (compare.py:289) è un OR su tutti i tensori e precede le altre classi in classify. Il report conserva relative_l2 e max_abs per tensore (`details`), quindi un lettore vede che l'eccesso viene da uno scalare. Lo `status` però resta HYPOTHESIS_UNSUPPORTED secondo la regola congelata (decisioni.md:8405/8407, doc riga 17).
- Probabilità complessiva stimata: circa 1-2,5% (due scalari, e solo se l'elemento cambia davvero chunk fra il layout a 2 bucket e quello a 7). Per un diagnostico con un solo tentativo è un esito attribuito male, ma raro. Non è un bloccante: non c'è un controesempio sul run vero, solo una probabilità bassa.

## Gravità
**Minore**. È un rischio reale ma quantificato e basso, ristretto ai due scalari, e visibile nei dettagli del report. Il rilievo originale, «importante» per tensori con forte cancellazione, va ridimensionato.

## Correzione minima (prima della misura, da registrare in decisioni.md e nel doc)
Si usa nel falsificatore il denominatore di scala locale: `excessive |= nd > LIMIT * max(nx, ny, ||Σ_r |local_r|/4||, 1e-12)`. Per calcolarlo si accumula Σ_r|local_r| nel ciclo sui rank (compare.py:252-292), con un accumulatore per tensore, prima di `del tensors`. Si riportano entrambi i rapporti. In alternativa, più semplice: per i tensori con numel=1 l'eccesso si riporta in forma descrittiva e non smentisce l'ipotesi.

## Verifiche positive
- Per i tensori con almeno 5 elementi e contributi indipendenti, la soglia 1e-6 è conservativa: il massimo osservato è 5,1e-7 e il valore tipico 5e-8.
- I tensori float64 non sono a rischio.
