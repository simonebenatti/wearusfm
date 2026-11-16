# Controllo interno della consegna contrastiva — 16/11/2026

Ambito: implementazione successiva alla review `924e19d`, per la nuova richiesta di Simone di confrontare entrambe le idee. Non è la review di Claude e non autorizza l'esecuzione.

Due letture indipendenti, sul disegno/strumenti e sul nucleo/runtime, non hanno trovato un nuovo difetto dimostrato che impedisca di pubblicare il pacchetto per Claude. Il percorso distribuito resta da verificare sperimentalmente: non si considera approvato per il cluster sulla sola algebra.

## Osservazioni chiuse nella consegna

- Il config conserva gli stessi due pesi calibrati in C0, CN e CR; cambia solo `arm`. Un peso non misurato resta nullo nei config di review, non assume silenziosamente il default tecnico 0,1.
- La calibrazione legge i primi 32 batch congiuntamente informativi entro il pannello fissato. Richiede 512 batch completi, zero update optimizer, identità iniziale/finale di student e teacher, quattro rank, sei worker per rank e copertura globale/per dataset.
- L'audit riproduce l'intercalazione dei 24 worker logici del loader. Il suo CLI ora verifica la ricetta congelata e i pin degli ingressi prima/dopo il replay, incluse le statistiche di posa. Non certifica proiezioni o gradienti dell'encoder.
- La politica dei duplicati RnC è descritta precisamente: rimuove la coppia dello stesso istante in entrambi i ruoli; non elimina le repliche dalla popolazione contro gli altri punti.
- Il pannello `fixed_eval` è dichiarato di sola loss di base e non aggiorna la memoria. Non si interpreta la sua curva come misura dell'obiettivo contrastivo totale.
- L'ingresso di campagna non può rilasciare training impostando soltanto uno stato e un walltime. La fase `training` è rifiutata finché i rapporti misurati dei tre benchmark e il validatore dei preventivi non saranno collegati e rivisti. Questo seguito operativo richiede le misure future; il trainer scientifico è implementato.

## Lettura statica delle riduzioni

InfoNCE usa il numeratore delle query locali e il numero globale delle finestre. RnC calcola uno scalare globale replicato: il denominatore nel contratto è il world-size, il backward del gather somma i contributi e la media DDP ricostruisce il gradiente globale. La diagnostica ripete la stessa normalizzazione prima della riduzione esplicita delle componenti. Teacher detached e commit delle chiavi pre-update risultano coerenti.

Questa lettura non sostituisce i sei test nuovi a più processi. In particolare l'interazione fra `autograd.grad`, activation checkpointing e backward DDP va eseguita, con rank senza posa e lunghezze differenti. Il cast delle maschere/target sui rank senza posa è incluso.

## Verifiche e confini

- Suite locale: **153 passati, 6 distribuiti deselezionati**, 150,08 s; un warning preesistente nel test della posa.
- Ultima riesecuzione dei due file modificati per ingresso/audit: **9 passati**, 2,30 s; è un sottoinsieme, non nove test aggiuntivi.
- Due test torchrun storici passati separatamente: controllano il percorso precedente, non il nuovo contrastivo.
- Test nuovi distribuiti non completati: la sandbox impedisce i socket locali; le richieste tecniche esterne sono state annullate e non ripetute.
- Nessuna CUDA/NCCL, misura di risorse, calibrazione sul train reale, nuova estrazione o sonda eseguita. Nessun job sottomesso.

Evidenza e SHA dei file: `results/infra/contrastive_comparison_20261116/verification.json`. Compatibilità C0 con il percorso storico verificata entro l'arrotondamento FP32; ripresa CN verificata bit per bit in CPU. Non estendere queste conclusioni al multi-GPU prima dei test.
