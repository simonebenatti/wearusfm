# TCN: diagnosi del controllo temporale e seguito proposto

Stato: proposta per review Claude, nessun nuovo job o cambiamento dei sorgenti. Base: d3e41d06afbc5d47cb988488784a4655f16584af, codice congelato a1ae9607681c1a181034d7f72770537054af419d. Restano validi esito e stop del 21/11.

## Che cosa e dimostrato

Il controllo non e il compito di posa reale. Sui latenti reali del solo train si costruisce un bersaglio noto: y_clean = L(z) + a T(z); y_observed = y_clean + sigma epsilon, con z = LN(lat). L e affine; T e una rete temporale congelata con hidden8, cinque ReLU, campo13 patch; epsilon e rumore AR(1) indipendente dal generatore, correlazione exp(-1/4) per passo. Nei due fold la calibrazione ha scelto a=0,4 e sigma=1,3. L e T sono normalizzati per giunto sul fitting; il coefficiente0,4 non equivale a una percentuale della varianza totale, perche le componenti possono essere correlate.

La TCN del lettore usa hidden64/32, affine diretta, LayerNorm apprendibile, due dropout0,1; circa73mila parametri. Non e l'intera TEMPONet raw del ramo Claude. Il generatore e piu stretto, ma strutturalmente contenuto nel lettore in valutazione. Il riferimento oracle conosce il generatore; la rete deve inferirne la funzione da etichette rumorose e da20 utenti di fitting per fold. Essere rappresentabile non garantisce recuperare il50% con questi dati e questa ricetta.

| Lettura pooled del controllo temporale | E_dyn |
|---|---:|
| riferimento clean_ls | 0,728479 |
| oracolo | 0,670659 |
| soglia della TCN per recuperare50% | 0,699569 |
| TCN scelta con MSE | 0,715926 |
| miglior E_dyn di TUTTE le curve salvate | 0,715826 |
| MLP scelta con MSE | 0,735130 |

La TCN migliora E_dyn del2,61% rispetto alla MLP; recupera21,71% del margine oracle rispetto a clean_ls. Anche scegliere retrospettivamente il minimo E_dyn di tutte le curve darebbe soltanto21,88%: non e una soluzione, non si cambia selettore. Le curve sono complete: LR1e-4 fino300epoche; 3e-4,1e-3,3e-3,1e-2 fino150. Il candidato scelto e LR0,003/esposizione31, non un training interrotto a31epoche. Nessun allarme di ottimizzazione del selettore non equivale alla prova di convergenza globale.

Al checkpoint scelto il pooled fitting e E_dyn0,685897/MSE1,698865, contro held dei fold0,715926/1,800145. E un divario descrittivo; fitting e held hanno rumore, utenti e denominatori differenti, quindi non e una dimostrazione isolata di overfitting. Sul bersaglio pulito, held E_dyn della TCN0,139236 contro MLP0,196055: anche questa lettura conferma apprendimento parziale. Sono tutti fold interni al train; nessuna lettura dei15 utenti di sviluppo.

## Prova costruttiva CPU eseguita

Script e risultato sotto results/infra/pose_temponet_execution_20261121_A1/analysis_20261122/. Usa i generatori produttivi con i due seed/fold previsti, feature artificiali FP16 di geometria160x384, e make_head produttivo. Inserisce la funzione L normalizzata nel ramo affine e la T normalizzata nei primi8 canali degli strati64/32, assorbendo medie, scale e coefficiente0,4 nell'uscita. LayerNorm identita affine; eval, dropout spento come nell'inferenza. Nessun fit, ottimizzatore o dato esterno.

Massimo errore assoluto crop6: 4,82e-6 e4,36e-6; RMSE relativo4,66e-7 in entrambi i casi. Conferma sperimentalmente l'inclusione della funzione nel lettore, gia prevista dal protocollo. NON attesta parita sugli input reali o su A100, capacita di apprendere quei pesi, ne ammissione al gate. I pesi del teacher non possono inizializzare il candidato che deve qualificarsi: sarebbe una scorciatoia del controllo.

## Diagnosi: fatti e ipotesi separate

Esclusi come spiegazione sufficiente: campo temporale troppo corto rispetto al generatore (stessoRF13), impossibilita della classe di funzioni, selezione MSE che nasconde un checkpoint dinamico sufficiente, collaudo interrotto troppo presto. Nessun difetto di implementazione dimostrato nella lettura di crop, maschere, normalizzazione o backpropagation. Non e una prova esaustiva dell'assenza di bug.

Ipotesi principale: difficolta statistica/di ottimizzazione di un ramo non lineare debole insieme al ramo affine, su latenti correlati e bersagli rumorosi. Non possiamo ancora separare difficolta di fit della funzione pulita, generalizzazione fra utenti, regolarizzazione e interazione tra rami. Dropout, weight decay, LayerNorm apprendibile e inizializzazione del ramo sono candidati da diagnosticare, non colpevoli dimostrati. La prova costruttiva vale in eval; il dropout rende diverso il problema di training. Non consiglio una griglia indiscriminata di loss, larghezze o LR, ne il riaddestramento del FM.

## Seguito finito proposto, da rivedere prima di implementare o inviare

1. **Diagnosi senza nuovi fit.** Sui soli artefatti gia pinnati: ricostruire esattamente y_clean/y_observed, rigiocare i checkpoint selezionati con dropout off, riportare fitting e held separatamente e decomporre errore totale, centrato nel tempo e di livello. Confrontare il contributo affine e quello deep, norme e deriva della LN. Le due branche non sono identificabili univocamente: il confronto con L e T e descrittivo, non una misura di informazione attribuibile. Registrare i numeri senza selezionare nuovi modelli. Estendere la prova di embedding ai componenti originali e agli stessi input/supporti verifica raggiungibilita numerica, non qualifica il candidato.
2. **Un controllo pulito per distinguere i problemi.** Se la ricostruzione e corretta, proporre una sola prova TCN sugli stessi y_clean senza epsilon, due fold originali, seed0, LR0,003, massimo150epoche, stessa testa e ricetta restante. Dose e letture fissate nel protocollo prima di inviare: fitting/held a esposizioni1,10,31,75,150, piu curve complete; nessuna estensione o ricerca aggiuntiva. Riportare errori rispetto alla funzione nota; non riutilizzare meccanicamente le soglie del controllo rumoroso (cambiano denominatori e margini). Il test e diagnostico, non sostituisce il gate originale. Fit pulito scarso indica difficolta di apprendimento/ottimizzazione o un difetto da cercare; fit buono e held scarso indica un problema di generalizzazione sul controllo; entrambi buoni ma controllo rumoroso negativo indirizza verso robustezza/statistica del problema rumoroso. Le combinazioni intermedie restano inconclusive.
3. **Solo dopo la diagnosi, un candidato motivato.** La mia prima opzione, se si conferma che le branche non si separano bene, e una lettura residuale: LN fissa, base affine/ridge stimata sul solo fitting con alpha selezionato nella CV interna; base congelata, ramo temporale allenato sul residuo. A inferenza si sommano base e correzione. Non si usano L/T noti per la base della ricetta candidata. Sottrarre la base non elimina epsilon e non aggiunge informazione: cambia l'ottimizzazione e la regolarizzazione, va dichiarato. E un nuovo protocollo con due vincoli congiunti, non un risultato automaticamente confrontabile con la ricetta precedente. Claude deve contestare se la diagnosi giustifica questi vincoli o se una modifica singola piu piccola e preferibile. Non aggiungere anche cambio di loss/dropout/larghezza o nuovi semi alla stessa prova.

Per il candidato restano obbligatori controlli lineare e temporale con soglie originali90% e50%, tutti i fold e pooled; stessa selezione MSE, nessun passaggio sui dati reali in caso di gate negativo. Va rieseguita almeno la qualificazione di tutte le teste che cambiano; se si impone una nuova ricetta comune, anche i relativi controlli. Nuova identita, piano, ID e benchmark/gate risorse per codice modificato; nessun riuso implicito di ricevute fuori dalla loro compatibilita revisionata.

Il solo controllo pulito a un LR/due fold dovrebbe costare molto meno della ricerca completa TCN da1,23GPUh: ordine0,2-0,4GPUh, NON preventivo ammesso e senza tetto firmato qui. Qualunque invio richiede protocollo con obiettivi/letture e preventivo misurato; i costi di estrazione dei diagnostici/inferenza e il codice operativo vanno contati separatamente. Nessun job nuovo e autorizzato da questa nota. La campagna A1 resta congelata e ferma.
