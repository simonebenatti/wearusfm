# Review richiesta: perche il controllo temporale TCN fallisce e come ripartire

Simone chiede una diagnosi e un seguito motivato, prima di nuove prove. Ti chiedo una review scientifica e del codice, non deployment o invii. La campagna A1 resta ferma al gate negativo; non abbassare le soglie, non alterare gli artefatti e non leggere i15 utenti di sviluppo.

Ramo GPT: origin/codex/pose-temponet-launcher-20261120. Parti dalla punta che contiene questa consegna; base della raccolta d3e41d06afbc5d47cb988488784a4655f16584af. Codice scientifico invariato a1ae9607681c1a181034d7f72770537054af419d, source_sha256 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea. Nessun nuovo job GPT dopo i sei controlli.

Leggi:
- docs/reviews/pose_temponet_controls_results_20261121.md;
- docs/reviews/pose_temponet_temporal_diagnosis_20261122.md, soprattutto la separazione fatti/ipotesi e il seguito proposto;
- results/infra/pose_temponet_execution_20261121_A1/launcher/positive_controls_gate_20261121.json, selection_temporal_TCN_post.json e le altre cinque selezioni, calibration_0.json e calibration_1.json;
- analysis_20261122/teacher_embedding_diagnostic.py e teacher_embedding_result.json sotto la stessa cartella di campagna;
- scripts/pose_temponet/{heads,controls,workflow,train,selection}.py e il protocollo congelato docs/pose_temponet_frozen_protocol_20261117.md.

Fatti da verificare indipendentemente:
1. Sei controlli COMPLETED0:0, sorgenti e contesto gpu8 attesi. Passano i tre lineari. Il controllo temporale TCN recupera24,17%/18,88%/21,71% del margine oracle nei due fold/pooled, richiesto50%; status recipe_not_admitted, senza optimizer_alarm TCN. Nessuna selezione reale o inferenza held inviata.
2. Pooled: clean_ls0,728479, oracle0,670659, TCN0,715926; per il gate servirebbe <=0,699569. La TCN migliora del2,61% rispetto alla MLP0,735130. Anche il minimo E_dyn retrospettivo su TUTTE le curve e0,715826: cambiare selettore non risolverebbe. Il candidato scelto a31epoche non significa che il job si e fermato a31: le curve150/300 sono complete.
3. Il controllo e y=L(z)+0,4T(z)+1,3epsilon in entrambi i fold; L/T normalizzati, epsilon AR(1). Teacher hidden8 e lettore hidden64/32 hanno lo stessoRF13. Ho inserito costruttivamente il teacher nel lettore produttivo senza fit: in eval, errore relativo CPU circa4,66e-7 su feature artificiali FP16. Non e una prova su cache reali/A100, ne un modo lecito per qualificare il candidato.
4. Al checkpoint scelto fitting E_dyn0,685897 contro held0,715926: divario descrittivo, non prova isolata di overfitting. La funzione pulita held e appresa in parte (E_dyn0,139236 della TCN contro0,196055 MLP). Nessun difetto di codice gia dimostrato; resta da separare ottimizzazione, rumore e generalizzazione.

Review richiesta:
A. Cerca un difetto concreto nella catena generazione-target, LN e supporto/crop, loss e selezione; verifica che il confronto clean_ls/oracle/TCN usi gli stessi pannelli. Se trovi un errore, indica file/riga e prova riproducibile. Non interpretare l'assenza di optimizer_alarm come convergenza matematica.
B. Controlla la prova di rappresentabilita, compreso assorbimento di media/scala, canali inattivi e dropout off; identifica cosa non puo dimostrare. Valuta anche se un generatore rappresentabile puo comunque essere difficile da apprendere da20 utenti rumorosi per fold: il gate non garantisce da solo la fattibilita statistica al budget scelto.
C. Valuta il percorso diagnostico finito proposto: prima rigiocare i checkpoint esistenti e decomporre gli errori senza nuovi fit; poi, solo con protocollo e preventivo, una TCN sul bersaglio pulito, stessi due fold/seed0/LR0,003, massimo150epoche, letture fissate. LR e scelta della prova sono informati dall'esito negativo: e diagnostica esplorativa, non conferma indipendente. Richiedo conclusioni condizionate a fit/held puliti, non un altro grid search; le soglie del controllo rumoroso non si trasferiscono automaticamente al pulito.
D. Contesta o approva come candidato condizionato la proposta residuale: LN fissa, base affine/ridge stimata e selezionata solo sul fitting/CV interna, base congelata, TCN sul residuo, somma a inferenza. Non e una correzione a costo zero: cambia ricetta e vincoli, non elimina rumore. Se una modifica singola piu piccola e scientificamente meglio motivata, proponila con il dato che la giustifica. Non mescolare anche loss/dropout/larghezza/semi nella stessa prova. Il candidato deve riqualificare le teste cambiate con soglie originali90%/50% e stessa selezione MSE; un gate negativo ferma ancora il seguito.
E. Indica quali ricevute e artefatti si possono riusare, quali vanno rigenerati con nuova identita/benchmark, e un preventivo distinto per diagnostici in inferenza, controllo pulito ed eventuale candidato. L'ordine0,2-0,4GPUh per la sola prova pulita e una stima, non un budget rilasciato. Non allargare il perimetro per far passare il controllo.

Consegna una review con verdetto distinto su diagnosi, verifiche proposte e candidato residuale, indicando blocchi e alternative. Puoi riprodurre le verifiche CPU locali e leggere le prove gia raccolte. Per ora nessuna modifica ai sorgenti congelati, nessun nuovo fit o invio, nessun contatto diretto con altri operatori. Registra e pubblica la review sul tuo ramo conservando gli SHA; Simone mi trasmettera il riferimento per convergere sul seguito.
