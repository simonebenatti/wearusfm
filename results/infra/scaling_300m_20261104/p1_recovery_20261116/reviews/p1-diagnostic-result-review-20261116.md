# Review indipendente dell'esito diagnostico P1, 16/11/2026

**Esito tecnico conforme. Risultato diagnostico: NOT_REPRODUCED / READERS_CLEAR_IN_TEST.** Il job 64610181 ha completato il disegno prespecificato, senza riprodurre il difetto. La causa dei due fallimenti di P1 resta aperta: non è una validazione di `os.sync` come rimedio e non rilascia automaticamente una terza estrazione.

La ricostruzione è indipendente dal codice del diagnostico: script Python di sola libreria standard, 617 controlli sui piccoli artefatti raccolti, senza importare le funzioni di valutazione e senza accessi remoti. Sono disponibili `p1-diagnostic-result-review-20261116.py` e il relativo JSON dettagliato.

- Diciotto fixture, sei cicli nell'ordine registrato. Per tutte: digest dell'oracolo in memoria uguale ai due digest osservati, dimensione entro l'1% prespecificato, descriptor/percorso concordi, validazione NPZ positiva. Nessuna variazione di oggetto, size, mtime o ctime nei punti osservati, né attraverso la validazione. Le variazioni osservate riguardano soltanto atime, escluso dalla guardia originale.
- Le sei fixture aged DB2/DB3 sono state lette dopo 729,642–774,074 secondi dalla chiusura, oltre i 720 richiesti. Le 18 fixture occupano 27.719.878.320 byte, sotto 32 GB. Durate di scrittura 0,318–3,253 secondi: il disegno comprende anche scritture da circa 1–2 secondi, ma non ricostruisce tutto il contesto del producer GPU originario.
- Due giri sui sei input reali: 12 confronti integri, guardie di classificazione e posa senza rifiuti e identità stabili anche fra i giri. I pin disponibili sono rispettati. DB6 del primo fallimento e DB3 del secondo non avevano digest storici: le loro nuove impronte restano baseline diagnostiche. DB2 del P1-r1 conserva il suo pin di provenienza, ma non viene promosso a cache scientifica accettata.
- Manifest uguale byte per byte al rilascio; 228 hash delle sorgenti congelate ricontrollati sui blob Git del commit dichiarato. Hash di helper, runner, launcher e attestazione CPU coerenti con il pacchetto rivisto. Eventi: 408, sequenza continua e monotonicità rispettata. Hash del file eventi uguale al pin nel report, artefatti locali uguali ai byte registrati nella raccolta remota.
- `sacct` grezzo: diagnostico COMPLETED 0:0, 977 secondi, 1 GPU, 8 CPU, 32 GiB; 0,271389 GPU-ore e 2,171111 core-ore. Checkout COMPLETED 0:0, 41 secondi su 2 CPU, 0,022778 core-ore. Totale CPU di questi due job: 2,193889 core-ore. MaxRSS batch 14,235 GiB. Deadline applicativa e tetti rispettati.

La politica candidata ha accettato soltanto casi stabili in questo job; la sua gestione delle transizioni mtime/ctime è coperta dai mutanti CPU, non da una riproduzione reale avvenuta qui. Il nullo non discrimina fra un problema dipendente dal nodo/carico, dal producer reale o da un altro elemento non ricreato dalle fixture. Le guardie vecchie risultano sane nel campione osservato, senza garantire stabilità per un futuro percorso CPU di 43 ore.

**Conseguenza operativa:** mantenere fermo il rilascio automatico di P1 e dei lettori dipendenti. Pubblicare il nullo e la causa ancora aperta; il prossimo accertamento richiede una proposta esplicita. Nessun allargamento della conclusione a «Lustre sano», «sync risolve» o «P1 esistente accettabile».

Limite della review: confrontati report, eventi, log, piano, pin, blob Git e accounting raccolti. Non sono state rilette le grandi NPZ su Leonardo, né ricalcolati da remoto i loro digest o il contenuto sintetico.

Nota di confezionamento GPT: lo script pubblicato deriva il percorso del repository dalla propria posizione e legge la review precedente dal pacchetto; scrive il risultato nella directory temporanea di sistema. Sono cambiati solo questi percorsi, non i controlli. Rieseguito con 617 controlli conformi.
