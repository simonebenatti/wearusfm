# Controprova R4 (impronte dei checkpoint a 576)

Esito: il rilievo regge nei fatti. Propongo gravità minore invece di importante.

## Verifiche fatte (worktree di release b58a66f6)
- Sequenza, righe 187-188: «le loro impronte e dimensioni sono già nelle ricevute e nel report».
- observe.py, righe 122-126: la ricevuta observation.rank*.json registra file, byte e sha256 di vectors.rank*.pt. Per i vettori la frase è vera.
- compare.py, riga 316: `endpoint_hashes[mode] = file_sha(actual)` è l'unico calcolo dello sha256 del file checkpoint.pt a 576. Entra nel report alle righe 326 e 386.
- compare.py, righe 401-405: se compare solleva un'eccezione, il report contiene solo status INCONCLUSIVE e reason. Se l'errore arriva prima della riga 316, le impronte mancano.
- contrastive_cn_restart.py, righe 167-175: se il job si ferma prima della fase compare (timeout, errore di una gamba), il report non viene scritto.
- run.py, righe 1843-1844: summary.json registra final_student_sha256 e final_teacher_sha256. Sono hash dello stato, non lo sha256 del file. Non sostituiscono l'impronta del file.
- Le dimensioni dei .pt restano coperte dal `ls -lR` del passo 7.

## Perché la gravità scende a minore
- I .pt restano su WORK, che non ha cancellazione automatica. Lo SHA si può calcolare in qualunque momento, anche al recupero del solo lettore (protocollo, riga 25).
- Nel ramo riuscito il report contiene le impronte. Per l'ordinaria, compare le confronta con il valore storico ENDPOINT.
- Seguire la frase alla lettera non fa fallire il lancio e non brucia un ID. Lascia solo una lacuna di documentazione nel ramo di errore.

## Correzione
Correggere la frase del passo 7 sui .pt. Se report.json manca o non contiene actual_endpoint_sha256, calcolare sul login lo sha256 dei checkpoint.pt delle due gambe e archiviarlo con i percorsi.
