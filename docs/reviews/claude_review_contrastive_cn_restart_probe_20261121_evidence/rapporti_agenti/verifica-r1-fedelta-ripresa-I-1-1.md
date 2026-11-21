# Verifica avversaria di I-1 (device_identity via libcuda)

Verdetto: il rilievo regge nei fatti, ma lo declasso da importante a **minore**. Non c'e' un controesempio concreto di fallimento sul cluster. La correzione proposta e' incompleta: con `gpu_uuid=None` il confronto fallirebbe comunque.

## Fatti confermati
- Su torch 2.2 la struttura `_CudaDeviceProperties` non ha `uuid`. Lo mostra `torch22_api_20261121.out.txt:4`, che elenca solo gcnArchName, is_integrated, is_multi_gpu_board, major, max_threads_per_multi_processor, minor, multi_processor_count, name e total_memory. Quindi gira sempre il ramo ctypes (`contrastive_cn_restart_leg.py:34-41`).
- Il ramo non ha precedenti nel repository: `grep libcuda|cuDeviceGetUuid` trova solo questo file. I test lo sostituiscono con `cpu_identity` (`tests/cpu/test_cn_restart_leg.py:85-88`).
- La chiamata avviene dentro `dispatch`, al primo backward del 545 (`leg.py:160-163`), dopo l'avvio e la validazione. Un errore solleva un'eccezione nel rank prima della scrittura di `boundary.rank*.json`, la gamba fallisce e l'esito e' INCONCLUSIVE.

## Perche' il fallimento non e' dimostrato (dedotto dal codice e dall'API del driver)
1. **Indice del dispositivo.** `device` vale `f"cuda:{ctx.local_rank}"` (`src/wearusfm/training/run.py:1297-1298`), quindi `index` e' un int e mai None. L'ordinale del driver e quello del runtime rispettano entrambi CUDA_VISIBLE_DEVICES.
2. **Firme.** Il ritorno predefinito di ctypes e' int, il che va bene per CUresult. `byref(c_int)` corrisponde a `CUdevice*`, `byref(c_ubyte*16)` a `CUuuid*`, e un `c_int` passato per valore a `CUdevice`. Ho eseguito una simulazione sul Mac (`agents/verifica-r1-fedelta-ripresa-I-1-1/sim_device_identity.py`), con props senza uuid e una libcuda finta. Il percorso con rc=0 restituisce `gpu_uuid` formattato, `device_index` 2 e il nome. Con rc!=0 restituisce `RuntimeError Unable to identify CUDA device UUID`. Gli argomenti arrivano come `int 2` e `c_int 2`. E' una simulazione: non prova il driver vero.
3. **cuInit.** Al 545 il processo ha gia' un contesto CUDA: set_device, modello e ottimizzatore sono sulla GPU e NCCL e' inizializzato. `CDLL("libcuda.so.1")` riprende per soname la libreria gia' caricata dal runtime, quindi lo stato inizializzato e' condiviso. Il caso CUDA_ERROR_NOT_INITIALIZED non e' plausibile.
4. **Simbolo.** `cuDeviceGetUuid` esiste nel driver da CUDA 9.2. Le versioni `_v2` sono aggiunte, non lo sostituiscono. Se il simbolo mancasse, l'errore sarebbe un AttributeError e non un RuntimeError, ma con lo stesso effetto.
5. **MIG.** Le A100 di boost_usr_prod non risultano partizionate. Anche con MIG, `cuDeviceGetUuid` riesce e restituisce l'identita' del dispositivo visibile.

Nessun altro controllo intercetta il problema prima del 545: `allocation.json` contiene l'inventario nvidia-smi (`contrastive_cn_restart.py:159-163`), ma non lo confronta con `gpu_uuid`.

## Correzione proposta del revisore: incompleta
`compare.py:297-299` richiede `all(p["gpu_uuid"] ...)` e quattro uuid distinti. Registrare `gpu_uuid=None` sposterebbe quindi l'INCONCLUSIVE nel confronto. Anticipare la chiamata all'avvio fa fallire prima, ma il tentativo resta perso comunque.

## Correzione minima che consiglio (economica, facoltativa)
- In `device_identity`, catturare OSError, AttributeError e rc!=0 e registrare `gpu_uuid=None` con `gpu_uuid_error`, `cuda_visible_devices` e `local_rank`.
- In `compare.py:297-299`, quando un uuid manca, accettare l'unicita' su (host, CUDA_VISIBLE_DEVICES, device_index) e tenere la richiesta degli uuid distinti quando sono presenti.
- In alternativa, dichiarare nel documento che il rischio e' accettato.

## Verifiche positive
- Le firme ctypes e la conversione dei tipi sono corrette.
- Il dispositivo ha un indice esplicito per rank.
- Il confronto tra le gambe su host, device_index, device_name e gpu_uuid (`compare.py:273-275`) e' coerente con quello che `device_identity` produce.
