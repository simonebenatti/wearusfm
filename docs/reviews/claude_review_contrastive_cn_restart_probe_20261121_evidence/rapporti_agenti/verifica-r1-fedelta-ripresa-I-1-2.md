# Verifica avversaria di I-1 (device_identity, ramo ctypes)

Esito: il rilievo regge come osservazione, ma va ridimensionato da importante a **minore**. Non c'e' un controesempio concreto.

## Che cosa ho eseguito
Script `agents/verifica-r1-fedelta-ripresa-I-1-2/sim_identity.py`. E' una SIMULAZIONE: libcuda finta, perche' il Mac non ha CUDA, e props senza `uuid`, come su torch 2.2. Chiama la vera `leg.device_identity("cuda:2")`.
- Con rc 0/0 la funzione restituisce `gpu_uuid='20212223-...-2e2f'` e `device_index=2`. La lettura via puntatori di `c_int` e `c_ubyte*16`, il passaggio di `c_int` per valore e `uuid.UUID(bytes=...)` sono corretti.
- Con rc diverso da 0, sia su cuDeviceGet sia su cuDeviceGetUuid, la funzione solleva `RuntimeError Unable to identify CUDA device UUID`. Lo scenario del revisore, nella sua meccanica, e' quindi vero.

## Che cosa deduco dal codice (non eseguito su CUDA)
- La chiamata avviene in `leg.py:160-163`, dentro `dispatch`, dopo il warm-up. Il codice gira nello stesso processo del trainer (`leg.main` -> `production.main`). A quel punto `run.py:1297-1298` ha gia' eseguito `torch.cuda.set_device`, e ci sono gia' stati il forward e l'inizializzazione di NCCL. Il runtime CUDA ha quindi gia' chiamato `cuInit` e caricato `libcuda.so.1`. `CDLL("libcuda.so.1")` riceve da dlopen l'oggetto gia' caricato, con lo stesso soname, e lo stato globale e' gia' inizializzato. `cuDeviceGet` richiede solo `cuInit`, non un contesto.
- `cuDeviceGetUuid` e' esportato da libcuda dalla versione 9.2 in poi. Il `_v2` del header e' un alias e il simbolo storico resta, per compatibilita' ABI. Torch 2.2 per CUDA 12 richiede comunque un driver recente.
- Con il default di ctypes (`restype c_int`, int Python -> `c_int`) le firme combaciano con `CUresult cuDeviceGet(CUdevice*, int)` e `cuDeviceGetUuid(CUuuid*, CUdevice)`. `device` e' la stringa `cuda:{local_rank}` (`run.py:1297`), quindi `index` non e' mai None.
- L'ordinale del driver e quello del runtime seguono lo stesso `CUDA_VISIBLE_DEVICES` e lo stesso ordinamento predefinito. Con MIG la chiamata riesce e restituisce l'UUID MIG; i nodi Booster non usano MIG.
- I casi di fallimento elencati («driver non inizializzato», «simbolo mancante») sono esclusi dallo stato del processo. Rimane solo un guasto generico del driver, che farebbe cadere anche il training stesso.
- Intercettazione: se la funzione fallisse, l'errore partirebbe prima della scrittura di `boundary.rank*.json`. Nessun altro controllo lo intercetta, e l'esito sarebbe `INCONCLUSIVE` tramite `execution_failure.json` (`contrastive_cn_restart.py`). La conseguenza e' giusta: non c'e' un'attribuzione errata, solo un tentativo perso.

## Gravita'
Minore. Il rischio e' teorico e non ho trovato un controesempio. Il codice pero' non e' mai stato eseguito su CUDA, e un fallimento costerebbe l'unico tentativo. Conviene quindi un irrobustimento a basso costo, non necessario per la release.

## Correzione minima (facoltativa)
1. Calcolare `device_identity` una sola volta in `leg.main`, oppure prima di `production.main` dopo `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))`, e riusarla in `dispatch`. Cosi' un guasto emerge in secondi, prima della validazione costosa.
2. Facoltativo: aggiungere `driver.cuInit(0)` esplicito. Non fa danni.
3. Attenzione alla correzione proposta dal revisore («gpu_uuid=None e si prosegue»). Da sola romperebbe `compare.py:297-298`, che richiede `all(p["gpu_uuid"])` e quattro UUID distinti, quindi l'esito sarebbe comunque INCONCLUSIVE. Se si adotta il ripiego, ci sono due strade:
   - il compare accetta `(host, CUDA_VISIBLE_DEVICES, device_index)` come chiave di posizionamento;
   - in alternativa, l'UUID viene ricavato dall'inventario `nvidia-smi` di `allocation.json` tramite `CUDA_VISIBLE_DEVICES[local_rank]`.

## Verifiche positive
- La meccanica ctypes regge nella simulazione, sia nel percorso di successo sia in quello di errore.
- `device` contiene sempre un indice.
- Il compare confronta lo stesso posizionamento fra le gambe (`compare.py:273-274`) e richiede un solo nodo con quattro UUID distinti.
