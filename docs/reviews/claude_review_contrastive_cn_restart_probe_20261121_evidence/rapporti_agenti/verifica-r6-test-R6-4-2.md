# Verifica avversaria R6-4 (ramo ctypes di device_identity)

Esito: il rilievo regge come lacuna dei test, non come rischio reale per il run. Gravita' corretta: minore (dichiarata: importante).

## Eseguito
- Simulazione `agents/verifica-r6-test-R6-4-2/sim_ctypes.py`: props senza `uuid` come in torch 2.2, `ctypes.CDLL` sostituito da un finto libcuda
  con la semantica di cuDeviceGet/cuDeviceGetUuid. Con `cuda:0..3`: dlopen("libcuda.so.1"), ordinali 0..3, quattro UUID distinti,
  formato `uuid.UUID`, device_index 0..3. Con codice di errore 3: RuntimeError "Unable to identify CUDA device UUID". SIM OK.

## Dedotto dal codice
- leg.py:30-43: firma ctypes adeguata (CUresult int, CUdevice int, puntatori byref, buffer di 16 byte).
- libcuda.so.1 c'e' per forza: cudart lo carica per far funzionare torch.
- cuInit e' gia' fatto: alla chiamata (leg.py:163, dentro `dispatch`, al primo update reale) modello, DDP e NCCL sono gia' su GPU.
- I simboli cuDeviceGet e cuDeviceGetUuid sono esportati da ogni driver dalla 9.2 in poi. Lo stack e' CUDA 12.1 (summary.json).
- device = `cuda:{local_rank}` (run.py:1296-1298). torchrun --nproc_per_node=4 con ntasks=1: tutti i rank vedono le stesse 4 GPU,
  quindi gli ordinali sono distinti, gli UUID sono distinti e compare.py:297-299 passa.
- compare non confronta il formato dell'UUID con l'inventario di nvidia-smi: la differenza del prefisso "GPU-" non conta.
- Non ho trovato un controesempio concreto.

## Correzione (facoltativa)
Aggiungere un test con un finto CDLL come la simulazione. Il fallback su nvidia-smi aggiungerebbe altro codice non provato.
