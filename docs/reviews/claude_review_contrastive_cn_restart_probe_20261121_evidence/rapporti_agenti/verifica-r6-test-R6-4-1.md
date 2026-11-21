# Verifica avversaria R6-4 (ramo ctypes di device_identity)

Esito: il rilievo regge solo come lacuna di test. Gravita' corretta: **minore** (dichiarata: importante). Non e' bloccante.

## Eseguito (simulazione sul Mac)
Ho scritto `agents/verifica-r6-test-R6-4-1/fakecuda.c` e l'ho compilato in `libfakecuda.dylib`. La libreria finta riproduce l'ABI del driver:
`int cuDeviceGet(int*, int)` e `int cuDeviceGetUuid(struct{char[16]}*, int)`, con una mappa dell'ordinale riordinata (2,0,3,1).
Nello script `sim.py` ho sostituito `ctypes.CDLL("libcuda.so.1")` con questa libreria e `torch.cuda.get_device_properties` con un oggetto senza `uuid`, come sul torch 2.2.
Poi ho chiamato il vero `leg.device_identity("cuda:r")`, con r = 0..3.
Esito: quattro UUID distinti, ben formati e coerenti con la mappa dell'ordinale. Con `cuda:7` si ottiene il RuntimeError previsto.
La conversione ctypes predefinita e' quindi corretta: `byref(c_int)` e `byref(c_ubyte*16)` come puntatori, `c_int` per valore e restype int.

## Dedotto dal codice
- In leg.py:163, `device` vale `f"cuda:{ctx.local_rank}"` (run.py:1297), quindi l'indice non e' mai None.
- Al passo 545 cuInit e' gia' stato chiamato: il modello e' su GPU, DDP ha gia' comunicato e `get_device_properties` ha inizializzato il runtime.
- `libcuda.so.1` e' gia' caricata nel processo da cudart. Il simbolo non versionato `cuDeviceGetUuid` resta esportato dai driver per l'ABI, e i nodi boost non usano MIG.
- compare.py:297-299 controlla soltanto che gli UUID non siano vuoti e che siano 4 distinti. Non li confronta con il formato `GPU-...` di nvidia-smi, quindi la differenza di prefisso non conta.
- Non ho trovato un controesempio concreto in cui il ramo fallisca sul cluster.
- La mutazione L21 (uuid costante) sopravvive perche' test_cn_restart_leg.py:84-88 sostituisce l'intera funzione. E' una lacuna di copertura, non un difetto di esecuzione.

## Correzione minima (facoltativa)
Aggiungere un test che sostituisca `ctypes.CDLL` con un finto libcuda e `get_device_properties` con un oggetto senza uuid, come in sim.py.
Il test deve verificare l'ordinale per rank, il formato e l'unicita' degli UUID. Il fallback da allocation.json non e' necessario.
