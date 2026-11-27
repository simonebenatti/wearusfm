# Prova in sola memoria (memfd, nessun file su disco) della chiamata corretta di load_observation
# sul torch vero del cluster: torch.save di viste che condividono lo storage del gruppo, poi
# torch.load(str(percorso), map_location="cpu", mmap=True, weights_only=True).
import hashlib, os, sys
from pathlib import Path
import torch

def _tensor_hash(tensor):  # copiata da contrastive_cn_restart_observe.py (8c72655c)
    data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()

print("torch", torch.__version__, "| python", sys.version.split()[0])
g = torch.Generator().manual_seed(545)
# due gruppi: uno in precisione singola e uno in doppia, come i gruppi di costruzione del modello vero
buffers = {"local": [torch.randn(4096, generator=g, dtype=torch.float32), torch.randn(64, generator=g, dtype=torch.float64)],
           "reduced": [torch.randn(4096, generator=g, dtype=torch.float32), torch.randn(64, generator=g, dtype=torch.float64)]}
# (gruppo, forma, passo, offset in elementi): matrice, vettore, bias di un elemento, matrice trasposta (non contigua), doppia precisione
members = [(0, (16, 32), (32, 1), 0), (0, (32,), (1,), 512), (0, (1,), (1,), 544), (0, (8, 4), (1, 8), 600),
           (1, (6,), (1,), 0), (1, (1,), (1,), 6), (1, (5,), (1,), 7)]
vectors = {stage: [bufs[b].as_strided(shape, stride, off) for b, shape, stride, off in members] for stage, bufs in buffers.items()}
expected = {stage: [(str(v.dtype), list(v.shape), v.storage_offset(), _tensor_hash(v)) for v in vs] for stage, vs in vectors.items()}

fd = os.memfd_create("vectors.rank0.pt")
with os.fdopen(fd, "wb", closefd=False) as stream:
    torch.save(vectors, stream)
path = f"/proc/self/fd/{fd}"
print("byte salvati:", os.path.getsize(path))
try:
    torch.load(Path(path), map_location="cpu", mmap=True, weights_only=True)
    print("Path + mmap: ACCETTATO (inatteso sul torch 2.2)")
except Exception as e:
    print("Path + mmap:", type(e).__name__, "|", e)
loaded = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
print("str + mmap + weights_only: caricato | chiavi", sorted(loaded), "| lunghezze", {k: len(v) for k, v in loaded.items()})
ok = True
for stage in ("local", "reduced"):
    for i, t in enumerate(loaded[stage]):
        got = (str(t.dtype), list(t.shape), t.storage_offset(), _tensor_hash(t))
        same = got == expected[stage][i] and t.is_floating_point()
        ok &= same
        print(f"  {stage}[{i}] dtype {got[0]} forma {got[1]} offset {got[2]} uguale all'originale: {same}")
print("TUTTO UGUALE:", ok)
os.close(fd)
