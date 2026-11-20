import sys, time
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura")
import torch
from heads import BUILD, D, J
print("thread torch:", torch.get_num_threads())
for T in (40, 160):
    x = torch.randn(32, T, D); y = torch.randn(32, T, J)
    for name in ("L0", "MLP", "L13", "TCN_post"):
        m = BUILD[name]().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        t0 = time.perf_counter()
        for _ in range(5):
            opt.zero_grad(); loss = (m(x)[:, 6:-6] - y[:, 6:-6]).pow(2).mean(); loss.backward(); opt.step()
        print(f"T={T} {name:9s} {(time.perf_counter() - t0) / 5 * 1000:.1f} ms/passo")
