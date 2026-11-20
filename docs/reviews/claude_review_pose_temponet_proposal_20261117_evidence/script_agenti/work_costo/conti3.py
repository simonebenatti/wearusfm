"""Conti aggiuntivi (secondo tentativo, dimensione 5): memoria proiettata, riferimenti storici, ordine di grandezza atteso.
Solo aritmetica su numeri letti nelle ricevute (citate nel rapporto) e sulle uscite dei benchmark sintetici di questo tentativo."""
GiB = 2**30
r = 18548 / 1907
print(f"r = 18548/1907 = {r:.6f}")

print("== ricevute storiche della valutazione lunga (2930d7d) ==")
bench_rss, fit_rss = 1383280640, 9513836544          # LONG-BENCH-pose (train-only, CPU) e LONG-FIT-G49046-pose (CPU, train+val)
print(f"  CPU: RSS benchmark train {bench_rss/GiB:.3f} GiB x r = {bench_rss*r/GiB:.3f} GiB proiettati; misurati nel fit {fit_rss/GiB:.3f} GiB;"
      f" sovrastima {bench_rss*r/fit_rss:.2f}x")
g_tr, g_va = 3097219072, 20192440320                  # estrazioni GPU LONG-G49046-pose-train / -val, peak host RSS
b = (g_va - g_tr) / (18548 - 1907)
a = g_tr - b * 1907
print(f"  GPU (estrazioni): RSS train {g_tr/GiB:.3f} GiB, val {g_va/GiB:.3f} GiB; rapporto {g_va/g_tr:.2f} (r = {r:.2f})")
print(f"     retta per i due punti: parte fissa {a/GiB:.3f} GiB, {b/1e6:.4f} MB per finestra")
print(f"     regola 'RSS_train x r' applicata alla GPU: {g_tr*r/GiB:.3f} GiB contro {g_va/GiB:.3f} misurati (sovrastima {g_tr*r/g_va:.2f}x)")
print(f"     0,85 x 32 GiB = {0.85*32:.1f} GiB (frazione 0,85 della lunga, BENCHMARK_POLICY memory_fraction)")
rv, pv = 1663 - 0, 0
print("  tempi storici: lettura+validazione train nel benchmark 5,94 s -> val stimata", f"{5.938640302978456*r:.1f} s;",
      "job di fit ridge completo (train+val, 2 profondita', 5 salvataggi) 184,7 s nello script, 199 s in sacct")

print("== ordine di grandezza atteso (non una misura su A100) ==")
steps = 75000
for ms in (4.1, 8.0, 20.0):
    print(f"  {steps} passi x {ms} ms = {steps*ms/1000:.0f} s = {steps*ms/3.6e6:.3f} h")
print("  passo eager TEMPONet raw su A100 senza vincolo deterministico: 4,1 ms (registro, voce TEMPONet DB8 collaudo 3);"
      " con deterministic=True 14,6 ms col CUDA graph contro 1,1 ms")
fit_lo, fit_hi = 0.135, 0.347   # proiezioni MPS: tentativo precedente (LETTO) e questo (ESEGUITO, macchina carica)
print(f"  fit (MPS sintetico): {fit_lo}-{fit_hi} h di solo calcolo; raddoppiati {2*fit_lo:.2f}-{2*fit_hi:.2f} h; margine sul tetto 3 h: {3/(2*fit_hi):.1f}-{3/(2*fit_lo):.1f}x")
gpu_lo, gpu_hi = 0.25, 0.6
cpu_lo, cpu_hi = 1.5, 3.5
print(f"  stima grossolana: GPU {gpu_lo}-{gpu_hi} GPUh (benchmark + fit + inferenza + avvii) = {8*gpu_lo:.0f}-{8*gpu_hi:.1f} ore locali;"
      f" CPU {cpu_lo}-{cpu_hi} core-ore; totale {8*gpu_lo+cpu_lo:.1f}-{8*gpu_hi+cpu_hi:.1f} ore locali contro il tetto di 44"
      f" ({44/(8*gpu_hi+cpu_hi):.1f}-{44/(8*gpu_lo+cpu_lo):.1f}x)")

print("== sforamento oltre il limite (OverTimeLimit 3 min) ==")
over = 184 / 3600  # 63593508: limite 1:30:00, durata 1:33:04
for gpu_jobs, cpu_list in ((3, (2, 8, 8, 2)), (7, (2, 8, 8, 2))):
    g = gpu_jobs * over
    c = sum(cpu_list) * over
    print(f"  {gpu_jobs} job GPU tutti in TIMEOUT: +{g:.3f} GPUh (riserva 0,25); job CPU {cpu_list}: +{c:.3f} core-ore (riserva 1)")
