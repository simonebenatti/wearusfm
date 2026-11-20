"""Verifica avversaria di T-07: rapporto fra eccesso residuo al LR piu' piccolo (3e-5) e margine di vittoria su 1e-4, per leggi
di plateau eccesso = c*LR^k (k = 0,5, 1, 2). Solo aritmetica. Scrive solo in questa cartella."""
OUT="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/verifica-tenuta/v_t07_plateau.out.txt"
L=[]
for k in (0.5,1.0,2.0):
    r=(3e-5)**k/((1e-4)**k-(3e-5)**k)
    L.append(f"k={k}: residuo/margine = {r:.3f}; con margine 0,014% -> residuo {0.014*r:.4f}%; con margine 1% -> {1*r:.3f}%")
open(OUT,"w").write("\n".join(L)+"\n"); print("\n".join(L))
