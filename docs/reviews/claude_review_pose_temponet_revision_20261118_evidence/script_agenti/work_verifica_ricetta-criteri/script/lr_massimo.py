# Combinatoria pura delle regole P:67-69 (nessun modello, nessun dato).
# Ogni LR ha un punteggio finale (minimo MSE pooled con i pareggi gia' risolti da durata e LR): basta un ordine totale.
# Si enumerano tutti gli ordini dei 6 LR possibili e si applica la regola: griglia iniziale di 4, aggiunta una volta per lato.
from itertools import permutations
LR = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]   # ordine crescente
init = ["1e-4", "3e-4", "1e-3", "3e-3"]
maxn, cases, alarm_lr = 0, 0, 0
both_sides = 0
for perm in permutations(range(6)):
    rank = {LR[i]: perm[i] for i in range(6)}          # rango piu' basso = MSE migliore
    grid = list(init); ext = {"low": False, "high": False}
    while True:
        win = min(grid, key=lambda l: rank[l])
        g = sorted(grid, key=LR.index)
        if win == g[0] and not ext["low"] and g[0] != "3e-5":
            grid.append("3e-5"); ext["low"] = True; continue
        if win == g[-1] and not ext["high"] and g[-1] != "1e-2":
            grid.append("1e-2"); ext["high"] = True; continue
        break
    g = sorted(grid, key=LR.index)
    at_edge_after_ext = (win == g[0] and ext["low"]) or (win == g[-1] and ext["high"])
    alarm_lr += at_edge_after_ext
    both_sides += ext["low"] and ext["high"]
    maxn = max(maxn, len(grid)); cases += 1
print("ordini enumerati:", cases)
print("massimo numero di LR provati:", maxn)
print("ordini con estensione di entrambi i lati:", both_sides)
print("ordini con allarme 'LR al bordo dopo estensione':", alarm_lr)
