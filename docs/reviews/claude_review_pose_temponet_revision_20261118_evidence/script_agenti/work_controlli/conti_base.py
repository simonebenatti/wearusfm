# Conti in forma chiusa per la revisione dei controlli positivi (nessun dato, nessuna simulazione).
import math
rho = math.exp(-0.25)
print("rho = exp(-1/4) =", rho)
print("1-rho^2 =", 1-rho**2, " sqrt =", math.sqrt(1-rho**2))
# stazionarieta': Var(eps_t) = rho^2*1 + (1-rho^2)*1 = 1
print("Var stazionaria:", rho**2*1 + (1-rho**2)*1)
print("tempo di correlazione (patch):", -1/math.log(rho), "-> ms:", -1/math.log(rho)*25)
def sum_rho(n):
    return sum((n-abs(k))*rho**abs(k) for k in range(-(n-1), n))
for n in (148, 124, 160):
    S = sum_rho(n)
    var_mean = S/n**2
    print(f"n={n}: sum_ts rho^|t-s| = {S:.2f}; Var(media finestra) = {var_mean:.4f}; c_n = 1-Var = {1-var_mean:.4f}; inflazione v_T = S/n = {S/n:.3f}")
print("(1+rho)/(1-rho) =", (1+rho)/(1-rho))
for phi in (0.0, 0.5, 0.8, 0.95, 1.0):
    print(f"inflazione AR(phi)xAR(rho) phi={phi}: {(1+rho*phi)/(1-rho*phi):.2f}")
# semi
for f in (0,1):
    print(f"fold {f}: L {20261118+1000*f}, T {20261118+1000*f+1}, rumore {20261118+1000*f+2}")
# parametri di T (larghezze 8) e di g (64/32)
def conv(cin,cout,k): return cin*cout*k + cout
T = conv(384,8,1)+conv(8,8,3)*3+conv(8,8,1)+conv(8,20,1)
g = conv(384,64,1)+conv(64,64,3)*3+conv(64,32,1)+conv(32,20,1)
print("parametri T (hidden 8):", T, " di cui pesi:", 384*8+3*8*8*3+8*8+8*20)
print("parametri g TCN:", g, " + ramo affine 7700 + LN 768 =", g+7700+768)
print("RF:", 1+2*(1+2+3))
# elementi per fold e per giunto
print("validi per finestra-giunto (val, dinamiche): 20119920/(8327*20) =", 20119920/(8327*20))
print("validi per finestra-giunto (val, tutte): 41194560/(16620*20) =", 41194560/(16620*20))
nw = 1907/2
print("finestre per fold ~", nw, "; elementi validi per giunto ~", nw*124)
