# Quando l'alpha ottimo unico (penalita' sulla SSE) non dipende da n: solo aritmetica in forma chiusa.
# rischio(alpha) = sum_k s_k*(alpha/(n*l_k+alpha))^2 + (sig2K/n)*(n*l_k/(n*l_k+alpha))^2
import math
def risk(alpha,n,s,l,sig2K):
    return sum(sk*(alpha/(n*lk+alpha))**2 + (sig2K/n)*(n*lk/(n*lk+alpha))**2 for sk,lk in zip(s,l))
def best_alpha(n,s,l,sig2K):
    lo,hi=math.log(1e-3),math.log(1e9)
    for _ in range(200):
        a=lo+(hi-lo)/3; b=hi-(hi-lo)/3
        if risk(math.exp(a),n,s,l,sig2K)<risk(math.exp(b),n,s,l,sig2K): hi=b
        else: lo=a
    return math.exp((lo+hi)/2)
p=384; sig2K=6.25*5
cases={'segnale e disegno isotropi':([1/p]*p,[1.0]*p),
       'segnale anisotropo, disegno isotropo':([0.9/50]*50+[0.1/334]*334,[1.0]*p),
       'segnale isotropo, disegno anisotropo':([1/p]*p,[3.0]*50+[0.55]*334)}
for name,(s,l) in cases.items():
    a_in=best_alpha(59000.0,s,l,sig2K); a_out=best_alpha(118000.0,s,l,sig2K)
    r_opt=1-risk(a_out,118000.0,s,l,sig2K); r_in=1-risk(a_in,118000.0,s,l,sig2K)
    print(f"{name}: alpha* 10 utenti {a_in:.1f}, 20 utenti {a_out:.1f}, rapporto {a_in/a_out:.3f}; recupero con alpha 20 {r_opt:.4f}, con alpha 10 non riscalato {r_in:.4f}")
