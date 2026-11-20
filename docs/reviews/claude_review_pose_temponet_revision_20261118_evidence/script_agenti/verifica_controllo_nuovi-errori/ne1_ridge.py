# NE-1: ridge a 10 e 20 utenti con due parametrizzazioni della penalita' (solo aritmetica in forma chiusa).
# Modello isotropo dei revisori (conti_rilettura.py): p=384 direzioni, segnale s=1/p per direzione,
# varianza OLS per direzione v(n) = sigma^2*K/n (n = numero di finestre), K_MSE=5, n_out=118000, n_in=n_out/2.
p=384; n_out=118000.0; n_in=n_out/2; K=5.0
s=1.0/p
def risk(n,alpha_sse,sigma,lam=1.0):
    # SSE + alpha*||b||^2 con design isotropo (autovalore per campione lam): shrink h = n*lam/(n*lam+alpha)
    v=sigma**2*K/n; h=n*lam/(n*lam+alpha_sse)
    return p*(s*(1-h)**2 + v*h**2)
for sigma in (2.5,3.0):
    v_out=sigma**2*K/n_out; v_in=sigma**2*K/n_in
    # alpha ottimo SSE: h*=s/(s+v) -> alpha = n*lam*v/s = sigma^2*K*lam/s (indipendente da n)
    a_sse_in=n_in*v_in/s; a_sse_out=n_out*v_out/s
    # loss media: (1/n)SSE + a'||b||^2 <=> SSE + n*a'||b||^2 ; a'* = alpha*/n
    ap_in=a_sse_in/n_in; ap_out=a_sse_out/n_out
    R_wls=1-p*v_out
    R_opt=1-risk(n_out,a_sse_out,sigma)
    R_sse_inner=1-risk(n_out,a_sse_in,sigma)            # alpha SSE scelto sugli inner, non riscalato
    R_mean_inner=1-risk(n_out,ap_in*n_out,sigma)         # alpha' (loss media) scelto sugli inner, non riscalato
    print(f"sigma={sigma}: alpha*_SSE inner={a_sse_in:.1f} outer={a_sse_out:.1f}; alpha'* loss media inner={ap_in:.3e} outer={ap_out:.3e} (rapporto {ap_in/ap_out:.2f})")
    print(f"   recupero MSE: WLS {R_wls:.4f} | ridge alpha ottimo {R_opt:.4f} | SSE alpha inner non riscalato {R_sse_inner:.4f} | loss media alpha inner non riscalato {R_mean_inner:.4f}")
