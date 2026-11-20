# Verifica aritmetica delle etichette di PROT:116-122 / PJ evaluation.outcome_priority (nessun dato del progetto).
import itertools

def label(g, lo, up):
    if g >= 2 and lo > 0: return 'beneficio_pratico_risolto'
    if g <= -2 and up < 0: return 'deterioramento_pratico'
    if up < 2: return 'beneficio_2pct_non_sostenuto'
    return 'inconclusivo'

def gain(e_ref, e_tcn):
    return 100 * (e_ref - e_tcn) / e_ref

# 1) Copertura: ogni terna (gain, lower, upper) ha esattamente una etichetta (per costruzione) - contiamo i casi limite
grid = [x / 4 for x in range(-24, 25)]  # -6..6 passo 0.25
cnt = {}
for g, lo, up in itertools.product(grid, grid, grid):
    if lo > up: continue
    cnt[label(g, lo, up)] = cnt.get(label(g, lo, up), 0) + 1
print('copertura griglia (lo<=up):', cnt, 'totale', sum(cnt.values()))

# 2) gain fra -2 e 2 con IC che esclude 0
for case in [(1.5, 0.5, 1.9), (1.5, 0.5, 2.5), (-1.5, -2.5, -0.5), (-1.5, -1.9, -0.8), (0.0, -1.0, 1.0)]:
    print('gain=%5.2f IC99=[%5.2f,%5.2f] ->' % case, label(*case))

# 3) bande che decidono la candidatura (contrasto 2 o 3)
print('banda A, MLP risolta migliore <2 pct: gain2=%.3f' % gain(0.740, 0.750), label(gain(0.740, 0.750), -1.9, -0.8))
print('banda B, MLP migliore >=2 pct non risolta: gain2=%.3f' % gain(0.731, 0.750), label(gain(0.731, 0.750), -5.0, 0.3))

# 4) esempi C-2 della review 32e5452 sotto la regola v2
E = dict(L0=0.785, L13=0.730, TCN=0.750, ridge=0.7859020004638595, head=0.7786289843513977, MLP=0.735)
for k in ['L0', 'MLP', 'L13', 'ridge', 'head']:
    print('contrasto TCN vs %-5s gain=%7.3f' % (k, gain(E[k], E['TCN'])))

# 5) punto estremo: stima fuori dall'IC percentile (possibile col percentile bootstrap)
print('gain=2.05 IC=[0.3,1.98] ->', label(2.05, 0.3, 1.98), '(precedenza: etichetta 1 anche con upper<2)')
