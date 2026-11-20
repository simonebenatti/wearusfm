"""Traccia meccanica delle regole §4.2 (punti 1-6 e S-2) su curve SCRITTE A MANO.
Nessun modello, nessun dato, nessun training: le curve sono sequenze arbitrarie che servono solo a
esercitare la logica di selezione, estensione e allarmi come e' scritta nel protocollo 3e007c8.
Scelte di lettura dichiarate:
  - checkpoint ammessi: 1/4, 1/2, 3/4, 1, 2, ..., H (il passo 0 e' escluso da selezione e minimi);
  - argmin = checkpoint piu' precoce in caso di pareggio (P:66);
  - trigger a 150: argmin MSE o argmin E_dyn in 146..150 -> orizzonte 300 (P:66);
  - selezione: min MSE fra tutti i LR e checkpoint; pareggi: durata minore, poi LR minore (P:67);
  - bordo LR: se vince il bordo della griglia attiva e quel lato non e' esteso, si aggiunge 3e-5 / 1e-2 (P:68);
  - allarmi sul LR scelto (P:69): bordo LR dopo l'estensione di quel lato; checkpoint 1/4; argmin MSE o
    E_dyn in 296..300 se l'orizzonte e' 300;
  - S-2 (P:72): distanza > 5 epoche e regret >= 2% rispetto al minimo E_dyn dello stesso LR.
"""
import math

GRID0 = [1e-4, 3e-4, 1e-3, 3e-3]
LOW, HIGH = 3e-5, 1e-2

def ckpts(H):
    return [0.25, 0.5, 0.75] + list(range(1, H + 1))

def argmin(curve, cps):
    best = min(curve(c) for c in cps)
    return next(c for c in cps if curve(c) == best), best

def run_lr(world, lr):
    mse, edy = world[lr]
    c150 = ckpts(150)
    am, _ = argmin(mse, c150); ae, _ = argmin(edy, c150)
    H = 300 if (146 <= am <= 150) or (146 <= ae <= 150) else 150
    return {"lr": lr, "H": H, "trigger": (am, ae)}

def select(world, runs):
    cands = []
    for lr, r in runs.items():
        for c in ckpts(r["H"]):
            cands.append((world[lr][0](c), c, lr))
    cands.sort(key=lambda t: (t[0], t[1], t[2]))  # MSE, durata minore, LR minore
    return cands[0]

def procedure(world, nome):
    grid = list(GRID0); ext = {"low": False, "high": False}
    runs = {lr: run_lr(world, lr) for lr in grid}
    log = []
    while True:
        m, c, lr = select(world, runs)
        log.append(f"selezione: LR {lr:g} checkpoint {c} MSE {m:.4f} (griglia {[f'{g:g}' for g in sorted(grid)]})")
        if lr == min(grid) and not ext["low"] and LOW in world:
            grid.append(LOW); ext["low"] = True; runs[LOW] = run_lr(world, LOW); log.append("  bordo inferiore -> aggiungo 3e-5"); continue
        if lr == max(grid) and not ext["high"] and HIGH in world:
            grid.append(HIGH); ext["high"] = True; runs[HIGH] = run_lr(world, HIGH); log.append("  bordo superiore -> aggiungo 1e-2"); continue
        break
    r = runs[lr]; mse, edy = world[lr]; cps = ckpts(r["H"])
    alarms = []
    if (lr == min(grid) and ext["low"]) or (lr == max(grid) and ext["high"]):
        alarms.append("LR ancora al bordo dopo l'estensione")
    if c == 0.25:
        alarms.append("checkpoint selezionato al primo quarto")
    am, _ = argmin(mse, cps); ae, emin = argmin(edy, cps)
    if r["H"] == 300 and ((296 <= am <= 300) or (296 <= ae <= 300)):
        alarms.append(f"minimo MSE ({am}) o E_dyn ({ae}) nelle ultime 5 di 300")
    esel = edy(c)
    if emin == 0:
        regret = math.inf if esel > 0 else float("nan")
    else:
        regret = (esel - emin) / emin
    dist = abs(ae - c)
    s2 = (dist > 5) and (regret >= 0.02)  # nan >= 0.02 -> False
    unsel = [f"{g:g}" for g, rr in runs.items() if g != lr and rr["H"] == 300 and
             (296 <= argmin(world[g][0], ckpts(300))[0] <= 300 or 296 <= argmin(world[g][1], ckpts(300))[0] <= 300)]
    print(f"--- {nome}")
    for x in log: print("  " + x)
    print(f"  orizzonti: { {f'{g:g}': rr['H'] for g, rr in runs.items()} }; LR provati {len(runs)}")
    print(f"  LR scelto {lr:g}, durata {c} ep -> rifit: " + (f"{c} ep x S_full" if c >= 1 else f"ceil({c}*S_full)={math.ceil(c*60)} passi"))
    print(f"  allarmi punto 5: {alarms if alarms else 'nessuno'}")
    print(f"  S-2: minimo E_dyn a {ae} (E={emin:.4f}), scelto E={esel:.4f}, distanza {dist}, regret {regret if not isinstance(regret, float) else round(100*regret, 2) if math.isfinite(regret) else regret}% -> objective_tradeoff={s2}")
    print(f"  LR scartati al bordo di 300 (solo registrati): {unsel if unsel else 'nessuno'}")

flat = lambda v: (lambda c: v)
def valley(e0, v0, k):  # minimo interno a e0
    return lambda c: v0 + k * (c - e0) ** 2
def decreasing(v_end, slope):  # strettamente decrescente fino a 300
    return lambda c: v_end + slope * (300 - c)

# Caso A: MSE con minimo interno (ep 93), E_dyn ancora in discesa fino a 300 (tipo mondo V oltre 150)
wA = {lr: (flat(0.80), flat(0.80)) for lr in GRID0}
wA[1e-3] = (valley(93, 0.7689, 1e-6), decreasing(0.70, 1e-4))
procedure(wA, "A: MSE interno, E_dyn in discesa al bordo (LR interno vince)")

# Caso A': E_dyn con minimo a 148 e poi in risalita (mondo V se la dinamica si ferma)
wA2 = dict(wA); wA2[1e-3] = (valley(93, 0.7689, 1e-6), valley(148, 0.7370, 1e-6))
procedure(wA2, "A': MSE interno (93), E_dyn minimo a 148 poi risale")

# Caso B: E_dyn con minimo interno (ep 60), MSE al bordo (ep 148) poi minimo a 220
wB = {lr: (flat(0.80), flat(0.80)) for lr in GRID0}
wB[3e-4] = (lambda c: 0.75 + 1e-7 * (c - 220) ** 2 if c > 1 else 0.9, valley(60, 0.74, 1e-6))
procedure(wB, "B: E_dyn interno (60), MSE ancora in discesa a 150 (minimo poi a 220)")

# Caso C: vince il bordo superiore prima e dopo l'estensione
wC = {lr: (flat(0.80), flat(0.80)) for lr in GRID0}
wC[3e-3] = (valley(40, 0.76, 1e-6), valley(40, 0.76, 1e-6)); wC[HIGH] = (valley(30, 0.75, 1e-6), valley(30, 0.75, 1e-6)); wC[LOW] = (flat(0.9), flat(0.9))
procedure(wC, "C: bordo superiore prima e dopo l'estensione")

# Caso C': vince il bordo inferiore; dopo 3e-5 vince di nuovo 1e-4 (interno): risolto, 5 LR
wC2 = {lr: (flat(0.80), flat(0.80)) for lr in GRID0}
wC2[1e-4] = (valley(120, 0.76, 1e-6), valley(120, 0.76, 1e-6)); wC2[LOW] = (valley(140, 0.77, 1e-6), valley(140, 0.77, 1e-6)); wC2[HIGH] = (flat(0.9), flat(0.9))
procedure(wC2, "C': bordo inferiore, poi 1e-4 interno")

# Caso D: pareggio esatto fra 3e-4 e 1e-3 alla stessa durata
wD = {lr: (flat(0.80), flat(0.80)) for lr in GRID0}
wD[3e-4] = (valley(50, 0.76, 1e-6), valley(50, 0.76, 1e-6)); wD[1e-3] = (valley(50, 0.76, 1e-6), valley(50, 0.76, 1e-6))
procedure(wD, "D: pareggio esatto 3e-4 / 1e-3 all'epoca 50")

# Caso E: checkpoint scelto a 1/4 di epoca
wE = {lr: (lambda c: 0.75 + 0.01 * c, lambda c: 0.75 + 0.01 * c) for lr in GRID0}
wE[1e-3] = (lambda c: 0.70 + 0.01 * c, lambda c: 0.70 + 0.01 * c)
procedure(wE, "E: minimo MSE al primo quarto")

# Caso G: LR scartato ancora al bordo (1e-4 lento) mentre 3e-4 vince all'interno
wG = {lr: (flat(0.80), flat(0.80)) for lr in GRID0}
wG[1e-4] = (decreasing(0.78, 1e-5), decreasing(0.78, 1e-5)); wG[3e-4] = (valley(100, 0.76, 1e-6), valley(100, 0.76, 1e-6))
procedure(wG, "G: LR scartato 1e-4 ancora al bordo di 300")

# Caso H: S-2 con E_dyn_min = 0 (degenere) e con E_dyn = 0 anche al checkpoint scelto
wH = {lr: (flat(0.80), flat(0.80)) for lr in GRID0}
wH[1e-3] = (valley(20, 0.5, 1e-6), lambda c: 0.0 if c >= 40 else 0.1)
procedure(wH, "H1: E_dyn_min = 0 a 40, scelto a 20 con E_dyn 0,1")
wH2 = dict(wH); wH2[1e-3] = (valley(50, 0.5, 1e-6), lambda c: 0.0 if c >= 40 else 0.1)
procedure(wH2, "H2: E_dyn = 0 anche al checkpoint scelto (0/0)")

# Caso L: testa lineare veloce (tipo L0 del mondo V) con profilo piatto verso i LR bassi.
# Minimi pooled scritti a mano: 3e-3 0,7330 (q 1/4), 1e-3 0,7290 (q 3/4), 3e-4 0,7272 (ep 6), 1e-4 0,7269 (ep 17), 3e-5 0,7268 (ep 57).
def vq(c0, v0, k):
    return lambda c: v0 + k * (c - c0) ** 2
wL = {3e-3: (vq(0.25, 0.7330, 1e-3), vq(0.25, 0.7330, 1e-3)),
      1e-3: (vq(0.75, 0.7290, 1e-3), vq(0.75, 0.7290, 1e-3)),
      3e-4: (vq(6, 0.7272, 1e-5), vq(6, 0.7272, 1e-5)),
      1e-4: (vq(17, 0.7269, 1e-6), vq(14, 0.7255, 1e-6)),
      LOW: (vq(57, 0.7268, 1e-7), vq(47, 0.7254, 1e-7)),
      HIGH: (flat(0.9), flat(0.9))}
procedure(wL, "L: L0 veloce, profilo piatto verso i LR bassi (differenza 0,0001 fra 3e-5 e 1e-4)")
