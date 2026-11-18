"""Verifica avversaria dei conti dell'avvocato di GPT (solo aritmetica sulle formule degli agenti, nessun dato).
Modello del revisore (work_verifica_architettura-controlli/v_scan_lineare.py) e modello ridge della rilettura
(rilettura_fedelta/conti_rilettura.py). Scrive solo in questa cartella."""
import math
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo/verifica-avvocato-gpt/verifica_conti.out.txt"
L = []
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2
grid = [1, 1.5, 2, 2.5, 3]
fine = [round(0.5 + 0.1 * i, 1) for i in range(36)]

def inner(f, s, Kd):
    v = s * s; N = v * c_n
    return (N + v * p * Kd / n_in) / (f + N)

def rec(f, s, Kd, Kt):
    v = s * s
    return 1 - v * p * Kd / n_out / f, 1 - v * p * Kt / n_out

def pick(f, Er, Kd, g):
    c = [(abs(inner(f, s, Kd) - Er), s) for s in g if abs(inner(f, s, Kd) - Er) <= 0.05]
    return min(c)[1] if c else None

def s_cont(f, Er, Kd):
    a = c_n + p * Kd / n_in
    return math.sqrt(Er * f / (a - Er * c_n))

L.append("== AVV-01: controesempio di BL-1 con griglia attuale, griglia 0,1 e sigma continua")
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f, Er in ((0.5, 0.92), (0.5, 0.94), (0.5, 0.95)):
        row = []
        for name, g in (("attuale", grid), ("0,1", fine)):
            s = pick(f, Er, Kd, g)
            if s is None:
                row.append("%s: non calibrato" % name); continue
            Rd, Rm = rec(f, s, Kd, Kt)
            row.append("%s: sigma %.2f inner %.4f R_dyn %.4f R_MSE %.4f %s" % (name, s, inner(f, s, Kd), Rd, Rm, "PASSA" if min(Rd, Rm) >= 0.9 else "FALLISCE"))
        s = s_cont(f, Er, Kd); Rd, Rm = rec(f, s, Kd, Kt)
        row.append("continua: sigma %.3f R_dyn %.4f R_MSE %.4f %s" % (s, Rd, Rm, "PASSA" if min(Rd, Rm) >= 0.9 else "FALLISCE"))
        L.append("K=(%.1f;%.1f) f'=%.1f E_real=%.2f | %s" % (Kd, Kt, f, Er, " | ".join(row)))
# con sigma continua R_dyn non dipende da f': formula chiusa
L.append("R_dyn con sigma continua = 1 - E_real*(p*Kd/n_out)/(c_n*(1-E_real)+p*Kd/n_in): non dipende da f'")
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    b = p * Kd / n_out; a = p * Kd / n_in
    # soglia R_dyn = 0.9: E*b = 0.1*(c_n(1-E)+a) -> E*(b+0.1 c_n) = 0.1(c_n+a)
    E = 0.1 * (c_n + a) / (b + 0.1 * c_n)
    L.append("  K_dyn %.1f: R_dyn < 0,90 per E_real > %.4f (vincolo su E_dyn)" % (Kd, E))
    for f in (0.3, 0.4, 0.5, 0.6):
        # vincolo MSE: 1 - v p Kt/n_out >= 0.9 con v = Er f/(c_n + a - Er c_n)
        # v_max = 0.1 n_out/(p Kt); Er_max: Er f = v_max (c_n + a - Er c_n) -> Er = v_max (c_n+a)/(f + v_max c_n)
        vmax = 0.1 * n_out / (p * Kt)
        Em = vmax * (c_n + a) / (f + vmax * c_n)
        L.append("    f'=%.1f: vincolo MSE fallisce per E_real > %.4f; soglia complessiva %.4f" % (f, Em, min(E, Em)))

L.append("")
L.append("== AVV-04: ridge con penalita' sulla somma dei quadrati (alpha* = sigma^2 K / tau^2, indipendente da n)")
def risk(sk, v, c):
    sh = sk / (sk + c * v); return sk * (1 - sh) ** 2 + v * sh ** 2
for sig in (2.5, 3.0):
    v_out = sig * sig * 5.0 / n_out; v_in = 2 * v_out
    spec = [1 / 384] * 384
    ols = len(spec) * v_out
    # alpha scelto sugli inner (ottimo a n_in) con penalita' SSE: lo shrink a n_out e' quello ottimo a n_out (c=1)
    opt = sum(risk(sk, v_out, 1) for sk in spec)
    # parametrizzazione a loss media: lambda ottimo a n_in applicato a n_out equivale a c=2
    dbl = sum(risk(sk, v_out, 2) for sk in spec)
    L.append("  isotropo sigma %.1f: WLS %.4f; ridge SSE alpha inner (= ottimo outer) %.4f; loss media senza riscalare (c=2) %.4f" % (sig, 1 - ols, 1 - opt, 1 - dbl))
L.append("  Con la ridge a 0,908 e la soglia 0,90: una L0 neurale fra 0,898 e 0,900 riceve «ricetta non ammessa»; fra 0,900 e 0,908 passa.")

L.append("")
L.append("== AVV-02: con E_dyn misurata su y* l'oracolo (= y*) ha errore nullo")
for g_ in (0.06, 0.08, 0.10):
    L.append("  banda richiesta %.2f: guadagno (E_LS - 0)/E_LS = 1 per ogni E_LS > 0 -> mai in [0,06; 0,10]" % g_)

L.append("")
L.append("== AVV-05: regole di BL-2 (cancello) composte con OB-1; recupero = (E_LS-E)/(E_LS-E_or); clausola = rec_TCN - rec_MLP >= q")
def stato(rec_tcn, al_tcn, rec_mlp, al_mlp, q=0.10):
    # bozza: passo 2 'non diagnostico' (clausola MLP) prevale; con allarme MLP la clausola non conta come superata
    clausola_num = (rec_tcn - rec_mlp) >= q
    clausola = clausola_num and not al_mlp
    if not clausola:
        bozza = "non diagnostico"
    else:
        bozza = "passa" if rec_tcn >= 0.5 else ("ottimizzazione non risolta" if al_tcn else "ricetta non ammessa")
    ob1 = "passa" if rec_tcn >= 0.5 else ("ottimizzazione non risolta" if al_tcn else "ricetta non ammessa")
    return bozza, ob1, clausola_num
for nome, args in (("A (avvocato)", (0.48, True, 0.52, False)), ("A' (MLP bassa)", (0.30, True, 0.35, False)),
                   ("B (avvocato)", (0.62, False, 0.40, True)), ("C senza allarmi, TCN sotto 50%", (0.30, False, 0.35, False)),
                   ("D caso di controlli-2", (0.55, False, 0.56, False))):
    b, o, cn = stato(*args)
    L.append("  caso %s: TCN %.2f allarme %s, MLP %.2f allarme %s -> bozza «%s»; OB-1 senza clausola «%s»; clausola numerica %s" % (
        nome, args[0], args[1], args[2], args[3], b, o, cn))

open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
