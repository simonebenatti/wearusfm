"""Avvocato di GPT: casi d'esempio (numeri scritti a mano, nessun dato) per OB-3, OB-4 e BL-2.
Regole lette in P:116-122 e J:85-102 (prima condizione vera). Scrive solo nella cartella avvocato-gpt."""
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo/avvocato-gpt/etichette_casi.out.txt"
L = []
def lab(g, lo, up):
    if g >= 2 and lo > 0: return "1 beneficio_pratico_risolto"
    if g <= -2 and up < 0: return "2 deterioramento_pratico"
    if up < 2: return "3 beneficio_2pct_non_sostenuto"
    return "4 inconclusivo"
# OB-3 (a): comparatore «risolto migliore ma di meno del 2%» con limite inferiore oltre -2
for g, lo, up in ((-1.5, -1.9, -0.8), (-1.5, -2.6, -0.4), (-1.9, -3.4, -0.3)):
    L.append("OB-3(a) contrasto 2 gain %.1f IC99 [%.1f; %.1f] -> etichetta %s; veto (a)=%s; veto (b)=%s; "
             "deterioramento >=2%% escluso dall'IC: %s" % (g, lo, up, lab(g, lo, up), lab(g, lo, up).startswith("2"), up < 0, lo > -2))
# OB-4 con OB-3(b): allarme sulla TCN, contrasto 2 con upper99<0 ma etichetta 3
g, lo, up = -1.5, -1.9, -0.8
L.append("OB-4+OB-3(b) allarme TCN, contrasto 2 gain %.1f [%.1f; %.1f]: etichetta %s; veto (b) senza allarme=%s; "
         "con il testo di OB-4 resta escluso solo il deterioramento_pratico -> veto conservato=%s" % (g, lo, up, lab(g, lo, up), up < 0, lab(g, lo, up).startswith("2")))
# BL-2: coerenza interna del controesempio (TCN 55%, MLP 0,869, TCN 0,870) con banda 6-10%
for gfrac in (0.06, 0.08, 0.10):
    # (E_LS - 0.870) = 0.55 * gfrac * E_LS  ->  E_LS = 0.870 / (1 - 0.55*gfrac)
    Els = 0.870 / (1 - 0.55 * gfrac); Eor = Els * (1 - gfrac)
    L.append("BL-2 banda %.2f: E_LS=%.4f E_oracolo=%.4f margine=%.4f; recupero MLP=%.3f, TCN=%.3f" % (
        gfrac, Els, Eor, Els - Eor, (Els - 0.869) / (Els - Eor), (Els - 0.870) / (Els - Eor)))
# BL-2 + OB-1: casi in cui la clausola MLP fallisce o «non conta come superata» con un allarme aperto
L.append("BL-2/OB-1 caso A: TCN recupero 0,48 con minimo E_dyn all'epoca 299/300 (allarme), MLP 0,52 senza allarme: "
         "OB-1 da solo -> «ottimizzazione non risolta»; con la clausola al passo 2 e la precedenza di BL-2 -> «non diagnostico»")
L.append("BL-2/OB-1 caso B: TCN recupero 0,62 senza allarme, MLP 0,40 con allarme: clausola numericamente superata con q<=0,22, "
         "ma per BL-2 «non conta come superata» -> «non diagnostico» e arresto, benche' l'unico stato aperto sia l'allarme di una testa senza cancello")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
