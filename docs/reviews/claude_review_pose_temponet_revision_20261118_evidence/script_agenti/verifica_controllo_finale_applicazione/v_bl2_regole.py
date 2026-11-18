"""Verifica avversaria di P-01: regole di BL-2 nella bozza congelata (B2:167-178), casi A, A', B, C, D di AVV-05/VA-N1
(numeri d'esempio, nessun dato). recupero = (E_LS*-E)/(E_LS*-E_or); clausola numerica: rec_TCN - rec_MLP >= q.
Due letture di B2:176 «non conta come superata»: (i) equivale a «non passa» (allora B2:174 da' l'etichetta);
(ii) e' distinta da «non passa» (allora B2:174, alla lettera, non da' etichetta)."""
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/verifica-applicazione/v_bl2_regole.out.txt"
q = 0.10
casi = {"A": (0.48, True, 0.52, False), "A'": (0.30, True, 0.35, False), "B": (0.62, False, 0.40, True),
        "C": (0.30, False, 0.35, False), "D": (0.55, False, 0.56, False), "E (allarme TCN, clausola fallisce)": (0.55, True, 0.50, False),
        "F (allarme MLP, clausola fallisce)": (0.60, False, 0.55, True)}
L = ["q = %.2f" % q]
for nome, (rt, at, rm, am) in casi.items():
    if rt < 0.5:
        et = "stati per testa di OB-1: " + ("ottimizzazione non risolta" if at else "ricetta non ammessa")
        L.append("%-38s TCN %.2f all=%s MLP %.2f all=%s -> %s; B2:168 non si applica" % (nome, rt, at, rm, am, et)); continue
    num = (rt - rm) >= q
    non_conta = am  # B2:176
    for lettura in ("i", "ii"):
        if lettura == "i": non_passa = (not num) or non_conta
        else: non_passa = not num
        if non_passa: et = "ottimizzazione non risolta" if (at or am) else "non diagnostico"
        elif non_conta: et = "NESSUNA ETICHETTA (B2:174 alla lettera non copre «non conta come superata»)"
        else: et = "passa"
        b168 = ""
        if (not num) or (lettura == "i" and non_conta):
            b168 = "; B2:168 dice «il margine si legge in modo istantaneo»: MLP %.2f (minimo garantito 0,5-q = %.2f se la clausola numerica fallisce)%s" % (
                rm, 0.5 - q, "  <-- la clausola numerica PASSA: la frase non e' stabilita" if num else "")
        L.append("%-38s lettura (%s): clausola numerica %s, non conta=%s -> %s%s" % (nome, lettura, num, non_conta, et, b168))
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
