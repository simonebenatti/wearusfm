# Ricalcolo dei numeri del gate R-1 dalle somme per utente della ricevuta FIT pinnata (sola lettura).
import json, sys, math
d = json.load(open(sys.argv[1]))
for cell in ["post_ridge", "trained_head", "pre_ridge"]:
    for grp in ["dynamic", "static", "all"]:
        g = d["cells"][cell]["analysis"]["groups"][grp]
        c = g["centered"]
        users = sorted(c["per_user"].keys())
        num = sum(c["per_user"][u]["sums"]["centered"] for u in users)
        den = sum(c["per_user"][u]["sums"]["vary"] for u in users)
        nw = sum(c["per_user"][u]["n_windows"] for u in users)
        nwj = sum(c["per_user"][u]["n_window_joints"] for u in users)
        ne = sum(c["per_user"][u]["n_valid_elements"] for u in users)
        print(f"{cell:13s} {grp:8s} E_dyn_ricalcolato={num/den!r} memorizzato={c['centered_mse_ratio']!r} "
              f"utenti={len(users)} n_selected_windows={g['n_selected_windows']} centered.n_windows={c['n_windows']} (somma {nw}) "
              f"coppie={c['n_window_joints']} (somma {nwj}) elementi={c['n_valid_elements']} (somma {ne})")
