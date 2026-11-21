"""Read-only analysis of archived control reports; no ML imports or fitting.

Run from any directory with Python 3. Outputs stay next to this script.
"""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
INPUT = ROOT / "results/infra/pose_temponet_execution_20261121_A1/launcher"


def values(s):
    return {"E_dyn": s["dynamic_sse"] / s["dynamic_target_ss"],
            "MSE": s["mse_sum"] / s["window_count"]}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


gate_path = INPUT / "positive_controls_gate_20261121.json"
gate = json.loads(gate_path.read_text())["gate"]
results = {"inputs": {str(gate_path.relative_to(ROOT)): sha(gate_path)},
           "method": "Recompute from JSON additive moments and pooled curves only; no fitting, no task-held scores.",
           "limitations": [
               "CV selection seed is 0; fold model initialization is shared, trajectories differ by fold.",
               "Saved full curves are pooled across two internal held folds, not per-fold curves.",
               "Fitting diagnostics exist only for the selected checkpoint and score the noisy fitted target.",
               "Thus no train/validation trajectory comparison or clean fitting error can be reconstructed.",
               "Absent optimizer alarms certify only the prespecified edge checks, not an optimum."],
           "heads": {}}

for p in sorted(INPUT.glob("selection_*.json")):
    d = json.loads(p.read_text())
    if "curves" not in d:
        continue
    results["inputs"][str(p.relative_to(ROOT))] = sha(p)
    r = {key: d[key] for key in ("world", "head", "selected", "optimizer_alarms",
                                "optimization_unresolved", "objective_tradeoff", "recipe")}
    r["fitting_at_selected"] = {f: values(s) for f, s in d["selected_fitting_diagnostics"].items()}
    r["outer_at_selected"] = d["selected_outer_values"]
    r["outer_clean_at_selected"] = d["selected_outer_clean_values"]
    r["curves"] = []
    for c in d["curves"]:
        selectable = [point for point in c["points"] if point["exposure"] > 0]
        best_mse = min(selectable, key=lambda point: (point["mse"], point["exposure"]))
        best_dynamic = min(selectable, key=lambda point: (point["edyn"], point["exposure"]))
        r["curves"].append({"lr": c["lr"], "horizon": c["horizon"],
                            "best_mse": best_mse, "best_dynamic": best_dynamic,
                            "last": c["points"][-1]})
    best = min(r["curves"], key=lambda c: (c["best_dynamic"]["edyn"], c["best_dynamic"]["exposure"], c["lr"]))
    r["best_dynamic_over_all_logged_checkpoints"] = {"lr": best["lr"], **best["best_dynamic"]}
    results["heads"][p.stem.removeprefix("selection_")] = r

refs = gate["reference_gate"]["temporal_records"]
tcn = results["heads"]["temporal_TCN_post"]
results["temporal_gate"] = {}
for fold, ref in refs.items():
    b, oracle = ref["clean_ls"], ref["oracle"]
    candidate = tcn["outer_at_selected"][fold]["E_dyn"]
    results["temporal_gate"][fold] = {
        **ref, "threshold_for_50pct_recovery": (b + oracle) / 2,
        "TCN_selected_E_dyn": candidate, "TCN_selected_recovery": (b - candidate) / (b - oracle)}
pool = results["temporal_gate"]["pooled"]
pool["best_logged_E_dyn"] = tcn["best_dynamic_over_all_logged_checkpoints"]["edyn"]
pool["best_logged_recovery"] = (pool["clean_ls"] - pool["best_logged_E_dyn"]) / (pool["clean_ls"] - pool["oracle"])
pool["TCN_relative_gain_vs_selected_L0"] = 1 - pool["TCN_selected_E_dyn"] / results["heads"]["temporal_L0"]["selected"]["edyn"]
pool["selection_MSE_instead_of_dynamic_cost_absolute"] = pool["TCN_selected_E_dyn"] - pool["best_logged_E_dyn"]
chosen_curve = next(c for c in tcn["curves"] if c["lr"] == tcn["selected"]["lr"])
pool["last_vs_selected_dynamic_relative_degradation"] = chosen_curve["last"]["edyn"] / tcn["selected"]["edyn"] - 1
pool["last_vs_selected_mse_relative_degradation"] = chosen_curve["last"]["mse"] / tcn["selected"]["mse"] - 1

results["closed_form_context_diagnostics"] = {}
for fold, diag in gate["linear_diagnostics"].items():
    result = {}
    for name in ("L0", "L13"):
        d = diag[name]
        result[name] = {
            "noisy_target_fit": {split: values(s["statistics"]) for split, s in d["worlds"]["temporal"].items()},
            "clean_target_fit_scored_on_noisy_fitting_target": values(d["temporal_margin_clean_fit_statistics"]),
            "clean_fit_temporal_recovery": d["temporal_margin_recovery_fitting"],
            "unread_T_fraction_fitting": d["T_unread_fraction_fitting"]}
    # Algebraic reconstruction is labelled; do NOT treat recovery on the fitted,
    # noisy target as a measure of learned clean temporal signal.
    baseline = result["L0"]["clean_target_fit_scored_on_noisy_fitting_target"]["E_dyn"]
    l13 = result["L13"]["clean_target_fit_scored_on_noisy_fitting_target"]["E_dyn"]
    oracle = baseline - (baseline - l13) / result["L13"]["clean_fit_temporal_recovery"]
    fitting_tcn = tcn["fitting_at_selected"][fold[-1]]["E_dyn"]
    result["derived_fitting_oracle"] = oracle
    result["TCN_recovery_on_seen_noisy_target_not_clean_recovery"] = (baseline - fitting_tcn) / (baseline - oracle)
    results["closed_form_context_diagnostics"][fold] = result

(OUT / "analysis.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

lines = ["# Diagnostica delle curve del controllo temporale", "",
         "Analisi dei soli JSON archiviati. Nessun training, invio, lettura del cluster o punteggio della posa vera.", "",
         "## Il risultato non dipende dalla scelta MSE invece di E_dyn", "",
         "| Pannello interno | Clean-LS | Oracolo | Limite per recuperare 50% | TCN selezionata | Recupero |",
         "|---|---:|---:|---:|---:|---:|"]
for fold, r in results["temporal_gate"].items():
    lines.append(f"| {fold} | {r['clean_ls']:.6f} | {r['oracle']:.6f} | {r['threshold_for_50pct_recovery']:.6f} | {r['TCN_selected_E_dyn']:.6f} | {100*r['TCN_selected_recovery']:.2f}% |")
lines += ["", f"Il minimo E_dyn fra **tutti** gli LR e checkpoint salvati è {pool['best_logged_E_dyn']:.9f}; recupera {100*pool['best_logged_recovery']:.2f}% del margine. Quindi nessun checkpoint registrato raggiunge il gate pooled. Il costo della selezione MSE è solo {pool['selection_MSE_instead_of_dynamic_cost_absolute']:.9f} di E_dyn.", "",
          "## Curve della TCN temporale", "",
          "| LR | Orizzonte | Epoca min MSE | Min MSE | E_dyn lì | Epoca min E_dyn | Min E_dyn | E_dyn finale |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
for c in tcn["curves"]:
    m, e, last = c["best_mse"], c["best_dynamic"], c["last"]
    lines.append(f"| {c['lr']:g} | {c['horizon']} | {m['exposure']:g} | {m['mse']:.6f} | {m['edyn']:.6f} | {e['exposure']:g} | {e['edyn']:.6f} | {last['edyn']:.6f} |")
lines += ["", "La selezione è LR 0,003, epoca 31; il minimo dinamico è all'epoca 27. Il LR 0,01 è stato provato e il LR 0,0001 ha già raggiunto 300 epoche. Non è il vecchio problema delle sole 50 epoche.",
          f"Alla fine delle 150 epoche del LR selezionato, E_dyn è peggiore del {100*pool['last_vs_selected_dynamic_relative_degradation']:.2f}% e MSE del {100*pool['last_vs_selected_mse_relative_degradation']:.2f}%. Questo sconsiglia una semplice continuazione; non dimostra che un altro schedule non possa aiutare.", "",
          "## Fitting e utenti interni esclusi, al checkpoint scelto", "",
          "| Testa | Fold | MSE fitting | MSE esclusi | E_dyn fitting | E_dyn esclusi | E_dyn esclusi contro target pulito |",
          "|---|---|---:|---:|---:|---:|---:|"]
for name in ("temporal_L0", "temporal_MLP", "temporal_TCN_post"):
    r = results["heads"][name]
    for fold in ("0", "1"):
        f, v, c = r["fitting_at_selected"][fold], r["outer_at_selected"]["fold" + fold], r["outer_clean_at_selected"]["fold" + fold]
        lines.append(f"| {name} | {fold} | {f['MSE']:.6f} | {v['MSE']:.6f} | {f['E_dyn']:.6f} | {v['E_dyn']:.6f} | {c['E_dyn']:.6f} |")
lines += ["", f"La TCN migliora E_dyn del {100*pool['TCN_relative_gain_vs_selected_L0']:.2f}% rispetto a L0 sulle finestre escluse. Quindi non è una testa che non impara: non recupera abbastanza del segnale temporale impiantato.",
          "Il divario fitting/esclusi e il peggioramento tardivo sono compatibili con un problema di generalizzazione, rumore e complessità del lettore. Non bastano per attribuire causalmente il fallimento a overfitting: mancano curve fitting e errori sul target pulito nel fitting.", "",
          "## Un indizio indipendente dal training neurale", "",
          "| Fold | Lettore in forma chiusa, fit su y | E_dyn fitting | E_dyn esclusi |",
          "|---|---|---:|---:|"]
for fold, r in results["closed_form_context_diagnostics"].items():
    for name in ("L0", "L13"):
        s = r[name]["noisy_target_fit"]
        lines.append(f"| {fold} | {name} | {s['fitting']['E_dyn']:.6f} | {s['held']['E_dyn']:.6f} |")
lines += ["", "L13 riduce l'errore sul fitting ma peggiora sugli utenti interni esclusi anche con soluzione in forma chiusa. È evidenza di un problema statistico del controllo con contesto, non una prova che la TCN abbia raggiunto il proprio ottimo.", "",
          "## Limiti e conseguenze", ""]
lines += ["- " + x for x in results["limitations"]]
lines += ["", "1. Non abbassare il gate e non cambiare la selezione dopo questi risultati. La ricetta corrente resta non ammessa; nessun risultato negativo sul compito reale o sull'encoder.",
          "2. Non proporre soltanto più epoche: lungo le traiettorie osservate il minimo è già interno. Un nuovo schedule o una nuova inizializzazione richiederebbero un emendamento e un controllo, non una garanzia di soluzione.",
          "3. Prima di nuovi tentativi, distinguere ottimizzazione/capacità e generalizzazione con lo stesso controllo temporale: errore pulito e rumoroso su fitting ed esclusi; traiettorie per fold, contributo del ramo lineare e di quello temporale; un controllo senza rumore, a architettura invariata. Sono proposte, non operazioni eseguite.",
          "4. Se la ricetta apprende il bersaglio pulito ma non generalizza sotto il rumore prescritto, valutare regolarizzazione e dimensione/struttura del lettore o più finestre indipendenti, mantenendo il gate e congelando la scelta prima del test reale. Se non apprende neppure il pulito, indagare architettura/ottimizzazione/supporto del bersaglio. La scelta dipende anche dalla costruzione del controllo, oggetto di un audit separato.", "",
          "Riproduzione: `python3 results/infra/pose_temponet_temporal_diagnosis_20261121/optimization/analyze.py`. Gli hash degli input e tutti i numeri non arrotondati sono in `analysis.json`.", ""]
(OUT / "README.md").write_text("\n".join(lines))
print(json.dumps({"outputs": [str(OUT / "analysis.json"), str(OUT / "README.md")],
                  "best_dynamic_recovery": pool["best_logged_recovery"]}))
