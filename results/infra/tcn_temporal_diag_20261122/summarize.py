"""Compact reading of the diagnostic results collected from Leonardo (D0, D1, D2), per the frozen protocol.

Usage: python summarize.py <folder with receipts/, d0/, d1/, d2/>. Reads JSON only, writes nothing.
"""
import json
import sys
from pathlib import Path

PANELS = ("fold0", "fold1", "pooled")


def pct(x):
    return "n.d." if x is None else "%.2f%%" % (100 * x)


def num(x, fmt="%.6f"):
    return "n.d." if x is None else fmt % x


def receipts(root):
    for path in sorted((root / "receipts").glob("*.json")):
        r = json.loads(path.read_text())
        d = r.get("details", {})
        print("%s: %s | gates %s | %.0f s | %s" % (r["job_id"], r["status"], r.get("gates"), d.get("elapsed_seconds", 0),
                                                 d.get("error", "")))


def shares(summary, split):
    rows = []
    for name in PANELS:
        e = summary[split]["noise_alignment"][name]["error_shares"]
        f = summary[split]["noise_alignment"][name]["noise_fit_fraction"]
        rows.append("%s E_dyn %s = pulito %s + rumore %s + allineamento %s; <e,n>/<n,n> %s" % (
            name, num(e["E_dyn"]), num(e["clean_error"]), num(e["noise_level"]), num(e["alignment"]), num(f, "%.4f")))
    return rows


def branches(per_fold, split):
    out = []
    for fold, row in enumerate(per_fold):
        b = row[split].get("branches")
        if not b:
            out.append("fold%d: nessun ramo profondo" % fold)
            continue
        last = b["relu_layers"][-1]
        dead = [layer["dead_units"] for layer in b["relu_layers"]]
        out.append("fold%d: unità mai attive per strato %s (ultimo %d/%d, attive %.3f); quota del ramo profondo "
                   "sull'energia centrata %s" % (fold, dead, last["dead_units"], last["units"], last["active_fraction"],
                                                   num(b["deep_share_of_centred_energy"], "%.3f")))
    return out


def d0(root):
    path = root / "d0" / "replay.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    print("\n== D0: parità %s, cancelli %s" % (r["parity_level"], r["gates"]))
    c = r["selected_comparison"]
    print("recuperi riletti (esclusi) %s, archiviati %s, limiti %s" % (
        {k: pct(v) for k, v in c["recovery_held"].items()}, {k: pct(v) for k, v in c["archived_recovery"].items()},
        {k: "%.2g" % v for k, v in c["recovery_bounds"].items()}))
    print("recuperi sul fitting allo stato scelto %s" % {k: pct(v) for k, v in c["recovery_fitting"].items()})
    for label in ("held_minus_fitting_selected", "held_minus_fitting_final"):
        print("%s:" % label)
        for name in PANELS:
            g = c[label][name]
            print("   %s ΔE_dyn %s = Δpulito %s + Δrumore %s + Δallineamento %s" % (
                name, num(g["E_dyn"]), num(g["clean_error"]), num(g["noise_level"]), num(g["alignment"])))
    for head, replay in r["replays"].items():
        print("-- %s (%s): parità %s" % (head, replay["scope"], replay["parity_level"]))
        for rec in replay["records"]:
            s = rec["summary"]
            print("   lr %g %s esp. %g: R esclusi %s | R fitting %s | E_dyn pulito esclusi %s, fitting %s" % (
                rec["lr"], rec["kind"], rec["exposure"],
                pct(s["held"]["recovery_on_observed"]["pooled"]["recovery"]),
                pct(s["fitting"]["recovery_on_observed"]["pooled"]["recovery"]),
                num(s["held"]["clean"]["pooled"]["E_dyn"]), num(s["fitting"]["clean"]["pooled"]["E_dyn"])))
            for line in branches(rec["per_fold"], "fitting"):
                print("      fitting " + line)
            scale = s["held"].get("clean_scale")
            if scale:
                print("      quota dell'errore pulito di cleanLS tolta (esclusi) %s" % {
                    k: pct(v["share_of_clean_ls_error_removed"]) for k, v in scale.items()})
            if rec["selected"]:
                for split in ("held", "fitting"):
                    for line in shares(s, split):
                        print("      %s %s" % (split, line))
    t = r.get("teacher_embedded_in_reader", {})
    if "summary" in t:
        s = t["summary"]
        print("-- generatore scritto nella testa: R esclusi %s | E_dyn pulito esclusi %s" % (
            {k: pct(s["held"]["recovery_on_observed"][k]["recovery"]) for k in PANELS},
            num(s["held"]["clean"]["pooled"]["E_dyn"], "%.3g")))
    else:
        print("-- generatore scritto nella testa: %s" % t.get("error"))
    refs = r["references"]
    cs = refs["references"].get("clean_scale", {})
    print("-- cleanLS ricalcolato: autocontrollo %s %s" % (cs.get("self_check_passed"), cs.get("error", "")))
    for row in refs.get("generator", []):
        print("-- generatore fold %s: unità attive per strato sul fitting %s" % (
            row["outer_fold"], [round(x, 3) for x in (row["hidden_active_fraction_fitting"] or [])]))


def d1(root):
    path = root / "d1" / "clean_fit.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    f = r["clean_fit"]
    print("\n== D1: lettura %s | parità del driver %s | cancelli %s" % (r["reading"], r["driver_parity_level"], r["gates"]))
    print("orizzonte %s (esteso %s), punto scelto %s, minimi %s, bordo %s" % (
        f["horizon"], f["extended_by_frozen_rule"], f["selected"], f["minimum_exposures"], f["late"]))
    print("R esclusi %s | R fitting %s" % ({k: pct(v) for k, v in r["recovery_held"].items()},
                                          {k: pct(v) for k, v in r["recovery_fitting"].items()}))
    for key, row in r["fixed_readings"].items():
        s = row["summary"]
        print("   esp. %s: R esclusi %s | R fitting %s | E_dyn pulito esclusi %s" % (
            key, pct(s["held"]["recovery_on_observed"]["pooled"]["recovery"]),
            pct(s["fitting"]["recovery_on_observed"]["pooled"]["recovery"]), num(s["held"]["clean"]["pooled"]["E_dyn"])))


def d2(root):
    path = root / "d2" / "noisy_extension.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    e = r["extension"]
    print("\n== D2: parità di partenza %s | cancelli %s" % (e["start_parity"].get("parity_level"), r["gates"]))
    if not e.get("records"):
        return
    print("punti nuovi %s; minimo E_dyn pooled nuovo %s (scelto archiviato %s); qualche epoca sotto: %s; soglia in tutti i "
          "pannelli: %s" % (e["new_points"], num(e["minimum_new_pooled_edyn"]), num(r["archived_selected"]["edyn"]),
                            e["any_new_pooled_edyn_below_archived_selected"],
                            e["any_new_point_reaches_threshold_in_every_panel"]))
    print("<e,n>/<n,n> pooled sul fitting a 150 e a 300: %s" % [num(x, "%.4f") for x in e["fitting_noise_fit_fraction_start_end"]])
    for rec in e["records"]:
        if rec["point"]["exposure"] in (150.0, 200.0, 250.0, 300.0):
            s = rec["summary"]
            print("   esp. %g: R esclusi %s | R fitting %s" % (rec["point"]["exposure"],
                  pct(s["held"]["recovery_on_observed"]["pooled"]["recovery"]),
                  pct(s["fitting"]["recovery_on_observed"]["pooled"]["recovery"])))


def d3(root):
    rows = {}
    for stage in ("d3t", "d3l", "d3s"):
        path = root / stage / "variant_summary.json"
        if path.exists():
            rows.update(json.loads(path.read_text())["rows"])
    if not rows:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from tcn_temporal_diag import core
    results = {}
    for row in rows.values():
        r = results.setdefault(row["weight_decay"], {"linear": {}, "valid": True})
        r["valid"] = r["valid"] and bool(row["reproduced"])
        if row["world"] == "temporal":
            r["temporal"], r["late"] = row["gate"], row["late"]
            r["selected"] = row["selected"]
        else:
            r["linear"][row["head"]] = row["gate"]
    print("\n== D3: esplorazione della dose di weight decay (LR 0,003, mondo originale)")
    for dose in sorted(results):
        r = results[dose]
        temporal = r.get("temporal")
        line = "wd %g:" % dose
        if temporal:
            line += " R temporale TCN %s, esposizione scelta %s, bordo %s;" % (
                {k: pct(v) for k, v in temporal["recovery"].items()}, r["selected"]["exposure"], r["late"])
        for head, gate in sorted(r["linear"].items()):
            worst = min(v for panel in gate["recovery"].values() for v in panel.values() if v is not None)
            line += " lineare %s %s (min %s);" % (head, gate["status"], pct(worst))
        print(line)
    complete = {d: r for d, r in results.items() if "temporal" in r and set(r["linear"]) == set(core.W.CONTROL_HEADS)}
    if len(complete) == len(core.VARIANT_DOSES):
        decision = core.choose_dose(complete)
        print("regola fissata prima: dose scelta %s, conferma su un mondo nuovo: %s" % (decision["chosen"], decision["confirm"]))
    else:
        print("regola non applicabile: dosi complete %s su %d" % (sorted(complete), len(core.VARIANT_DOSES)))


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    receipts(folder)
    d0(folder)
    d1(folder)
    d2(folder)
    d3(folder)
