import json, math, hashlib
from pathlib import Path
ROOT = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1")
L = ROOT/"launcher"
J = lambda name: json.loads((L/name).read_text())
def pool(records):
    return {k: (math.fsum(r[k] for r in records) if k != "window_count" else sum(r[k] for r in records))
            for k in ("mse_sum", "window_count", "dynamic_sse", "dynamic_target_ss")}
edyn = lambda r: r["dynamic_sse"]/r["dynamic_target_ss"]
mse = lambda r: r["mse_sum"]/r["window_count"]
rec = lambda b, c, r: (b-c)/(b-r)
cal = [J("calibration_0.json"), J("calibration_1.json")]
gate = J("positive_controls_gate_20261121.json")["gate"]

print("=== G. controllo lineare: riferimenti e teste (dagli accumulatori)")
refl = {}
for name in ("constant", "ridge", "oracle", "clean_ls"):
    rows = [c["outer_reference_statistics"]["linear"][name] for c in cal]
    refl[name] = {"fold0": rows[0], "fold1": rows[1], "pooled": pool(rows)}
for split in ("fold0", "fold1", "pooled"):
    for mname, m in (("MSE", mse), ("E_dyn", edyn)):
        c, r, o = (m(refl[k][split]) for k in ("constant", "ridge", "oracle"))
        print(f" riferimento {split} {mname}: costante {c:.9f} ridge {r:.9f} oracolo {o:.9f} -> recupero ridge {100*rec(c, r, o):.6f}% (gate JSON {100*gate['reference_gate']['linear_gate']['recovery'][split][mname]:.6f}%)")
allrec = []
for head in ("L0", "MLP", "TCN_post"):
    s = J(f"selection_linear_{head}.json")
    st = dict(s["selected_outer_statistics"]); st["pooled"] = pool([st["fold0"], st["fold1"]])
    for split in ("fold0", "fold1", "pooled"):
        for mname, m in (("MSE", mse), ("E_dyn", edyn)):
            c, r = m(refl["constant"][split]), m(refl["ridge"][split])
            v = rec(c, m(st[split]), r); allrec.append((v, head, split, mname))
            g = gate["linear_head_gates"][head]["recovery"][split][mname]
            print(f" {head:8s} {split:6s} {mname:5s}: testa {m(st[split]):.9f} recupero {100*v:.6f}%  (gate JSON {100*g:.6f}%, diff {abs(v-g):.2e})  target_ss {st[split]['dynamic_target_ss']!r} vs ref {refl['ridge'][split]['dynamic_target_ss']!r}")
    print(f"   {head}: status gate {gate['linear_head_gates'][head]['status']} alarms {gate['linear_head_gates'][head]['optimization_alarms']} selected {s['selected']}")
print(" minimo dei 18 recuperi lineari:", min(allrec), " massimo:", max(allrec))

print("\n=== H. ricevute dei sei controlli")
for w in ("linear", "temporal"):
    for h in ("L0", "MLP", "TCN_post"):
        name = f"TCNH-controls-{w}-{h}-20261120"
        r = J(name+".json")
        raw = (L/(name+".json")).read_bytes()
        sac = (L/(name+".sacct.tsv")).read_text().strip()
        hw = r["numeric_context"]["hardware"]
        print(f" {name}: sacct {sac} | status {r['status']} complete {r['complete']} stage {r['stage']} | source {r['source_sha256'][:12]} numeric {r['numeric_sha256'][:12]} kind {hw['kind']} gpu {hw['gpu_name']} cap {hw['gpu_capability']} bytes {hw['gpu_bytes']} cpu '{hw['cpu_model']}' intraop {r['numeric_context']['threads']['intraop']} | code_sha {r['code_sha'][:12]} reviewed {r['reviewed_code_sha'][:12]} | elapsed {r['details']['elapsed_seconds']:.1f}s gpus {r['details']['allocation']['gpus']} cpus {r['details']['allocation']['cpus']} | file sha256 {hashlib.sha256(raw).hexdigest()[:12]} bytes {len(raw)} | gates {r['gates']}")
plan = json.loads((ROOT/"plan_controls.json").read_text())
print(" plan_controls numeric_pins:", plan.get("numeric_pins"))
print(" gate source_sha256:", J("positive_controls_gate_20261121.json")["source_sha256"], "status", gate["status"])
acc = J("accounting_through_controls_20261121.json")
six = [r for r in acc["rows"] if r["slurm_id"] in ("65035622","65035623","65035626","65035628","65035629","65035631")]
print(" durate:", [(r["slurm_id"], r["seconds"], r["state"], r["exit_code"]) for r in six])
print(" GPUh sei controlli:", sum(r["seconds"] for r in six)/3600, " TCN temporale:", 4427/3600, " TCN lineare:", 3710/3600)
print(" tsv sacct:")
for line in (L/"sacct_through_controls_20261121.tsv").read_text().splitlines():
    f = line.split("|")
    if f[0] in ("65035622","65035623","65035626","65035628","65035629","65035631"):
        print("   ", f[0], f[1], f[2], f[3], "s", f[5])
