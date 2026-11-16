"""Independent numerical checks of review D1/D2/D3, without training or signals.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. This script
does not import the runtime, comparator or reviewer implementation. It reads
historical identity metadata, gradient summaries and subject/class counts only.
Raw indices and per-subject counts are never emitted into the repository.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(repo, local, manifest_path=None):
    paths = {
        "manifest": manifest_path or local / "manifests_derived_20261015/derived_final/manifest_v1_1_nested_1.json.gz",
        "window_seconds": local / "manifests_derived_20261015/window_seconds_59425316.json",
        "metrics": local / "logs/review6_remote/control30m/metrics.jsonl",
        "classification": repo / "results/infra/logsquare_evaluation_20261107/final/classification/report.json.gz",
    }
    manifest = json.loads(gzip.decompress(paths["manifest"].read_bytes()))
    rows = [dict(zip(manifest["columns"], r)) for r in manifest["rows"]]
    rows = [r for r in rows if r["split"] == "pretraining"]
    windows = json.loads(paths["window_seconds"].read_text())["window_s"]
    assert sha(paths["window_seconds"]) == manifest["params"]["derived_inputs"]["window_seconds_sha256"]
    records = [json.loads(line) for line in paths["metrics"].read_text().splitlines()]
    report = json.loads(gzip.decompress(paths["classification"].read_bytes()))
    provenance = {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()}
    return rows, windows, records, report, provenance


def coverage(rows, window_seconds, records):
    """Reconstruct the full ideal-validity replay using independent FIFO maps.

    The log lacks stops. Recompute both 4 s and 1 s duration sensitivity on all
    queries; exact CapgMyo intervals are unnecessary because all same-recording
    candidates are excluded at either duration. Keep all duplicate queries in
    denominators but use one candidate per physical identity.
    """
    weights = defaultdict(float)
    time_weights = defaultdict(float)
    for r in rows:
        key = "/".join(str(r[k]) for k in ("dataset", "subject", "session"))
        time_weights[r["dataset"]] += r["weight"]
        duration = window_seconds[key]
        weights[r["dataset"]] += r["weight"] / duration if duration else 0.0
    normalizer = sum(weights.values())
    expected = sorted(d for d, w in weights.items() if w > 0)
    quota = {d: weights[d] / normalizer for d in expected}
    exempt = [d for d in expected if quota[d] * 32 * 256 < 33]
    memory = {d: {} for d in expected}
    recordings = defaultdict(dict)
    expiry = deque()
    per_step = []
    neg_hist = {dur: defaultdict(Counter) for dur in (1., 4.)}

    def remove(ident):
        session = ident[0]
        d = rows[session]["dataset"]
        memory[d].pop(ident)
        recordings[session].pop(ident)

    for step, record in enumerate(records):
        while expiry and step - expiry[0][0] > 256:
            old_step, ident = expiry.popleft()
            d = rows[ident[0]]["dataset"]
            if memory[d].get(ident) == old_step:
                remove(ident)
        current = list(dict.fromkeys(tuple(o) for o in record["origin"]))
        for ident in current:
            s = ident[0]
            d = rows[s]["dataset"]
            if ident in memory[d]:
                remove(ident)
            memory[d][ident] = step
            recordings[s][ident] = step
            expiry.append((step, ident))
        summaries = defaultdict(lambda: [0, 0, 0])
        for s, trial, start in record["origin"]:
            d = rows[s]["dataset"]
            fs = float(rows[s]["fs_hz"])
            t0 = start / fs
            different_record = len(memory[d]) - len(recordings[s])
            nums = {}
            for dur in (1., 4.):
                same_record = sum(
                    tr == trial and max(st / fs, t0) - min(st / fs + dur, t0 + dur) >= 4.
                    for _, tr, st in recordings[s]
                )
                nums[dur] = different_record + same_record
                if step >= 256:
                    neg_hist[dur][d][nums[dur]] += 1
            summaries[d][0] += 1
            summaries[d][1] += nums[4.] >= 32
            summaries[d][2] += nums[1.] >= 32
        # The current batch is included in snapshots before capacity truncation.
        for d in {rows[i[0]]["dataset"] for i in current}:
            while len(memory[d]) > 1024:
                remove(next(iter(memory[d])))
        per_step.append({"datasets": dict(summaries), "pose_active": record.get("pose_windows", 0) >= 1})

    def summarize(start, stop):
        counts = {d: [0, 0, 0] for d in expected}
        for r in per_step[start:stop]:
            for d, nums in r["datasets"].items():
                counts[d] = [a+b for a, b in zip(counts[d], nums)]
        total, eligible4, eligible1 = map(sum, zip(*counts.values()))
        failed = [d for d, (n, e4, _) in counts.items() if n < 32 or e4 / n < .8]
        pose_fraction = sum(r["pose_active"] for r in per_step[start:stop]) / (stop-start)
        return {"start_batch_inclusive": start, "stop_batch_exclusive": stop,
                "total_queries": total, "eligible_duration4": eligible4,
                "eligible_duration1": eligible1, "nce_fraction_duration4": eligible4 / total,
                "pose_active_fraction_proxy": pose_fraction, "datasets": counts,
                "failed_original": failed,
                "passed_after_manifest_exemption": not (set(failed)-set(exempt)) and eligible4 / total >= .9 and pose_fraction >= .9}

    blocks = [summarize(s, s+1024) for s in range(256, len(records)-1024+1, 1024)]
    hist = neg_hist[4.]["capgmyo"]
    cumulative = np.repeat(list(hist), list(hist.values()))
    return {"rule": "quota_window * batch_global * max_age < min_negatives + 1",
            "quota_definition": "sum(weight / frozen_window_seconds), normalized over pretraining rows; time_weighted loader",
            "quota_window": quota, "manifest_time_weight": dict(time_weights),
            "expected_windows_per_max_age": {d: quota[d]*32*256 for d in expected},
            "exempt_only_from_per_dataset_gates": exempt,
            "overall_includes_exempt": True,
            "calibration_panel": summarize(256, 512), "audit_panel": summarize(256, 1024),
            "full_postfill": summarize(256, len(records)), "complete_blocks": blocks,
            "blocks_pass_original": sum(not b["failed_original"] and b["nce_fraction_duration4"] >= .9 and b["pose_active_fraction_proxy"] >= .9 for b in blocks),
            "blocks_pass_exemption": sum(b["passed_after_manifest_exemption"] for b in blocks),
            "capgmyo_negative_histogram_duration4": dict(sorted(hist.items())),
            "capgmyo_negative_median": float(np.median(cumulative)),
            "limits": ["All candidates assumed QC/visibility valid; therefore an optimistic structural replay, not a real-loader audit.",
                       "Pose activity uses historical pose_windows >= 1, not actual RnC anchor validity.",
                       "Durations approximated as 1 or 4 seconds; new-run stream identity remains to verify.",
                       "Expected number of windows is not a count of distinct admissible keys; exemption is an a priori feasibility screen, not a guarantee."]}


def gradients(records):
    g = np.asarray([r["grad_norm"] for r in records], dtype=np.float64)
    g0 = float(g[:8].mean())
    reference = {"original": g0, "option1_end_ramp_updates257_512": float(g[256:512].mean()),
                 "option2_active_updates513_19618": float(g[512:].mean())}
    options = {}
    for name, ref in reference.items():
        rho = .1 * ref / g
        slices = {"first_full_513_1024": slice(512, 1024), "active_513_19618": slice(512, None),
                  "last100_19519_19618": slice(-100, None)}
        options[name] = {"reference_base_norm": ref, "multiplicative_factor": ref/g0,
                         "calibration_fraction": .1*ref/g0,
                         "proxy_rho": {k: {"mean_of_ratios": float(rho[s].mean()),
                                           "ratio_of_means": float(.1*ref/g[s].mean()),
                                           "median": float(np.median(rho[s]))} for k, s in slices.items()},
                         "ramp257_512_mean_proxy_rho": float((rho[256:512] * np.arange(1,257)/256).mean())}
    return {"updates": len(g), "baseline_first8_mean": g0, "options": options,
            "clipped_if_threshold1": int((g > 1).sum()),
            "recommended_default": "option2_active_updates513_19618",
            "recommendation_rationale": "Small auxiliary contribution over the full active run, using only preexisting data; option1 targets the ramp boundary and implies approximately twice the mean proxy ratio.",
            "limits": ["Historical grad_norm covers all parameters; calibration uses only shared parameters.",
                       "Every rho here assumes constant contrastive gradient norm relative to calibration and historical base trajectory.",
                       "These are scale scenarios, not measured contrastive norms or predictions of AdamW update ratios.",
                       "Arithmetic active mean gives ratio-of-means 0.1, but mean-of-ratios 0.11359; no exact time-average 10% guarantee."]}


def classification(report, n_boot):
    arms = ("T25-L", "T100-L", "T100-G", "T100-LQ")
    datasets = ("epn612", "ninapro_db2", "ninapro_db3", "ninapro_db6", "uci_emg")
    rng = np.random.default_rng(20261116)
    points, samples, dataset_reports = {}, {}, {}
    for ds in datasets:
        stats = [report["datasets"][ds]["paired_counts"][a] for a in arms]
        users = sorted(stats[0]["subjects"])
        den = np.asarray([stats[0]["subjects"][u]["n"] for u in users], dtype=np.float64)
        hits = [np.asarray([s["subjects"][u]["correct"] for u in users], dtype=np.float64) for s in stats]
        for stat in stats:
            assert sorted(stat["subjects"]) == users
            assert np.array_equal(den, [stat["subjects"][u]["n"] for u in users])
        draws = rng.integers(0, len(users), size=(n_boot, len(users)))
        values = np.empty((n_boot, len(arms)))
        for start in range(0, n_boot, 256):
            chunk = draws[start:start+256]
            d = den[chunk].sum(1)
            present = d > 0
            for j, correct in enumerate(hits):
                values[start:start+len(chunk), j] = np.divide(correct[chunk].sum(1), d, out=np.zeros_like(d), where=present).sum(1)/present.sum(1)
        total = den.sum(0)
        point = np.asarray([np.mean(h.sum(0)[total>0]/total[total>0]) for h in hits])
        points[ds], samples[ds] = point, values
        published = np.asarray([report["datasets"][ds]["cells"][a]["balanced_accuracy"] for a in arms])
        assert np.allclose(point, published, rtol=0, atol=1e-12)
        dataset_reports[ds] = {"subjects": len(users), "windows": int(den.sum()),
                               "balanced_accuracy_percent": dict(zip(arms, (100*point).tolist())),
                               "published_max_abs_difference": float(np.max(np.abs(point-published)))}
    for key, source, weights in (("P1", datasets[1:4], (8/13,3/13,2/13)), ("P2", ("epn612","uci_emg"), (.5,.5))):
        points[key] = sum(w*points[d] for w,d in zip(weights,source))
        samples[key] = sum(w*samples[d] for w,d in zip(weights,source))
    contrasts = {}
    summary = {}
    for endpoint in ("P1", "P2", "epn612", "uci_emg"):
        results = []
        for i,j in itertools.permutations(range(4),2):
            draw = 100*(samples[endpoint][:,i]-samples[endpoint][:,j])
            point = float(100*(points[endpoint][i]-points[endpoint][j]))
            lo,hi = np.quantile(draw,[.05/18,1-.05/18])
            results.append({"pair": [arms[i],arms[j]], "difference_pp": point,
                            "ci95_pp": np.quantile(draw,[.025,.975]).tolist(),
                            "ci_simultaneous_nominal_pp": [float(lo),float(hi)],
                            "ni_margin1_pp": bool(lo > -1), "useful_gain_half_pp": bool(point >= .5 and lo > 0),
                            "half_width_pp": float((hi-lo)/2), "positive_estimate_needed_for_ni1_pp": float(point-lo-1)})
        contrasts[endpoint]=results
        summary[endpoint]={"ordered_contrasts":len(results), "ni_margin1_count":sum(r["ni_margin1_pp"] for r in results),
                           "useful_gain_count":sum(r["useful_gain_half_pp"] for r in results),
                           "half_width_range_pp":[min(r["half_width_pp"] for r in results),max(r["half_width_pp"] for r in results)],
                           "ni_counts_if_margin_changed":{str(m):sum(r["ci_simultaneous_nominal_pp"][0]>-m for r in results) for m in (1.,2.,3.)}}
    shares=[]
    for i,j in itertools.combinations(range(4),2):
        v_uci=np.var(samples["uci_emg"][:,i]-samples["uci_emg"][:,j],ddof=1)
        v_epn=np.var(samples["epn612"][:,i]-samples["epn612"][:,j],ddof=1)
        shares.append({"pair":[arms[i],arms[j]],"uci_variance_fraction_independent_datasets":float(v_uci/(v_uci+v_epn))})
    return {"replicates":n_boot,"seed":20261116,"family_size_for_width_comparison":9,
            "method":"Independent paired subject bootstrap from sufficient counts; pooled macro recall, all draws retained; no imports from comparator or reviewer.",
            "datasets":dataset_reports,"summary":summary,"contrasts":contrasts,"p2_variance":shares,
            "recommendation":{"retain_historical_P2":"0.5 EPN612 + 0.5 UCI, name and composition unchanged; report descriptively.",
                              "decision_endpoints":["E_dyn","P1","EPN612"],
                              "uci":"descriptive endpoint, cannot claim broad classification noninferiority from EPN612 alone",
                              "margins":"Keep original endpoint-appropriate gain/NI margins; do not widen to match sampling error.",
                              "profiles":"Endpoint-specific benefits, material harm, bounded-small-effect, unresolved, and tradeoff; head-to-head recipe evidence, never a universal winner."},
            "limits":["Existing historic outcomes informed this amendment, which precedes all new-arm data; exploratory development selection must be disclosed.",
                      "Subject bootstrap is conditional on one trained model per arm and does not measure pretraining-seed variance.",
                      "Nominal Bonferroni percentile intervals do not guarantee actual 95% joint coverage.",
                      "No resimulation of review's power probabilities or pose triplet results; only classification widths/counts independently recomputed."]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument("--local",type=Path,default=Path.home()/"wearusfm_local")
    parser.add_argument("--manifest",type=Path,help="Exact execution manifest; omit to use the archived historical local copy")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--n-boot",type=int,default=100000)
    args=parser.parse_args()
    rows,windows,records,report,provenance=read_inputs(args.repo,args.local,args.manifest)
    result={"status":"SCIENTIFIC_PROPOSAL_NOT_AUTHORIZATION","provenance":provenance,
            "D1":coverage(rows,windows,records),"D3":gradients(records),"D2":classification(report,args.n_boot)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"D1_exempt":result["D1"]["exempt_only_from_per_dataset_gates"],
                      "D1_blocks_pass":result["D1"]["blocks_pass_exemption"],"D2":result["D2"]["summary"]},indent=2))


if __name__=="__main__":
    main()
