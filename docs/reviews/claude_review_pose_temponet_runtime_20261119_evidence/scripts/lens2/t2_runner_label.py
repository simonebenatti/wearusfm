from pose_temponet import runner as R, controls as C

def stats(mse_sum, n, dsse, dss):
    return {"mse_sum": mse_sum, "window_count": n, "dynamic_sse": dsse, "dynamic_target_ss": dss}

def fold(f):
    # linear ridge recovers only 50% of constant->oracle margin -> closed-form non-diagnostic
    lin = {"constant": stats(10., 10, 10., 10.), "oracle": stats(0., 10, 0., 10.),
           "clean_ls": stats(5., 10, 5., 10.), "ridge": stats(5., 10, 5., 10.)}
    tmp = {"constant": stats(10., 10, 10., 10.), "oracle": stats(9.2, 10, 9.2, 10.),
           "clean_ls": stats(10., 10, 10., 10.)}
    return {"outer_fold": f, "calibration": {"status": "calibrated"}, "outer_scored_after_inner_selection": True,
            "outer_reference_statistics": {"linear": lin, "temporal": tmp}}

values = [fold(0), fold(1)]
print("closed-form status:", C.combine_reference_folds(values)["status"])
receipts = {"a": {"stage": "wls_calibration", "v": values[0]}, "b": {"stage": "wls_calibration", "v": values[1]}}
R._torch_artifact = lambda r, name: r["v"]
try:
    R._calibrations(receipts)
except Exception as e:
    print("raised:", type(e).__name__, "| GateNotPassed?", isinstance(e, R.GateNotPassed), "|", e)
