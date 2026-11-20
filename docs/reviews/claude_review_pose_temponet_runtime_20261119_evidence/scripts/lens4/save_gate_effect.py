"""Lens 4: effect of a one-off first save sample (dataset identity hash) on the neural GPU gate."""
import sys, copy
sys.path.insert(0, "tests/cpu")
from test_pose_temponet_resources import receipts
from pose_temponet.resources import gate_and_pack, GIB, HEADS
def run(first_fold, first_full, steady=.012, train=.30, ev=.45):
    cpu, gpu = receipts()
    for head in HEADS:
        for part, c in gpu["cells"][head].items():
            c["train_seconds"] = [train*(2 if part == "full" else 1)]*10
            c["eval_seconds"] = [ev*(2 if part == "full" else 1)]*10
            first = first_full if part == "full" else first_fold
            c["save_seconds"] = [steady+first] + [steady]*9
            c["load_state_seconds"] = [.01]*10
    g = gate_and_pack(cpu, gpu, work_free_bytes=20*GIB, pending_write_bytes=0, persistent_output_bytes=7*GIB)
    neural = sum(j["gpu_hours"] for j in g["jobs"] if j["phase"] == "neural")
    return round(neural, 2), g["gates"]["neural_gpu_hours"], g["all_passed"]
print("steady saves only           (neural GPUh, gate, all):", run(0, 0))
print("first save +0.05 s / +0.10 s (neural GPUh, gate, all):", run(.05, .10))
print("first save +0.10 s / +0.20 s (neural GPUh, gate, all):", run(.10, .20))
