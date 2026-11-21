"""Constructive representation check; CPU forward only, no training or real data.

Teacher/reader are imported unchanged from the checked-in runtime. This is NOT
an admission test: it places the known teacher weights into the reader and only
establishes representability, not learnability from noisy finite data.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from torch import nn
from pose_temponet import controls, heads


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(20261121)
    x = rng.normal(size=(8, 160, 384)).astype(np.float16)
    metadata = [{"user": f"synthetic_user_{k}", "window": k} for k in range(8)]
    support = np.ones((8, 148, 20), dtype=bool)
    results = []
    for fold in (0, 1):
        c = controls.generate_components(x, metadata, support, np.arange(4), np.arange(4, 8),
                                        [m["user"] for m in metadata], outer_fold=fold)
        a = 0.4  # Both calibrated real-latent worlds used this coefficient.
        clean, _ = controls.synthetic_target(c, a, 0.0)
        student = heads.make_head("TCN_post", seed=0).eval()
        teacher_state = c["generator_states"]["T"]
        student_convs = [m for m in student.deep if isinstance(m, nn.Conv1d)]
        teacher_keys = ["0", "2", "4", "6", "8", "10"]
        l_norm, t_norm = c["normalization"]["L"], c["normalization"]["T"]
        with torch.no_grad():
            for i, (layer, key) in enumerate(zip(student_convs, teacher_keys)):
                layer.weight.zero_(); layer.bias.zero_()
                w, b = teacher_state[key + ".weight"], teacher_state[key + ".bias"]
                if i < 5:
                    layer.weight[:w.shape[0], :w.shape[1]].copy_(w)
                    layer.bias[:b.shape[0]].copy_(b)
                else:
                    scale = torch.tensor(t_norm["std"], dtype=torch.float32)
                    mean = torch.tensor(t_norm["mean"], dtype=torch.float32)
                    layer.weight[:, :8].copy_(a * w / scale[:, None, None])
                    layer.bias.copy_(a * (b - mean) / scale)
            ls = torch.tensor(l_norm["std"], dtype=torch.float32)
            lm = torch.tensor(l_norm["mean"], dtype=torch.float32)
            student.affine.weight.copy_(c["generator_states"]["L"]["weight"] / ls[:, None])
            student.affine.bias.copy_((c["generator_states"]["L"]["bias"] - lm) / ls)
            prediction = student(torch.tensor(x.astype(np.float32))).numpy().astype(np.float64)
        delta = prediction - clean
        central = delta[:, 6:-6]
        results.append({"fold": fold, "a": a, "max_abs_error_all": float(np.abs(delta).max()),
                        "max_abs_error_central": float(np.abs(central).max()),
                        "rmse_central": float(np.sqrt(np.mean(central**2))),
                        "clean_rms_central": float(np.sqrt(np.mean(clean[:, 6:-6]**2))),
                        "teacher_hidden_activation_fraction": c["hidden_active_fraction_fitting"],
                        "reader_parameters": sum(p.numel() for p in student.parameters()),
                        "teacher_state_hashes": c["generator_hashes"]})
        assert np.max(np.abs(delta)) < 5e-5, results[-1]
    result = {"purpose": "constructive representability only; no optimization/admission claim",
              "source": "unchanged runtime imports controls.generate_components and heads.make_head",
              "data": "8 synthetic random feature windows, not real latents",
              "operations": "known teacher state embedded into reader; no backward or optimizer",
              "dtype": "FP32 forward, float64 normalized target comparison",
              "teacher_width": 8, "reader_width": "64 then 32", "receptive_field_patches": 13,
              "dropout": "disabled at inference", "torch_version": torch.__version__,
              "numpy_version": np.__version__, "results": results}
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
