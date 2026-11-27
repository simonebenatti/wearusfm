"""C4 reviewer additions: end-to-end paths the 8c72655c suite does not reach (reviewer copy, not in the clone)."""
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/cpu"))
import test_cn_restart_end_to_end as e2e
reader = e2e.reader


def run_cli(tmp_path, monkeypatch, o, w, a, c):
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--ordinary", str(o), "--warm", str(w),
                                      "--historical-ordinary", str(a), "--historical-continuous", str(c),
                                      "--report", str(report)])
    reader.main()
    return reader.read(report)


def rewrite_rank(folder, rank, edit):
    """Edit one rank's vectors and keep receipt and grad pins consistent (as a real writer would)."""
    vec = folder / f"vectors.rank{rank}.pt"
    rec_path = folder / f"observation.rank{rank}.json"
    vectors = torch.load(str(vec), weights_only=True)
    edit(vectors)
    vec.unlink()
    torch.save(vectors, vec)
    rec = json.loads(rec_path.read_text())
    from contrastive_cn_restart_observe import _tensor_hash, _norm, _file_hash
    for row in rec["parameters"]:
        for s in ("local", "reduced"):
            t = vectors[s][row["id"]]
            row[s + "_sha256"], row[s + "_norm"] = _tensor_hash(t), _norm(t)
    rec["vectors"].update(bytes=vec.stat().st_size, sha256=_file_hash(vec))
    rec_path.write_text(json.dumps(rec))


@pytest.mark.parametrize("case, expected", [
    ("local_diff", "HYPOTHESIS_UNSUPPORTED"), ("no_discrepancy", "HYPOTHESIS_UNSUPPORTED"),
    ("first_update", "PARTIAL_SUPPORT"), ("dormant_differs_both", "INCONCLUSIVE"),
    ("ordinary_metrics", "INCONCLUSIVE"), ("near_limit_scalar", "HYPOTHESIS_UNSUPPORTED"),
])
def test_missing_paths(tmp_path, monkeypatch, case, expected):
    o, w, a, c = e2e.make_case(tmp_path, monkeypatch, "ordinary_rounding")
    if case == "local_diff":
        # Bucket storage is shared: change one element of a non-scalar local view on rank 2.
        rewrite_rank(w, 2, lambda v: v["local"][0].view(-1)[0].add_(1.))
    elif case == "no_discrepancy":
        for r in range(4):
            rewrite_rank(w, r, lambda v: v["reduced"][253].zero_())
    elif case == "first_update":
        rows = [json.loads(x) for x in (w / "metrics.jsonl").read_text().splitlines()]
        for row in rows:
            if row["step"] == 545:
                row["grad_norm"] = row["grad_norm"] * (1 + 1e-6)
        e2e.json_lines(w / "metrics.jsonl", rows)
    elif case == "dormant_differs_both":
        state = torch.load(str(w / "checkpoint.pt"), weights_only=False)
        state["cuda_rng"][2] = torch.full((16,), 99, dtype=torch.uint8)
        (w / "checkpoint.pt").unlink()
        torch.save(state, w / "checkpoint.pt")
    elif case == "ordinary_metrics":
        rows = [json.loads(x) for x in (o / "metrics.jsonl").read_text().splitlines()]
        rows[3]["grad_norm"] = rows[3]["grad_norm"] * (1 + 1e-6)
        e2e.json_lines(o / "metrics.jsonl", rows)
    elif case == "near_limit_scalar":
        # Screen denominator is (1+2^-24)/2: a difference of 2e-6 * that is 2x the frozen limit.
        for r in range(4):
            rewrite_rank(w, r, lambda v: v["reduced"][253].fill_(2e-6 * (1 + 2.**-24) / 2))
    if case in ("local_diff", "no_discrepancy", "near_limit_scalar"):
        # p.grad pins are in the receipt only through reduced_sha256; post_update unchanged.
        pass
    result = run_cli(tmp_path, monkeypatch, o, w, a, c)
    assert result["status"] == expected, result["reason"]
