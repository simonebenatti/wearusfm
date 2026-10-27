"""Read-only reproductions for E2 pose evaluation; pass the reviewed checkout path.

These assertions document the defects in c5e42cb/d43dbcf, not desired behavior.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve() / "src"))
    from wearusfm.harness import pose_eval as p

    shape = (3, 4, 20)
    wave = np.asarray([-1, 1, -1, 1], dtype=np.float32)[None, :, None]
    target = np.broadcast_to(wave + np.asarray([1, 2, 100])[:, None, None], shape).copy()
    pred = target.copy()
    pred[2] += 100
    # First two windows are eight seconds apart in the same recording;
    # the third is in a different stage and has no eligible partner.
    meta = [dict(user="u", session="s", stage="A" if i < 2 else "B", side="left",
                 recording=0 if i < 2 else 1, start=i * 16000 if i < 2 else 0,
                 fs=2000) for i in range(3)]
    data = dict(meta=meta, lat=np.zeros((3, 4, 2)), target=target, pred=pred,
                mask=np.ones(shape, dtype=bool), std=np.ones((3, 20)),
                pose_var=target.var(axis=1).mean(axis=1), has_head=True,
                n_recordings=2, config={})
    report = p.report(data)
    reported_gap = report["E2"]["gap_mae_per_user"]["u"]
    partners, _ = p.misaligned_pairs(meta, target, data["mask"])
    keep = partners >= 0
    paired_mask = data["mask"][keep] & data["mask"][partners[keep]]
    aligned = np.abs(pred[keep] - target[keep])[paired_mask].mean()
    misaligned = np.abs(pred[keep] - target[partners[keep]])[paired_mask].mean()
    correct_gap = float(misaligned - aligned)
    assert np.isclose(reported_gap, -32.333333333333336), reported_gap
    assert np.isclose(correct_gap, 1.0), correct_gap
    print({"reported_gap": reported_gap, "correct_gap_on_matched_support": correct_gap})

    # Now two simultaneous recordings, one for each hand, in the same session.
    meta[1].update(side="right", recording=1, start=0)
    partners, info = p.misaligned_pairs(meta[:2], target[:2], data["mask"][:2])
    assert partners.tolist() == [1, 0], partners
    print({"opposite_hands_same_session_same_start_partners": partners.tolist(), "pairing": info})
    print("PASS: both current-behavior reproductions confirmed; checkout unchanged.")


if __name__ == "__main__":
    main()
