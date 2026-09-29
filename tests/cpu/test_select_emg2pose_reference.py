import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "select_emg2pose_reference", Path(__file__).resolve().parents[2] / "scripts" / "select_emg2pose_reference.py"
)
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)


def _csv(tmp_path, users=8, files=4):
    lines = ["session,user,stage,start,end,side,filename,moving_hand,held_out_user,held_out_stage,split,generalization"]
    for u in range(users):
        for j in range(files):
            split = "train" if j < files - 1 else "test"
            lines.append(f"s{u}{j},u{u},st,0,1,left,rec-u{u}-{j}_left.hdf5,both,False,False,{split},none")
    lines.append("sx,lonely,st,0,1,left,rec-lonely-0.hdf5,both,False,False,train,none")  # 1 sola registrazione
    p = tmp_path / "meta.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_select_is_deterministic_train_only_and_per_user(tmp_path):
    p = _csv(tmp_path)
    a = sel.select(p, n_users=5, per_user=3, seed=0)
    b = sel.select(p, n_users=5, per_user=3, seed=0)
    assert a == b and len(a) == 15
    assert len({u for u, _ in a}) == 5 and "lonely" not in {u for u, _ in a}
    assert all(m.startswith("emg2pose_data/rec-") and m.endswith(".hdf5") for _, m in a)
    assert all("-3_left" not in m for _, m in a)  # la quarta registrazione di ogni utente e' 'test'
    assert sel.select(p, n_users=5, per_user=3, seed=1) != a


def test_select_rejects_too_few_users(tmp_path):
    with pytest.raises(ValueError, match="ne servono"):
        sel.select(_csv(tmp_path, users=3), n_users=5, per_user=3, seed=0)
