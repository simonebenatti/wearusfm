"""scripts/inspect_zhang_csv.py su un CSV sintetico nella forma vista su Zhang: intestazione su piu' righe, separatore ';', virgola decimale, una
colonna a 4000 Hz piena e una a 2000 Hz piena solo a righe alterne, una IMU rada che comincia piu' tardi."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("inspect_zhang_csv", Path(__file__).resolve().parents[2] / "scripts" / "inspect_zhang_csv.py")
IZ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IZ)


def _csv(path, muscles=("FCR", "FDS")):
    # come Zhang: sensore, tipo, muscolo, frequenza, colore/numero (con NaN), numero di serie (tutto numerico), unita' di misura
    lines = ["S1;S2;NaN", "EMG;EMG;ACC X", f"{muscles[0]};{muscles[1]};Wrist", "4000 Hz;2000 Hz;148,1481 Hz", "1;blue;NaN", "73242;70786;70786",
             "mV;mV;G"]
    for i in range(200):
        a = f"{i * 0.5:.1f}".replace(".", ",")
        b = f"{i:.1f}".replace(".", ",") if i % 2 == 0 else ""
        c = "1,25" if i >= 7 and (i - 7) % 27 == 0 else ""
        lines.append(f"{a};{b};{c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_layout_rates_origin_and_decimal_comma(tmp_path):
    r = IZ.inspect_file(_csv(tmp_path / "a.csv"), 20000)
    assert r["delimiter"] == ";" and r["n_header_lines"] == 7 and r["n_data_rows"] == 200 and r["n_cols"] == 3
    c4k, c2k, imu = r["columns"]
    assert c4k["header"] == ["S1", "EMG", "FCR", "4000 Hz", "1", "73242", "mV"] and c4k["n_nonempty"] == 200 and c4k["steps_first_rows"] == {1: 199}
    assert c2k["n_nonempty"] == 100 and c2k["first_row"] == 0 and c2k["last_row"] == 198 and c2k["steps_first_rows"] == {2: 99}
    assert imu["first_row"] == 7 and imu["steps_first_rows"] == {27: 7}
    assert c2k["frac_comma"] == 1.0 and c2k["n_not_numeric"] == 0 and c2k["samples"] == ["0,0", "2,0", "4,0"]


def test_emg_header_comparison_between_files(tmp_path):
    a = IZ.inspect_file(_csv(tmp_path / "a.csv"), 100)
    b = IZ.inspect_file(_csv(tmp_path / "b.csv"), 100)
    c = IZ.inspect_file(_csv(tmp_path / "c.csv", muscles=("FCU", "FDS")), 100)
    assert IZ.compare_headers([a, b]) == {"emg_columns_per_file": [2, 2], "all_equal": True}
    assert IZ.compare_headers([a, c])["all_equal"] is False


def test_main_prints_compact_columns(tmp_path, capsys):
    import json

    IZ.main([str(_csv(tmp_path / "a.csv")), str(_csv(tmp_path / "b.csv"))])
    out = json.loads(capsys.readouterr().out)
    assert out["files"][0]["header_rows"][2] == ["FCR", "FDS", "Wrist"] and len(out["files"][0]["columns"]) == 3
    assert "columns" not in out["files"][1] and out["emg_header_comparison"]["all_equal"] is True
