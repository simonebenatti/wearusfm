import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_quotes.py"
_spec = importlib.util.spec_from_file_location("check_quotes", SCRIPT)
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)

PAGE = """<html><head><style>.x{color:red}</style><script>var q="Columns 9-16 represent";</script></head><body>
<!-- Creative Commons Attribution No Derivatives 4.0 International -->
<p>Columns 9-16 represent the second <b>Myo</b>, tilted by 22.5&nbsp;degrees clockwise.</p>
<p>The dataset includes 6 repetitions of 52 different movements. It’s sampled at 200 Hz.</p></body></html>"""


def test_verbatim_quote_found_across_tags_entities_and_case():
    r = Q.check("pagina", ["columns 9-16 represent the second Myo, tilted by 22.5 degrees clockwise.", "It's sampled at 200 Hz."], raw=PAGE)
    assert [x["found"] for x in r] == [True, True]


def test_paraphrase_is_not_found_and_context_is_given():
    r = Q.check("pagina", ["The dataset includes Six repetitions of 52 different movements."], raw=PAGE)
    assert r[0]["found"] is False and "6 repetitions" in r[0]["nearby"]


def test_text_in_comments_and_scripts_is_not_visible():
    r = Q.check("pagina", ["Creative Commons Attribution No Derivatives", 'var q="Columns'], raw=PAGE)
    assert [x["found"] for x in r] == [False, False]


def test_local_file_and_exit_code(tmp_path, capsys):
    f = tmp_path / "p.html"
    f.write_text(PAGE)
    assert Q.main(["--url", str(f), "--quote", "tilted by 22.5 degrees clockwise"]) == 0
    assert Q.main(["--url", str(f), "--quote", "tilted by 45 degrees"]) == 1
