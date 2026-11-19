"""Apply one mutation to contrastive_prepare.py in MY temporary clone only (tests untouched).
  ZINSTR  behaviour-preserving instrumentation: log the test name when the receipt-guard branch (r.383) runs
          and when the final comparison (r.389) runs with a receipt source different from source_digest()
  F       remove the receipt guard call (r.384-385), keep the reader_provenance substitution
  H       remove the receipt/recalculation comparison (r.389-390)
"""
import sys
from pathlib import Path
clone, name, mark = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
assert '/scratchpad/rev_targets/tmp/vp2n1s1/' in str(clone)
path = clone / 'scripts/contrastive_prepare.py'
text = path.read_text()
def rep(old, new):
    global text
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)
guard = ('        from contrastive_source_compat import validate_historical_evidence\n'
         '        validate_historical_evidence("calibration_receipt", receipt)\n')
compare = ('    if _jsonable(receipt) != comparable:\n'
           '        raise ValueError("calibration receipt differs from authenticated recalculation")\n')
if name == 'ZINSTR':
    log = '__import__("pathlib").Path(%r).open("a").write("%%s %%s\\n" %% (%r, __import__("os").environ.get("PYTEST_CURRENT_TEST", "?")))\n'
    rep(guard, '        ' + log % (mark, 'BRANCH') + guard)
    rep(compare, '    if receipt.get("source_sha256") != source_digest():\n        ' + log % (mark, 'COMPARE_FOREIGN_SOURCE') + compare)
elif name == 'F':
    rep(guard, '        pass\n')
elif name == 'H':
    rep(compare, '')
else:
    raise SystemExit('unknown ' + name)
path.write_text(text)
print('applied', name)
