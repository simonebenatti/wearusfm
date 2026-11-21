import sys, pathlib
src = pathlib.Path(sys.argv[1]); t = pathlib.Path(sys.argv[2]).read_text(); m = sys.argv[3]
R = {"M5": ('        if before != after:\n            raise ValueError("Null warm-up', '        if False:\n            raise ValueError("Null warm-up'),
     "M6": ('        if not before["grad_none"]:', '        if False:'),
     "M8": ('            if not count:\n', '            if False:\n'),
     "M9": ('        if count or len(optimizers) != 1 or len(micros) != 1:', '        if False:'),
     "M16": ('    if count != 1:\n', '    if False:\n')}
a, b = R[m]; assert t.count(a) == 1, m; src.write_text(t.replace(a, b))
