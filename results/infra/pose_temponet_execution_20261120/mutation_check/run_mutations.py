"""Mutation check of the new tests: each mutation must make at least one of them fail. Runs in a scratch worktree."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PROVENANCE = ROOT/"scripts/pose_temponet/provenance.py"
EXECUTION = ROOT/"scripts/pose_temponet/execution.py"
ORIGINAL = {PROVENANCE: PROVENANCE.read_text(), EXECUTION: EXECUTION.read_text()}
MUTATIONS = [
    ("a: historical_producer returns the directory names (first delivery behaviour)", PROVENANCE,
     "    users = sorted(cache_user(directory) for directory in directories)\n",
     "    users = sorted(directories)\n"),
    ("b: second check of read_verified_inputs removed", PROVENANCE,
     '    require(sorted(storage_directory(user) for user in cache_users) == producer["storage_directories"],\n'
     '            "cache users do not map onto the producer storage directories")\n', ""),
    ("c: cache_user strips the first character without any check", PROVENANCE,
     '    require(isinstance(directory,str) and re.fullmatch(r"u[0-9A-Za-z.@-]+",directory) is not None,\n'
     '            "producer storage directory has no unique cache user ID")\n'
     '    user = directory[1:]\n'
     '    require(storage_directory(user) == directory, "producer storage directory does not round-trip")\n'
     '    return user\n',
     '    return directory[1:]\n'),
    ("d: delivery identity back to 20261119", EXECUTION, 'DELIVERY = "20261120"\n', 'DELIVERY = "20261119"\n'),
    ("e: storage_directory forgets the u prefix", PROVENANCE,
     '    return "u"+_STORAGE_UNSAFE.sub("_",user)\n', '    return _STORAGE_UNSAFE.sub("_",user)\n'),
    ("f: first check of read_verified_inputs removed", PROVENANCE,
     '    require(cache_users == producer["users"], "cache user population differs from original producer")\n', ""),
    ("g: cache_user accepts an underscore", PROVENANCE, 'r"u[0-9A-Za-z.@-]+"', 'r"u[0-9A-Za-z_.@-]+"'),
    ("h: round-trip check of cache_user removed", PROVENANCE,
     '    require(storage_directory(user) == directory, "producer storage directory does not round-trip")\n', ""),
]
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
COMMAND = [str(Path.home()/".venvs/wearusfm-tok/bin/python"), "-c",
           "import sys; sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', "
           "'--tb=no', 'tests/cpu/test_pose_temponet_provenance.py', 'tests/cpu/test_pose_temponet_execution.py']))"]


def run():
    result = subprocess.run(COMMAND, cwd=ROOT, env=ENV, capture_output=True, text=True)
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) \S+::(\S+)", result.stdout, re.M)))
    summary = [line for line in result.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line]
    return result.returncode, failed, summary[-1] if summary else result.stdout[-300:]


report = []
code, failed, summary = run()
report.append({"mutation": "none (baseline)", "returncode": code, "failed": failed, "summary": summary})
for name, path, old, new in MUTATIONS:
    text = ORIGINAL[path]
    assert text.count(old) == 1, name
    path.write_text(text.replace(old, new))
    try:
        code, failed, summary = run()
    finally:
        path.write_text(text)
    report.append({"mutation": name, "returncode": code, "failed": failed, "summary": summary, "caught": bool(failed)})
assert all(p.read_text() == t for p, t in ORIGINAL.items())
print(json.dumps(report, indent=1))
