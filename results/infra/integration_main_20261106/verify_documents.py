"""Verify that integration retained every journal/decision block verbatim."""
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def blocks(text):
    return [s.strip("\n") for s in re.split(r"(?=^### )", text, flags=re.M) if s.strip("\n")]

def sections(text):
    before, second = text.split("## 2) Problemi trovati", 1)
    prefix, first = before.split("## 1) Cosa ho fatto", 1)
    return prefix, first, second

def verify():
    refs = json.loads((HERE / "inputs.json").read_text())
    result = {}
    for name in ("docs/claude_GPT_shared", "docs/decisioni.md"):
        final = (ROOT / name).read_text()
        if name.endswith("claude_GPT_shared"):
            assert final.count("## 1) Cosa ho fatto") == final.count("## 2) Problemi trovati") == 1
        for label, sha in refs.items():
            original = subprocess.check_output(["git", "show", sha + ":" + name], cwd=ROOT, text=True)
            if name.endswith("claude_GPT_shared"):
                old = sections(original)
                assert sections(final)[0] == old[0]
                expected = blocks(old[1]) + blocks(old[2])
            else:
                expected = blocks(original)
            missing = [s.splitlines()[0] for s in expected if s not in final]
            result[label + ":" + name] = {"blocks": len(expected), "missing": missing}
            assert not missing, (label, name, missing)
    return result

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
