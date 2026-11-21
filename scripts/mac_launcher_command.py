"""Render a released WearUsFM submission for the shared Mac. Never executes it."""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


WORKSPACE = Path.home() / "Documents/ChatGPT/FM per EMG"
SUBMITTER = WORKSPACE / "wearusfm-gpt-science/scripts/submit_job.py"
PYTHON = Path.home() / ".venvs/wearusfm-tok/bin/python"


def render(worktree, commit, argv):
    worktree = worktree.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("serve lo SHA completo del codice della campagna")
    if len(argv) < 4 or argv[0] not in ("python3", str(PYTHON)):
        raise ValueError("atteso un comando diretto del submitter")
    original = (worktree / argv[1]).resolve(strict=True)
    local = worktree / "scripts/submit_job.py"
    if original not in (local, SUBMITTER):
        raise ValueError("percorso del submitter estraneo alla campagna")
    options = argv[2:argv.index("--")] if "--" in argv else argv[2:]
    if "--id" not in options or options.index("--id") + 1 >= len(options):
        raise ValueError("il comando deve contenere un ID di sottomissione")
    frozen = subprocess.check_output(
        ["git", "-C", str(worktree), "show", f"{commit}:scripts/submit_job.py"]
    )
    if original.read_bytes() != frozen or SUBMITTER.read_bytes() != frozen:
        raise ValueError("il submitter richiesto e quello condiviso devono coincidere con il commit")
    direct = [str(PYTHON), str(SUBMITTER), *argv[2:]]
    return {"workdir": str(worktree), "cmd": shlex.join(direct)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path)
    parser.add_argument("commit")
    parser.add_argument("argv", nargs=argparse.REMAINDER, help="-- seguito dagli argomenti del piano")
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    try:
        print(json.dumps(render(args.worktree, args.commit, argv), ensure_ascii=False))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
