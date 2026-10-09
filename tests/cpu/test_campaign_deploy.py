"""Checkout di campagna e codice congelato dei launcher (decisione del 09/10/2026, CLAUDE.md «Checkout di campagna»):
`scripts/slurm/campaign_checkout.sbatch` e `scripts/slurm/lib_code_dir.sh`, provati con bash e git su un $WORK finto."""

import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECKOUT = REPO / "scripts" / "slurm" / "campaign_checkout.sbatch"
LIB = REPO / "scripts" / "slurm" / "lib_code_dir.sh"
REAL = {name: shutil.which(name) for name in ("git", "chmod", "find")}

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None, reason="servono git e bash")


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def work(tmp_path):
    """$WORK finto col repo canonico: due commit, il secondo e' quello della campagna."""
    canon = tmp_path / "wearusfm"
    (canon / "scripts" / "slurm").mkdir(parents=True)
    (canon / "src" / "wearusfm").mkdir(parents=True)
    shutil.copy(LIB, canon / "scripts" / "slurm" / "lib_code_dir.sh")
    (canon / "src" / "wearusfm" / "__init__.py").write_text("V = 1\n")
    _git(canon, "init", "-q", "-b", "main")
    _git(canon, "add", "-A")
    _git(canon, "commit", "-q", "-m", "uno")
    first = _git(canon, "rev-parse", "HEAD")
    (canon / "src" / "wearusfm" / "__init__.py").write_text("V = 2\n")
    _git(canon, "commit", "-q", "-am", "due")
    yield tmp_path, canon, first, _git(canon, "rev-parse", "HEAD")
    for root, dirs, files in os.walk(tmp_path):  # i checkout sono in sola lettura: si riabilita la scrittura per la pulizia
        for name in dirs + files:
            p = Path(root) / name
            if not p.is_symlink():
                p.chmod(p.stat().st_mode | stat.S_IWUSR)


def _checkout(work_dir, name, sha, **extra):
    env = {**os.environ, "WORK": str(work_dir), "CAMPAIGN": name, "CAMPAIGN_SHA": sha, "SLURM_JOB_ID": "1", **extra}
    return subprocess.run(["bash", str(CHECKOUT)], env=env, capture_output=True, text=True)


def _fake(tmp, name, body):
    """Un `name` finto in testa al PATH (l'originale resta raggiungibile come REAL[name]); ritorna il PATH da passare."""
    bin_dir = tmp / "bin"
    bin_dir.mkdir(exist_ok=True)
    p = bin_dir / name
    p.write_text("#!/bin/sh\n" + body.replace("@REAL@", shlex.quote(REAL[name])) + "\n")
    p.chmod(0o755)
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


# git che fallisce sui comandi il cui elenco di argomenti corrisponde al pattern in FAKE_GIT_FAIL (glob di `case`)
FAKE_GIT = 'case " $* " in $FAKE_GIT_FAIL) echo "fatal: errore simulato di git ($*)" >&2; exit 128;; esac\nexec @REAL@ "$@"'


def _writable(dest):
    """File e cartelle del codice ancora scrivibili (escluso il file .git del worktree)."""
    return [p for p in [dest, *dest.rglob("*")] if p != dest / ".git" and not p.is_symlink() and p.stat().st_mode & 0o222]


def _setup(work_dir, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA", "PYTHONPATH")}
    env.update({"WORK": str(work_dir), **extra})
    script = f'source "{LIB}" && code_dir_setup && echo "PWD=$PWD" && echo "PP=$PYTHONPATH" && echo "SHA=$WEARUSFM_CODE_SHA"'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)


def test_campaign_checkout_is_frozen_and_never_reused(work):
    tmp, canon, first, sha = work
    r = _checkout(tmp, "w1_prova", first)
    assert r.returncode == 0, r.stdout + r.stderr
    dest = tmp / "wearusfm_campaigns" / "w1_prova"
    assert _git(dest, "rev-parse", "HEAD") == first
    assert (dest / "src" / "wearusfm" / "__init__.py").read_text() == "V = 1\n"
    assert not os.access(dest / "src" / "wearusfm" / "__init__.py", os.W_OK)  # sola lettura
    assert not _writable(dest)  # nessun file o cartella del codice resta scrivibile
    assert _git(canon, "rev-parse", "HEAD") == sha and not _git(canon, "status", "--porcelain")  # il canonico non cambia
    assert "locked" in _git(canon, "worktree", "list", "--porcelain")
    again = _checkout(tmp, "w1_prova", sha)  # la stessa campagna non si riscrive, nemmeno con un altro commit
    assert again.returncode == 2 and "esiste gia'" in again.stdout
    assert _git(dest, "rev-parse", "HEAD") == first


def test_campaign_checkout_refuses_bad_inputs(work):
    tmp, _, _, sha = work
    assert _checkout(tmp, "../fuori", sha).returncode == 2
    assert _checkout(tmp, ".nascosta", sha).returncode == 2
    assert _checkout(tmp, "ok", sha[:12]).returncode == 2  # SHA abbreviato
    missing = _checkout(tmp, "ok", "0" * 40)  # commit non pubblicato
    assert missing.returncode == 2 and "non presente" in missing.stdout
    assert not (tmp / "wearusfm_campaigns" / "ok").exists()


def test_launcher_uses_the_frozen_code(work):
    tmp, canon, first, sha = work
    assert _checkout(tmp, "c1", first).returncode == 0
    dest = tmp / "wearusfm_campaigns" / "c1"
    ok = _setup(tmp, CODE_DIR=str(dest), EXPECTED_SHA=first)
    assert ok.returncode == 0, ok.stderr
    out = dict(line.split("=", 1) for line in ok.stdout.splitlines() if "=" in line)
    assert Path(out["PWD"]).resolve() == dest.resolve() and out["PP"].split(":")[0] == f"{dest}/src" and out["SHA"] == first
    assert _setup(tmp, CODE_DIR=str(dest)).returncode == 2  # campagna senza SHA atteso
    assert _setup(tmp, CODE_DIR=str(dest), EXPECTED_SHA=sha).returncode == 2  # SHA diverso
    assert _setup(tmp, CODE_DIR=str(dest), EXPECTED_SHA=first[:12]).returncode == 2  # SHA abbreviato
    assert _setup(tmp, CODE_DIR=str(tmp / "nessuno"), EXPECTED_SHA=first).returncode == 2  # non e' un checkout
    target = dest / "src" / "wearusfm" / "__init__.py"  # un checkout di campagna scrivibile (rimasto a meta'): rifiutato
    target.chmod(target.stat().st_mode | stat.S_IWUSR)
    half = _setup(tmp, CODE_DIR=str(dest), EXPECTED_SHA=first)
    assert half.returncode == 2 and "scrivibile" in half.stderr and "__init__.py" in half.stderr


def test_training_launcher_refuses_run_dir_inside_code_and_unpinned_campaign(work):
    tmp, _, first, _ = work
    assert _checkout(tmp, "c2", first).returncode == 0
    launcher = REPO / "scripts" / "slurm" / "sanity_jepa.sbatch"
    base = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA")}
    for inside in (tmp / "wearusfm_campaigns" / "c2" / "out", tmp / "wearusfm" / "out"):
        r = subprocess.run(["bash", str(launcher), "m.json.gz", "s.json", str(inside)], env={**base, "WORK": str(tmp)},
                           capture_output=True, text=True)
        assert r.returncode == 2 and "dentro il codice" in r.stdout
    # RUN_DIR fuori dal codice, ma checkout di campagna senza SHA atteso: si ferma in code_dir_setup, prima di python
    r = subprocess.run(["bash", str(launcher), "m.json.gz", "s.json", str(tmp / "runs" / "x")],
                       env={**base, "WORK": str(tmp), "CODE_DIR": str(tmp / "wearusfm_campaigns" / "c2")}, capture_output=True, text=True)
    assert r.returncode == 2 and "senza EXPECTED_SHA" in r.stderr


def test_canonical_default_and_dirty_checkout(work):
    tmp, canon, _, sha = work
    plain = _setup(tmp)  # default: il canonico, come prima della regola; SHA facoltativo
    assert plain.returncode == 0 and f"SHA={sha}" in plain.stdout
    assert _setup(tmp, EXPECTED_SHA=sha).returncode == 0
    (canon / "nuovo.txt").write_text("x")  # un file non tracciato rende il checkout sporco
    dirty = _setup(tmp)
    assert dirty.returncode == 2 and "non e' pulito" in dirty.stderr


# Review GPT (P1): un comando che fallisce non e' mai un checkout pulito o pronto, anche se il suo stdout e' vuoto.

@pytest.mark.parametrize("fail_on", ["* status *", "* rev-parse HEAD *", "* rev-parse --is-inside-work-tree *"],
                         ids=["status", "rev-parse-head", "inside-work-tree"])
def test_code_dir_setup_stops_on_git_errors(work, fail_on):
    tmp, _, _, sha = work
    r = _setup(tmp, EXPECTED_SHA=sha, PATH=_fake(tmp, "git", FAKE_GIT), FAKE_GIT_FAIL=fail_on)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "codice:" not in r.stdout and "PWD=" not in r.stdout


def test_code_dir_setup_refuses_the_git_dir(work):
    tmp, canon, _, sha = work
    # dentro .git rev-parse risponde (HEAD giusto) ma `git status` fallisce con stdout vuoto: non e' un checkout pulito
    r = _setup(tmp, CODE_DIR=str(canon / ".git"), EXPECTED_SHA=sha)
    assert r.returncode == 2 and "codice:" not in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("fail_on", ["* worktree add *", "* worktree lock *", "* rev-parse HEAD *", "* status *"],
                         ids=["worktree-add", "worktree-lock", "rev-parse-head", "status"])
def test_campaign_checkout_stops_on_git_errors(work, fail_on):
    tmp, _, first, _ = work
    r = _checkout(tmp, "c_err", first, PATH=_fake(tmp, "git", FAKE_GIT), FAKE_GIT_FAIL=fail_on)
    assert r.returncode == 1 and "errore simulato" in r.stderr, r.stdout + r.stderr
    assert "checkout di campagna pronto" not in r.stdout


CHMOD_FAILS = 'echo "chmod: errore simulato" >&2; exit 1'
CHMOD_SKIPS_ONE = ('mode="$1"; shift\nfor f in "$@"; do\n  case "$f" in *__init__.py) ;; *) @REAL@ "$mode" "$f" || exit 1;; esac\n'
                   'done\nexit 0')  # esce con 0 ma lascia scrivibile un file
FIND_FAILS = 'echo "find: errore simulato" >&2; exit 1'
FIND_FAILS_ON_CHECK = 'case " $* " in *" -perm "*) echo "find: errore simulato" >&2; exit 1;; esac\nexec @REAL@ "$@"'


@pytest.mark.parametrize("tool,body", [("chmod", CHMOD_FAILS), ("chmod", CHMOD_SKIPS_ONE), ("find", FIND_FAILS),
                                       ("find", FIND_FAILS_ON_CHECK)],
                         ids=["chmod-fails", "chmod-skips-one", "find-fails", "find-fails-on-check"])
def test_campaign_checkout_never_ready_with_writable_code(work, tool, body):
    tmp, _, first, _ = work
    r = _checkout(tmp, "c_rw", first, PATH=_fake(tmp, tool, body))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "checkout di campagna pronto" not in r.stdout
    if body == CHMOD_SKIPS_ONE:  # chmod ha detto si', la verifica finale trova il file rimasto scrivibile
        assert [p.name for p in _writable(tmp / "wearusfm_campaigns" / "c_rw")] == ["__init__.py"]
        assert "__init__.py" in r.stdout
