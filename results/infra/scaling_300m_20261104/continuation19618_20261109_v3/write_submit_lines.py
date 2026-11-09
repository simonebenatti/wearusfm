"""Write training_submit_lines.txt of the recovery for a deploy commit: the four lines come from the generator of the chain
(`scaling_plans19618.recovery_lines`), never by hand; this script only adds the comments, from the same identity.

Usage (from the repository root): python -B <this folder>/write_submit_lines.py <deploy commit>
Refuses a commit that does not carry the plan of the working tree, and an existing file.
"""
import hashlib
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_plans19618 as S  # noqa: E402


def main(commit):
    plan_blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{S.RECOVERY['plan']}"], capture_output=True, check=True).stdout
    assert plan_blob == (ROOT / S.RECOVERY["plan"]).read_bytes(), "the deploy commit does not carry the plan of the working tree"
    lines, campaign = S.recovery_lines(commit, ROOT), S.RECOVERY["campaign"]
    checkout_dir, precheck_dir = S.EVIDENCE_DIRS["checkout-training"], S.EVIDENCE_DIRS["identity-precheck"]
    text = f"""# Linea 1, recupero della continuazione 300M 5.000 -> 19.618: {campaign}. Lanci autorizzati da Simone il 09/11
# («allora puoi lanciare, no need per la review di gpt»; per il v3: «Sì, prepara il v3»), dopo la rilettura in sola lettura
# superata (remote_read.sh, poi make_training_plan.py <listing> <sha256> --check con "passed": true).
# Commit di deploy (codice e piano): {commit}
# Piano: {S.RECOVERY['plan']}, SHA-256 {hashlib.sha256(plan_blob).hexdigest()}
# Prodotte da: scripts/scaling_plans19618.py recovery-lines --sha {commit}
# Una riga alla volta, cosi' com'e'; ognuna solo dopo che la precedente e' finita come atteso ed e' stata letta; stop al
# primo esito diverso; nessun retry, nessuna catena afterok.
# 1) Riferimento di deploy su leonardo (non tocca il working tree di $WORK/wearusfm; mai main). Atteso: ramo nuovo; poi
#    `git ls-remote leonardo refs/heads/deploy/{campaign}` uguale al commit di deploy.
{lines[0]}
# 2) Checkout di campagna congelato: corsia seriale, 2 CPU, 4G, 5 minuti, 0 GPU-ore. Atteso: `sottomesso`, poi
#    COMPLETED 0:0; poi `scaling_plans19618.py collect checkout-training` (prove in evidence/{checkout_dir}/).
{lines[1]}
# 3) Pre-controllo dell'identita': corsia seriale, 4 CPU, 24G, 20 minuti, 0 GPU-ore. Atteso: `sottomesso`, poi COMPLETED 0:0;
#    poi `scaling_plans19618.py collect identity-precheck` (prove in evidence/{precheck_dir}/) e nella ricevuta
#    "identical": true e "numpy_loaded_before_torch": false.
{lines[2]}
# 4) Training 5.000 -> 19.618: 1 nodo, 4 GPU, 32 CPU, 480G, 14:45:00, QoS normal; al massimo 59 GPU-ore. Prima dell'invio: suite
#    completa tests/cpu verde sul commit di deploy e, in sola lettura, cartelle del run e delle ricevute ancora assenti.
#    Atteso: `sottomesso`; a fine job `scaling_plans19618.py collect training`.
{lines[3]}
"""
    out = HERE / "training_submit_lines.txt"
    with out.open("x") as stream:
        stream.write(text)
    print(out.relative_to(ROOT), hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
