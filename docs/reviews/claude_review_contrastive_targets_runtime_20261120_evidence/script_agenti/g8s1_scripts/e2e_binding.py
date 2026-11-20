"""G-8 scettico 1: prova end-to-end del blocco di review sul pacchetto reale (clone temporaneo).

Uso: python -B e2e_binding.py <clone>
Crea commit SOLO nel clone temporaneo indicato.
"""
import json
import subprocess
import sys
import traceback
from pathlib import Path

clone = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(clone / "scripts"))
sys.path.insert(0, str(clone / "src"))
import contrastive_release as C  # noqa: E402

assert Path(C.__file__).resolve().is_relative_to(clone), C.__file__


def git(*args, author="Fixture"):
    return subprocess.check_output(
        ["git", "-c", "user.name=" + author, "-c", "user.email=fixture@example.invalid", *args],
        cwd=clone, stderr=subprocess.PIPE).decode().strip()


pin = json.loads((clone / "results/infra/contrastive_targets_20261118/amendment_20261119/delivery_pin.json").read_text())
real = {"schema": C.REVIEW_SCHEMA, "status": "APPROVED", "reviewed_commit": pin["content_commit"],
        "source_sha256": pin["source_sha256"], "science_plan_sha256": pin["science_plan_sha256"],
        "preflight_plan_sha256": pin["preflight_plan_sha256"]}
print("source_digest attuale == delivery_pin:", C.source_digest() == pin["source_sha256"])
print("plan sha == pin:", C.file_sha(C.PLAN) == pin["science_plan_sha256"],
      "preflight sha == pin:", C.file_sha(C.PREFLIGHT) == pin["preflight_plan_sha256"])

prompt_text = (clone / "docs/prompt_claude_contrastive_targets_runtime_20261119.md").read_text()
template_start = prompt_text.index("````text")
template_end = prompt_text.index("````", template_start + 8) + 4
template = prompt_text[template_start:template_end]

block = "```contrastive-release-review\n" + json.dumps(real, indent=2) + "\n```\n"
cases = {
    "A_blocco_reale_solo": "# Review\n\nVerdetto.\n\n" + block,
    "B_reale_piu_modello_del_prompt": "# Review\n\nModello richiesto:\n\n" + template + "\n\n" + block,
    "C_solo_modello_del_prompt": "# Review\n\n" + template + "\n",
    "D_reale_con_scope": "# Review\n\n```contrastive-release-review\n" + json.dumps(dict(real, scope="audit_only"), indent=2) + "\n```\n",
}
reviews = clone / "docs/reviews/g8_probe"
reviews.mkdir(parents=True, exist_ok=True)
for name, text in cases.items():
    (reviews / (name + ".md")).write_text(text)
git("add", "docs/reviews/g8_probe")
git("commit", "-qm", "g8 probe reviews (temp clone only)", author="GPT")
review_commit = git("rev-parse", "HEAD")
print("commit della review (autore GPT, solo clone temporaneo):", review_commit)

bindings = {}
for name in cases:
    rel = "docs/reviews/g8_probe/" + name + ".md"
    try:
        binding = C.bind_review(review_commit, rel, clone / rel)
        bindings[name] = binding
        print(name, "-> ACCETTATA", binding["review_payload"]["reviewed_commit"])
    except Exception as error:  # noqa: BLE001
        print(name, "-> RIFIUTATA:", type(error).__name__, error)

binding = bindings["A_blocco_reale_solo"]
plan = json.loads(C.PREFLIGHT.read_text())
for cell in plan["cells"]:
    evidence_sets = {"vuota": {}}
    if cell["mode"] != "audit":
        some = C.pin(C.PLAN)  # file qualunque al posto del report di audit
        evidence_sets["audit_senza_accounting"] = {"audit": some, "audit_sacct": None}
    for label, evidence in evidence_sets.items():
        try:
            job = C.make_job(binding, cell["id"], evidence, "/remote/out/" + cell["id"])
            print("make_job", cell["mode"], cell["id"], label, "-> CREATO", job["status"])
        except Exception as error:  # noqa: BLE001
            print("make_job", cell["mode"], cell["id"], label, "-> RIFIUTATO:", type(error).__name__, str(error)[:90])

# training: modalita' fuori dal preflight
print("modi ammessi in make_job (riga 227):", sorted({"audit", "calibration", "cuda", "full", "resume", "readout"}))
print("modi presenti nel preflight:", sorted({c["mode"] for c in plan["cells"]}))
