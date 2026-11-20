"""Independent reproduction for R3-N1 (skeptic 1). Usage: python -B vr3n1_bind.py <clone> [case ...]"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()
ONLY = set(sys.argv[2:])
sys.path[:0] = [str(CLONE / "src"), str(CLONE / "scripts")]
import contrastive_release as C  # noqa: E402
assert C.ROOT == CLONE, (C.ROOT, CLONE)

ENV = dict(os.environ, GIT_AUTHOR_NAME="vr3n1", GIT_AUTHOR_EMAIL="vr3n1@invalid",
           GIT_COMMITTER_NAME="vr3n1", GIT_COMMITTER_EMAIL="vr3n1@invalid")
FULL = {"fbf": "fbfde797b48644744c36547533acf0e8a7793535",
        "0ef": "0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba",
        "438": "438c8d65cf3b60a92af4f602869aa29bb328efca"}
SRC = "40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0"
PLAN = "b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a"
PRE = "e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92"


def git(*args, data=None):
    return subprocess.run(["git", *args], cwd=CLONE, input=data, env=ENV, check=True,
                          capture_output=True).stdout


def orphan_commit(blob_bytes, name):
    blob = git("hash-object", "-w", "--stdin", data=blob_bytes).decode().strip()
    tree = git("mktree", data=f"100644 blob {blob}\treview.md\n".encode()).decode().strip()
    return git("commit-tree", tree, "-m", "vr3n1 " + name).decode().strip()


def body(status='"APPROVED"', commit=FULL["fbf"], src=SRC, indent="", extra_lines=None):
    lines = ["{",
             '  "schema": "contrastive-target-release-review-v1",',
             f'  "status": {status},',
             f'  "reviewed_commit": "{commit}",',
             f'  "source_sha256": "{src}",',
             f'  "science_plan_sha256": "{PLAN}",',
             f'  "preflight_plan_sha256": "{PRE}"',
             "}"]
    if extra_lines:
        lines[2:2] = extra_lines
    return "\n".join(indent + line for line in lines)


B = body()
F = "```contrastive-release-review"
MODEL = ("````text\n" + F + "\n{\n  \"schema\": \"contrastive-target-release-review-v1\",\n  \"status\": \"APPROVED\",\n"
         "  \"reviewed_commit\": \"SHA_COMPLETO_DEI_CONTENUTI_RIVISTI\",\n  \"source_sha256\": \"SHA256_DEI_SORGENTI\",\n"
         "  \"science_plan_sha256\": \"SHA256_DEL_PLAN_JSON\",\n  \"preflight_plan_sha256\": \"SHA256_DEL_PREFLIGHT_PLAN_JSON\"\n}\n```\n````\n")
REAL = F + "\n" + B + "\n```\n"
HEAD = "# Review Claude\n\nVerdetto: APPROVATO.\n\n"
cases = {
    # name: (text, copy_mutator or None, expected)
    "C00_canonical": (HEAD + REAL, None, "ACCEPT"),
    "C01_midline_open": (HEAD + "Testo " + F + "\n" + B + "\n```\n", None, "ACCEPT"),
    "C02_four_backticks": (HEAD + "`" + F + "\n" + B + "\n````\n", None, "ACCEPT"),
    "C03_indent2_open": (HEAD + "  " + F + "\n" + B + "\n```\n", None, "ACCEPT"),
    "C03b_indent4_open_body": (HEAD + "    " + F + "\n" + body(indent="    ") + "\n```\n", None, "ACCEPT"),
    "C03c_indent3_open_and_close": (HEAD + "   " + F + "\n" + body(indent="   ") + "\n   ```\n", None, "REJECT"),
    "C04_dup_status_rej_then_app": (HEAD + F + "\n" + body(extra_lines=['  "status": "REJECTED",']) + "\n```\n", None, "ACCEPT"),
    "C04r_dup_status_app_then_rej": (HEAD + F + "\n" + body(status='"APPROVED"', extra_lines=[]).replace(
        '"preflight_plan_sha256": "' + PRE + '"', '"preflight_plan_sha256": "' + PRE + '",\n  "status": "REJECTED"') + "\n```\n", None, "REJECT"),
    "C05_escape_status": (HEAD + F + "\n" + body(status='"\\u0041PPROVED"') + "\n```\n", None, "ACCEPT"),
    "C05b_escape_in_hash": (HEAD + F + "\n" + body(src="\\u0034" + SRC[1:]) + "\n```\n", None, "ACCEPT"),
    "C06_prompt_model_plus_real": (HEAD + MODEL + "\n" + REAL, None, "REJECT"),
    "C06b_prompt_model_alone": (HEAD + MODEL, None, "REJECT"),
    "C07_example_in_text3_plus_real": (HEAD + "Esempio:\n\n```text\n" + REAL + "```\n\n" + REAL, None, "REJECT"),
    "C08_blockquote": (HEAD + "\n".join("> " + l for l in REAL.rstrip("\n").split("\n")) + "\n", None, "REJECT"),
    "C09_brace_fence_same_line": (HEAD + F + "\n" + B + "```\n", None, "REJECT"),
    "C10_unfavorable_with_single_example": ("# Review Claude\n\nVerdetto: NON APPROVATO, DA RIVEDERE (due bloccanti).\n\n"
                                            "Quando i bloccanti saranno chiusi, il blocco avra' questa forma:\n\n" + REAL, None, "ACCEPT"),
    "C11_close_with_info": (HEAD + F + "\n" + B + "\n```json\n", None, "ACCEPT"),
    "C11b_close_with_text": (HEAD + F + "\n" + B + "\n``` fine blocco\n", None, "ACCEPT"),
    "C12_space_after_fence": (HEAD + "``` contrastive-release-review\n" + B + "\n```\n", None, "REJECT"),
    "C13_trailing_ws_after_name": (HEAD + F + " \t \n" + B + "\n```\n", None, "ACCEPT"),
    "C14a_commit_0efa": (HEAD + F + "\n" + body(commit=FULL["0ef"]) + "\n```\n", None, "ACCEPT"),
    "C14b_commit_base438": (HEAD + F + "\n" + body(commit=FULL["438"]) + "\n```\n", None, "REJECT"),
    "C15_uppercase_hex": (HEAD + F + "\n" + body(src=SRC.upper()) + "\n```\n", None, "REJECT"),
    "C16_latin1_accents": ((HEAD + "Perche' cosi' e' piu' chiaro: ").encode("utf-8") + "perché".encode("latin-1") + b"\n\n" + REAL.encode(), None, "REJECT"),
    "C16b_utf8_accents": (HEAD + "Perché così è più chiaro.\n\n" + REAL, None, "ACCEPT"),
    "C17_copy_not_identical": (HEAD + REAL, lambda b: b + b"\n", "REJECT"),
    "C18_inline_mentions_plus_real": (HEAD + "Il blocco ```contrastive-release-review``` e il nome `contrastive-release-review` "
                                      "e 'contrastive-release-review' citati.\n\n" + REAL, None, "ACCEPT"),
    "C19_stray_open_without_later_fence": (HEAD + REAL + "\nNota: " + F + "\n(nessun recinto dopo)\n", None, "ACCEPT?"),
    "C19b_stray_open_with_later_fence": (HEAD + REAL + "\nNota: " + F + "\n(testo)\n\n```bash\nls\n```\n", None, "REJECT?"),
    "C20_realistic": ("# Review Claude del pacchetto RMS100/fast125\n\nVerdetto: **APPROVATO**. Perché: parità, ripresa e release verificate.\n\n"
                      "```bash\nPYTHONPATH=src:scripts python scripts/contrastive_release.py bind-review --help\n```\n\n"
                      "```text\nesito: 465 passed\n```\n\n## Blocco di release\n\n" + REAL +
                      "\n## Note\n\n```python\nprint('ok')\n```\n", None, "ACCEPT"),
    "C21_crlf_file": ((HEAD + REAL).replace("\n", "\r\n"), None, "ACCEPT?"),
}

out = []
for name, (text, mutate, expected) in cases.items():
    if ONLY and name not in ONLY:
        continue
    data = text if isinstance(text, bytes) else text.encode("utf-8")
    try:
        quick = C.review_payload(data.decode("utf-8"))
        quick_res = "payload-ok status=" + quick["status"]
    except Exception as e:  # noqa: BLE001
        quick_res = f"payload-{type(e).__name__}: {str(e)[:90]}"
    commit = orphan_commit(data, name)
    copy = CLONE / "docs/reviews" / f"vr3n1_{name}.md"
    copy.write_bytes(mutate(data) if mutate else data)
    t0 = time.time()
    try:
        binding = C.bind_review(commit, "review.md", copy)
        again = C.validate_review_binding(binding)  # runtime check (verify_git default True)
        res = "ACCEPT"
        detail = f"status={again['status']} reviewed_commit={again['reviewed_commit'][:8]}"
    except Exception as e:  # noqa: BLE001
        res = "REJECT"
        detail = f"{type(e).__name__}: {str(e)[:110]}"
    finally:
        copy.unlink()
    line = f"{name}\texpected={expected}\tbind={res}\t{time.time()-t0:.1f}s\t{detail}\t| {quick_res}"
    print(line, flush=True)
    out.append(line)
