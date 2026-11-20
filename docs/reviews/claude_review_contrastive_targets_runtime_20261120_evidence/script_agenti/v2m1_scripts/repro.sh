#!/bin/bash
# Riproduzione indipendente di R3-M1 (scettico n. 2). Solo clone temporaneo, nessun push.
set -u
BASE=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad
SRC=$BASE/rev_targets/clone
T=$BASE/rev_runtime/tmp/v2m1_clone
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
PIN=0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba
OLDBASE=438c8d65cf3b60a92af4f602869aa29bb328efca
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
G() { git -C "$T" -c user.name=skeptic -c user.email=skeptic@example.invalid -c commit.gpgsign=false "$@"; }
REL=results/infra/contrastive_targets_20261118/releases_v2m1
check() {  # $1 etichetta; esegue check-binding come farebbe il job (PYTHONPATH=src, sys.path[0]=scripts)
  local out rc
  out=$(cd "$T" && PYTHONPATH="$T/src" "$PY" -B scripts/contrastive_release.py check-binding --binding "$REL/binding.json" 2>&1); rc=$?
  echo "[$1] rc=$rc HEAD=$(G rev-parse --short HEAD) ancestor_of_PIN=$(G merge-base --is-ancestor $PIN HEAD && echo si || echo no)"
  printf "%s\n" "$out" | grep -E "INJECTED|Error|error|differ" | head -5; printf "   last: %s\n" "$(printf "%s" "$out" | tail -1 | cut -c1-110)"
}
rm -rf "$T"
git clone -q --shared "$SRC" "$T" && G checkout -q --detach $PIN || exit 1
# 1) review finta nel suo commit (ramo laterale), con il blocco richiesto
read SRCD SCI PRE < <(cd "$T" && PYTHONPATH="$T/src:$T/scripts" "$PY" -B -c "
from contrastive_prepare import source_digest
from contrastive_release import file_sha, PLAN, PREFLIGHT
print(source_digest(), file_sha(PLAN), file_sha(PREFLIGHT))")
echo "source_digest=$SRCD science=$SCI preflight=$PRE"
G checkout -q -b v2m1-review $PIN
mkdir -p "$T/docs/reviews"
cat > "$T/docs/reviews/v2m1_fake_review.md" <<EOF
# review di prova (solo clone temporaneo)
\`\`\`contrastive-release-review
{"schema": "contrastive-target-release-review-v1", "status": "APPROVED", "reviewed_commit": "$PIN", "source_sha256": "$SRCD", "science_plan_sha256": "$SCI", "preflight_plan_sha256": "$PRE"}
\`\`\`
EOF
G add docs/reviews/v2m1_fake_review.md && G commit -q -m "fake review" && REVIEW=$(G rev-parse HEAD)
# 2) commit di release figlio del commit rivisto: copia della review + binding (metadati)
G checkout -q --detach $PIN
mkdir -p "$T/$REL" && cp "$T/docs/reviews/v2m1_fake_review.md" /dev/null 2>/dev/null
G show "$REVIEW:docs/reviews/v2m1_fake_review.md" > "$T/$REL/review_copy.md"
(cd "$T" && PYTHONPATH="$T/src" "$PY" -B scripts/contrastive_release.py bind-review --review-commit "$REVIEW" \
   --review-path docs/reviews/v2m1_fake_review.md --review-copy "$REL/review_copy.md" --out "$REL/binding.json") || { echo "bind-review fallito"; exit 1; }
G add "$REL" && G commit -q -m "release metadata only" && META=$(G rev-parse HEAD)
check "M0 solo metadati (controllo positivo)"
# 3) stessa release + src/sitecustomize.py (eseguito all'avvio via PYTHONPATH)
cat > "$T/src/sitecustomize.py" <<'EOF'
import sys
sys.stderr.write("V2M1 INJECTED sitecustomize eseguito\n")
EOF
G add src/sitecustomize.py && G commit -q -m "release + sitecustomize"
check "M7 + src/sitecustomize.py"
# 4) + scripts/argparse.py che oscura la stdlib (sys.path[0] = scripts/)
cat > "$T/scripts/argparse.py" <<'EOF'
import sys, importlib.util, sysconfig, os
sys.stderr.write("V2M1 INJECTED scripts/argparse.py eseguito al posto della stdlib\n")
_spec = importlib.util.spec_from_file_location("argparse", os.path.join(sysconfig.get_paths()["stdlib"], "argparse.py"))
_mod = importlib.util.module_from_spec(_spec); sys.modules["argparse"] = _mod; _spec.loader.exec_module(_mod)
EOF
G add scripts/argparse.py && G commit -q -m "release + argparse shadow"
INJ=$(G rev-parse HEAD)
check "M8 + scripts/argparse.py"
# 5) controlli negativi: modifica di un file dell'inventario e nuovo modulo contrastive_* devono fallire
G checkout -q --detach $META
printf '\n# drift\n' >> "$T/scripts/contrastive_job.py"; G commit -q -am "drift in inventory"
check "N1 modifica di scripts/contrastive_job.py (atteso rifiuto)"
G checkout -q --detach $META
printf 'X = 1\n' > "$T/scripts/contrastive_extra.py"; G add scripts/contrastive_extra.py; G commit -q -m "new contrastive module"
check "N2 nuovo scripts/contrastive_extra.py (atteso rifiuto)"
# 6) ascendenza: commit orfano con lo stesso albero della release con iniezione, e commit con genitore 438c8d65
ORPH=$(G commit-tree "$INJ^{tree}" -m orphan); G checkout -q --detach "$ORPH"
check "A1 orfano, albero = release iniettata"
SIDE=$(G commit-tree "$INJ^{tree}" -p $OLDBASE -m "parent 438c8d65"); G checkout -q --detach "$SIDE"
check "A2 genitore 438c8d65, albero = release iniettata"
# 7) il launcher reale: python3 scripts/contrastive_job.py --help con PYTHONPATH=src esegue il codice iniettato prima di ogni controllo
G checkout -q --detach "$INJ"
(cd "$T" && PYTHONPATH="$T/src" "$PY" -B scripts/contrastive_job.py --help 2>&1 | grep -E "INJECTED|usage" | head -4)
echo "status pulito: [$(G status --porcelain --untracked-files=all | head -3)]"
