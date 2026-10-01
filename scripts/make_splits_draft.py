#!/usr/bin/env python3
"""Split dei soggetti per il manifest D9, punto (d) della BOZZA `docs/proposta_manifest_d9.md` (non firmata): genera una proposta da rivedere, NON il
manifest congelato.

Regole proposte:
- Kaifosh: tutto benchmark (proposta b), nessun soggetto nel pretraining; si conserva lo split ufficiale 80/10/10;
- split ufficiali dove esistono: emg2qwerty (gli 8 utenti di test di config/user/user0-7.yaml), emg2pose (utenti con held_out_user = True, e per
  sessione le registrazioni del test per fasi nuove: decisione di Simone del 02/10/2026);
- NinaPro DB4, DB5, DB7, DB8: interi nel pretraining (sovrapposizioni di soggetti dichiarate o probabili, fatto n. 2); i test di NinaPro solo da DB2, DB3,
  DB6, DB10;
- altrove: il 20% dei soggetti, almeno 2, arrotondato per eccesso, stratificato per gruppo (amputati / normodotati dove il dataset li mescola), con un
  seme per dataset derivato da `--seed` e dal nome del dataset (crc32);
- manifest sottocampionati per soggetti (asse di v10 §10.6): prefissi annidati di una permutazione dei soggetti di pretraining, per dataset, al 12,5, 25 e
  50% (almeno 1 soggetto).

Ingressi: elenco dei soggetti per dataset (jsonl dal processato), gli split ufficiali in `splits/official/`, e per i dataset non ancora ingeriti del tutto
l'elenco atteso (DB10 dai nomi dei file grezzi; emg2pose dal CSV dei metadati).

  python3 scripts/make_splits_draft.py --subjects ~/wearusfm_local/reports/passo4/subjects_by_dataset.jsonl --out splits/draft/splits_draft.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "splits" / "official"
TEST_FRACTION = 0.2
MIN_TEST = 2
MIN_TEST_OVERRIDE = {"csl_hdemg": 1}  # Simone, 01/10/2026: «CSL-hdemg: un solo soggetto di test» (5 soggetti)
NESTED = (0.125, 0.25, 0.5)
NINAPRO_ALL_PRETRAINING = ("ninapro_db4", "ninapro_db5", "ninapro_db7", "ninapro_db8")
BENCHMARK_ONLY = ("kaifosh",)
# DB10 MDS1: soggetti dai file grezzi (S010-S040 senza S025, S101-S115; verificato il 01/10/2026), cartelle del processato s<NNN>
DB10_EXPECTED = [f"s{n:03d}" for n in list(range(10, 41)) if n != 25] + [f"s{n:03d}" for n in range(101, 116)]


def amputee_group(dataset: str, subject: str) -> str:
    """Gruppo di stratificazione: amputati e normodotati dove un dataset li mescola."""
    if dataset == "ninapro_db10":
        return "amputee" if int(subject[1:]) >= 101 else "able"
    if dataset == "ninapro_db7":
        return "amputee" if subject in ("s21", "s22") else "able"
    if dataset == "ninapro_db8":
        return "amputee" if subject in ("s11", "s12") else "able"
    return "all"


def _rng(seed: int, dataset: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(dataset.encode())])


def split_random(dataset: str, subjects: list[str], seed: int) -> tuple[list[str], list[str]]:
    """(pretraining, test): 20% (almeno 2, per eccesso) per gruppo di stratificazione, a caso col seme del dataset."""
    rng = _rng(seed, dataset)
    groups: dict[str, list[str]] = {}
    for s in sorted(subjects):
        groups.setdefault(amputee_group(dataset, s), []).append(s)
    test = []
    for g in sorted(groups):
        members = groups[g]
        min_test = MIN_TEST_OVERRIDE.get(dataset, MIN_TEST) if len(groups) == 1 else 1
        k = min(len(members) - 1, max(min_test, math.ceil(TEST_FRACTION * len(members))))
        test += [members[i] for i in sorted(rng.permutation(len(members))[:k])]
    test = sorted(test)
    return sorted(s for s in subjects if s not in test), test


def nested(dataset: str, pretraining: list[str], seed: int) -> dict[str, list[str]]:
    perm = [pretraining[i] for i in _rng(seed + 1, dataset).permutation(len(pretraining))]
    return {f"{f:g}": sorted(perm[: max(1, math.ceil(f * len(perm)))]) for f in NESTED}


def official_test(dataset: str, subjects: list[str]) -> list[str] | None:
    if dataset == "emg2qwerty":
        ids = json.loads((OFFICIAL / "emg2qwerty_users.json").read_text())["test_users"].values()
        return sorted(f"u{i}" for i in ids)
    if dataset == "emg2pose":
        return sorted(f"u{i}" for i in json.loads((OFFICIAL / "emg2pose_users.json").read_text())["held_out_users"])
    return None


def expected_subjects(dataset: str, found: list[str]) -> tuple[list[str], str]:
    if dataset == "ninapro_db10":
        return DB10_EXPECTED, "attesi dai file grezzi (ingest in corso)"
    if dataset == "emg2pose":
        return sorted(f"u{i}" for i in json.loads((OFFICIAL / "emg2pose_users.json").read_text())["all_users"]), "dal CSV dei metadati (ingest in corso)"
    return sorted(found), "cartelle del processato"


def make(subjects_by_dataset: dict[str, list[str]], seed: int) -> dict:
    out = {}
    for ds in sorted(subjects_by_dataset):
        subjects, source = expected_subjects(ds, subjects_by_dataset[ds])
        if ds in BENCHMARK_ONLY:
            out[ds] = {"role": "benchmark", "benchmark": subjects, "pretraining": [], "test": [], "source": source,
                       "rule": "D3b proposta: tutto benchmark; split ufficiale in splits/official/kaifosh_users.json"}
            continue
        off = official_test(ds, subjects)
        if off is not None:
            missing = sorted(set(off) - set(subjects))
            if missing:
                raise ValueError(f"{ds}: utenti di test ufficiali non trovati {missing[:5]}")
            test, rule = off, "split ufficiale"
        elif ds in NINAPRO_ALL_PRETRAINING:
            test, rule = [], "tutto nel pretraining (sovrapposizione di soggetti con altri DB, fatto n. 2)"
        else:
            _, test = split_random(ds, subjects, seed)
            rule = f"{int(TEST_FRACTION * 100)}% per gruppo, almeno {MIN_TEST_OVERRIDE.get(ds, MIN_TEST)}, seme {seed}"
        pre = sorted(s for s in subjects if s not in test)
        out[ds] = {"role": "pretraining", "pretraining": pre, "test": sorted(test), "nested": nested(ds, pre, seed), "source": source, "rule": rule}
        if ds == "emg2pose":  # decisione di Simone del 02/10/2026: fuori dal pretraining anche il test ufficiale per fasi nuove (per sessione)
            out[ds]["test_sessions"] = json.loads((OFFICIAL / "emg2pose_users.json").read_text())["held_out_stage_test_sessions"]
            out[ds]["rule"] += " + test per fasi nuove (held_out_stage) per sessione"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=Path, required=True, help="jsonl con {dataset, subjects} per riga (dal processato)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    by = {}
    for line in args.subjects.read_text().splitlines():
        d = json.loads(line)
        by[d["dataset"]] = d["subjects"]
    res = {"draft": True, "proposal": "docs/proposta_manifest_d9.md, punto (d) - NON firmata", "seed": args.seed, "datasets": make(by, args.seed)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print("| Dataset | Soggetti | Pretraining | Test | 12,5% | 25% | 50% | Regola |")
    print("|---|---|---|---|---|---|---|---|")
    for ds, r in res["datasets"].items():
        n = len(r["pretraining"]) + len(r["test"]) + len(r.get("benchmark", []))
        ns = r.get("nested", {})
        print(f"| {ds} | {n} | {len(r['pretraining'])} | {len(r['test'])} | {len(ns.get('0.125', []))} | {len(ns.get('0.25', []))} | "
              f"{len(ns.get('0.5', []))} | {r['rule']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
