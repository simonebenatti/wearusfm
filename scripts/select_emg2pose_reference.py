"""Sottoinsieme di riferimento di emg2pose per il passo 1-bis (D5a: "sottoinsieme fisso, seed 0").

Il mini di emg2pose ha UN solo utente (verificato il 29/09/2026: 30 file, tutti di d387095792),
quindi non basta: la sonda dataset-ID di V3 vuole >= 3 soggetti per dataset e il riferimento di
V2 deve venire da piu' persone. Il dataset completo ha 25.253 registrazioni di 193 utenti (CSV
di metadati). Qui si scelgono, con seed fisso, `n_users` utenti fra quelli con almeno
`per_user` registrazioni di split 'train' e `per_user` registrazioni per utente; l'elenco dei
membri del tar esce su file, uno per riga (`emg2pose_data/<stem>.hdf5`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

TAR_ROOT = "emg2pose_data"


def select(csv_path: Path, n_users: int, per_user: int, seed: int, split: str = "train") -> list[tuple[str, str]]:
    """Ritorna [(utente, membro_del_tar)], ordinato per utente e nome (stabile a parita' di seed)."""
    by_user: dict[str, list[str]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                by_user.setdefault(row["user"], []).append(Path(row["filename"]).stem)
    eligible = sorted(u for u, files in by_user.items() if len(set(files)) >= per_user)
    if len(eligible) < n_users:
        raise ValueError(f"solo {len(eligible)} utenti con >= {per_user} registrazioni '{split}', ne servono {n_users}")
    rng = np.random.default_rng(seed)
    users = sorted(rng.choice(eligible, size=n_users, replace=False).tolist())
    out = []
    for u in users:
        files = sorted(set(by_user[u]))
        pick = sorted(rng.choice(files, size=per_user, replace=False).tolist())
        out += [(u, f"{TAR_ROOT}/{stem}.hdf5") for stem in pick]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--n-users", type=int, default=16)
    p.add_argument("--per-user", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="un membro del tar per riga")
    args = p.parse_args()
    chosen = select(args.csv, args.n_users, args.per_user, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(m for _, m in chosen) + "\n")
    print(f"{len(chosen)} file di {len({u for u, _ in chosen})} utenti -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
