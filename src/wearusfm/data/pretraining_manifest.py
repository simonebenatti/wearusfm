"""Manifest del pretraining (D9, passo 4; v10 §2.8, §10.3): dalle sessioni processate e dagli split dei soggetti a una riga per sessione con classe di
quota, peso di campionamento, contabilita' D_t / D_c e stato dell'ancora RVQ. Le scelte (classi, quote, tetto, pesi, RVQ) sono quelle della BOZZA
`docs/proposta_manifest_d9.md` finche' Simone non firma: qui sono parametri, non decisioni.

Contabilita' (v10 §10.3): D_t = time-patch unici = durata / patch; D_c = source-channel-patch unici = (durata x canali validi - tempo-canale escluso dai
buchi) / patch, dopo il QC e prima di ogni augmentation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

# Classi della bozza D9 (v10 §2.8: la classe e' del montaggio). Unita' di allocazione = dataset, o dataset:modo per Zhang.
CLASS_BY_UNIT = {
    "ninapro_db2": "A", "ninapro_db3": "A", "ninapro_db4": "A", "ninapro_db6": "A", "ninapro_db7": "A", "camargo2021": "A",
    "zhang2026:anatomical": "A",
    "kaifosh": "B", "emg2qwerty": "B", "emg2pose": "B", "grabmyo": "B", "putemg": "B", "ninapro_db5": "B", "ninapro_db8": "B", "ninapro_db10": "B",
    "zhang2026:random": "B",
    "capgmyo": "C", "csl_hdemg": "C", "hyser": "C",
}
# Ancora RVQ (D5b firmata, docs/decisioni.md): accesa, spenta, o da misurare (dataset nuovi: V2 e conferma del ramo)
RVQ_ON = {"camargo2021", "capgmyo", "emg2pose", "emg2qwerty", "grabmyo", "hyser", "kaifosh", "ninapro_db2", "ninapro_db3", "ninapro_db4", "ninapro_db6",
          "ninapro_db7", "ninapro_db8", "zhang2026"}  # DB8 (run 59104658) e Zhang (run 59108493): V2 e ramo 0 confermati
RVQ_OFF = {"putemg", "csl_hdemg", "ninapro_db5"}


def unit_of(dataset: str, session: str) -> str:
    if dataset == "zhang2026":
        return f"zhang2026:{session.split('/')[-1]}"
    return dataset


def rvq_status(dataset: str) -> str:
    return "on" if dataset in RVQ_ON else "off" if dataset in RVQ_OFF else "pending"


@dataclass
class SessionRow:
    dataset: str
    subject: str
    session: str
    split: str  # pretraining | test | benchmark
    nested: list[str]  # frazioni dei manifest sottocampionati a cui appartiene (solo pretraining)
    n_samples: int
    fs_hz: float
    n_channels: int
    n_valid: int
    excluded_channel_samples: int  # campioni x canale dei buchi su canali validi
    n_segments: int
    unit: str
    quota_class: str
    rvq: str
    sidecar_sha256: str
    weight: float = 0.0

    @property
    def hours(self) -> float:
        return self.n_samples / self.fs_hz / 3600

    def d_t(self, patch_s: float) -> float:
        return self.n_samples / self.fs_hz / patch_s

    def d_c(self, patch_s: float) -> float:
        return (self.n_samples * self.n_valid - self.excluded_channel_samples) / self.fs_hz / patch_s


def check_splits(splits: dict) -> None:
    """Le liste di un dataset devono essere disgiunte (un soggetto in test e in pretraining sarebbe un errore silenzioso)."""
    for ds, d in splits["datasets"].items():
        seen: dict[str, str] = {}
        for role in ("benchmark", "test", "pretraining"):
            for s in d.get(role, []):
                if s in seen:
                    raise ValueError(f"{ds}/{s}: sia in {seen[s]} sia in {role}")
                seen[s] = role


def split_of(splits: dict, dataset: str, subject: str, session: str | None = None) -> tuple[str, list[str]]:
    """Ruolo di una sessione: benchmark, test o pretraining per soggetto; `test_sessions` sposta in test singole sessioni di soggetti di pretraining
    (emg2pose: le registrazioni del test ufficiale per fasi nuove, decisione di Simone del 02/10/2026)."""
    d = splits["datasets"].get(dataset)
    if d is None:
        raise KeyError(f"{dataset}: nessuno split")
    if session is not None and session in set(d.get("test_sessions", [])):
        return "test", []
    if subject in d.get("benchmark", []):
        return "benchmark", []
    if subject in d.get("test", []):
        return "test", []
    if subject in d.get("pretraining", []):
        return "pretraining", sorted(k for k, v in d.get("nested", {}).items() if subject in v)
    raise KeyError(f"{dataset}/{subject}: soggetto assente dagli split")


def allocate(units: dict[str, tuple[str, float]], quota: dict[str, float], alpha: float, max_passes: float, epochs: float = 4.0,
             hours_total: float | None = None) -> dict:
    """Quota di campioni per unita' {unita': (classe, ore)}: dentro ogni classe pesi proporzionali a ore^alpha, con un tetto di `max_passes`
    passaggi per unita' (con `epochs` epoche di consumo delle ore totali); l'eccesso di un'unita' al tetto si ridistribuisce alle altre della classe.
    `hours_total` (default: somma delle ore delle unita') e' la base del consumo totale.
    Ritorna {"per_unit": {unita': (quota, passaggi)}, "unused_quota": {classe: quota non assegnabile}}."""
    if hours_total is None:
        hours_total = sum(h for _, h in units.values())
    budget = epochs * hours_total
    out, unused = {}, {}
    for cls, q in quota.items():
        members = {u: h for u, (c, h) in units.items() if c == cls and h > 0}
        cap = {u: max_passes * h / budget for u, h in members.items()}
        share, free, left = {}, dict(members), q
        while free and left > 1e-12:
            w = {u: h ** alpha for u, h in free.items()}
            s = sum(w.values())
            over = {u for u in free if left * w[u] / s > cap[u] + 1e-15}
            if not over:
                share.update({u: left * w[u] / s for u in free})
                left = 0.0
                break
            for u in over:
                share[u] = cap[u]
                left -= cap[u]
                del free[u]
        unused[cls] = max(left, 0.0)
        for u, h in members.items():
            out[u] = (share.get(u, 0.0), share.get(u, 0.0) * budget / h)
    return {"per_unit": out, "unused_quota": unused}


def assign_weights(rows: list[SessionRow], quota: dict[str, float], alpha: float, max_passes: float, epochs: float = 4.0) -> dict:
    """Peso di campionamento per sessione di pretraining: quota dell'unita' (allocate) ripartita fra le sue sessioni in proporzione alle ore; 0 per
    test e benchmark. Ritorna l'esito di allocate."""
    pre = [r for r in rows if r.split == "pretraining"]
    units: dict[str, tuple[str, float]] = {}
    for r in pre:
        c, h = units.get(r.unit, (r.quota_class, 0.0))
        units[r.unit] = (c, h + r.hours)
    alloc = allocate(units, quota, alpha, max_passes, epochs)
    for r in rows:
        if r.split != "pretraining":
            r.weight = 0.0
            continue
        share, _ = alloc["per_unit"][r.unit]
        r.weight = share * r.hours / units[r.unit][1]
    return alloc


def sort_rows(rows: list[SessionRow]) -> list[SessionRow]:
    """Ordine canonico: i pesi (somme in virgola mobile) si calcolano sempre nello stesso ordine, cosi' l'hash non dipende dall'ordine di scoperta."""
    return sorted(rows, key=lambda r: (r.dataset, r.subject, r.session))


def manifest_hash(params: dict, rows: Iterable[SessionRow]) -> str:
    """sha256 del contenuto canonico (parametri e righe ordinate). Le righe vanno pesate DOPO `sort_rows`: cosi' l'hash non dipende dall'ordine di
    scoperta delle sessioni (le somme dei pesi dipendono dall'ordine; review del codice, 02/10/2026)."""
    body = {"params": params, "rows": sorted((asdict(r) for r in rows), key=lambda d: (d["dataset"], d["subject"], d["session"]))}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
