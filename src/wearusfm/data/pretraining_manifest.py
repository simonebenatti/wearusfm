"""Manifest del pretraining (D9, passo 4; v10 §2.8, §10.3): dalle sessioni processate e dagli split dei soggetti a una riga per sessione con classe di
quota, peso di campionamento, contabilita' D_t / D_c e stato dell'ancora RVQ. Le scelte (classi, quote, tetto, pesi, RVQ) sono quelle della BOZZA
`docs/proposta_manifest_d9.md` finche' Simone non firma: qui sono parametri, non decisioni.

Contabilita' (v10 §10.3): D_t = time-patch unici = durata / patch; D_c = source-channel-patch unici = (durata x canali validi - tempo-canale escluso dai
buchi) / patch, dopo il QC e prima di ogni augmentation.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

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
          "ninapro_db7", "ninapro_db8", "ninapro_db10", "zhang2026"}  # DB8 (run 59104658), Zhang (59108493), DB10 (59183099): V2 e ramo 0 confermati
RVQ_OFF = {"putemg", "csl_hdemg", "ninapro_db5"}
# Finestre (s) per la tabella di sensibilita' del tempo inutilizzabile (bozza D9 §f, contesto di D10): NON e' una soglia, nessuna scelta dipende da qui
WINDOWS_S = (1.0, 2.0, 4.0, 8.0)


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
    # salti dell'asse dei tempi (dal campo time_axis del sidecar, dove c'e'): oggi NON spezzano i segmenti, una finestra puo' attraversarli
    n_time_gaps: int = 0
    missing_time_s: float = 0.0  # durata coperta dai timestamp meno durata dei campioni
    max_gap_s: float = 0.0
    gaps_truncated: bool = False  # il sidecar elenca al piu' 10.000 salti
    gaps_out_of_range: int = 0  # posizioni di salti fuori dall'array (incoerenza fra sidecar e dati: va guardata)
    # tempo (s) in tratti contigui piu' corti di ciascuna finestra di WINDOWS_S: solo prove, e prove spezzate anche ai salti elencati
    short_segments_s: list[float] = field(default_factory=list)
    short_segments_gaps_s: list[float] = field(default_factory=list)
    outside_trials_s: float = 0.0  # tempo dell'array fuori da ogni prova (contato in D_t, ma non dentro un tratto)

    @property
    def hours(self) -> float:
        return self.n_samples / self.fs_hz / 3600

    def d_t(self, patch_s: float) -> float:
        return self.n_samples / self.fs_hz / patch_s

    def d_c(self, patch_s: float) -> float:
        return (self.n_samples * self.n_valid - self.excluded_channel_samples) / self.fs_hz / patch_s


def segment_lengths(n_samples: int, trials: list[dict] | None, gap_indices: Iterable[int] = ()) -> list[int]:
    """Lunghezze (campioni) dei tratti contigui: le prove (`offset`, `n_samples`) se ci sono, altrimenti l'array intero; ogni tratto si spezza agli
    indici dei salti (indice del primo campione dopo il salto) che cadono strettamente al suo interno."""
    spans = [(int(t["offset"]), int(t["offset"]) + int(t["n_samples"])) for t in trials] if trials else [(0, int(n_samples))]
    cuts = sorted(set(int(i) for i in gap_indices))
    out = []
    for a, b in spans:
        inner = cuts[bisect.bisect_right(cuts, a):bisect.bisect_left(cuts, b)]
        edges = [a, *inner, b]
        out.extend(e2 - e1 for e1, e2 in zip(edges, edges[1:]))
    return out


def short_time_s(lengths: Iterable[int], fs_hz: float, windows_s: Iterable[float] = WINDOWS_S) -> list[float]:
    """Per ogni finestra: secondi contenuti in tratti piu' corti della finestra (una finestra non ci sta, quel tempo non si campiona mai)."""
    lens = [int(n) for n in lengths]
    return [sum(n for n in lens if n < w * fs_hz) / fs_hz for w in windows_s]


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


# --- manifest annidati per soggetto (D16) --------------------------------------------------------------------------------------------------

NESTED_FRACTIONS = ("0.125", "0.25", "0.5")
NESTED_RULE = ("regole di D9 rifatte sulle ore del sottoinsieme (allocate: quote di classe, ore^alpha, tetto di passaggi); la quota che una "
               "classe non puo' assegnare perche' tutte le sue unita' sono al tetto va alle unita' sotto il tetto delle altre classi, in "
               "proporzione alla loro quota, senza superare il tetto; nessun peso di un'unita' al tetto aumenta (GPT e Claude, 15/10/2026)")


def assign_weights_nested(rows: list[SessionRow], quota: dict[str, float], alpha: float, max_passes: float, epochs: float = 4.0) -> dict:
    """Come `assign_weights`, piu' la gestione esplicita del residuo: con `allocate` una classe le cui unita' sono tutte al tetto lascia quota
    non assegnata, e la normalizzazione del loader la spalmerebbe anche sulle unita' al tetto, oltre `max_passes`. Qui il residuo va alle
    unita' sotto il tetto (di qualunque classe), in proporzione alla loro quota, fino al tetto; le quote finali sommano a 1, quindi la
    distribuzione campionata e' esattamente quella dei pesi. Senza residuo il risultato e' quello di `assign_weights` (stesse operazioni).
    Ritorna {"per_unit": {unita': (quota, passaggi)}, "residual": {classe: quota ridistribuita}, "receivers": [unita']}."""
    pre = [r for r in rows if r.split == "pretraining"]
    units: dict[str, tuple[str, float]] = {}
    for r in pre:
        c, h = units.get(r.unit, (r.quota_class, 0.0))
        units[r.unit] = (c, h + r.hours)
    alloc = allocate(units, quota, alpha, max_passes, epochs)
    share = {u: s for u, (s, _) in alloc["per_unit"].items()}
    residual = {c: q for c, q in alloc["unused_quota"].items() if q > 1e-15}
    budget = epochs * sum(h for _, h in units.values())
    cap = {u: max_passes * h / budget for u, (_, h) in units.items()}
    left, receivers = sum(residual.values()), set()
    while left > 1e-15:
        free = {u: share[u] for u in share if share[u] < cap[u] - 1e-15}
        if not free:
            raise ValueError(f"quota residua {left:.3g} non assegnabile: tutte le unita' sono al tetto di {max_passes} passaggi")
        total = sum(free.values())
        given = 0.0
        for u, s in free.items():
            add = min(left * s / total, cap[u] - s)
            share[u] += add
            given += add
            receivers.add(u)
        left -= given
    for r in rows:
        if r.split != "pretraining":
            r.weight = 0.0
            continue
        r.weight = share[r.unit] * r.hours / units[r.unit][1]
    return {"per_unit": {u: (share[u], share[u] * budget / units[u][1]) for u in units}, "residual": residual, "receivers": sorted(receivers)}


def sampled_distribution(rows: list[SessionRow], epochs: float = 4.0) -> dict:
    """La distribuzione che campiona il loader (pesi normalizzati sulle righe di pretraining, `ManifestIndex.load`): quote per unita' e per
    classe, passaggi per unita' a `epochs` epoche, ore, soggetti e sessioni. I controlli dei manifest si fanno su questa, non sui conti
    intermedi di `allocate`."""
    pre = [r for r in rows if r.split == "pretraining"]
    total_w = sum(r.weight for r in pre)
    hours = sum(r.hours for r in pre)
    units: dict[str, dict] = {}
    for r in pre:
        u = units.setdefault(r.unit, {"class": r.quota_class, "w": 0.0, "hours": 0.0})
        u["w"] += r.weight / total_w
        u["hours"] += r.hours
    classes: dict[str, float] = {}
    for u in units.values():
        classes[u["class"]] = classes.get(u["class"], 0.0) + u["w"]
    passes = {k: u["w"] * epochs * hours / u["hours"] for k, u in units.items() if u["hours"] > 0}
    return {"classes": classes, "shares": {k: u["w"] for k, u in units.items()}, "passes": passes, "max_passes": max(passes.values()),
            "hours": hours, "sessions": len(pre), "subjects": len({(r.dataset, r.subject) for r in pre}), "weight_sum": total_w}


def nested_manifest(doc: dict, fraction: str, source_sha256: str, code_commit: str = "", source_content_hash: str | None = None) -> dict:
    """Il manifest annidato di `fraction` dal manifest congelato `doc` (D16): le sessioni di pretraining dei soggetti del sottoinsieme
    (`nested` della riga, dagli split firmati), pesate con `assign_weights_nested`; test e benchmark invariati (peso 0); le altre sessioni di
    pretraining fuori. Parametri del manifest d'origine piu' frazione, origine, regola e residuo; hash come `manifest_hash`."""
    if fraction not in NESTED_FRACTIONS:
        raise ValueError(f"frazione {fraction!r}: una di {NESTED_FRACTIONS}")
    if doc["params"].get("nested"):
        raise ValueError("il manifest d'origine e' gia' annidato")
    rows = [SessionRow(**dict(zip(doc["columns"], r))) for r in doc["rows"]]
    keep = sort_rows([r for r in rows if r.split != "pretraining" or fraction in r.nested])
    p = doc["params"]
    alloc = assign_weights_nested(keep, p["quota"], p["alpha"], p["max_passes"], p.get("epochs", 4.0))
    params = {**p, "nested": fraction, "nested_rule": NESTED_RULE,
              "nested_residual": {"per_class": alloc["residual"], "receivers": alloc["receivers"]},
              "derived_from": {"file_sha256": source_sha256, "hash": doc.get("hash"), "content_hash": source_content_hash,
                               "version": p.get("version"), "revision": p.get("revision", {}).get("version")}}
    dist = sampled_distribution(keep, p.get("epochs", 4.0))
    if dist["max_passes"] > p["max_passes"] * (1 + 1e-9):
        raise ValueError(f"frazione {fraction}: {dist['max_passes']:.6f} passaggi nella distribuzione campionata, oltre il tetto {p['max_passes']}")
    summary = {"sampled": {"classes": dist["classes"], "max_passes": dist["max_passes"], "hours": dist["hours"], "sessions": dist["sessions"],
                           "subjects": dist["subjects"], "weight_sum": dist["weight_sum"]},
               "per_unit": {u: {"share": dist["shares"][u], "passes": dist["passes"][u]} for u in sorted(dist["shares"])}}
    return {"params": params, "hash": manifest_hash(params, keep), "code_commit": code_commit, "created_s": None, "summary": summary,
            "columns": list(asdict(keep[0]).keys()) if keep else [], "rows": [list(asdict(r).values()) for r in keep]}
