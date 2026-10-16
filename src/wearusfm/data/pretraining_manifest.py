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
import math
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
            over = [u for u in free if left * w[u] / s > cap[u] + 1e-15]
            if not over:
                share.update({u: left * w[u] / s for u in free})
                left = 0.0
                break
            # ordine FISSO delle sottrazioni (review del critico, 16/10/2026): prima era un set, il cui ordine dipende da PYTHONHASHSEED, e
            # con due o piu' unita' al tetto nella stessa iterazione `left` cambiava all'ultima cifra (il derivato al 25% aveva due contenuti
            # possibili). Alfabetico inverso: fra gli ordini fissi e' quello che riproduce i quattro derivati gia' costruiti sul cluster il
            # 15/10 (hash di contenuto 80a214fd / 13d3dce8 / 52ec61f0 / 167923c3), quindi nessun file cambia
            for u in sorted(over, reverse=True):
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


# --- manifest derivati e annidati per soggetto (D16) ---------------------------------------------------------------------------------------

SUBSET_FRACTIONS = ("0.125", "0.25", "0.5")
DERIVED_FRACTIONS = SUBSET_FRACTIONS + ("1",)  # "1": il derivato al 100%, stessa regola sul corpus intero (accordo del 15/10/2026)
# campi della configurazione del loader da cui dipendono le durate medie delle finestre (`pretraining_loader.mean_window_s`)
WINDOW_CONFIG_FIELDS = ("min_window_s", "max_window_s", "align_ms", "patch_ms", "run_margin_s", "split_at_gaps")
DERIVED_RULE = ("base temporale: ore delle sessioni campionabili secondo window_s congelato (valore positivo; null = nessuna finestra, peso 0); "
                "D9 su quelle ore (pesi nella classe proporzionali a ore^alpha, tetto di max_passes passaggi a epochs epoche); se la classe C "
                "non raggiunge la sua quota perche' tutte le sue unita' sono al tetto, A resta alla sua quota e il solo residuo di C va a B, "
                "riallocato con gli stessi pesi e lo stesso tetto; A o B senza capacita': errore esplicito; nessun fattore comune sui pesi "
                "(accordo GPT-Claude del 15/10/2026, docs/accordo_operativo_finestra1_20261015.md)")


def session_key(dataset: str, subject: str, session: str) -> str:
    return f"{dataset}/{subject}/{session}"


def samplable_seconds(rows: list[SessionRow], window_s: dict) -> dict[str, bool]:
    """Per ogni riga di pretraining: campionabile (durata media delle finestre positiva e finita) o no (null dichiarato da
    `scripts/window_seconds.py`: nessuna finestra). Chiave mancante, valore non numerico, non finito, zero o negativo: errore di input."""
    out = {}
    for r in rows:
        if r.split != "pretraining":
            continue
        k = session_key(r.dataset, r.subject, r.session)
        if k not in window_s:
            raise ValueError(f"{k}: chiave assente da window_s (errore di input: serve la durata, anche null, per ogni sessione di pretraining)")
        v = window_s[k]
        if v is None:
            out[k] = False
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{k}: durata media delle finestre {v!r} non valida (positiva e finita, oppure null)")
        out[k] = True
    return out


def allocate_c_to_b(units: dict[str, tuple[str, float]], quota: dict[str, float], alpha: float, max_passes: float, epochs: float) -> dict:
    """`allocate` con la regola del residuo dell'accordo: A e B devono raggiungere la loro quota (altrimenti errore); la quota che C non puo'
    assegnare va a B, riallocata con gli stessi pesi e lo stesso tetto (se B non la assorbe: errore). Senza residuo e' `allocate`."""
    eps = 1e-12
    first = allocate(units, quota, alpha, max_passes, epochs)
    unused = first["unused_quota"]
    for cls in ("A", "B"):
        if unused.get(cls, 0.0) > eps:
            raise ValueError(f"la classe {cls} non sostiene la quota {quota[cls]}: {unused[cls]:.6g} non assegnabile col tetto di {max_passes} passaggi")
    others = sorted(c for c, q in unused.items() if q > eps and c not in ("A", "B", "C"))
    if others:
        raise ValueError(f"quota non assegnabile nelle classi {others}: la regola prevede solo il residuo di C")
    residual = unused.get("C", 0.0)
    if residual <= eps:
        return {**first, "residual_c_to_b": 0.0}
    second = allocate(units, {**quota, "B": quota["B"] + residual, "C": quota["C"] - residual}, alpha, max_passes, epochs)
    if second["unused_quota"].get("B", 0.0) > eps:
        raise ValueError(f"la classe B non assorbe il residuo di C ({residual:.6g}) col tetto di {max_passes} passaggi")
    return {**second, "residual_c_to_b": residual}


def assign_weights_derived(rows: list[SessionRow], window_s: dict, quota: dict[str, float], alpha: float, max_passes: float,
                           epochs: float = 4.0) -> dict:
    """Pesi dei manifest derivati (`DERIVED_RULE`): base temporale = ore campionabili; righe di pretraining non campionabili, test e benchmark
    a peso 0. Senza sessioni escluse e senza residuo, stesse operazioni e stessi pesi di `assign_weights` (test). Ritorna l'allocazione per
    unita' (quota, passaggi sulle ore campionabili), il residuo C->B e le ore nominali/campionabili/escluse."""
    ok = samplable_seconds(rows, window_s)
    units: dict[str, tuple[str, float]] = {}
    nominal = excluded = 0.0
    n_excluded = 0
    for r in rows:
        if r.split != "pretraining":
            continue
        nominal += r.hours
        if not ok[session_key(r.dataset, r.subject, r.session)]:
            excluded += r.hours
            n_excluded += 1
            continue
        c, h = units.get(r.unit, (r.quota_class, 0.0))
        units[r.unit] = (c, h + r.hours)
    alloc = allocate_c_to_b(units, quota, alpha, max_passes, epochs)
    for r in rows:
        if r.split != "pretraining" or not ok[session_key(r.dataset, r.subject, r.session)]:
            r.weight = 0.0
            continue
        share, _ = alloc["per_unit"][r.unit]
        r.weight = share * r.hours / units[r.unit][1]
    return {"per_unit": alloc["per_unit"], "residual_c_to_b": alloc["residual_c_to_b"], "hours_nominal": nominal,
            "hours_samplable": nominal - excluded, "hours_excluded": excluded, "sessions_excluded": n_excluded}


def time_shares(rows: list[SessionRow], window_s: dict) -> list[float]:
    """Le trasformazioni del sampler sui pesi (stessa aritmetica di `ManifestIndex.load` e `pretraining_loader.time_weighted_probs`, che i test
    confrontano): normalizzazione sulle righe di pretraining, probabilita' di estrazione = peso / durata media (0 senza finestre), di nuovo
    normalizzate; quota nel tempo della sessione = probabilita' x durata, normalizzata. Ritorna la quota nel tempo di ogni riga di pretraining."""
    pre = [r for r in rows if r.split == "pretraining"]
    w = [r.weight for r in pre]
    total = sum(w)
    p = []
    for r, x in zip(pre, w):
        d = window_s.get(session_key(r.dataset, r.subject, r.session))
        p.append((x / total) / d if d else 0.0)
    sp = sum(p)
    p = [x / sp for x in p]
    t = [x * (window_s.get(session_key(r.dataset, r.subject, r.session)) or 0.0) for r, x in zip(pre, p)]
    st = sum(t)
    return [x / st for x in t]


def sampled_distribution(rows: list[SessionRow], window_s: dict, epochs: float = 4.0, batch: int = 32) -> dict:
    """Massa, quote per classe e per unita', passaggi attesi per unita' (sulle ore campionabili) e endpoint, DOPO le trasformazioni del
    sampler (`time_shares`). Endpoint: passi a E epoche = floor(E x finestre per epoca / batch), con un solo floor sul totale, cosi' la dose
    non supera l'esposizione nominale; dose effettiva = passi x batch / finestre per epoca."""
    pre = [r for r in rows if r.split == "pretraining"]
    t = time_shares(rows, window_s)
    units: dict[str, dict] = {}
    hours = 0.0
    inv_window = 0.0  # finestre per secondo di tempo campionato: somma delle quote nel tempo / durata
    for r, x in zip(pre, t):
        d = window_s.get(session_key(r.dataset, r.subject, r.session))
        if not d:
            continue
        hours += r.hours
        inv_window += x / d
        u = units.setdefault(r.unit, {"class": r.quota_class, "share": 0.0, "hours": 0.0})
        u["share"] += x
        u["hours"] += r.hours
    classes: dict[str, float] = {}
    for u in units.values():
        classes[u["class"]] = classes.get(u["class"], 0.0) + u["share"]
    passes = {k: u["share"] * epochs * hours / u["hours"] for k, u in units.items()}
    windows_per_epoch = hours * 3600.0 * inv_window
    steps = math.floor(epochs * windows_per_epoch / batch)
    return {"mass": sum(t), "classes": dict(sorted(classes.items())), "shares": {k: units[k]["share"] for k in sorted(units)},
            "passes": {k: passes[k] for k in sorted(passes)}, "max_passes": max(passes.values()), "hours_samplable": hours,
            "mean_window_s": 1.0 / inv_window, "windows_per_epoch": windows_per_epoch, "batch": batch, "epochs": epochs,
            "steps_at_epochs": steps, "effective_epochs": steps * batch / windows_per_epoch,
            "sessions": len(pre), "subjects": len({(r.dataset, r.subject) for r in pre})}


def derived_checks(rows: list[SessionRow], window_s: dict, params: dict, residual: float, tol: float = 1e-9) -> list[str]:
    """Le verifiche di un derivato (problemi; vuota se tutto torna), sul file e DOPO le trasformazioni del sampler (`sampled_distribution`):
    durate valide per ogni riga di pretraining (`samplable_seconds`: chiave mancante o valore non valido = problema), pesi finiti e non
    negativi, peso 0 sulle righe non campionabili, somma 1 dei pesi di pretraining prima della normalizzazione del loader, quote dopo il
    sampler A = quota_A, B = quota_B + residuo, C = quota_C - residuo, passaggi entro il tetto (review del critico, 16/10/2026)."""
    try:
        ok = samplable_seconds(rows, window_s)
    except ValueError as e:
        return [f"durate: {e}"]
    problems = []
    pre = [r for r in rows if r.split == "pretraining"]
    weights = [r.weight for r in rows]
    if not all(isinstance(w, (int, float)) and math.isfinite(w) and w >= 0 for w in weights):
        return ["pesi non finiti o negativi"]
    bad = [session_key(r.dataset, r.subject, r.session) for r in pre if not ok[session_key(r.dataset, r.subject, r.session)] and r.weight != 0]
    if bad:
        problems.append(f"righe senza finestre con peso diverso da 0: {bad[:3]}")
    if abs(sum(r.weight for r in pre) - 1.0) > tol:
        problems.append(f"somma dei pesi di pretraining {sum(r.weight for r in pre)!r}, attesa 1")
    dist = sampled_distribution(rows, window_s, params.get("epochs", 4.0))
    q = params["quota"]
    expected = {"A": q["A"], "B": q["B"] + residual, "C": q["C"] - residual}
    for cls, want in expected.items():
        got = dist["classes"].get(cls, 0.0)
        if abs(got - want) > tol:
            problems.append(f"quota di {cls} {got!r} dopo il sampler, attesa {want!r}")
    if dist["max_passes"] > params["max_passes"] * (1 + tol):
        problems.append(f"{dist['max_passes']:.6f} passaggi dopo il sampler, oltre il tetto {params['max_passes']}")
    return problems


def derived_manifest(doc: dict, fraction: str, window_s: dict, inputs: dict, code_commit: str = "") -> dict:
    """Il manifest derivato di `fraction` (D16; "1" = corpus intero) dal manifest congelato `doc`: sessioni di pretraining dei soggetti del
    sottoinsieme firmato (`nested` della riga; tutte per "1"), pesate con `assign_weights_derived`; test e benchmark invariati (peso 0); le
    altre sessioni di pretraining fuori. `inputs`: provenienza (SHA del genitore, hash dichiarato e di contenuto, window_s, configurazione
    delle finestre). Le verifiche di massa, quote e tetto si fanno sulla distribuzione campionata (`sampled_distribution`)."""
    if fraction not in DERIVED_FRACTIONS:
        raise ValueError(f"frazione {fraction!r}: una di {DERIVED_FRACTIONS}")
    if doc["params"].get("nested"):
        raise ValueError("il manifest d'origine e' gia' derivato")
    rows = [SessionRow(**dict(zip(doc["columns"], r))) for r in doc["rows"]]
    keep = sort_rows([r for r in rows if r.split != "pretraining" or fraction == "1" or fraction in r.nested])
    p = doc["params"]
    out = assign_weights_derived(keep, window_s, p["quota"], p["alpha"], p["max_passes"], p.get("epochs", 4.0))
    dist = sampled_distribution(keep, window_s, p.get("epochs", 4.0))
    problems = derived_checks(keep, window_s, p, out["residual_c_to_b"])
    if problems:
        raise ValueError(f"frazione {fraction}: " + "; ".join(problems))
    params = {**p, "nested": fraction, "derived_rule": DERIVED_RULE, "derived_inputs": inputs,
              "derived_hours": {k: out[k] for k in ("hours_nominal", "hours_samplable", "hours_excluded", "sessions_excluded")},
              "residual_c_to_b": out["residual_c_to_b"],
              "virtual_montages": "D6a: le quote sono della classe d'origine; i montaggi virtuali delle griglie HD restano in classe C"}
    summary = {"sampled": {k: dist[k] for k in ("mass", "classes", "max_passes", "hours_samplable", "mean_window_s", "windows_per_epoch",
                                                 "sessions", "subjects")},
               "per_unit": {u: {"share": dist["shares"][u], "passes": dist["passes"][u]} for u in dist["shares"]}}
    return {"params": params, "hash": manifest_hash(params, keep), "code_commit": code_commit, "created_s": None, "summary": summary,
            "columns": list(asdict(keep[0]).keys()) if keep else [], "rows": [list(asdict(r).values()) for r in keep]}
