"""Dataloader del pretraining (passo 6; v10 §4.3, §4.6, §6.3-6.4): dal manifest (D9) ai batch impacchettati per `WearUsFM` e per il ciclo JEPA.
Solo numpy (la conversione in tensori e' in `to_model_inputs`, che importa torch): la logica si prova sul Mac.

Per ogni campione del batch:
1. **sessione** estratta con i pesi del manifest (righe di pretraining; le quote per classe e il tetto sono gia' nei pesi);
2. **finestra**: lunghezza fra `min_window_s` e `max_window_s` (proposta D10: 1-4 s, contesto variabile), inizio su un multiplo di 200 ms
   dall'inizio della prova (griglia del tokenizer RVQ), dentro una prova e, se `split_at_gaps` (proposta D9 §f), dentro un tratto senza salti
   dell'asse dei tempi; una finestra che tocca un tratto costante («buco») su un canale valido si riestrae (il campionamento li esclude);
3. **filtro** opzionale (passa-banda e notch a 50 e 60 Hz, v10 §4.3) con un margine ai lati, dentro il tratto: **i dati processati NON sono
   filtrati all'ingest**, e v10 vuole il filtraggio offline; finche' non c'e' una copia filtrata su disco, il filtro si applica qui (decisione di
   Simone: copia filtrata o filtro nel loader);
4. **normalizzazione di sessione**: un solo numero per sessione, condiviso fra i canali (v10 §4.3: mediana dei MAD dei canali validi, mai per
   finestra), stimato su tratti sparsi della sessione e messo in cache (`scale_cache`, da riempire offline con `estimate_session_scale`);
5. **target delle ancore** sul segnale pulito (`anchor_targets`), **maschera** (`training.masking`), codici di canale e insiemi di attenzione.

**Lettura a blocchi** (`block_s`): ogni processo tiene `block_pool` blocchi in memoria. Gli inizi di finestra ammessi di ogni tratto sono divisi
in **tessere fisse** di `block_s` secondi; un blocco e' una tessera di una sessione estratta coi pesi del manifest, scelta con probabilita'
proporzionale ai suoi inizi ammessi (= la tessera che contiene un inizio estratto uniformemente nella sessione), letta una volta con i dati fino alla
fine dell'ultima finestra possibile e al margine del filtro. Da ogni blocco si estraggono `windows_per_block` finestre con inizio uniforme nella
tessera e la stessa lunghezza della lettura diretta, poi si sostituisce. Cosi' **la distribuzione delle finestre e' quella della lettura
diretta**: la probabilita' di ogni sessione per finestra resta quella del manifest (ogni blocco da' lo stesso numero di finestre) e, dentro la
sessione, ogni inizio ammesso ha la stessa probabilita'; ogni finestra ha lo stesso margine del filtro della lettura diretta. Cambia solo che le
finestre di un batch vengono da un gruppo piu' piccolo di sessioni. (La prima versione, 03/10, cominciava il blocco su un inizio estratto e lo
tagliava alla fine del tratto: copriva poco l'inizio dei tratti e troppo la fine, e dava finestre corte in eccesso; review del 03/10.)

**Tratti costanti** (decisione di Simone del 01/10, `docs/decisioni.md`): una finestra non tocca un tratto costante di un canale valido, **con un
margine di 200 ms per lato** (`run_margin_s`, per il transitorio del filtro). **Integrita' del manifest:** se la riga ha `sidecar_sha256`, il
`metadata.json` della sessione deve avere quell'hash (un sidecar cambiato dopo il congelamento cambierebbe QC e tratti costanti senza dirlo).

Montaggi virtuali e sottocampionamento HD al volo (D6a) non ci sono ancora: il montaggio e' quello nativo.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

from wearusfm.data.processed import read_scale
from wearusfm.model import anchor_targets as AT
from wearusfm.model import channel_codes as CC
from wearusfm.training import masking as MK


@dataclass(frozen=True)
class LoaderConfig:
    min_window_s: float  # D10, proposta 1 s
    max_window_s: float  # D10, proposta 4 s
    split_at_gaps: bool  # D9 §f, proposta: si'
    mask: MK.MaskSpec  # D10
    k_neighbors: int
    filter_band_hz: tuple[float, float] | None  # None = nessun filtro; v10 §4.3: (20, 450)
    patch_ms: float = 25.0
    align_ms: float = 200.0  # griglia del tokenizer
    notch_hz: tuple[float, ...] = (50.0, 60.0)  # v10 §4.3: sempre entrambi
    filter_margin_s: float = 0.5
    run_margin_s: float = 0.2  # margine attorno ai tratti costanti (decisione del 01/10: «un margine di una patch da 200 ms per lato»)
    max_window_attempts: int = 20
    max_session_attempts: int = 20
    scale_chunks: int = 32
    scale_chunk_s: float = 2.0
    open_sessions: int = 64  # sessioni tenute aperte (memmap) per processo
    # lettura a blocchi (03/10/2026: la misura del job 59254061 dava 1,5-2,4 finestre/s per processo con una lettura piccola per finestra su
    # Lustre, contro ~8,5 richieste): None = una lettura per finestra (come prima)
    block_s: float | None = 30.0
    windows_per_block: int = 8
    block_pool: int = 16


# --- manifest ------------------------------------------------------------------------------------------------------------------------

@dataclass
class ManifestIndex:
    rows: list[dict]  # solo pretraining
    weights: np.ndarray  # normalizzati
    roots: list[Path]

    @classmethod
    def load(cls, manifest_path: Path, roots: list[Path], nested: str | None = None) -> "ManifestIndex":
        """`nested`: '0.125' / '0.25' / '0.5' per i manifest sottocampionati per soggetti (D16); None = tutto il pretraining."""
        doc = json.loads((gzip.open if str(manifest_path).endswith(".gz") else open)(manifest_path, "rb").read())
        cols = doc["columns"]
        rows = [dict(zip(cols, r)) for r in doc["rows"]]
        rows = [r for r in rows if r["split"] == "pretraining" and (nested is None or nested in r["nested"])]
        w = np.asarray([r["weight"] for r in rows], dtype=np.float64)
        if not len(rows) or w.sum() <= 0:
            raise ValueError("nessuna riga di pretraining con peso positivo")
        return cls(rows, w / w.sum(), [Path(p) for p in roots])

    def path_of(self, row: dict) -> Path:
        for root in self.roots:
            base = root / row["dataset"] / row["subject"]
            for cand in (base / row["session"], base) if row["session"] == "s" else (base / row["session"],):
                if (cand / "metadata.json").exists():
                    return cand
        raise FileNotFoundError(f"{row['dataset']}/{row['subject']}/{row['session']}: sessione non trovata nelle radici")


# --- sessione ------------------------------------------------------------------------------------------------------------------------

@dataclass
class SessionView:
    """Una sessione aperta in memmap: tratti campionabili e tratti costanti per canale, in coordinate dell'array."""

    path: Path
    meta: dict
    arr: np.ndarray  # (T, C) o (n_prove, T, C), int16 memmap
    fs: float
    scale: object  # None, scalare o (C,)
    qc_valid: np.ndarray
    spans: list[tuple[int | None, int, int, int]]  # (prova se 3D, inizio, fine, ancora della griglia)
    runs: list[tuple[int | None, int, int, int]] = field(default_factory=list)  # (prova se 3D, canale, inizio, fine)
    cache: dict = field(default_factory=dict)  # inizi accettati per configurazione (calcolati una volta per sessione aperta)

    @classmethod
    def open(cls, path: Path, split_at_gaps: bool, expected_sha256: str | None = None) -> "SessionView":
        raw = (path / "metadata.json").read_bytes()
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"{path}: metadata.json diverso da quello del manifest (sha256): sidecar cambiato dopo il congelamento")
        meta = json.loads(raw)
        arr = np.load(path / "data_int16.npy", mmap_mode="r")
        qc = np.asarray([c["qc_valid"] for g in meta["montage"]["groups"] for c in g["channels"]], dtype=bool)
        if len(qc) != arr.shape[-1]:
            raise ValueError(f"{path}: {len(qc)} flag qc_valid per {arr.shape[-1]} colonne")
        spans: list[tuple[int | None, int, int, int]] = []
        runs = []
        if arr.ndim == 3:
            spans = [(k, 0, arr.shape[1], 0) for k in range(arr.shape[0])]
            for e in meta.get("constant_runs") or []:
                runs.append((int(e["trial"]), int(e["channel"]), int(e["start"]), int(e["start"]) + int(e["n_samples"])))
        else:
            trials = meta.get("trials") or []
            bounds = [(int(t["offset"]), int(t["offset"]) + int(t["n_samples"])) for t in trials] if trials and all("offset" in t for t in trials) \
                else [(0, arr.shape[0])]
            cuts = sorted(int(g["index"]) for g in (meta.get("time_axis") or {}).get("gaps") or []) if split_at_gaps else []
            for a, b in bounds:
                inner = [c for c in cuts if a < c < b]
                edges = [a, *inner, b]
                spans.extend((None, s, e, a) for s, e in zip(edges, edges[1:]))
            for e in meta.get("constant_runs") or []:
                runs.append((None, int(e["channel"]), int(e["start"]), int(e["start"]) + int(e["n_samples"])))
        runs = [r for r in runs if qc[r[1]]]  # i buchi sui canali scartati non contano
        return cls(path, meta, arr, float(meta["native_fs_hz"]), read_scale(meta), qc, spans, runs)

    @property
    def montage(self) -> dict:
        return self.meta["montage"]

    def read(self, trial: int | None, start: int, stop: int) -> np.ndarray:
        """(C, n) float32 in unita' fisiche (dati / int16_scale)."""
        x = np.asarray(self.arr[trial, start:stop] if trial is not None else self.arr[start:stop], dtype=np.float32).T
        if self.scale is not None:
            x = x / (np.asarray(self.scale, dtype=np.float32)[:, None] if np.ndim(self.scale) else np.float32(self.scale))
        return x

    def band_limit_hz(self) -> np.ndarray:
        hi = [float(c["effective_band_hz"][1]) for g in self.montage["groups"] for c in g["channels"]]
        return np.minimum(np.asarray(hi), self.fs / 2.0)


def _filter(x: np.ndarray, fs: float, band: tuple[float, float], notch: tuple[float, ...]) -> np.ndarray:
    hi = min(band[1], 0.45 * fs)  # il passa-banda si adatta alla Nyquist (v10 §4.3: «adattato alla Nyquist effettiva»)
    sos = sps.butter(4, [band[0], hi], btype="bandpass", fs=fs, output="sos")
    for f0 in notch:
        if f0 < fs / 2.0 - 1.0:
            b, a = sps.iirnotch(f0, 30.0, fs=fs)
            sos = np.vstack([sos, sps.tf2sos(b, a)])
    return sps.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def read_window(view: SessionView, trial, start: int, stop: int, span: tuple[int, int], cfg: LoaderConfig) -> np.ndarray:
    """La finestra [start, stop) filtrata (se richiesto) con un margine dentro il tratto `span`, poi tagliata."""
    if cfg.filter_band_hz is None:
        return view.read(trial, start, stop)
    m = int(round(cfg.filter_margin_s * view.fs))
    a, b = max(span[0], start - m), min(span[1], stop + m)
    y = _filter(view.read(trial, a, b), view.fs, cfg.filter_band_hz, cfg.notch_hz)
    return y[:, start - a: start - a + (stop - start)]


def scale_seed(key: str) -> int:
    """Seme della stima della scala dalla chiave della sessione (dataset/soggetto/sessione): ogni processo e ogni rilancio danno la stessa scala."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def estimate_session_scale(view: SessionView, cfg: LoaderConfig, rng: np.random.Generator) -> float:
    """Mediana dei MAD dei canali validi (v10 §4.3), su `scale_chunks` tratti da `scale_chunk_s` sparsi nella sessione, filtrati come le
    finestre. Un numero per sessione, condiviso dai canali."""
    n = max(1, int(round(cfg.scale_chunk_s * view.fs)))
    # solo i tratti da cui si estraggono finestre (>= min_window_s): fra due salti ravvicinati restano tratti di pochi campioni, che il filtro
    # rifiuta (job 59254061: 195 sessioni fallite cosi', «padlen 39»)
    min_len = cfg.min_window_s * view.fs
    spans = [sp for sp in view.spans if sp[2] - sp[1] >= min_len]
    if not spans:
        raise ValueError(f"{view.path}: nessun tratto lungo almeno {cfg.min_window_s} s")
    lengths = np.array([e - s for _, s, e, _ in spans], dtype=np.float64)
    pieces = []
    for _ in range(cfg.scale_chunks):
        i = int(rng.choice(len(spans), p=lengths / lengths.sum()))
        trial, s, e, _ = spans[i]
        ln = min(n, e - s)
        t0 = int(rng.integers(s, e - ln + 1))
        pieces.append(read_window(view, trial, t0, t0 + ln, (s, e), cfg))
    x = np.concatenate(pieces, axis=1)[view.qc_valid]
    mad = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    scale = float(np.median(mad))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{view.path}: scala di sessione nulla o non finita")
    return scale


# --- finestre ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    trial: int | None
    start: int
    stop: int
    span: tuple[int, int]
    n_patches: int


def _accepted(view: SessionView, cfg: LoaderConfig, trial, s: int, e: int, anchor: int, ks: np.ndarray) -> np.ndarray:
    """Gli inizi `ks` la cui finestra (`window_at`) esiste: stessi conti di `window_at`, vettoriali (un test li confronta uno per uno)."""
    fs, patch = view.fs, cfg.patch_ms / 1000.0
    st = np.maximum(np.round(anchor + ks * (cfg.align_ms / 1000.0 * fs)).astype(np.int64), s)
    n_patch = np.floor(np.minimum(cfg.max_window_s * fs, e - st) / fs / patch + 1e-9).astype(np.int64)
    stop = st + np.ceil(n_patch * patch * fs - 1e-9).astype(np.int64)
    over = stop > e
    n_patch[over] -= 1
    stop[over] = st[over] + np.ceil(n_patch[over] * patch * fs - 1e-9).astype(np.int64)
    ok = n_patch * patch >= cfg.min_window_s - 1e-9
    m = int(math.ceil(cfg.run_margin_s * fs - 1e-9))
    for r in view.runs:
        if r[0] == trial:
            ok &= ~((r[2] - m < stop) & (r[3] + m > st))
    return ks[ok]


def admissible_starts(view: SessionView, cfg: LoaderConfig) -> list[tuple]:
    """Per tratto con almeno un inizio accettato: (prova, inizio, fine, ancora, k0, ks). Gli inizi ammessi sono `ancora + k * align` per k da k0,
    multipli di `align_ms` dall'ancora della prova con almeno `min_window_s` prima della fine del tratto; `ks` sono quelli la cui finestra non
    tocca un tratto costante (col margine). Estrarre k uniforme fra tutti i `ks` della sessione = la lettura diretta, senza tentativi."""
    key = (cfg.min_window_s, cfg.max_window_s, cfg.align_ms, cfg.patch_ms, cfg.run_margin_s)
    if key in view.cache:
        return view.cache[key]
    fs = view.fs
    align = cfg.align_ms / 1000.0 * fs
    min_n = cfg.min_window_s * fs
    out = []
    for trial, s, e, anchor in view.spans:
        k0 = math.ceil((s - anchor) / align - 1e-9)
        k1 = math.floor((e - min_n - anchor) / align + 1e-9)
        if k1 >= k0:
            ks = _accepted(view, cfg, trial, s, e, anchor, np.arange(k0, k1 + 1, dtype=np.int64))
            if len(ks):
                out.append((trial, s, e, anchor, k0, ks))
    view.cache[key] = out
    return out


def window_at(view: SessionView, cfg: LoaderConfig, cand: tuple, k: int) -> Window | None:
    """La finestra che comincia all'inizio ammesso k del tratto `cand`: lunghezza = il massimo fino a `max_window_s`, in patch intere. None se
    tocca un tratto costante di un canale valido (con `run_margin_s` per lato) o se resta piu' corta di `min_window_s`."""
    trial, s, e, anchor = cand[:4]
    fs, patch = view.fs, cfg.patch_ms / 1000.0
    st = max(int(round(anchor + k * cfg.align_ms / 1000.0 * fs)), s)
    n_patch = int(math.floor(min(cfg.max_window_s * fs, e - st) / fs / patch + 1e-9))
    stop = st + int(math.ceil(n_patch * patch * fs - 1e-9))
    if stop > e:
        n_patch -= 1
        stop = st + int(math.ceil(n_patch * patch * fs - 1e-9))
    if n_patch * patch < cfg.min_window_s - 1e-9:
        return None
    m = int(math.ceil(cfg.run_margin_s * fs - 1e-9))
    if any(r[0] == trial and r[2] - m < stop and r[3] + m > st for r in view.runs):
        return None  # tocca (col margine) un tratto costante su un canale valido
    return Window(trial, st, stop, (s, e), n_patch)


def sample_window(view: SessionView, cfg: LoaderConfig, rng: np.random.Generator) -> Window | None:
    """Inizio uniforme fra gli inizi accettati della sessione (`admissible_starts`). None se la sessione non ne ha."""
    cands = admissible_starts(view, cfg)
    if not cands:
        return None
    counts = np.array([len(c[5]) for c in cands], dtype=np.float64)
    cand = cands[int(rng.choice(len(cands), p=counts / counts.sum()))]
    return window_at(view, cfg, cand, int(cand[5][rng.integers(len(cand[5]))]))


@dataclass
class Block:
    """Gli inizi accettati `ks` di una tessera del tratto `cand`, coi dati grezzi (unita' fisiche) letti una volta su [data_lo, data_lo + n):
    bastano per ogni finestra che comincia nella tessera, col suo margine del filtro."""

    row: dict
    view: SessionView
    cand: tuple  # (prova, inizio, fine, ancora, k0, ks) del tratto
    ks: np.ndarray
    data_lo: int
    data: np.ndarray
    uses_left: int

    @property
    def trial(self):
        return self.cand[0]


def read_window_block(block: Block, win: Window, cfg: LoaderConfig) -> np.ndarray:
    """Come `read_window`, ma dai dati del blocco: margine del filtro dentro il tratto e dentro i dati letti."""
    a0 = block.data_lo
    if cfg.filter_band_hz is None:
        return block.data[:, win.start - a0: win.stop - a0]
    m = int(round(cfg.filter_margin_s * block.view.fs))
    a = max(win.span[0], win.start - m, a0)
    b = min(win.span[1], win.stop + m, a0 + block.data.shape[1])
    y = _filter(block.data[:, a - a0: b - a0], block.view.fs, cfg.filter_band_hz, cfg.notch_hz)
    return y[:, win.start - a: win.start - a + (win.stop - win.start)]


# --- batch ---------------------------------------------------------------------------------------------------------------------------

@dataclass
class PretrainBatch:
    signals: list[np.ndarray]  # (C_s, T_s) float32 normalizzati
    fs: list[float]
    counts: list[int]
    qc_valid: np.ndarray  # (C_tot,)
    codes: CC.AnatomyCodes  # impacchettati
    sets: CC.AttentionSets  # impacchettati
    visible: np.ndarray  # (C_tot, P_max)
    kind: np.ndarray  # (C_tot, P_max) tipo di maschera
    rvq_on: np.ndarray  # (C_tot,)
    anchor_targets: list  # AnchorTargets per campione
    n_patches: list[int]
    rows: list[dict]  # righe del manifest (dataset, unita', classe...) per il logging per topologia
    windows: list[Window]
    skipped_sessions: int = 0  # sessioni saltate finora da questo processo (scala non calcolabile)


class PretrainLoader:
    def __init__(self, index: ManifestIndex, cfg: LoaderConfig, scale_cache: dict | None = None):
        self.index, self.cfg = index, cfg
        self.scale_cache = scale_cache if scale_cache is not None else {}
        self._open: OrderedDict = OrderedDict()
        self.timings: dict[str, float] = {}  # secondi cumulati per fase (misura del ritmo)
        self.skipped: dict[str, str] = {}  # sessioni saltate perche' la scala non si calcola (es. MAD nullo): contate, mai in silenzio
        self._pool: list[Block] = []
        self.blocks_read = 0

    def _tick(self, name: str, t0: float) -> float:
        t1 = time.perf_counter()
        self.timings[name] = self.timings.get(name, 0.0) + (t1 - t0)
        return t1

    def view(self, row: dict) -> SessionView:
        key = (row["dataset"], row["subject"], row["session"])
        if key in self._open:
            self._open.move_to_end(key)
            return self._open[key]
        v = SessionView.open(self.index.path_of(row), self.cfg.split_at_gaps, row.get("sidecar_sha256"))
        self._open[key] = v
        if len(self._open) > self.cfg.open_sessions:
            self._open.popitem(last=False)
        return v

    def scale(self, row: dict, view: SessionView, rng: np.random.Generator) -> float:
        key = f"{row['dataset']}/{row['subject']}/{row['session']}"
        if key not in self.scale_cache:  # seme dalla chiave, non dal generatore del processo: la stessa sessione ha la stessa scala ovunque
            self.scale_cache[key] = estimate_session_scale(view, self.cfg, np.random.default_rng(scale_seed(key)))
        return self.scale_cache[key]

    def _session(self, rng: np.random.Generator) -> tuple[dict, SessionView]:
        """Una sessione estratta coi pesi del manifest, con la scala e almeno un inizio di finestra ammesso; quelle senza scala si saltano e si
        contano (mai in silenzio)."""
        for _ in range(self.cfg.max_session_attempts):
            row = self.index.rows[int(rng.choice(len(self.index.rows), p=self.index.weights))]
            key = f"{row['dataset']}/{row['subject']}/{row['session']}"
            if key in self.skipped:
                continue
            view = self.view(row)
            try:
                self.scale(row, view, rng)
            except (ValueError, FloatingPointError) as e:
                self.skipped[key] = f"{type(e).__name__}: {e}"
                continue
            return row, view
        raise RuntimeError(f"nessuna sessione con la scala in {self.cfg.max_session_attempts} estrazioni")

    def sample(self, rng: np.random.Generator) -> tuple[dict, SessionView, Window]:
        for _ in range(self.cfg.max_session_attempts):
            row, view = self._session(rng)
            win = sample_window(view, self.cfg, rng)
            if win is not None:
                return row, view, win
            self.skipped[f"{row['dataset']}/{row['subject']}/{row['session']}"] = "nessun inizio di finestra accettato"
        raise RuntimeError(f"nessuna finestra valida in {self.cfg.max_session_attempts} sessioni estratte")

    def _new_block(self, rng: np.random.Generator) -> Block | None:
        """La tessera che contiene un inizio estratto uniformemente fra gli inizi accettati della sessione (= tessera con probabilita'
        proporzionale ai suoi inizi accettati), coi dati fino alla fine dell'ultima finestra possibile piu' il margine del filtro."""
        row, view = self._session(rng)
        cands = admissible_starts(view, self.cfg)
        if not cands:
            self.skipped[f"{row['dataset']}/{row['subject']}/{row['session']}"] = "nessun inizio di finestra accettato"
            return None
        counts = np.array([len(c[5]) for c in cands], dtype=np.float64)
        cand = cands[int(rng.choice(len(cands), p=counts / counts.sum()))]
        trial, s, e, anchor, k0, ks = cand
        per = max(1, int(round(self.cfg.block_s / (self.cfg.align_ms / 1000.0))))  # inizi per tessera (tessere fisse dall'inizio del tratto)
        tile = (ks - k0) // per
        tile_ks = ks[tile == tile[rng.integers(len(ks))]]
        k_lo, k_hi = int(tile_ks[0]), int(tile_ks[-1])
        fs = view.fs
        align = self.cfg.align_ms / 1000.0 * fs
        m = int(round(self.cfg.filter_margin_s * fs)) if self.cfg.filter_band_hz is not None else 0
        lo = max(int(round(anchor + k_lo * align)), s)
        hi = min(e, int(round(anchor + k_hi * align)) + int(math.ceil(self.cfg.max_window_s * fs)) + 1)
        a, b = max(s, lo - m), min(e, hi + m)
        self.blocks_read += 1
        return Block(row, view, cand, tile_ks, a, view.read(trial, a, b), self.cfg.windows_per_block)

    def sample_from_blocks(self, rng: np.random.Generator) -> tuple[dict, Block, Window]:
        for _ in range(100 * self.cfg.max_session_attempts):
            while len(self._pool) < self.cfg.block_pool:
                blk = self._new_block(rng)
                if blk is not None:
                    self._pool.append(blk)
            i = int(rng.integers(len(self._pool)))
            blk = self._pool[i]
            win = window_at(blk.view, self.cfg, blk.cand, int(blk.ks[rng.integers(len(blk.ks))]))  # sempre valida: inizi accettati
            blk.uses_left -= 1
            if win is None or blk.uses_left <= 0:
                self._pool.pop(i)
            if win is not None:
                return blk.row, blk, win
        raise RuntimeError("nessuna finestra valida dai blocchi")

    def batch(self, batch_size: int, rng: np.random.Generator) -> PretrainBatch:
        t = time.perf_counter()
        if self.cfg.block_s is not None:
            picked = [self.sample_from_blocks(rng) for _ in range(batch_size)]
        else:
            picked = [self.sample(rng) for _ in range(batch_size)]
        t = self._tick("sessione_e_finestra", t)
        p_max = max(w.n_patches for _, _, w in picked)
        signals, fs, counts, qc, codes, sets, vis, kind, rvq, targets, rows, wins = ([] for _ in range(12))
        for row, src, win in picked:
            view = src.view if isinstance(src, Block) else src
            s = self.scale(row, view, rng)
            t = self._tick("scala", t)
            raw = read_window_block(src, win, self.cfg) if isinstance(src, Block) else \
                read_window(view, win.trial, win.start, win.stop, win.span, self.cfg)
            x = raw / np.float32(s)
            t = self._tick("lettura_e_filtro", t)
            layout = CC.layout_from_montage(view.montage)
            code = CC.anatomy_codes(view.montage)
            v, k = MK.generate_mask(layout, code.compartment_weights.argmax(axis=1), win.n_patches, p_max, row["rvq"] == "on", self.cfg.mask, rng)
            t = self._tick("maschera", t)
            signals.append(x)
            fs.append(view.fs)
            counts.append(layout.n_channels)
            qc.append(view.qc_valid)
            codes.append(code)
            sets.append(CC.attention_sets(layout, self.cfg.k_neighbors))
            t = self._tick("codici_e_vicini", t)
            vis.append(v)
            kind.append(k)
            rvq.append(np.full(layout.n_channels, row["rvq"] == "on"))
            targets.append(AT.anchor_targets(x, view.fs, view.band_limit_hz(), patch_ms=self.cfg.patch_ms))
            t = self._tick("target_ancore", t)
            rows.append(row)
            wins.append(win)
        packed_codes = CC.AnatomyCodes(*[np.concatenate([getattr(c, f) for c in codes]) for f in
                                         ("region", "compartment_weights", "muscle", "muscle_known", "topology")])
        return PretrainBatch(signals, fs, counts, np.concatenate(qc), packed_codes, CC.pack_attention_sets(sets), np.concatenate(vis),
                             np.concatenate(kind), np.concatenate(rvq), targets, [w.n_patches for w in wins], rows, wins, len(self.skipped))


def to_model_inputs(batch: PretrainBatch):
    """(ModelInputs, visible, rvq_on) in tensori per `WearUsFM` e `training.jepa`."""
    import torch

    from wearusfm.model.channel_identity import codes_to_tensors
    from wearusfm.model.fm import ModelInputs
    from wearusfm.model.local_encoder import sets_to_tensors

    inp = ModelInputs([torch.as_tensor(x) for x in batch.signals], list(batch.fs), codes_to_tensors(batch.codes), sets_to_tensors(batch.sets),
                      list(batch.counts), torch.as_tensor(batch.qc_valid))
    return inp, torch.as_tensor(batch.visible), torch.as_tensor(batch.rvq_on)
