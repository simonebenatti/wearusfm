"""Codici di canale dai metadati di montaggio dei sidecar (passo 6; v10 §3.4-3.6, §4.5, §5.2-5.4). Solo numpy: si prova sul Mac senza torch.

Due percorsi d'identita' (v10 §3.5), letti dal dizionario `montage` del sidecar (`ingest/common.py`, `montage_to_dict`):

- **geometrico relativo**: angolo sull'anello, riga e colonna sulla griglia, nulla per i canali sparsi. Le grandezze sono **a coppie** e definite
  solo fra canali dello **stesso gruppo metrico** (anello o griglia): un angolo assoluto, o una distanza fra gruppi diversi, non hanno significato
  (la fascia puo' essere ruotata, l'orientamento e' spesso ignoto, i gruppi non hanno distanze dichiarate nei sidecar). Cosi' la simmetria
  dichiarata emerge da sola (v10 §3.4, «symmetry-capable senza essere symmetry-imposing»): ruotare tutti gli angoli di un anello, o traslare una
  griglia, non cambia nulla; la riflessione cambia il segno degli spostamenti angolari (equivarianza, non invarianza).
- **anatomico**: regione, compartimento (al polso: settore) e muscolo, al livello realmente noto (`precision`), con i pesi soft della funzione
  atlante dove ci sono (v10 §4.5). Mai etichette inventate: una chiave fuori dalla tassonomia e' un errore.

Unita': gli spostamenti angolari sono in radianti sul cerchio unitario, quelli di griglia in passi di indice. I sidecar non portano distanze in mm
(`inter_electrode_distance_mm` non e' serializzato): le due scale non sono confrontabili fra topologie, e il bias di §5.3 va appreso per topologia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wearusfm.metadata import taxonomy as T

TOPOLOGIES = ("ring", "grid_2d", "sparse")
UNK = "<unk>"
# Vocabolari dell'embedding anatomico: indice 0 = ignoto, uno per livello (v10 §5.2, «embedding UNK appreso per livello»)
REGION_KEYS = (UNK, *T.REGIONS)
COMPARTMENT_KEYS = (UNK, *T.COMPARTMENTS)  # comprende i quattro settori del polso
MUSCLE_KEYS = (UNK, *T.MUSCLES)
SIDE_KEYS = (UNK, "left", "right")  # lato dell'arto (campo `chirality` del sidecar; v10 §3.6, coordinate anatomiche)
MAX_GROUPS = 8  # ordine del gruppo nel montaggio (sistema del sensore, v10 §3.6); oltre, l'ultimo indice
N_SENSOR_POS = 8  # posizione nel sistema del sensore: anello cos/sin(m*angolo), m = 1..4; griglia cos/sin(pi*m*x) per riga e colonna, m = 1, 2
_ROUND = 9  # cifre per confrontare distanze uguali a meno del rumore di virgola mobile (rotazioni, griglie)


@dataclass(frozen=True)
class ChannelLayout:
    """Un canale per riga, nell'ordine delle colonne dell'array (gruppi, poi canali: come `qc_valid`)."""

    group: np.ndarray  # (C,) indice del gruppo nel montaggio
    topology: np.ndarray  # (C,) indice in TOPOLOGIES
    ring_angle: np.ndarray  # (C,) radianti in [0, 2pi), NaN fuori dagli anelli
    grid_rc: np.ndarray  # (C, 2) riga e colonna, NaN fuori dalle griglie
    qc_valid: np.ndarray  # (C,) bool
    chirality: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return int(len(self.group))


def layout_from_montage(montage: dict) -> ChannelLayout:
    group, topo, ang, rc, qc, chir = [], [], [], [], [], []
    for gi, g in enumerate(montage["groups"]):
        t = g["topology"]
        if t not in TOPOLOGIES:
            raise ValueError(f"gruppo {g.get('group_id')!r}: topologia {t!r} sconosciuta")
        for c in g["channels"]:
            sc = c["sensor_coords"]
            a, r, col = sc.get("ring_angle_deg"), sc.get("grid_row"), sc.get("grid_col")
            if t == "ring":
                if a is None:
                    raise ValueError(f"gruppo {g.get('group_id')!r}: canale di anello senza ring_angle_deg (mai geometria inventata)")
                ang.append(math.radians(float(a) % 360.0))
                rc.append((math.nan, math.nan))
            elif t == "grid_2d":
                if r is None or col is None:
                    raise ValueError(f"gruppo {g.get('group_id')!r}: canale di griglia senza grid_row/grid_col")
                ang.append(math.nan)
                rc.append((float(r), float(col)))
            else:
                ang.append(math.nan)
                rc.append((math.nan, math.nan))
            group.append(gi)
            topo.append(TOPOLOGIES.index(t))
            qc.append(bool(c.get("qc_valid", True)))
            chir.append(str(c.get("chirality", "unknown")))
    return ChannelLayout(np.asarray(group, dtype=np.int64), np.asarray(topo, dtype=np.int64), np.asarray(ang, dtype=np.float64),
                         np.asarray(rc, dtype=np.float64).reshape(-1, 2), np.asarray(qc, dtype=bool), tuple(chir))


def _wrap(x: np.ndarray) -> np.ndarray:
    """Angolo in [-pi, pi)."""
    return (x + math.pi) % (2 * math.pi) - math.pi


def relative_geometry(layout: ChannelLayout) -> dict[str, np.ndarray]:
    """Grandezze a coppie (C, C), dal canale i (riga) al canale j (colonna), definite solo dentro lo stesso gruppo metrico; altrove NaN.

    `d_angle` = angolo di j meno angolo di i, in [-pi, pi); `d_row`, `d_col` = spostamento di griglia; `distance` = |d_angle| sugli anelli,
    norma euclidea in passi di indice sulle griglie (scale diverse: vedi il docstring del modulo)."""
    same = layout.group[:, None] == layout.group[None, :]
    ring = same & (layout.topology[:, None] == TOPOLOGIES.index("ring"))
    grid = same & (layout.topology[:, None] == TOPOLOGIES.index("grid_2d"))
    with np.errstate(invalid="ignore"):
        d_angle = np.where(ring, _wrap(layout.ring_angle[None, :] - layout.ring_angle[:, None]), np.nan)
        d_row = np.where(grid, layout.grid_rc[None, :, 0] - layout.grid_rc[:, None, 0], np.nan)
        d_col = np.where(grid, layout.grid_rc[None, :, 1] - layout.grid_rc[:, None, 1], np.nan)
        distance = np.where(ring, np.abs(d_angle), np.where(grid, np.hypot(d_row, d_col), np.nan))
    return {"metric": ring | grid, "ring": ring, "grid": grid, "d_angle": d_angle, "d_row": d_row, "d_col": d_col, "distance": distance}


def neighbors(layout: ChannelLayout, k: int, *, only_valid: bool = True) -> np.ndarray:
    """(C, k) indici dei k vicini piu' vicini di ogni canale nello stesso gruppo metrico, escluso il canale stesso; -1 dove non ce ne sono
    abbastanza. I canali sparsi non hanno vicini metrici (v10 §5.4: su Camargo il vicinato metrico e' vuoto). Con `only_valid` i canali scartati
    dal QC non sono vicini di nessuno e non hanno vicini.

    A parita' di distanza si sceglie con la geometria relativa, mai con l'indice di colonna: prima lo spostamento angolare positivo (anelli),
    poi riga e colonna crescenti (griglie). Cosi' l'uscita dipende solo dalla geometria relativa: ruotare un anello o traslare una griglia non la
    cambia, e permutare le colonne la permuta allo stesso modo."""
    if k < 1:
        raise ValueError("k deve essere >= 1")
    rel = relative_geometry(layout)
    c = layout.n_channels
    ok = rel["metric"] & ~np.eye(c, dtype=bool)
    if only_valid:
        ok &= layout.qc_valid[:, None] & layout.qc_valid[None, :]
    dist = np.round(np.where(ok, rel["distance"], np.inf), _ROUND)
    tie_a = np.round(np.nan_to_num(-rel["d_angle"], nan=0.0), _ROUND)  # angolo positivo prima
    tie_r = np.nan_to_num(rel["d_row"], nan=0.0)
    tie_c = np.nan_to_num(rel["d_col"], nan=0.0)
    out = np.full((c, k), -1, dtype=np.int64)
    for i in range(c):
        order = np.lexsort((tie_c[i], tie_r[i], tie_a[i], dist[i]))
        chosen = [j for j in order[:k] if np.isfinite(dist[i, j])]
        out[i, : len(chosen)] = chosen
    return out


@dataclass(frozen=True)
class AnatomyCodes:
    """Indici nei vocabolari REGION_KEYS / COMPARTMENT_KEYS / MUSCLE_KEYS (0 = ignoto) e pesi sui compartimenti (una riga per canale, somma 1:
    one-hot dove il compartimento e' noto, pesi della funzione atlante dove ci sono, tutto su «ignoto» altrimenti)."""

    region: np.ndarray  # (C,)
    compartment_weights: np.ndarray  # (C, len(COMPARTMENT_KEYS))
    muscle: np.ndarray  # (C,)
    muscle_known: np.ndarray  # (C,) bool: dove si applica il dropout del livello muscolo (v10 §5.2)
    topology: np.ndarray  # (C,) indice in TOPOLOGIES
    side: np.ndarray  # (C,) indice in SIDE_KEYS
    group: np.ndarray  # (C,) ordine del gruppo nel montaggio, al piu' MAX_GROUPS - 1
    sensor_pos: np.ndarray  # (C, N_SENSOR_POS) posizione nel sistema del sensore (zeri sui canali sparsi)


def _region_from(chain_region: str, field: str | None, where: str) -> int:
    if field is not None and field != chain_region:
        raise ValueError(f"{where}: regione {field!r} incoerente con la tassonomia ({chain_region!r})")
    return REGION_KEYS.index(chain_region)


def anatomy_codes(montage: dict) -> AnatomyCodes:
    """Codici anatomici per canale, secondo `precision` (v10 §4.5, «Regola di annotazione»): muscolo -> muscolo, compartimento e regione per
    lookup; compartimento o settore -> compartimento e regione; regione -> regione dal campo (la funzione atlante dell'avambraccio distale usa
    chiavi di compartimento che nell'albero stanno sotto l'avambraccio prossimale: vale il campo, `taxonomy.py` punto 4) e pesi soft se ci sono;
    ignoto -> tutto ignoto."""
    n_comp = len(COMPARTMENT_KEYS)
    region, weights, muscle, known, topo, side, group, pos = [], [], [], [], [], [], [], []
    for gi, g in enumerate(montage["groups"]):
        for i, c in enumerate(g["channels"]):
            a = c["anatomical_identity"]
            where = f"{g.get('group_id')}[{i}]"
            prec = a.get("precision", "unknown")
            w = np.zeros(n_comp)
            r, m = 0, 0
            if prec == "muscle":
                key = a.get("muscle")
                chain = T.ancestors(key)  # KeyError se la chiave non e' nella tassonomia
                if key not in T.MUSCLES:
                    raise ValueError(f"{where}: precisione «muscle» ma {key!r} non e' un muscolo")
                r, m = _region_from(chain[2], a.get("region"), where), MUSCLE_KEYS.index(key)
                w[COMPARTMENT_KEYS.index(chain[1])] = 1.0
            elif prec in ("compartment", "sector"):
                key = a.get("compartment") if prec == "compartment" else a.get("sector")
                chain = T.ancestors(key)
                if key not in T.COMPARTMENTS:
                    raise ValueError(f"{where}: precisione {prec!r} ma {key!r} non e' un compartimento")
                r = _region_from(chain[1], a.get("region"), where)
                w[COMPARTMENT_KEYS.index(key)] = 1.0
            elif prec == "region":
                if a.get("region") not in T.REGIONS:
                    raise KeyError(f"{where}: regione sconosciuta {a.get('region')!r} (mai etichette inventate)")
                r = REGION_KEYS.index(a["region"])
                soft = a.get("soft_compartment_weights")
                if soft:
                    for comp, v in soft.items():
                        w[COMPARTMENT_KEYS.index(comp) if comp in T.COMPARTMENTS else _unknown_compartment(comp, where)] += float(v)
                    w /= w.sum()
                else:
                    w[0] = 1.0
            elif prec == "unknown":
                w[0] = 1.0
            else:
                raise ValueError(f"{where}: precisione {prec!r} sconosciuta")
            region.append(r)
            weights.append(w)
            muscle.append(m)
            known.append(m != 0)
            topo.append(TOPOLOGIES.index(g["topology"]))
            side.append(SIDE_KEYS.index(c.get("chirality")) if c.get("chirality") in SIDE_KEYS[1:] else 0)
            group.append(min(gi, MAX_GROUPS - 1))
        pos.append(_sensor_pos(g))
    return AnatomyCodes(np.asarray(region, dtype=np.int64), np.asarray(weights, dtype=np.float64).reshape(-1, n_comp),
                        np.asarray(muscle, dtype=np.int64), np.asarray(known, dtype=bool), np.asarray(topo, dtype=np.int64),
                        np.asarray(side, dtype=np.int64), np.asarray(group, dtype=np.int64),
                        np.concatenate(pos, axis=0).reshape(-1, N_SENSOR_POS) if pos else np.zeros((0, N_SENSOR_POS)))


def _sensor_pos(g: dict) -> np.ndarray:
    """(C_g, N_SENSOR_POS) posizione di ogni canale nel sistema del sensore del suo gruppo (v10 §3.6). Serve al decoder a query: i latenti del
    Perceiver non hanno canali, e senza una posizione per canale tutti i canali di un anello avrebbero la stessa query (review del 03/10: su
    emg2qwerty 32 canali, 1 identita'). Anello: cos/sin(m*angolo), m = 1..4, cosi' una rotazione della fascia e' uno sfasamento (la simmetria
    ciclica e' rappresentabile, non imposta). Griglia: cos/sin(pi*m*x) con x = indice / indice massimo in [0, 1], per riga e colonna, m = 1, 2
    (nessun avvolgimento: i bordi della griglia non sono vicini). Sparsi: zeri (l'identita' e' anatomica, v10 §3.5)."""
    chans = g["channels"]
    out = np.zeros((len(chans), N_SENSOR_POS))
    if g["topology"] == "ring":
        th = np.radians([float(c["sensor_coords"]["ring_angle_deg"]) for c in chans])
        m = np.arange(1, N_SENSOR_POS // 2 + 1)
        out[:] = np.concatenate([np.cos(th[:, None] * m), np.sin(th[:, None] * m)], axis=1)
    elif g["topology"] == "grid_2d":
        cols = []
        ext = g.get("grid_extent")  # montaggio virtuale (`data.virtual_montage`): indici massimi della griglia d'origine, stesse posizioni
        for k, key in enumerate(("grid_row", "grid_col")):
            v = np.asarray([float(c["sensor_coords"][key]) for c in chans])
            top = float(ext[k]) if ext else v.max()
            x = v / top if top > 0 else np.full_like(v, 0.5)
            m = np.arange(1, N_SENSOR_POS // 4 + 1)
            cols += [np.cos(np.pi * x[:, None] * m), np.sin(np.pi * x[:, None] * m)]
        out[:] = np.concatenate(cols, axis=1)
    return out


CODE_FIELDS = ("region", "compartment_weights", "muscle", "muscle_known", "topology", "side", "group", "sensor_pos")


def pack_codes(codes: list[AnatomyCodes]) -> AnatomyCodes:
    """Piu' montaggi in una sola sequenza di canali (packing), nello stesso ordine di `pack_attention_sets`."""
    return AnatomyCodes(*[np.concatenate([getattr(c, f) for c in codes]) for f in CODE_FIELDS])


def _unknown_compartment(comp: str, where: str) -> int:
    raise KeyError(f"{where}: compartimento sconosciuto nei pesi soft {comp!r} (mai etichette inventate)")


# --- Insiemi di attenzione dell'encoder locale (v10 §5.3-5.4) ------------------------------------------------------------------------------

PAIR_SELF, PAIR_RING, PAIR_GRID, PAIR_SET = 0, 1, 2, 3  # tipo di coppia (canale, chiave); -1 = posto vuoto


@dataclass(frozen=True)
class AttentionSets:
    """Per ogni canale (riga) le chiavi dell'attenzione locale (colonne; -1 = vuoto) e la geometria della coppia.

    Colonna 0 = il canale stesso. Canali di anello o griglia: i k vicini metrici (`neighbors`). Canali sparsi: tutti gli altri canali validi del
    montaggio, senza geometria (v10 §5.4: «set encoder su identita' anatomica», il vicinato metrico e' vuoto). *Scelta di AG, da confermare:* i
    canali metrici non guardano i canali sparsi dello stesso montaggio (la sinergia fra gruppi passa dal Perceiver e dal backbone); i canali sparsi
    guardano tutti. Le distanze sono in **passi d'elettrodo** del gruppo (anello: |angolo| / passo angolare minimo; griglia: passi di indice), cosi'
    la componente fissa del bias ha la stessa scala sulle due topologie."""

    index: np.ndarray  # (C, K)
    pair_type: np.ndarray  # (C, K)
    dist: np.ndarray  # (C, K) distanza in passi d'elettrodo, NaN se la coppia non e' metrica
    d_row: np.ndarray  # (C, K) spostamento di riga CON SEGNO (chiave meno query), NaN fuori dalle griglie
    d_col: np.ndarray  # (C, K) spostamento di colonna con segno, NaN fuori dalle griglie
    d_ring: np.ndarray  # (C, K) spostamento angolare con segno in passi d'elettrodo, NaN fuori dagli anelli (review del 03/10: col solo modulo
    #                     il verso di propagazione non era rappresentabile, e la riflessione dell'anello era un'invarianza imposta)

    @property
    def n_channels(self) -> int:
        return int(self.index.shape[0])


def _ring_pitch(layout: ChannelLayout, rel: dict) -> np.ndarray:
    """(C,) passo angolare minimo del gruppo di ogni canale di anello (NaN altrove)."""
    pitch = np.full(layout.n_channels, np.nan)
    for g in np.unique(layout.group[layout.topology == TOPOLOGIES.index("ring")]):
        members = np.flatnonzero(layout.group == g)
        d = rel["distance"][np.ix_(members, members)]
        positive = d[d > 10.0 ** -_ROUND]
        pitch[members] = positive.min() if positive.size else np.nan
    return pitch


def attention_sets(layout: ChannelLayout, k: int) -> AttentionSets:
    rel = relative_geometry(layout)
    nb = neighbors(layout, k)
    pitch = _ring_pitch(layout, rel)
    sparse = layout.topology == TOPOLOGIES.index("sparse")
    rows = []
    for i in range(layout.n_channels):
        if not layout.qc_valid[i]:
            keys = [i]
        elif sparse[i]:
            keys = [i, *[j for j in range(layout.n_channels) if j != i and layout.qc_valid[j]]]
        else:
            keys = [i, *[int(j) for j in nb[i] if j >= 0]]
        rows.append(keys)
    width = max(len(r) for r in rows)
    c = layout.n_channels
    index = np.full((c, width), -1, dtype=np.int64)
    pair_type = np.full((c, width), -1, dtype=np.int64)
    dist, d_row, d_col, d_ring = (np.full((c, width), np.nan) for _ in range(4))
    for i, keys in enumerate(rows):
        for col, j in enumerate(keys):
            index[i, col] = j
            if col == 0:
                pair_type[i, col], dist[i, col] = PAIR_SELF, 0.0
            elif rel["ring"][i, j]:
                pair_type[i, col], dist[i, col] = PAIR_RING, rel["distance"][i, j] / pitch[i]
                d_ring[i, col] = rel["d_angle"][i, j] / pitch[i]
            elif rel["grid"][i, j]:
                pair_type[i, col], dist[i, col] = PAIR_GRID, rel["distance"][i, j]
                d_row[i, col], d_col[i, col] = rel["d_row"][i, j], rel["d_col"][i, j]
            else:
                pair_type[i, col] = PAIR_SET
    return AttentionSets(index, pair_type, dist, d_row, d_col, d_ring)


def pack_attention_sets(sets: list[AttentionSets]) -> AttentionSets:
    """Piu' montaggi in una sola sequenza di canali (packing, D6b): indici spostati di quanti canali precedono, larghezza al massimo. Nessuna
    chiave attraversa due montaggi."""
    width = max(s.index.shape[1] for s in sets)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ("index", "pair_type", "dist", "d_row", "d_col", "d_ring")}
    offset = 0
    for s in sets:
        pad = width - s.index.shape[1]
        idx = np.where(s.index >= 0, s.index + offset, -1)
        parts["index"].append(np.pad(idx, ((0, 0), (0, pad)), constant_values=-1))
        parts["pair_type"].append(np.pad(s.pair_type, ((0, 0), (0, pad)), constant_values=-1))
        for name in ("dist", "d_row", "d_col", "d_ring"):
            parts[name].append(np.pad(getattr(s, name), ((0, 0), (0, pad)), constant_values=np.nan))
        offset += s.n_channels
    return AttentionSets(**{name: np.concatenate(v, axis=0) for name, v in parts.items()})
