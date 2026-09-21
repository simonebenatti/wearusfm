"""Schema dei metadati di sensore per l'ingest (v10 §4.5, §3.6).

Decisione dichiarata dal piano operativo come "la piu' costosa da cambiare a posteriori":
questo file si ferma qui, in attesa della firma di Simone (D7a, piano_operativo_v10.md
§11 passo 2) prima di qualunque ingest reale. Nessun dataset viene letto o validato qui:
solo la definizione dello schema e l'esportazione JSON Schema per la validazione a valle.

Principio guida (v10 §3.4, §4.5): la topologia e la simmetria sono proprieta' del
GRUPPO di canali, non del dataset; un montaggio e' un insieme di gruppi (es. il montaggio
NinaPro a 12 elettrodi ha un gruppo ad anello da 8 e un gruppo mirato da 4, v10 §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Topology(str, Enum):
    """v10 §3.4: tre casi, tre simmetrie. Proprieta' del gruppo, non del dataset."""

    RING = "ring"  # D_n, spostamento angolare relativo
    GRID_2D = "grid_2d"  # traslazione lungo gli assi, asse fibre privilegiato
    SPARSE = "sparse"  # nessuna simmetria, identita' del muscolo


class RawOrEnvelope(str, Enum):
    """v10 §2.4: determina se il dataset entra nel front-end. ENVELOPE non ci entra mai."""

    RAW = "raw"
    ENVELOPE = "envelope"


class AnatomicalPrecision(str, Enum):
    """v10 §4.5: livello piu' fine REALMENTE noto per il canale. Mai etichette inventate."""

    MUSCLE = "muscle"
    COMPARTMENT = "compartment"
    SECTOR = "sector"  # solo al polso: volare/dorsale x radiale/ulnare
    REGION = "region"
    UNKNOWN = "unknown"


class Chirality(str, Enum):
    """v10 §3.4: il mirroring L<->R inverte la chiralita' della propagazione."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class CalibrationStatus(str, Enum):
    """v10 §3.2: mai dare in input la scala di sessione non calibrata."""

    AVAILABLE = "available"
    ABSENT = "absent"


@dataclass
class SensorCoordinates:
    """v10 §3.6, "sistema del sensore": indice di canale, riga/colonna, angolo sul ring."""

    channel_index: int
    grid_row: int | None = None
    grid_col: int | None = None
    ring_angle_deg: float | None = None


@dataclass
class AnatomicalCoordinates:
    """v10 §3.6, "coordinate anatomiche": muscolo, compartimento, lato, orientamento fibre."""

    muscle: str | None = None
    compartment: str | None = None
    side: str | None = None  # "left" | "right" | None
    fiber_orientation_deg: float | None = None


@dataclass
class RepositioningTransform:
    """v10 §3.6, "trasformazioni da riposizionamento": rotazione, traslazione, chiralita'.

    Una permutazione ciclica ideale e uno spostamento fisico reale del bracciale non sono
    lo stesso fenomeno (v10 §3.6): questo campo descrive il secondo, non il primo.
    """

    rotation_deg: float | None = None
    translation_mm: tuple[float, float] | None = None
    chirality_flip: bool = False


@dataclass
class AnatomicalIdentity:
    """v10 §4.5: identita' anatomica gerarchica del canale (regione -> compartimento ->
    muscolo). Annotata UNA VOLTA SOLA al livello piu' fine realmente noto; gli antenati si
    derivano per lookup (v10 §4.5, "Regola di annotazione")."""

    region: str | None = None
    compartment: str | None = None
    sector: str | None = None  # solo al polso (v10 §4.5, "Due cautele")
    muscle: str | None = None
    precision: AnatomicalPrecision = AnatomicalPrecision.UNKNOWN
    muscle_ontology_id: str | None = None  # ID FMA o UBERON (v10 §4.5, "Vocabolario")
    soft_compartment_weights: dict[str, float] | None = None  # funzione atlante, v10 §4.5
    nominal: bool = False  # v10 §4.5: True per amputati (DB3, DB7) - anatomia alterata


@dataclass
class ChannelMetadata:
    """Un canale dentro un gruppo. Campi elencati in v10 §4.5."""

    sensor_coords: SensorCoordinates
    electrode_type: str
    native_fs_hz: float
    effective_band_hz: tuple[float, float]
    mains_frequency_hz: int  # 50 o 60 (v10 §4.3)
    raw_or_envelope: RawOrEnvelope
    anatomical_identity: AnatomicalIdentity
    chirality: Chirality = Chirality.UNKNOWN
    anatomical_coords: AnatomicalCoordinates | None = None
    repositioning: RepositioningTransform | None = None
    inter_electrode_distance_mm: float | None = None
    bipolar_orientation: str | None = None  # canonicalizzazione di polarita', v10 §3.3
    calibration_status: CalibrationStatus = CalibrationStatus.ABSENT
    mvc_reference: float | None = None  # solo se calibration_status == AVAILABLE
    qc_valid: bool = True  # flag di validita' per canale (QC, v10 §4.4)
    body_region: str = ""

    def __post_init__(self) -> None:
        if self.mains_frequency_hz not in (50, 60):
            raise ValueError(f"mains_frequency_hz deve essere 50 o 60, non {self.mains_frequency_hz}")
        if self.calibration_status == CalibrationStatus.ABSENT and self.mvc_reference is not None:
            raise ValueError("mvc_reference impostato ma calibration_status e' ABSENT")


@dataclass
class ChannelGroup:
    """v10 §3.4, §4.5: un montaggio e' un insieme di gruppi. Topologia e simmetria sono
    proprieta' DEL GRUPPO. `band_orientation_deg=None` significa "ignoto" (v10 §4.5,
    caso EPN-612: varia per utente -> l'identita' arriva dal percorso geometrico)."""

    group_id: str
    topology: Topology
    symmetry: str  # es. "D_8", "translational_2d", "none" (v10 §3.4)
    channels: list[ChannelMetadata]
    band_orientation_deg: float | None = None  # None = ignoto

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError(f"gruppo {self.group_id!r} senza canali")
        if self.topology == Topology.SPARSE and self.symmetry != "none":
            raise ValueError("un gruppo sparso non ha simmetria (v10 §3.4): symmetry deve essere 'none'")


@dataclass
class MontageMetadata:
    """Un montaggio = un insieme di gruppi di canali (v10 §3.4). Il campione che entra nel
    modello resta il montaggio intero (dataloader_bench_spec.md §2), MAI un singolo gruppo.

    `subject_id` deve essere comparabile FRA dataset per rilevare sovrapposizioni di
    soggetti (es. fra i DB NinaPro, v10 §2.1) - non basta che sia univoco dentro un
    dataset. `session_id`/`day_index` servono ai dataset multi-giorno (GRABMyo, v10 §2.1)
    e agli split cross-sessione (v10 §8, asse 2).
    """

    dataset_name: str
    subject_id: str
    session_id: str
    groups: list[ChannelGroup]
    day_index: int | None = None

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError(f"montaggio del dataset {self.dataset_name!r} senza gruppi")
        ids = [g.group_id for g in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError(f"group_id duplicati nel montaggio {self.dataset_name!r}: {ids}")

    @property
    def n_channels(self) -> int:
        return sum(len(g.channels) for g in self.groups)


# ---------------------------------------------------------------------------
# Esportazione JSON Schema (scritta a mano: lo schema e' piccolo e fisso, un
# introspettore generico su dataclass/Enum aggiungerebbe complessita' senza bisogno).
# ---------------------------------------------------------------------------

_ENUM_STRING = lambda enum_cls: {"type": "string", "enum": [e.value for e in enum_cls]}  # noqa: E731

_SENSOR_COORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "channel_index": {"type": "integer"},
        "grid_row": {"type": ["integer", "null"]},
        "grid_col": {"type": ["integer", "null"]},
        "ring_angle_deg": {"type": ["number", "null"]},
    },
    "required": ["channel_index"],
}

_ANATOMICAL_COORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "muscle": {"type": ["string", "null"]},
        "compartment": {"type": ["string", "null"]},
        "side": {"type": ["string", "null"]},
        "fiber_orientation_deg": {"type": ["number", "null"]},
    },
}

_REPOSITIONING_SCHEMA = {
    "type": "object",
    "properties": {
        "rotation_deg": {"type": ["number", "null"]},
        "translation_mm": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "chirality_flip": {"type": "boolean"},
    },
}

_ANATOMICAL_IDENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "region": {"type": ["string", "null"]},
        "compartment": {"type": ["string", "null"]},
        "sector": {"type": ["string", "null"]},
        "muscle": {"type": ["string", "null"]},
        "precision": _ENUM_STRING(AnatomicalPrecision),
        "muscle_ontology_id": {"type": ["string", "null"]},
        "soft_compartment_weights": {
            "type": ["object", "null"],
            "additionalProperties": {"type": "number"},
        },
        "nominal": {"type": "boolean"},
    },
    "required": ["precision", "nominal"],
}

_CHANNEL_SCHEMA = {
    "type": "object",
    "properties": {
        "sensor_coords": _SENSOR_COORDS_SCHEMA,
        "electrode_type": {"type": "string"},
        "native_fs_hz": {"type": "number"},
        "effective_band_hz": {
            "type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
        },
        "mains_frequency_hz": {"type": "integer", "enum": [50, 60]},
        "raw_or_envelope": _ENUM_STRING(RawOrEnvelope),
        "anatomical_identity": _ANATOMICAL_IDENTITY_SCHEMA,
        "chirality": _ENUM_STRING(Chirality),
        "anatomical_coords": {"anyOf": [_ANATOMICAL_COORDS_SCHEMA, {"type": "null"}]},
        "repositioning": {"anyOf": [_REPOSITIONING_SCHEMA, {"type": "null"}]},
        "inter_electrode_distance_mm": {"type": ["number", "null"]},
        "bipolar_orientation": {"type": ["string", "null"]},
        "calibration_status": _ENUM_STRING(CalibrationStatus),
        "mvc_reference": {"type": ["number", "null"]},
        "qc_valid": {"type": "boolean"},
        "body_region": {"type": "string"},
    },
    "required": [
        "sensor_coords", "electrode_type", "native_fs_hz", "effective_band_hz",
        "mains_frequency_hz", "raw_or_envelope", "anatomical_identity",
    ],
}

_CHANNEL_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "group_id": {"type": "string"},
        "topology": _ENUM_STRING(Topology),
        "symmetry": {"type": "string"},
        "channels": {"type": "array", "items": _CHANNEL_SCHEMA, "minItems": 1},
        "band_orientation_deg": {"type": ["number", "null"]},
    },
    "required": ["group_id", "topology", "symmetry", "channels"],
}

MONTAGE_JSON_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WearUsFM Montage Metadata",
    "description": "Schema dei metadati di sensore per l'ingest (v10 §4.5, §3.6). "
    "Vincolante solo dopo la firma di Simone (D7a).",
    "type": "object",
    "properties": {
        "dataset_name": {"type": "string"},
        "subject_id": {"type": "string"},
        "session_id": {"type": "string"},
        "day_index": {"type": ["integer", "null"]},
        "groups": {"type": "array", "items": _CHANNEL_GROUP_SCHEMA, "minItems": 1},
    },
    "required": ["dataset_name", "subject_id", "session_id", "groups"],
}


def export_json_schema() -> dict:
    """Ritorna lo schema JSON completo (deep copy non necessaria: e' usato read-only)."""
    return MONTAGE_JSON_SCHEMA
