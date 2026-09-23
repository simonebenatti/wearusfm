"""Tassonomia anatomica condivisa e funzione atlante (v10 §4.5).

BOZZA — revisione umana di Simone non ancora fatta (D7b, docs/decisioni.md). Non vincolante:
qualunque ingest che la usa va rifatto se la revisione cambia le mappature.

Albero regione -> compartimento (al polso: settore) -> muscolo, dalla tabella di v10 §4.5.
Non definisce un tipo parallelo: produce `AnatomicalIdentity` di schema.py.

ID ontologici letti il 23/09/2026 dall'Ontology Lookup Service EBI
(https://www.ebi.ac.uk/ols4/api/ontologies/{uberon,fma}/terms), UBERON release 2026-06-19,
FMA 5.1.0. Un ID entra qui solo se le due ontologie concordano: la voce UBERON riporta come
`database_cross_reference` esattamente l'ID FMA della voce FMA con lo stesso nome. Un ID
non verificato cosi' resta None, mai un numero plausibile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wearusfm.metadata.schema import AnatomicalIdentity, AnatomicalPrecision


@dataclass(frozen=True)
class Region:
    key: str
    description: str


@dataclass(frozen=True)
class Compartment:
    key: str
    region: str
    description: str
    is_sector: bool = False  # v10 §4.5, "Due cautele": settori solo al polso


@dataclass(frozen=True)
class Muscle:
    key: str
    name: str  # etichetta FMA
    compartment: str
    uberon_id: str | None
    fma_id: str | None
    note: str = ""

    @property
    def ontology_id(self) -> str | None:
        """UBERON prima di FMA: e' cross-specie e ha una release recente (2026-06-19 su OLS4).
        Scelta da confermare in revisione: schema.py ha un solo campo per l'ID."""
        return self.uberon_id or self.fma_id


REGIONS: dict[str, Region] = {r.key: r for r in (
    Region("forearm_proximal", "avambraccio prossimale"),
    Region("forearm_distal", "avambraccio distale"),
    Region("wrist", "polso"),
    Region("upper_arm", "braccio"),
    Region("lower_limb", "arto inferiore (solo Camargo)"),
    # Fuori dalla tabella di v10 §4.5: vedi l'obliquo esterno di Camargo sotto.
    Region("trunk", "tronco (solo Camargo, obliquo esterno)"),
)}

COMPARTMENTS: dict[str, Compartment] = {c.key: c for c in (
    Compartment("flexor_pronator_ulnar", "forearm_proximal", "flessori-pronatori, lato ulnare"),
    Compartment("flexor_pronator_radial", "forearm_proximal", "flessori-pronatori, lato radiale"),
    Compartment("radial_group", "forearm_proximal", "gruppo radiale"),
    Compartment("dorsal_extensors", "forearm_proximal", "estensori dorsali"),
    Compartment("dorsoradial_deep_outcropping", "forearm_distal",
                "dorsale-radiale, profondi affioranti"),
    Compartment("wrist_volar_ulnar", "wrist", "settore volare-ulnare", is_sector=True),
    Compartment("wrist_volar_radial", "wrist", "settore volare-radiale", is_sector=True),
    Compartment("wrist_dorsal_radial", "wrist", "settore dorsale-radiale", is_sector=True),
    Compartment("wrist_dorsal_ulnar", "wrist", "settore dorsale-ulnare", is_sector=True),
    Compartment("arm_anterior", "upper_arm", "anteriore"),
    Compartment("arm_posterior", "upper_arm", "posteriore"),
    # Arto inferiore: v10 §4.5 da' solo il livello muscolo. Compartimenti dell'agente, da
    # rivedere; v10 ("Perche' non solo compartimento") impone solo che VM/VL/RF, BF/ST e
    # gastrocnemio/soleo condividano un compartimento, ed e' cosi'.
    Compartment("thigh_anterior", "lower_limb", "coscia, loggia anteriore"),
    Compartment("thigh_medial", "lower_limb", "coscia, loggia mediale"),
    Compartment("thigh_posterior", "lower_limb", "coscia, loggia posteriore"),
    Compartment("leg_anterior", "lower_limb", "gamba, loggia anteriore"),
    Compartment("leg_posterior", "lower_limb", "gamba, loggia posteriore superficiale"),
    Compartment("gluteal", "lower_limb", "regione glutea"),
    Compartment("abdominal_wall_anterolateral", "trunk", "parete addominale anterolaterale"),
)}

MUSCLES: dict[str, Muscle] = {m.key: m for m in (
    # Avambraccio prossimale e distale, braccio: le 16 coppie UBERON/FMA sono concordi su OLS4.
    Muscle("FCU", "Flexor carpi ulnaris", "flexor_pronator_ulnar", "UBERON:0001522", "FMA:38465"),
    Muscle("PL", "Palmaris longus", "flexor_pronator_ulnar", "UBERON:0016493", "FMA:38462"),
    Muscle("FDS", "Flexor digitorum superficialis", "flexor_pronator_ulnar",
           "UBERON:0003222", "FMA:38469",
           note="solo la parte ulnare in questo compartimento (v10 §4.5); l'ID e' del muscolo intero"),
    Muscle("FCR", "Flexor carpi radialis", "flexor_pronator_radial", "UBERON:0001521", "FMA:38459"),
    Muscle("PT", "Pronator teres", "flexor_pronator_radial", "UBERON:0001520", "FMA:38450"),
    Muscle("BRD", "Brachioradialis", "radial_group", "UBERON:0011011", "FMA:38485"),
    Muscle("ECRL", "Extensor carpi radialis longus", "radial_group", "UBERON:0001524", "FMA:38494"),
    Muscle("ECRB", "Extensor carpi radialis brevis", "radial_group", "UBERON:0001525", "FMA:38497"),
    Muscle("EDC", "Extensor digitorum", "dorsal_extensors", "UBERON:0007612", "FMA:38500"),
    Muscle("EDM", "Extensor digiti minimi", "dorsal_extensors", "UBERON:0007614", "FMA:38503"),
    Muscle("ECU", "Extensor carpi ulnaris", "dorsal_extensors", "UBERON:0001526", "FMA:38506"),
    Muscle("APL", "Abductor pollicis longus", "dorsoradial_deep_outcropping",
           "UBERON:0001527", "FMA:38515"),
    Muscle("EPB", "Extensor pollicis brevis", "dorsoradial_deep_outcropping",
           "UBERON:0017618", "FMA:38518"),
    Muscle("BB", "Biceps brachii", "arm_anterior", "UBERON:0001507", "FMA:37670"),
    Muscle("BRA", "Brachialis", "arm_anterior", "UBERON:0001506", "FMA:37667"),
    Muscle("TB", "Triceps brachii", "arm_posterior", "UBERON:0001509", "FMA:37688"),
    # Camargo 2021, 11 muscoli: vedi CAMARGO_EMG_COLUMNS per le fonti. ID concordi UBERON/FMA
    # su OLS4.
    Muscle("GASMED", "Medial head of gastrocnemius", "leg_posterior", "UBERON:0011907", "FMA:45956"),
    Muscle("TA", "Tibialis anterior", "leg_anterior", "UBERON:0001385", "FMA:22532"),
    Muscle("SOL", "Soleus", "leg_posterior", "UBERON:0001389", "FMA:22542"),
    Muscle("VM", "Vastus medialis", "thigh_anterior", "UBERON:0001380", "FMA:22432"),
    Muscle("VL", "Vastus lateralis", "thigh_anterior", "UBERON:0001379", "FMA:22431"),
    Muscle("RF", "Rectus femoris", "thigh_anterior", "UBERON:0001378", "FMA:22430"),
    Muscle("BF", "Biceps femoris", "thigh_posterior", "UBERON:0001374", "FMA:22356"),
    Muscle("ST", "Semitendinosus", "thigh_posterior", "UBERON:0001375", "FMA:22357"),
    Muscle("GRA", "Gracilis", "thigh_medial", "UBERON:0000950", "FMA:43882"),
    Muscle("GMED", "Gluteus medius", "gluteal", "UBERON:0001371", "FMA:22315"),
    # "rightexternaloblique": muscolo del tronco, non dell'arto inferiore. La tabella di v10
    # lo metterebbe sotto "arto inferiore, 11 etichette": non entra, si segnala (CLAUDE.md).
    Muscle("EO", "External oblique", "abdominal_wall_anterolateral", "UBERON:0005442", "FMA:13335",
           note="lato destro secondo il nome della colonna di Camargo"),
)}

# Colonne EMG di Camargo 2021 -> chiave muscolo, nell'ordine del file. Due fonti concordi,
# stesso ordine (fatto n. 6 di docs/fatti_da_verificare.md: raccolto, non firmato):
# - nomi delle colonne della table MATLAB in
#   $WORK/data/raw/camargo2021/Subjects_Part1_AB06-AB14.zip ->
#   AB06/10_09_18/levelground/emg/levelground_ccw_fast_01_01.mat (letto via leonardo-ops);
# - https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/ (laboratorio degli
#   autori): "EMG electrodes are placed on the following muscle groups".
# Il README del dataset dice solo "Electromyography from 11 muscles"; il paper non e' letto.
CAMARGO_EMG_COLUMNS: dict[str, str] = {
    "gastrocmed": "GASMED",
    "tibialisanterior": "TA",
    "soleus": "SOL",
    "vastusmedialis": "VM",
    "vastuslateralis": "VL",
    "rectusfemoris": "RF",
    "bicepsfemoris": "BF",
    "semitendinosus": "ST",
    "gracilis": "GRA",
    "gluteusmedius": "GMED",
    "rightexternaloblique": "EO",
}


def ancestors(key: str) -> tuple[str, ...]:
    """Catena dal nodo alla regione, nodo incluso. Gli antenati si derivano solo per lookup
    (v10 §4.5, "Regola di annotazione")."""
    if key in MUSCLES:
        comp = MUSCLES[key].compartment
        return (key, comp, COMPARTMENTS[comp].region)
    if key in COMPARTMENTS:
        return (key, COMPARTMENTS[key].region)
    if key in REGIONS:
        return (key,)
    raise KeyError(f"etichetta anatomica sconosciuta: {key!r} (mai etichette inventate)")


def identity(key: str, *, nominal: bool = False) -> AnatomicalIdentity:
    """Annotazione a mano al livello piu' fine realmente noto (elettrodi mirati NinaPro,
    Camargo). `nominal=True` per gli amputati (DB3, DB7)."""
    chain = ancestors(key)
    if key in MUSCLES:
        muscle = MUSCLES[key]
        return AnatomicalIdentity(
            region=chain[2], compartment=chain[1], muscle=key,
            precision=AnatomicalPrecision.MUSCLE, muscle_ontology_id=muscle.ontology_id,
            nominal=nominal,
        )
    if key in COMPARTMENTS:
        if COMPARTMENTS[key].is_sector:
            return AnatomicalIdentity(
                region=chain[1], sector=key, precision=AnatomicalPrecision.SECTOR, nominal=nominal,
            )
        return AnatomicalIdentity(
            region=chain[1], compartment=key, precision=AnatomicalPrecision.COMPARTMENT,
            nominal=nominal,
        )
    return AnatomicalIdentity(region=key, precision=AnatomicalPrecision.REGION, nominal=nominal)


# ---------------------------------------------------------------------------
# Funzione atlante per anelli e fasce (v10 §4.5)
#
# Approssimazione di ingegneria, non un dato anatomico misurato. Assunzioni, tutte da
# rivedere in D7b:
#
# 1. Angolo nel sistema anatomico, non del sensore: gradi dal margine sottocutaneo
#    dell'ulna, crescenti in direzione volare (0 = ulna, verso FCU). Cosi' non dipende dal
#    lato: convertire `ring_angle_deg` + `band_orientation_deg` in questo angolo, tenendo
#    conto del verso di numerazione e della chiralita', e' compito dell'adattatore di ogni
#    dataset (fatto n. 8, orientamento delle fasce, ancora da raccogliere).
# 2. Livello prossimo-distale in [0, 1]: 0 = piega del gomito, 1 = linea degli stiloidi.
#    Confini fra regioni a 0.5 e 0.9, scelti a mano.
# 3. Circonferenza divisa in archi UGUALI, uno per posizione dell'ordine di
#    v10 §4.5 (FCU -> PL/FDS -> FCR -> PT -> BRD -> ECRL -> ECRB -> EDC -> EDM -> ECU ->
#    ulna). Le sezioni trasverse reali hanno ventri di larghezza diversa: e' il primo punto
#    da raffinare con sezioni anatomiche vere.
# 4. Avambraccio distale: stesso ordine con APL/EPB inseriti fra ECRB ed EDC, dove
#    affiorano. v10 elenca per l'avambraccio distale solo il compartimento nuovo; gli altri
#    proseguono (ventri distali, tendini) e mantengono le loro chiavi, che nell'albero
#    appartengono a `forearm_proximal`. La regione del canale e' quella nel campo `region`,
#    non quella ricavata per lookup da un peso: incoerenza da decidere in revisione.
# 5. Polso: quattro quadranti da 90 gradi con il bordo radiale opposto all'ulnare (180).
# 6. Pesi = massa di una normale avvolta centrata sull'elettrodo che cade sull'arco di ogni
#    compartimento: gli archi partizionano il cerchio, quindi i pesi sommano a 1 e un
#    compartimento largo pesa piu' di uno stretto. `ATLAS_SIGMA_DEG` somma diffusione della
#    raccolta dell'elettrodo e incertezza sull'orientamento della fascia; il valore e'
#    arbitrario.
# ---------------------------------------------------------------------------

ATLAS_SIGMA_DEG = 20.0
PROXIMAL_DISTAL_BOUNDARY = 0.5
WRIST_BOUNDARY = 0.9

_PROXIMAL_RING: tuple[str, ...] = (
    "flexor_pronator_ulnar",   # FCU
    "flexor_pronator_ulnar",   # PL / FDS
    "flexor_pronator_radial",  # FCR
    "flexor_pronator_radial",  # PT
    "radial_group",            # BRD
    "radial_group",            # ECRL
    "radial_group",            # ECRB
    "dorsal_extensors",        # EDC
    "dorsal_extensors",        # EDM
    "dorsal_extensors",        # ECU
)

_DISTAL_RING: tuple[str, ...] = (
    _PROXIMAL_RING[:7] + ("dorsoradial_deep_outcropping",) + _PROXIMAL_RING[7:]  # APL / EPB
)

_WRIST_RING: tuple[str, ...] = (
    "wrist_volar_ulnar", "wrist_volar_radial", "wrist_dorsal_radial", "wrist_dorsal_ulnar",
)

_RING_BY_REGION: dict[str, tuple[str, ...]] = {
    "forearm_proximal": _PROXIMAL_RING,
    "forearm_distal": _DISTAL_RING,
    "wrist": _WRIST_RING,
}


def region_at_level(level: float) -> str:
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"livello prossimo-distale fuori da [0, 1]: {level}")
    if level < PROXIMAL_DISTAL_BOUNDARY:
        return "forearm_proximal"
    if level < WRIST_BOUNDARY:
        return "forearm_distal"
    return "wrist"


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _wrapped_normal_arc_mass(center: float, start: float, end: float, sigma: float) -> float:
    wraps = math.ceil(6.0 * sigma / 360.0) + 1
    return sum(
        _phi((end - center + 360.0 * k) / sigma) - _phi((start - center + 360.0 * k) / sigma)
        for k in range(-wraps, wraps + 1)
    )


def atlas_weights(
    angle_deg: float | None, level: float | None, sigma_deg: float = ATLAS_SIGMA_DEG,
) -> dict[str, float] | None:
    """Pesi soft sui compartimenti (al polso: settori) della regione a quel livello.
    None se angolo o livello sono ignoti: nessun peso a caso."""
    if angle_deg is None or level is None:
        return None
    if not math.isfinite(angle_deg):
        raise ValueError(f"angolo non finito: {angle_deg}")
    if sigma_deg <= 0:
        raise ValueError(f"sigma_deg deve essere positivo: {sigma_deg}")
    ring = _RING_BY_REGION[region_at_level(level)]
    width = 360.0 / len(ring)
    center = angle_deg % 360.0
    weights = dict.fromkeys(ring, 0.0)
    for i, comp in enumerate(ring):
        weights[comp] += _wrapped_normal_arc_mass(center, i * width, (i + 1) * width, sigma_deg)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def atlas_identity(
    angle_deg: float | None, level: float | None, *, nominal: bool = False,
    sigma_deg: float = ATLAS_SIGMA_DEG,
) -> AnatomicalIdentity:
    """Identita' di un canale di anello o fascia.

    La precisione resta REGION anche con i pesi: il compartimento non e' noto davvero, e'
    stimato dall'atlante, e v10 §4.5 vuole l'etichetta al livello realmente noto con i pesi
    soft come informazione aggiuntiva (da confermare in revisione). Livello ignoto ->
    UNKNOWN anche se l'angolo e' noto; angolo ignoto -> «compartimento ignoto» (EPN-612).
    """
    if level is None:
        return AnatomicalIdentity(precision=AnatomicalPrecision.UNKNOWN, nominal=nominal)
    return AnatomicalIdentity(
        region=region_at_level(level), precision=AnatomicalPrecision.REGION,
        soft_compartment_weights=atlas_weights(angle_deg, level, sigma_deg), nominal=nominal,
    )
