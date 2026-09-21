#!/usr/bin/env python3
"""Esporta lo schema dei metadati (v10 §4.5) come file JSON Schema, per la revisione
umana richiesta da D7a (piano_operativo_v10.md §11, passo 2) prima di qualunque ingest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.metadata.schema import export_json_schema  # noqa: E402

if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "docs" / "montage_metadata.schema.json"
    out_path.write_text(json.dumps(export_json_schema(), indent=2, ensure_ascii=False) + "\n")
    print(f"scritto {out_path}")
