"""Creation and persistence of the post-certification JSON contract."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


def get_algeria_datetime() -> str:
    return datetime.now(timezone(timedelta(hours=1))).isoformat()


def create_certified_json(extracted_json: dict[str, Any], user_id: str, user_name: str, commentaire_global: str = "", nb_valeurs_validees: int = 0, nb_valeurs_corrigees: int = 0, nb_anomalies_acceptees: int = 0, decisions_analyste: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Preserve extraction data and append the certification payload."""
    certified_json = copy.deepcopy(extracted_json)
    certified_json["workflow"] = {"statut": "CERTIFIE", "certifie": True}
    certified_json["certification"] = {
        "statut": "CERTIFIE", "certifie": True,
        "certifie_par": {"user_id": user_id, "nom": user_name},
        "date_certification": get_algeria_datetime(), "version": 1,
        "commentaire_global": commentaire_global,
        "nb_valeurs_validees": nb_valeurs_validees,
        "nb_valeurs_corrigees": nb_valeurs_corrigees,
        "nb_anomalies_acceptees": nb_anomalies_acceptees,
    }
    certified_json["decisions_analyste"] = decisions_analyste or []
    return certified_json


def save_certified_json(certified_json: dict[str, Any], output_path: str | Path) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(certified_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
