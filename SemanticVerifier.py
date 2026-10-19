"""Deterministic and qualitative review for Series G extraction JSON."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

YEAR_N_COLUMNS = {"montant_brut", "amortissements_provisions_pertes", "net_n", "n", "n_debit", "n_credit"}
YEAR_N1_COLUMNS = {"net_n1", "n1", "n1_debit", "n1_credit"}
COHERENT_STATUSES = {"COHERENT", "COHERENT_ARRONDI"}
ARITHMETIC_WORDS = re.compile(r"\b(sum|total|addition|soustraction|calcul|arithmetic|mismatch|inconsisten|écart)\b", re.I)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

LLM_SYSTEM_PROMPT = """You are a qualitative audit assistant for Algerian Series G tax filings.
You never perform arithmetic, reconcile totals, compare years, or contradict deterministic controls.
Review only qualitative anomalies: implausible magnitude, label/value mismatch, suspicious debit/credit direction,
an OCR digit-looking anomaly, or a missing expected line. Do not invent facts.
Year N columns are montant_brut, amortissements_provisions_pertes, net_n, n, n_debit and n_credit.
Year N-1 columns are net_n1, n1, n1_debit and n1_credit. A difference between N and N-1 is normal and is NEVER an error.
Use evidence.value_a for the exact value at path and evidence.value_b as null. Return an empty findings list when
nothing is genuinely suspicious; an empty list is the expected answer for a consistent statement."""

LLM_USER_TEMPLATE = """Audit only this single table. The rows are independent extraction facts; do not add or compare values.

Table rows:
{rows}

Authoritative deterministic controls already run (do not re-verify or challenge them):
{deterministic_results}

Return only JSON matching the required schema."""

LLM_OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["findings"], "properties": {"findings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["path", "table", "poste", "column", "reason", "evidence", "severity", "confidence"], "properties": {"path": {"type": "string"}, "table": {"type": "string", "enum": ["ACTIF", "PASSIF", "TCR"]}, "poste": {"type": "string"}, "column": {"type": "string"}, "reason": {"type": "string"}, "evidence": {"type": "object", "additionalProperties": False, "required": ["value_a", "value_b"], "properties": {"value_a": {}, "value_b": {}}}, "severity": {"type": "string", "enum": ["warning", "error"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}}}}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tolerance(data: dict[str, Any]) -> float:
    value = data.get("controles", {}).get("tolerance_da", 1)
    return float(value) if _number(value) else 1.0


def cell_path(table: str, poste: str, column: str | None = None) -> str:
    return " > ".join(part for part in (table, poste, column) if part)


def _issue(severity: str, title: str, message: str, path: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"severity": severity, "title": title, "message": message, "path": path, "source": "rule", "evidence": evidence or {}, "status": "open"}


def _table_cells(data: dict[str, Any]) -> Iterable[tuple[int, str, str, str, Any]]:
    for page_index, page in enumerate(data.get("pages", [])):
        for table, rows in page.get("donnees", {}).items():
            if isinstance(rows, dict):
                for poste, columns in rows.items():
                    if isinstance(columns, dict):
                        for column, value in columns.items():
                            yield page_index, table, poste, column, value


def resolve_cell_path(data: dict[str, Any], path: str) -> tuple[int, str, str, str, Any] | None:
    """Resolve a public ``TABLE > poste > column`` path to a real source cell."""
    parts = [part.strip() for part in path.split(">")]
    if len(parts) != 3 or not all(parts):
        return None
    for cell in _table_cells(data):
        if cell[1:4] == tuple(parts):
            return cell
    return None


def _control_entries(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for page in data.get("pages", []):
        for table, rows in page.get("controles_donnees", {}).items():
            if isinstance(rows, dict):
                for poste, columns in rows.items():
                    if isinstance(columns, dict):
                        for column, control in columns.items():
                            if isinstance(control, dict):
                                yield {**control, "tableau": table, "poste": poste, "colonne": column}
    for control in data.get("controles", {}).get("formules", []):
        if isinstance(control, dict):
            yield control


def _table(data: dict[str, Any], table_name: str) -> dict[str, Any]:
    return next((page.get("donnees", {}).get(table_name) for page in data.get("pages", []) if isinstance(page.get("donnees", {}).get(table_name), dict)), {})


def _add_once(issues: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if not any(issue["path"] == item["path"] and issue["title"] == item["title"] for issue in issues):
        issues.append(item)


def _has_coherent_control(data: dict[str, Any], table: str, poste: str, column: str) -> bool:
    for control in _control_entries(data):
        if (control.get("tableau"), control.get("poste"), control.get("colonne")) == (table, poste, column):
            if str(control.get("statut", "")).upper() in COHERENT_STATUSES:
                return True
            if _number(control.get("ecart")) and abs(float(control["ecart"])) <= _tolerance(data):
                return True
    return False


def deterministic_checks(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Perform all arithmetic, subtotal and reconciliation checks in code."""
    issues: list[dict[str, Any]] = []
    tolerance = _tolerance(data)
    identity = data.get("identite", {})
    if _number(identity.get("annee_n")) and _number(identity.get("annee_n1")) and identity["annee_n"] != identity["annee_n1"] + 1:
        _add_once(issues, _issue("error", "Exercices incohérents", "annee_n doit suivre annee_n1.", "identite.annee_n"))
    nif = str(identity.get("nif_15", "")).strip()
    if nif and not re.fullmatch(r"\d{15}", nif):
        _add_once(issues, _issue("warning", "Format NIF à vérifier", "Le NIF doit comporter 15 chiffres.", "identite.nif_15"))
    for _page, table, poste, column, value in _table_cells(data):
        row = _table(data, table).get(poste, {})
        if column == "net_n" and all(_number(item) for item in (row.get("montant_brut"), row.get("amortissements_provisions_pertes"), value)):
            expected = row["montant_brut"] - row["amortissements_provisions_pertes"]
            if abs(value - expected) > tolerance:
                _add_once(issues, _issue("error", "Montant net incohérent", "Le net ne correspond pas au brut moins amortissements/provisions.", cell_path(table, poste, column), {"value_a": value, "value_b": expected}))
    for control in _control_entries(data):
        table, poste, column = control.get("tableau"), control.get("poste"), control.get("colonne")
        if not all(isinstance(value, str) and value for value in (table, poste, column)):
            continue
        path, resolved, expected = cell_path(table, poste, column), resolve_cell_path(data, cell_path(table, poste, column)), control.get("valeur_recalculee")
        if resolved and _number(resolved[4]) and _number(expected) and abs(resolved[4] - expected) > tolerance:
            _add_once(issues, _issue("error", "Écart de formule", "La valeur extraite ne correspond pas au résultat déterministe du contrôle.", path, {"value_a": resolved[4], "value_b": expected}))
        status = str(control.get("statut", "")).upper()
        if status and status not in COHERENT_STATUSES | {"NON_VERIFIABLE", "NON_APPLICABLE"}:
            _add_once(issues, _issue("error", "Contrôle en écart", f"Le contrôle déterministe est marqué « {status} ».", path, {"value_a": control.get("valeur_extraite"), "value_b": expected}))
        if str(control.get("sens_attendu", "")) == "DEBIT_OU_CREDIT" and resolved:
            opposite = column.replace("credit", "debit") if "credit" in column else column.replace("debit", "credit")
            opposite_value = _table(data, table).get(poste, {}).get(opposite)
            if resolved[4] is None and _number(opposite_value):
                _add_once(issues, _issue("error", "Sens débit/crédit inversé", "Le montant attendu dans cette colonne est présent dans la colonne opposée.", path, {"value_a": None, "value_b": opposite_value}))
    actif, passif, tcr = _table(data, "ACTIF"), _table(data, "PASSIF"), _table(data, "TCR")
    for a_col, p_col in (("net_n", "n"), ("net_n1", "n1")):
        active, passive = actif.get("total_general_actif", {}).get(a_col), passif.get("total_general_passif", {}).get(p_col)
        if _number(active) and _number(passive) and abs(active - passive) > tolerance:
            _add_once(issues, _issue("error", "Équilibre Actif = Passif rompu", "Le total général Actif ne correspond pas au total général Passif pour la même année.", cell_path("ACTIF", "total_general_actif", a_col), {"value_a": active, "value_b": passive}))
    for p_col, t_col in (("n", "n_credit"), ("n1", "n1_credit")):
        passive, result = passif.get("resultat_net", {}).get(p_col), tcr.get("resultat_net", {}).get(t_col)
        if _number(passive) and _number(result) and abs(passive - result) > tolerance:
            _add_once(issues, _issue("error", "Résultat net Passif/TCR incohérent", "Le résultat net Passif ne correspond pas au résultat net TCR pour la même année.", cell_path("TCR", "resultat_net", t_col), {"value_a": result, "value_b": passive}))
    return issues


def flatten_table(data: dict[str, Any], table_name: str) -> list[dict[str, Any]]:
    """Flatten a table while retaining every cell's explicit year."""
    return [{"table": table, "poste": poste, "column": column, "year": "N-1" if column in YEAR_N1_COLUMNS else "N" if column in YEAR_N_COLUMNS else "non précisé", "value": value} for _page, table, poste, column, value in _table_cells(data) if table == table_name]


def _qualitative_prompt(data: dict[str, Any], table: str, rule_issues: list[dict[str, Any]]) -> str:
    controls = [{"path": issue["path"], "status": "deterministic finding"} for issue in rule_issues if issue["path"].startswith(table + " >")]
    return LLM_USER_TEMPLATE.format(rows=json.dumps(flatten_table(data, table), ensure_ascii=False), deterministic_results=json.dumps(controls, ensure_ascii=False))


def _ollama_checks(data: dict[str, Any], model: str, rule_issues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    candidates, raw = [], []
    try:
        tables = sorted({table for _page, table, _poste, _column, _value in _table_cells(data)} & {"ACTIF", "PASSIF", "TCR"})
        for table in tables:
            request = Request(f"{OLLAMA_BASE_URL}/api/generate", data=json.dumps({"model": model, "system": LLM_SYSTEM_PROMPT, "prompt": _qualitative_prompt(data, table, rule_issues), "stream": False, "format": LLM_OUTPUT_SCHEMA, "options": {"temperature": 0}}).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode())
            raw.append(str(payload["response"]))
            parsed = json.loads(raw[-1])
            if isinstance(parsed.get("findings"), list):
                candidates.extend(item for item in parsed["findings"] if isinstance(item, dict))
        return candidates, model, raw
    except (URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError):
        return [], None, raw


def _different_year_comparison(data: dict[str, Any], table: str, poste: str, column: str, finding: dict[str, Any]) -> bool:
    other = resolve_cell_path(data, finding["comparison_path"]) if isinstance(finding.get("comparison_path"), str) else None
    if other:
        return (column in YEAR_N_COLUMNS and other[3] in YEAR_N1_COLUMNS) or (column in YEAR_N1_COLUMNS and other[3] in YEAR_N_COLUMNS)
    value_b = finding.get("evidence", {}).get("value_b") if isinstance(finding.get("evidence"), dict) else None
    if value_b is not None:
        return any(value == value_b and ((column in YEAR_N_COLUMNS and name in YEAR_N1_COLUMNS) or (column in YEAR_N1_COLUMNS and name in YEAR_N_COLUMNS)) for name, value in _table(data, table).get(poste, {}).items())
    return False


def validate_llm_findings(data: dict[str, Any], candidates: list[dict[str, Any]], rule_issues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only grounded qualitative findings; preserve all rejections for logs."""
    accepted, rejected, rule_paths = [], [], {item["path"] for item in rule_issues}
    required = {"path", "table", "poste", "column", "reason", "evidence", "severity", "confidence"}
    for finding in candidates:
        reason, path = ("missing required structured fields", "") if not required.issubset(finding) else (None, str(finding["path"]))
        resolved = resolve_cell_path(data, path) if not reason else None
        if not reason and not resolved: reason = "path does not resolve to an existing field"
        if not reason and (finding["table"], finding["poste"], finding["column"]) != resolved[1:4]: reason = "table/poste/column do not match path"
        evidence = finding.get("evidence", {})
        if not reason and (not isinstance(evidence, dict) or evidence.get("value_a") != resolved[4] or evidence.get("value_b") is not None): reason = "evidence does not exactly match the field value"
        if not reason and _different_year_comparison(data, resolved[1], resolved[2], resolved[3], finding): reason = "comparison crosses year N and N-1"
        if not reason and _has_coherent_control(data, resolved[1], resolved[2], resolved[3]): reason = "matching deterministic control is coherent within tolerance"
        if not reason and ARITHMETIC_WORDS.search(str(finding["reason"])) and path not in rule_paths: reason = "arithmetic claim is not confirmed by deterministic checks"
        severity, confidence = str(finding.get("severity", "")).lower(), finding.get("confidence")
        if not reason and (severity not in {"warning", "error"} or not _number(confidence) or not 0 <= confidence <= 1): reason = "invalid severity or confidence"
        if reason:
            rejected.append({"finding": finding, "status": "rejected by validator", "reason": reason})
        else:
            accepted.append({"severity": severity, "title": "Point qualitatif à vérifier", "message": str(finding["reason"]), "path": path, "source": "llm", "evidence": evidence, "confidence": confidence, "status": "open"})
    return accepted, rejected


def build_cell_issues(data: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for finding in findings:
        resolved = resolve_cell_path(data, str(finding.get("path", "")))
        if resolved:
            _page, table, poste, column, value = resolved
            result.setdefault(table, {}).setdefault(poste, {})[column] = {"severity": finding.get("severity", "warning"), "message": finding.get("message", ""), "source": finding.get("source", "rule"), "status": finding.get("status", "open"), "value": value, "expected": finding.get("evidence", {}).get("value_b")}
            continue
        parts = [part.strip() for part in str(finding.get("path", "")).split(">")]
        if len(parts) == 2 and isinstance(_table(data, parts[0]).get(parts[1]), dict):
            result.setdefault(parts[0], {}).setdefault(parts[1], {})["__row__"] = {"severity": finding.get("severity", "warning"), "message": finding.get("message", ""), "source": finding.get("source", "rule"), "status": finding.get("status", "open"), "value": None, "expected": None}
    for control in _control_entries(data):
        status, table, poste, column = str(control.get("statut", "")).upper(), control.get("tableau"), control.get("poste"), control.get("colonne")
        if status and status not in COHERENT_STATUSES and all(isinstance(part, str) for part in (table, poste, column)) and resolve_cell_path(data, cell_path(table, poste, column)):
            result.setdefault(table, {}).setdefault(poste, {})[column] = {"severity": "error", "message": f"Contrôle: {status}", "source": "rule", "status": "open", "value": control.get("valeur_extraite"), "expected": control.get("valeur_recalculee")}
    return result


def _write_analysis_log(entry: dict[str, Any]) -> Path:
    path = Path(__file__).resolve().parent / "logs" / "llm_analysis.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as file: file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path



def analyse_document(data: dict[str, Any], use_ai: bool = True, json_name: str = "unknown.json", model: str | None = None) -> dict[str, Any]:
    rule_issues = deterministic_checks(data)
    candidates, raw, model_used = [], [], None
    if use_ai:
        candidates, model_used, raw = _ollama_checks(data, model or os.getenv("SEMANTIC_VERIFIER_MODEL", "qwen2.5:3b"), rule_issues)
    llm_issues, rejected = validate_llm_findings(data, candidates, rule_issues)
    issues = [*rule_issues, *llm_issues]
    log_entry = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "json_name": json_name, "model": model_used, "raw_llm_output": raw, "llm_candidates": candidates, "llm_findings": llm_issues, "rejected_findings": rejected, "rule_findings": rule_issues}
    log_path = _write_analysis_log(log_entry)
    return {"issues": issues, "cell_issues": build_cell_issues(data, issues), "valid": not any(item["severity"] == "error" for item in issues), "model_used": model_used, "rule_findings": len(rule_issues), "llm_findings": len(llm_issues), "rejected_llm_findings": len(rejected), "log_entry": log_entry, "log_path": str(log_path)}
