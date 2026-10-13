"""Human review UI for OCR/VL fiscal-document extractions."""

from __future__ import annotations

import base64

import copy

import json

import os
import urllib.error
import urllib.request

from pathlib import Path

from typing import Any, Iterator

import hashlib

import streamlit as st

from certification import create_certified_json, save_certified_json

from SemanticVerifier import analyse_document
from rag import build_prompt, build_rag, pdf_fingerprint, references_from_chunks, retrieve

ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = ROOT / "documents"

CERTIFIED_DIR = ROOT / "certified"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

DOCUMENTS_DIR.mkdir(exist_ok=True)

CERTIFIED_DIR.mkdir(exist_ok=True)


@st.cache_resource(show_spinner="Indexation des PDF pour la recherche…")
def load_rag_index(_folder: str, fingerprint: str):
    """Keep the expensive model and FAISS index in memory until PDFs change."""
    del fingerprint  # It is deliberately part of the Streamlit cache key.
    return build_rag(_folder)

st.set_page_config(page_title="CertiFlow | Revue documentaire", page_icon=":material/fact_check:", layout="wide")

st.markdown("""<style>

    .block-container {max-width:1520px; padding-top:1.15rem; padding-bottom:3rem;}

    .hero {padding:1.25rem 1.5rem;border-radius:16px;background:linear-gradient(105deg,#062b52,#08708a);color:#fff;margin-bottom:1rem;}

    .hero h1 {font-size:2rem;margin:0}.hero p {margin:.25rem 0 0;opacity:.84}

    .issue {border-left:5px solid #D9363E;background:#FFF1F2;padding:.75rem 1rem;border-radius:7px;margin:.45rem 0;color:#642029;}

    .warning {border-left-color:#D68500;background:#FFF7E8;color:#674300;}

    .path {font-family:monospace;font-size:.8rem;opacity:.76}

    div[class*="st-key-cell_issue_error"] [data-baseweb="input"] {background:#fff1f2;border:1px solid #dc2626;border-radius:.4rem;}

    div[class*="st-key-cell_issue_error"] input {color:#991b1b;font-weight:650;}

    div[class*="st-key-cell_issue_warning"] [data-baseweb="input"] {background:#fff7ed;border:1px solid #f59e0b;border-radius:.4rem;}

    div[class*="st-key-cell_issue_warning"] input {color:#9a3412;font-weight:650;}

    div[class*="st-key-cell_focus"] [data-baseweb="input"] {box-shadow:0 0 0 3px #fbbf24;}

</style>""", unsafe_allow_html=True)

def pairs() -> list[dict[str, Path | str]]:

    return [{"name": pdf.stem, "pdf": pdf, "json": pdf.with_suffix(".json")} for pdf in sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda p: p.name.lower()) if pdf.with_suffix(".json").exists()]

def certified_path(name: str) -> Path:

    return CERTIFIED_DIR / f"{name}_certified.json"

def widget_key(path: str) -> str:

    return "field_" + "".join(char if char.isalnum() else "_" for char in path)

def edit_value(value: Any, path: str, label: str) -> Any:

    """Render all JSON values without flattening their types."""

    if isinstance(value, dict):

        result = {}

        for key, child in value.items():

            child_path = f"{path}.{key}"

            if isinstance(child, (dict, list)):

                with st.expander(f"{key} {'· ' + str(len(child)) if isinstance(child, list) else ''}", expanded=path.count(".") < 2):

                    result[key] = edit_value(child, child_path, key)

            else:

                result[key] = edit_value(child, child_path, key)

        return result

    if isinstance(value, list):

        result = []

        for index, child in enumerate(value):

            item_path = f"{path}[{index}]"

            if isinstance(child, (dict, list)):

                with st.expander(f"Élément {index + 1}", expanded=False):

                    result.append(edit_value(child, item_path, f"Élément {index + 1}"))

            else:

                result.append(edit_value(child, item_path, f"Élément {index + 1}"))

        return result

    key = widget_key(path)

    if isinstance(value, bool):

        return st.checkbox(label, value=value, key=key)

    if isinstance(value, int):

        return st.number_input(label, value=value, step=1, key=key)

    if isinstance(value, float):

        return st.number_input(label, value=value, step=0.01, key=key)

    if value is None:

        kind = st.selectbox(label, ["Vide", "Nombre", "Texte"], key=key + "_kind")

        return None if kind == "Vide" else (st.number_input(label + " - valeur", value=0, key=key + "_number") if kind == "Nombre" else st.text_input(label + " - valeur", key=key + "_text"))

    return st.text_area(label, value=value, height=68, key=key) if isinstance(value, str) and len(value) > 110 else st.text_input(label, value=value, key=key)

TABLE_LABELS = {

    "ACTIF": "Actif",

    "PASSIF": "Passif",

    "TCR": "Compte de résultat",

}

COLUMN_LABELS = {

    "montant_brut": "Montant brut",

    "amortissements_provisions_pertes": "Amortissements / provisions",

    "net_n": "Net N",

    "net_n1": "Net N-1",

    "n": "N",

    "n1": "N-1",

    "n_debit": "N · Débit",

    "n_credit": "N · Crédit",

    "n1_debit": "N-1 · Débit",

    "n1_credit": "N-1 · Crédit",

}

def humanize_label(value: str) -> str:

    """Turn a JSON row code into a readable accounting label."""

    return value.replace("_", " ").capitalize()

def render_financial_table(

    table_name: str,

    values: dict[str, Any],

    state_id: str,

    cell_issues: dict[str, dict[str, Any]],

    focus_path: str | None,

) -> dict[str, Any]:

    """Render an editable grid whose flagged inputs retain their own visual state."""

    if not values:

        st.info("Aucune donnée extraite pour ce tableau.")

        return values

    row_codes = list(values)

    columns = list(dict.fromkeys(column for row in values.values() if isinstance(row, dict) for column in row))

    widths = [2.35, *([1] * len(columns))]

    header = st.columns(widths, gap="xsmall")

    header[0].markdown("**Poste**")

    for column, slot in zip(columns, header[1:]):

        slot.markdown(f"**{COLUMN_LABELS.get(column, humanize_label(column))}**")

    edited: dict[str, Any] = {}

    for row_code in row_codes:

        row = values[row_code]

        row_issues = cell_issues.get(row_code, {})

        slots = st.columns(widths, gap="xsmall", vertical_alignment="center")

        row_issue = row_issues.get("__row__")

        row_label = f":material/warning: {humanize_label(row_code)}" if row_issue else humanize_label(row_code)

        slots[0].markdown(f"**{row_label}**" if row_code.startswith("total_") else row_label)

        edited[row_code] = {}

        for column, slot in zip(columns, slots[1:]):

            issue = row_issues.get(column)

            level = issue.get("severity", "warning") if issue else "clear"

            path = f"{table_name} > {row_code} > {column}"

            focus_class = "_focus" if focus_path == path else ""

            cell_key = f"cell_issue_{level}{focus_class}_{widget_key(state_id + path)}"

            container = slot.container(key=cell_key)

            source_value = row.get(column)

            tooltip = None

            if issue:

                expected = issue.get("expected")

                tooltip = f"{issue['message']}\nSource: {issue['source']}\nExtrait: {issue['value']}\nAttendu: {expected}"

            with container:

                edited[row_code][column] = st.number_input(

                    f"{humanize_label(row_code)} — {COLUMN_LABELS.get(column, column)}",

                    value=source_value if isinstance(source_value, (int, float)) else None,

                    step=1 if isinstance(source_value, int) else 0.01,

                    format="%d" if isinstance(source_value, int) else None,

                    placeholder="—",

                    icon=":material/warning:" if issue else None,

                    help=tooltip,

                    label_visibility="collapsed",

                    key=f"value_{widget_key(state_id + path)}",

                )

                if issue:

                    st.button(

                        "Afficher l’anomalie",

                        icon=":material/warning:",

                        key=f"issue_action_{widget_key(state_id + path)}",

                        # This button starts from a cell already on screen: it
                        # must therefore take the reviewer to its explanation,
                        # rather than trying to focus the same cell again.
                        on_click=lambda issue_path=path: st.session_state.update(
                            focus_path=None,
                            selected_finding_path=issue_path,
                        ),

                    )

                if focus_path == path:

                    # The target is generated from a trusted internal path, not model text.

                    st.html(

                        f"<script>requestAnimationFrame(() => document.querySelector(\"div[class*='st-key-{cell_key}']\")?.scrollIntoView({{behavior: 'smooth', block: 'center'}}));</script>",

                        unsafe_allow_javascript=True,

                    )

    return edited

def render_statement_editor(edited: dict[str, Any], state_id: str) -> None:

    """Offer one page and one financial statement at a time to reduce visual noise."""

    pages = edited.get("pages", [])

    usable_pages = [page for page in pages if isinstance(page, dict) and page.get("donnees")]

    if not usable_pages:

        st.info("Aucun tableau financier n’a été trouvé dans ce JSON.")

        return

    page_choices = list(range(len(usable_pages)))

    focus_path = st.session_state.get("focus_path")

    focus_table = None

    if isinstance(focus_path, str):

        focus_parts = [part.strip() for part in focus_path.split(">")]

        if len(focus_parts) == 3:

            focus_table = focus_parts[0]

            for index, candidate in enumerate(usable_pages):

                if focus_table in candidate.get("donnees", {}):

                    st.session_state[f"page_{widget_key(state_id)}"] = index

                    break

    selected_page = st.selectbox(

        "Page à réviser",

        page_choices,

        format_func=lambda index: f"Page {usable_pages[index].get('numero_page_imprimee', index + 1)} · {usable_pages[index].get('classification', {}).get('titre_page', 'Tableaux financiers')}",

        key=f"page_{widget_key(state_id)}",

    )

    page = usable_pages[selected_page]

    statements = [name for name in ("ACTIF", "PASSIF", "TCR") if isinstance(page.get("donnees", {}).get(name), dict)]

    if not statements:

        st.info("Cette page ne contient pas de tableau ACTIF, PASSIF ou TCR exploitable.")

        return

    statement_key = f"statement_{widget_key(state_id)}_{selected_page}"

    if focus_table in statements:

        st.session_state[statement_key] = focus_table

    table_name = st.segmented_control(

        "Tableau à réviser",

        statements,

        default=statements[0],

        required=True,

        format_func=lambda name: TABLE_LABELS.get(name, name),

        key=statement_key,

        width="stretch",

    )

    if table_name:

        table = page["donnees"][table_name]

        st.caption(f"{len(table)} postes · Modifiez directement les montants. Les cellules vides restent vides.")

        active_issues = (st.session_state.get("analysis") or {}).get("cell_issues", {}).get(table_name, {})

        page["donnees"][table_name] = render_financial_table(table_name, table, state_id, active_issues, focus_path)

def leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:

    if isinstance(value, dict):

        for key, child in value.items():

            yield from leaves(child, f"{path}.{key}" if path else key)

    elif isinstance(value, list):

        for index, child in enumerate(value):

            yield from leaves(child, f"{path}[{index}]")

    else:

        yield path, value

def decisions_from_changes(original: dict[str, Any], edited: dict[str, Any], source_name: str) -> list[dict[str, Any]]:

    original_values = dict(leaves(original))

    changed = [

        (path, original_values.get(path), new)

        for path, new in leaves(edited)

        if original_values.get(path) != new

    ]

    document = edited.get("document", {})

    decisions: list[dict[str, Any]] = []

    for path, old, new in changed:

        parts = path.replace("]", "").replace("[", ".").split(".")

        page = next((part for part in parts if part.isdigit()), None)

        table = next((part for part in parts if part in {"ACTIF", "PASSIF", "TCR"}), None)

        decisions.append({

            "tableau": table, "row_code": parts[-2] if len(parts) > 1 else None, "colonne": parts[-1], "annee": edited.get("identite", {}).get("annee_n"), "unite": "DZD",

            "role_champ": "SAISIE", "sens_attendu": None, "valeur_extraite": old, "valeur_recalculee": None, "valeur_corrigee": new, "valeur_retenue": new,

            "statut_controle": "CORRIGE", "decision_analyste": "CORRIGEE", "correction": {"avant": old, "apres": new, "motif": "CORRECTION_ANALYSTE"},

            "source": {"document_id": document.get("document_id"), "fichier": source_name, "page": int(page) + 1 if page else None},

        })

    return decisions

def count_reviewable_values(data: dict[str, Any]) -> int:

    """Count extracted cell values, excluding document metadata and controls."""

    total = 0

    for page in data.get("pages", []):

        for _path, value in leaves(page.get("donnees", {})):

            if value is not None:

                total += 1

    return total

def show_pdf(pdf_bytes: bytes) -> None:

    encoded = base64.b64encode(pdf_bytes).decode("ascii")

    st.markdown(f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="760" style="border:1px solid #D9E3EA;border-radius:10px"></iframe>', unsafe_allow_html=True)

def show_analysis() -> None:

    result = st.session_state.get("analysis")

    if not result:

        return

    model_used = result.get("model_used")

    if model_used:

        st.success(f"LLM response received: {model_used}", icon=":material/smart_toy:")

    else:

        st.warning("No LLM response received - only deterministic rules were run.", icon=":material/smart_toy:")

    st.caption(f"{result.get('rule_findings', 0)} rule findings · {result.get('llm_findings', 0)} validated LLM findings · {result.get('rejected_llm_findings', 0)} rejected")

    log_entry = result.get("log_entry", {})

    with st.expander("LLM audit log", expanded=False, icon=":material/receipt_long:"):

        st.caption(f"JSON: {log_entry.get('json_name', 'unknown')} · UTC: {log_entry.get('timestamp_utc', 'unknown')}")

        raw_output = log_entry.get("raw_llm_output")

        if raw_output:

            st.code(json.dumps(raw_output, ensure_ascii=False, indent=2), language="json")

        else:

            st.info("No raw LLM response was recorded for this run.")

        rejected = log_entry.get("rejected_findings", [])

        if rejected:

            st.caption(f"{len(rejected)} finding(s) rejected by validator")

            st.code(json.dumps(rejected, ensure_ascii=False, indent=2), language="json")

        st.caption(f"Persistent log: {result.get('log_path', 'not available')}")

    issues = result["issues"]

    if not issues:

        st.success("Aucune incohérence logique détectée.")

        return

    st.markdown("#### Analysis results")

    # A path can legitimately have findings from more than one validator (for
    # example, a deterministic rule and the LLM).  Streamlit widget keys must
    # still be unique within this render, so include the finding's position in
    # the action key rather than using only its cell path.
    for issue_index, issue in enumerate(issues):

        finding_key = f"analysis_finding_{widget_key(issue.get('path', ''))}_{issue_index}"

        with st.container(border=True, key=finding_key):

            if st.session_state.get("selected_finding_path") == issue.get("path"):

                st.caption(":material/ads_click: Cellule sélectionnée dans le tableau")

                # The analysis block is rendered in a reserved slot above the
                # editor.  Scrolling to this keyed container makes the cell's
                # “Afficher l’anomalie” action visibly reveal its explanation.
                st.html(

                    f"<script>requestAnimationFrame(() => document.querySelector(\"div[class*='st-key-{finding_key}']\")?.scrollIntoView({{behavior: 'smooth', block: 'center'}}));</script>",

                    unsafe_allow_javascript=True,

                )

            st.badge(issue.get("source", "rule"), color="blue" if issue.get("source", "rule") == "rule" else "violet")

            message = f"**{issue['title']}**  \n{issue['message']}  \n`{issue['path']}`"

            if issue["severity"] == "error":

                st.error(message, icon=":material/error:")

            else:

                st.warning(message, icon=":material/warning:")

            if str(issue.get("path", "")).count(">") == 2:

                st.button(

                    "Ouvrir la cellule",

                    icon=":material/ads_click:",

                    key=f"focus_{widget_key(issue['path'])}_{issue_index}",

                    on_click=lambda path=issue["path"]: st.session_state.update(focus_path=path, selected_finding_path=path),

                )

def review_workspace(name: str, pdf_bytes: bytes, original: dict[str, Any], output: Path | None, state_id: str, json_name: str | None = None) -> None:

    if st.session_state.get("active_document") != state_id:

        st.session_state.active_document = state_id

        st.session_state.original_json = copy.deepcopy(original)

        st.session_state.working_json = copy.deepcopy(original)

        st.session_state.analysis = None

    edited = st.session_state.working_json

    headline = edited.get("identite", {}).get("entreprise") or "Entreprise non renseignée"

    st.markdown(f'<div class="hero"><h1>Revue de {name}</h1><p>{headline} · Vérifiez chaque valeur extraite avant certification.</p></div>', unsafe_allow_html=True)

    # Reserve the review controls at the top. They are populated after the

    # editor has reconstructed the newest JSON from its widget values.

    toolbar_slot = st.empty()

    spinner_slot = st.empty()

    analysis_slot = st.empty()

    left, right = st.columns([0.72, 1.28], gap="large")

    with left:

        st.subheader("Document source", icon=":material/picture_as_pdf:")

        show_pdf(pdf_bytes)

    with right:

        st.subheader("Tableaux extraits", icon=":material/table_chart:")

        st.caption("Choisissez une page et un état financier. Les modifications sont tracées dans decisions_analyste.")

        previous = copy.deepcopy(edited)

        render_statement_editor(edited, state_id)

        st.session_state.working_json = edited

        edited = st.session_state.working_json

        if edited != previous:

            # A previous result may no longer describe the corrected payload.

            st.session_state.analysis = None

    # Render this visual block in the reserved top slot only after every

    # editable widget has returned its current value. This guarantees that one

    # click analyses the exact JSON currently shown in the UI.

    corrections = len(decisions_from_changes(st.session_state.original_json, edited, name))

    with toolbar_slot.container():

        a, b, c, action = st.columns([1, 1, 1, 1.45], vertical_alignment="center")

        a.metric("Status", "Certified" if output and output.exists() else "Ready for review")

        b.metric("Corrections", corrections)

        c.metric("Extracted pages", len(edited.get("pages", [])))

        analyse_clicked = action.button(

            "Analyse the JSON with AI",

            icon=":material/auto_awesome:",

            type="primary",

            width="stretch",

            help="Check the current edited JSON for logical inconsistencies.",

        )

    if analyse_clicked:

        # This placeholder was created before the PDF/JSON columns, so the

        # spinner appears above both panels while the analysis is running.

        with spinner_slot.container():

            with st.spinner("Vérification des cohérences…"):

                st.session_state.analysis = analyse_document(

                    edited,

                    json_name=json_name or f"{name}.json",

                )

        st.rerun()

    with analysis_slot.container(border=True):

        st.caption("AI and accounting consistency checks")

        show_analysis()

    st.subheader("Certification")

    with st.form("certification_" + state_id):

        col1, col2, col3 = st.columns(3)

        user_id = col1.text_input("Identifiant analyste", value="ANALYSTE")

        user_name = col2.text_input("Nom de l’analyste", value="Analyste Risque")

        comment = col3.text_input("Commentaire de certification", value="Liasse contrôlée sur le PDF.")

        accept_anomalies = st.checkbox("J’ai examiné les alertes et confirme la décision de certification.")

        submitted = st.form_submit_button("Approuver et générer le JSON certifié", icon=":material/task_alt:", type="primary", width="stretch", disabled=bool(output and output.exists()))

    if submitted:

        result = st.session_state.get("analysis") or {"issues": []}

        has_errors = any(issue["severity"] == "error" for issue in result["issues"])

        if has_errors and not accept_anomalies:

            st.error("Examinez les alertes rouges puis cochez la confirmation avant de certifier.")

            return

        decisions = decisions_from_changes(st.session_state.original_json, edited, name)

        accepted_anomalies = len(result["issues"]) if accept_anomalies else 0

        validated_values = max(0, count_reviewable_values(edited) - len(decisions))

        certified = create_certified_json(edited, user_id, user_name, comment, validated_values, len(decisions), accepted_anomalies, decisions)

        target = output or (ROOT / "output" / "modele_apres_certification.json")

        saved = save_certified_json(certified, target)

        st.success(f"JSON certifié enregistré : {saved}")

        st.download_button("Télécharger le JSON certifié", data=json.dumps(certified, ensure_ascii=False, indent=2), file_name=saved.name, mime="application/json")

        st.rerun()

def call_assistant_api(messages: list[dict[str, str]], document_question: str | None = None) -> str:
    """Chat with the same local Ollama model used by SemanticVerifier.py."""
    model = os.getenv("SEMANTIC_VERIFIER_MODEL", "qwen2.5:3b")
    system_prompt = (
        "You are CertiFlow Assistant, a helpful conversational assistant in an app for reviewing "
        "OCR and vision-language extractions from financial documents. Answer clearly and honestly. "
        "Do not invent financial values or claim to have inspected a document unless its content "
        "has actually been provided in the conversation. Reply in the language used by the user."
    )
    request_messages = [{"role": "system", "content": system_prompt}, *messages]
    if document_question:
        request_messages.append({"role": "user", "content": document_question})
    payload = {
        "model": model,
        "messages": request_messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama a renvoyé HTTP {error.code} pour le modèle « {model} ». {details[:500]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Impossible de joindre Ollama sur {OLLAMA_BASE_URL}. "
            "Vérifiez qu’Ollama est démarré et que le modèle utilisé par SemanticVerifier.py est disponible."
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama a renvoyé une réponse JSON invalide.") from error

    try:
        content = result["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Format de réponse inattendu depuis l’API de chat Ollama.") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Le modèle a renvoyé une réponse vide.")
    return content.strip()


def render_references(references: list[dict[str, object]]) -> None:
    """Show the exact PDF pages used for a response, including on replay."""
    if not references:
        return
    with st.expander("Références PDF", expanded=False, icon=":material/book_2:"):
        for reference in references:
            st.markdown(f"- `{reference['source']}` · page {reference['page']}")

def render_assistant() -> None:
    """Render the Assistant page and preserve chat history for this session."""
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = []

    st.markdown(
        """<div class="hero"><h1>Assistant</h1>
        <p>Discutez avec votre assistant IA. Votre conversation reste disponible pendant cette session.</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<style>
        div[data-testid="stChatMessage"] { max-width: 900px; margin-left: auto; margin-right: auto; }
        div[data-testid="stChatInput"] { max-width: 900px; margin-left: auto; margin-right: auto; }
        </style>""",
        unsafe_allow_html=True,
    )

    st.caption("Connecté automatiquement au même modèle Ollama que la vérification documentaire.")

    title_col, action_col = st.columns([5, 1])
    with title_col:
        st.subheader("Conversation")
    with action_col:
        if st.button("Effacer", icon=":material/delete_sweep:", width="stretch"):
            st.session_state.assistant_messages = []
            st.rerun()

    messages = st.session_state.assistant_messages
    if not messages:
        st.markdown(
            """<div style="text-align:center; padding:2.5rem 1rem 1rem; color: #64748b;">
            <div style="font-size:2.2rem; margin-bottom:.5rem;">✦</div>
            <div style="font-size:1.15rem; font-weight:600; color:inherit;">Comment puis-je vous aider ?</div>
            <div style="margin-top:.4rem;">Posez une question ou démarrez une nouvelle conversation.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_references(message.get("references", []))

    prompt = st.chat_input("Envoyer un message à Assistant…")
    if prompt:
        user_message = {"role": "user", "content": prompt}
        st.session_state.assistant_messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Assistant réfléchit…"):
                try:
                    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
                    references: list[dict[str, object]] = []
                    document_question = None
                    if pdf_files:
                        rag = load_rag_index(str(DOCUMENTS_DIR), pdf_fingerprint(DOCUMENTS_DIR))
                        retrieved_chunks = retrieve(prompt, rag)
                        references = references_from_chunks(retrieved_chunks)
                        document_question = build_prompt(prompt, retrieved_chunks)
                    answer = call_assistant_api(
                        st.session_state.assistant_messages[:-1],
                        document_question=document_question,
                    )
                except Exception as error:
                    st.error(f"Impossible d’obtenir une réponse : {error}")
                else:
                    st.markdown(answer)
                    render_references(references)
                    st.session_state.assistant_messages.append(
                        {"role": "assistant", "content": answer, "references": references}
                    )


st.sidebar.markdown("## CertiFlow")

page = st.sidebar.radio(
    "Navigation",
    ["Revue documentaire", "Assistant"],
    index=0,
    key="main_page_navigation",
)

if page == "Assistant":
    render_assistant()
    st.stop()

st.sidebar.caption("Contrôle humain des extractions OCR et vision-language")

mode = st.sidebar.radio("Mode de travail", ["Production - dossiers", "Démo - import"], label_visibility="collapsed")

st.sidebar.markdown("**Production**  \nDéposez `nom.pdf` et `nom.json` dans `documents/`.  \nLe résultat est créé dans `certified/`.")

if mode == "Production - dossiers":

    document_pairs = pairs()

    if not document_pairs:

        st.info("Aucun couple trouvé. Ajoutez un PDF et son JSON de même nom dans le dossier documents/.")

    else:

        names = [item["name"] for item in document_pairs]

        selected = st.sidebar.selectbox("Dossier à réviser", names)

        item = document_pairs[names.index(selected)]

        try:

            source = json.loads(Path(item["json"]).read_text(encoding="utf-8"))

        except json.JSONDecodeError as error:

            st.error(f"Le JSON associé est invalide : {error}")

        else:

            review_workspace(str(item["name"]), Path(item["pdf"]).read_bytes(), source, certified_path(str(item["name"])), "folder_" + str(item["name"]), Path(item["json"]).name)

else:

    uploaded_pdf = st.sidebar.file_uploader("PDF source", type="pdf")

    uploaded_json = st.sidebar.file_uploader("JSON extrait", type="json")

    if not uploaded_pdf:

        st.info("Importez un PDF dans la barre latérale pour commencer la démonstration.")

    elif not uploaded_json:

        st.info("Importez également le JSON extrait correspondant au PDF pour commencer la révision.")

    else:

        try:

            source = json.loads(uploaded_json.getvalue())

        except json.JSONDecodeError as error:

            st.error(f"Le JSON importé est invalide : {error}")

        else:

            json_bytes = uploaded_json.getvalue()

            json_hash = hashlib.sha256(json_bytes).hexdigest()[:12]

            json_state_id = f"{uploaded_json.name}_{len(json_bytes)}_{json_hash}"

            review_workspace(

                Path(uploaded_pdf.name).stem,

                uploaded_pdf.getvalue(),

                source,

                None,

                f"demo_{uploaded_pdf.name}_{uploaded_pdf.size}_{json_state_id}",

                uploaded_json.name,

            )
