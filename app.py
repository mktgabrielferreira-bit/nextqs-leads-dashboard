import base64
import io
import html
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

import gspread
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Relatório de Leads",
    page_icon="📊",
    layout="wide",
)

COMPANIES = {
    "nextqs": {
        "slug": "nextqs",
        "nome": "NextQS",
        "base_spreadsheet_id": "1dw5ssrZu9UfzymB7GLs0rqZf0LggvKAnC5Tek3go1cM",
        "logo_path": "assets/logo-nextqs.png",
        "titulo": "Relatório de Leads no Site NextQS",
    },
    "starled": {
        "slug": "starled",
        "nome": "StarLed",
        "base_spreadsheet_id": "1OmdviqRkQdsznC7BbyVwCxRxJQr7Bw7RNItKyYPfwEs",
        "logo_path": "assets/logo-starled.png",
        "titulo": "Relatório de Leads no Site StarLed",
    },
}

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SMARK_SPREADSHEET_ID = "1sYMJcxzmDQ1r-KPAZ38TNlrQeg7NQMrk-Yk26pyzMBg"
SMARK_WORKSHEET_GID = 841055934
SMARK_EMAIL_COLUMN = "E-mail Contato"
BASE_QUALIFIED_COLUMN = "qualificado"
OPPORTUNITIES_SHEET_NAME = "oportunidades"

# Nome da aba do SMARK no Google Sheets (onde o CSV será colado)
SMARK_SHEET_TAB_NAME = "smark_data"

SHEETS_REQUIRED = [
    "leads_site",
]

OPTIONAL_SHEETS = [
    "oportunidades",
    "leads_meta_whatsapp",
    "leads_meta_formulario",
]

MESES_LABEL = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}

DIA_SEMANA_LABEL = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}

GREEN_COLOR = "#83C9FF"
COMPARE_COLOR_1 = "#2563EB"
COMPARE_COLOR_2 = "#F97316"
FUNNEL_COLOR = "#1896D8"


def login():
    st.title("🔒 Acesso restrito")
    senha = st.text_input("Digite a senha:", type="password")

    if st.button("Entrar"):
        if senha == st.secrets["SENHA_DASH"]:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Senha incorreta")


if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

if not st.session_state["autenticado"]:
    login()
    st.stop()


def get_selected_company() -> dict:
    empresa = st.query_params.get("empresa", "nextqs")
    if empresa not in COMPANIES:
        empresa = "nextqs"
    return COMPANIES[empresa]


def render_company_selector(current_slug: str):
    def _select_company(slug: str):
        st.session_state["dashboard_view"] = "leads"
        st.query_params["empresa"] = slug
        st.rerun()

    if hasattr(st, "popover"):
        with st.popover("Selecionar Empresa", use_container_width=True):
            if st.button("NEXTQS", use_container_width=True, disabled=current_slug == "nextqs"):
                _select_company("nextqs")
            if st.button("STARLED", use_container_width=True, disabled=current_slug == "starled"):
                _select_company("starled")
    else:
        with st.expander("Selecionar Empresa", expanded=False):
            if st.button("NEXTQS", use_container_width=True, disabled=current_slug == "nextqs"):
                _select_company("nextqs")
            if st.button("STARLED", use_container_width=True, disabled=current_slug == "starled"):
                _select_company("starled")


def normalize_empty(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in ["false", "none", "null", "undefined", "", "nan"]:
        return None
    return text


def normalize_email(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in ["", "none", "null", "nan", "undefined"]:
        return None
    return text


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def strip_accents(value) -> str:
    text = normalize_text(value).lower()
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def has_qualified_marker(value) -> bool:
    return normalize_text(value).lower() in {"sim", "duplicado"}


def normalize_origin(value) -> str:
    text = normalize_text(value)
    return text if text else "Origem não identificada"


def normalize_campaign(value) -> str:
    text = normalize_text(value)
    if text.lower() in {"", "none", "null", "nan", "undefined", "false", "{campaignname}", "(not set)", "(notset)"}:
        return "Campanha não identificada"
    return text


def get_today_local() -> date:
    try:
        return datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    except Exception:
        return date.today()


def month_label_to_num(mes_label: str) -> int:
    return [k for k, v in MESES_LABEL.items() if v == mes_label][0]


def filter_by_year_month(df: pd.DataFrame, ano: int, mes_num: int) -> pd.DataFrame:
    if df.empty or "ano" not in df.columns or "mes" not in df.columns:
        return df.copy()
    return df[(df["ano"] == int(ano)) & (df["mes"] == int(mes_num))].copy()


def filter_by_custom_selection(df: pd.DataFrame, ano: int, mes_label: str) -> pd.DataFrame:
    if df.empty or "ano" not in df.columns:
        return df.copy()
    if mes_label == "Todo o ano":
        return df[df["ano"] == int(ano)].copy()
    if "mes" not in df.columns:
        return df.copy()
    return filter_by_year_month(df, ano, month_label_to_num(mes_label))


def trigger_sheet_reload():
    load_sheet.clear()
    st.rerun()


def label_evento(v):
    v_str = str(v).strip().lower()
    if "whats" in v_str:
        return "WhatsApp"
    if "form" in v_str:
        return "Formulário"
    return str(v).title()


def _get_creds():
    if "gcp_service_account" in st.secrets:
        service_info = dict(st.secrets["gcp_service_account"])
        return Credentials.from_service_account_info(service_info, scopes=SCOPES)
    return Credentials.from_service_account_file("credenciais_sheets.json", scopes=SCOPES)


def get_gspread_client():
    creds = _get_creds()
    return gspread.authorize(creds)


def get_base_spreadsheet(company_slug: str):
    client = get_gspread_client()
    spreadsheet_id = COMPANIES[company_slug]["base_spreadsheet_id"]
    return client.open_by_key(spreadsheet_id)


def get_worksheet_by_gid(spreadsheet, gid: int):
    for ws in spreadsheet.worksheets():
        if getattr(ws, "id", None) == gid:
            return ws
    raise ValueError(f"Não foi possível localizar a aba com gid={gid} na planilha do SMARK.")


def get_or_create_worksheet(spreadsheet, title: str, rows: int = 1000, cols: int = 20):
    try:
        return spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)


def ensure_column_exists(ws, column_name: str) -> int:
    headers = ws.row_values(1)
    if column_name in headers:
        return headers.index(column_name) + 1
    new_col_idx = len(headers) + 1
    ws.update_cell(1, new_col_idx, column_name)
    return new_col_idx


def ensure_headers_exist(ws, required_headers: list[str]) -> dict:
    headers = ws.row_values(1)
    changed = False
    for header in required_headers:
        if header not in headers:
            headers.append(header)
            changed = True

    if changed:
        ws.update(f"A1:{gspread.utils.rowcol_to_a1(1, len(headers))}", [headers])

    return {header: idx + 1 for idx, header in enumerate(headers)}


def format_date_br_from_any(value) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    for fmt in [
        "%d/%m/%Y - %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except Exception:
            pass

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return parsed.strftime("%d/%m/%Y")
    return text


def format_smark_swapped_date_to_br(value) -> str:
    return format_date_br_from_any(value)


def parse_date_for_sort(value):
    text = normalize_text(value)
    if not text:
        return pd.Timestamp.max

    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.max
    return parsed


def clean_status_funil(value) -> str:
    text = normalize_text(value)
    return re.sub(r"^\s*\d+\s*-\s*", "", text)


def _phone_variants_from_digits(digits: str) -> set[str]:
    digits = re.sub(r"\D", "", digits or "")
    if not digits:
        return set()

    variants = {digits}

    if digits.startswith("55") and len(digits) > 2:
        variants.add(digits[2:])

    for item in list(variants):
        if len(item) in {10, 11}:
            variants.add(f"55{item}")

    for item in list(variants):
        core = item[2:] if item.startswith("55") and len(item) in {12, 13} else item
        if len(core) == 10:
            with_nine = core[:2] + "9" + core[2:]
            variants.add(with_nine)
            variants.add("55" + with_nine)
        elif len(core) == 11 and core[2] == "9":
            without_nine = core[:2] + core[3:]
            variants.add(without_nine)
            variants.add("55" + without_nine)

    return {v for v in variants if v}


def extract_phone_candidates(value) -> set[str]:
    text = normalize_text(value)
    if not text:
        return set()

    candidates = set()

    pattern = re.compile(r"(?:\+?55\D*)?(?:\(?0?(\d{2})\)?\D*)?(9?\d{4,5})\D*(\d{4})")
    for match in pattern.finditer(text):
        ddd, prefix, suffix = match.groups()
        phone_local = re.sub(r"\D", "", f"{prefix}{suffix}")
        if len(phone_local) in {8, 9}:
            candidates.update(_phone_variants_from_digits(phone_local))
            if ddd and len(ddd) == 2:
                ddd_digits = re.sub(r"\D", "", ddd)
                national = f"{ddd_digits}{phone_local}"
                candidates.update(_phone_variants_from_digits(national))

    raw_digits = re.sub(r"\D", "", text)
    for size in (13, 12, 11, 10, 9, 8):
        if len(raw_digits) >= size:
            tail = raw_digits[-size:]
            candidates.update(_phone_variants_from_digits(tail))

    return {re.sub(r"\D", "", c) for c in candidates if re.sub(r"\D", "", c)}


def normalize_phone_for_lookup(value) -> str | None:
    digits = re.sub(r"\D", "", normalize_text(value))
    if not digits:
        return None

    if digits.startswith("55") and len(digits) in {12, 13}:
        digits = digits[2:]

    if len(digits) == 10:
        digits = digits[:2] + "9" + digits[2:]

    return digits


def build_smark_email_map(smark_records: list[dict]) -> dict:
    email_map = {}
    for row in smark_records:
        email_norm = normalize_email(row.get(SMARK_EMAIL_COLUMN))
        if email_norm and email_norm not in email_map:
            email_map[email_norm] = row
    return email_map


def build_smark_phone_map(smark_records: list[dict]) -> dict:
    phone_map = {}
    for row in smark_records:
        for phone in extract_phone_candidates(row.get("Telefones")):
            phone_map.setdefault(phone, row)
    return phone_map


def build_opportunity_row(data_lead, canal: str, user_id, origem, campanha, smark_row: dict) -> list[str]:
    return [
        format_smark_swapped_date_to_br(smark_row.get("Data Oportunidade")),
        format_date_br_from_any(data_lead),
        canal,
        normalize_origin(origem),
        normalize_text(campanha),
        normalize_text(user_id),
        normalize_text(smark_row.get("Cod. Oportunidade")),
        normalize_text(smark_row.get("Área de Atuação")),
        normalize_text(smark_row.get("Nome Colaborador Responsável")),
        clean_status_funil(smark_row.get("Funil de Venda")),
        format_smark_swapped_date_to_br(smark_row.get("Data Encerramento")),
    ]


def build_opportunity_record(data_lead, canal: str, user_id, origem, campanha, smark_row: dict) -> dict[str, str]:
    row = build_opportunity_row(data_lead, canal, user_id, origem, campanha, smark_row)
    return {
        "data_oportunidade": row[0],
        "data_lead": row[1],
        "canal": row[2],
        "origem": row[3],
        "campanha": row[4],
        "user_id": row[5],
        "oportunidade": row[6],
        "area_atuação": row[7],
        "consultor": row[8],
        "status_funil": row[9],
        "data_encerramento": row[10],
    }


# ---------------------------------------------------------------------------
# Função: upload CSV do SMARK e cola na planilha do SMARK no Sheets
# ---------------------------------------------------------------------------

# Mapeamento de colunas: nome no CSV → nome esperado internamente pelo sistema
# Se algum campo tiver nome diferente no CSV, ajuste aqui.
SMARK_CSV_TO_SHEET_COLUMN_MAP = {
    # CSV col name       : Sheet col name (se iguais, não precisa mapear)
    # Exemplo de divergência: "E-mail": "E-mail Contato"
    # Adicione aqui se necessário. Por ora, usamos pass-through.
}

SMARK_REQUIRED_COLUMNS = [
    "E-mail Contato",
    "Telefones",
    "Cod. Oportunidade",
    "Data Oportunidade",
    "Área de Atuação",
    "Nome Colaborador Responsável",
    "Funil de Venda",
    "Data Encerramento",
]


def upload_csv_to_smark_sheet(csv_file) -> dict:
    """
    Lê o CSV exportado do SMARK, cola os dados na aba do SMARK no Google Sheets
    e retorna informações sobre o upload (última data, colunas divergentes).
    """
    csv_file.seek(0)
    file_bytes = csv_file.getvalue()

    encodings_to_try = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    separators_to_try = [";", ","]

    df_csv = None
    last_error = None

    for encoding in encodings_to_try:
        for sep in separators_to_try:
            try:
                df_test = pd.read_csv(
                    io.BytesIO(file_bytes),
                    sep=sep,
                    encoding=encoding,
                )

                # evita falso positivo quando o separador está errado
                if df_test.shape[1] <= 1 and sep == ";":
                    continue

                df_csv = df_test
                break
            except Exception as e:
                last_error = e

        if df_csv is not None:
            break

    if df_csv is None:
        raise ValueError(
            f"Não foi possível ler o CSV do SMARK. "
            f"Tente exportar novamente em CSV UTF-8. Erro original: {last_error}"
        )

    # Aplica mapeamento de colunas divergentes (se houver)
    if SMARK_CSV_TO_SHEET_COLUMN_MAP:
        df_csv = df_csv.rename(columns=SMARK_CSV_TO_SHEET_COLUMN_MAP)

    divergent_cols = [c for c in SMARK_REQUIRED_COLUMNS if c not in df_csv.columns]

    ultima_data_str = None
    if "Data Oportunidade" in df_csv.columns:
        datas = pd.to_datetime(df_csv["Data Oportunidade"], dayfirst=True, errors="coerce")
        datas = datas.dropna()
        if not datas.empty:
            ultima_data_str = datas.max().strftime("%d/%m/%Y")

    client = get_gspread_client()
    smark_spreadsheet = client.open_by_key(SMARK_SPREADSHEET_ID)
    smark_ws = get_or_create_worksheet(
        smark_spreadsheet,
        SMARK_SHEET_TAB_NAME,
        rows=max(5000, len(df_csv) + 10),
        cols=max(30, len(df_csv.columns) + 2),
    )

    df_str = df_csv.fillna("").astype(str)
    payload = [df_str.columns.tolist()] + df_str.values.tolist()

    smark_ws.clear()
    end_cell = gspread.utils.rowcol_to_a1(len(payload), len(df_csv.columns))
    smark_ws.update(f"A1:{end_cell}", payload, value_input_option="USER_ENTERED")

    return {
        "rows": len(df_csv),
        "cols": len(df_csv.columns),
        "ultima_data": ultima_data_str,
        "divergent_cols": divergent_cols,
    }


def get_smark_ultima_data_from_sheet() -> str | None:
    """
    Lê a aba smark_data da planilha do SMARK e retorna a última 'Data Oportunidade'.
    Retorna None se não encontrar.
    """
    try:
        client = get_gspread_client()
        smark_spreadsheet = client.open_by_key(SMARK_SPREADSHEET_ID)
        try:
            ws = smark_spreadsheet.worksheet(SMARK_SHEET_TAB_NAME)
        except gspread.exceptions.WorksheetNotFound:
            # Tenta usar a aba original pelo GID
            try:
                ws = get_worksheet_by_gid(smark_spreadsheet, SMARK_WORKSHEET_GID)
            except Exception:
                return None

        records = ws.get_all_records()
        if not records:
            return None
        df = pd.DataFrame(records)
        if "Data Oportunidade" not in df.columns:
            return None
        datas = pd.to_datetime(df["Data Oportunidade"], dayfirst=True, errors="coerce").dropna()
        if datas.empty:
            return None
        return datas.max().strftime("%d/%m/%Y")
    except Exception:
        return None


def sync_opportunities_with_smark(company_slug: str, smark_records_override: list[dict] | None = None) -> dict:
    """
    Sincroniza oportunidades. Se smark_records_override for fornecido (após upload de CSV),
    usa esses registros em vez de buscar da planilha.
    """
    client = get_gspread_client()

    base_spreadsheet = client.open_by_key(COMPANIES[company_slug]["base_spreadsheet_id"])
    leads_ws = base_spreadsheet.worksheet("leads_site")
    opportunities_ws = get_or_create_worksheet(base_spreadsheet, OPPORTUNITIES_SHEET_NAME)

    try:
        leads_meta_whatsapp_ws = base_spreadsheet.worksheet("leads_meta_whatsapp")
        instagram_records = leads_meta_whatsapp_ws.get_all_records()
    except gspread.exceptions.WorksheetNotFound:
        leads_meta_whatsapp_ws = None
        instagram_records = []

    try:
        leads_meta_formulario_ws = base_spreadsheet.worksheet("leads_meta_formulario")
        formulario_records = leads_meta_formulario_ws.get_all_records()
    except gspread.exceptions.WorksheetNotFound:
        leads_meta_formulario_ws = None
        formulario_records = []

    leads_records = leads_ws.get_all_records()

    # Obtém registros do SMARK (da planilha smark_data ou da original)
    if smark_records_override is not None:
        smark_records = smark_records_override
    else:
        smark_spreadsheet = client.open_by_key(SMARK_SPREADSHEET_ID)
        try:
            smark_ws_read = smark_spreadsheet.worksheet(SMARK_SHEET_TAB_NAME)
        except gspread.exceptions.WorksheetNotFound:
            smark_ws_read = get_worksheet_by_gid(smark_spreadsheet, SMARK_WORKSHEET_GID)
        smark_records = smark_ws_read.get_all_records()

    opp_required_headers = [
        "data_oportunidade",
        "data_lead",
        "canal",
        "origem",
        "campanha",
        "user_id",
        "oportunidade",
        "area_atuação",
        "consultor",
        "status_funil",
        "data_encerramento",
    ]

    df_leads = pd.DataFrame(leads_records)
    df_smark = pd.DataFrame(smark_records)
    df_instagram = pd.DataFrame(instagram_records)
    df_formulario = pd.DataFrame(formulario_records)

    required_smark_columns = [
        SMARK_EMAIL_COLUMN,
        "Telefones",
        "Cod. Oportunidade",
        "Data Oportunidade",
        "Área de Atuação",
        "Nome Colaborador Responsável",
        "Funil de Venda",
        "Data Encerramento",
    ]
    missing_smark = [col for col in required_smark_columns if col not in df_smark.columns]
    if missing_smark:
        raise ValueError("A planilha do SMARK não possui as colunas obrigatórias: " + ", ".join(missing_smark))

    if not df_leads.empty:
        base_email_col = "email" if "email" in df_leads.columns else "user_id_email" if "user_id_email" in df_leads.columns else None
        if base_email_col is None:
            raise ValueError("A planilha base precisa ter a coluna 'email' ou 'user_id_email'.")
        if "user_id_email" not in df_leads.columns:
            raise ValueError("A aba 'leads_site' precisa ter a coluna 'user_id_email' para preencher 'user_id' em oportunidades.")
    else:
        base_email_col = None

    if leads_meta_whatsapp_ws is not None and not df_instagram.empty:
        required_instagram_columns = ["telefone", "user_id_cel", "data", "campanha"]
        missing_instagram = [col for col in required_instagram_columns if col not in df_instagram.columns]
        if missing_instagram:
            raise ValueError("A aba 'leads_meta_whatsapp' não possui as colunas obrigatórias: " + ", ".join(missing_instagram))

    if leads_meta_formulario_ws is not None and not df_formulario.empty:
        required_formulario_columns = ["email", "user_id_email", "data_hora", "origem", "campanha"]
        missing_formulario = [col for col in required_formulario_columns if col not in df_formulario.columns]
        if missing_formulario:
            raise ValueError("A aba 'leads_meta_formulario' não possui as colunas obrigatórias: " + ", ".join(missing_formulario))

    ensure_column_exists(leads_ws, BASE_QUALIFIED_COLUMN)
    ensure_headers_exist(opportunities_ws, opp_required_headers)

    instagram_header_map = {}
    if leads_meta_whatsapp_ws is not None:
        instagram_header_map = ensure_headers_exist(leads_meta_whatsapp_ws, ["qualificado", "consultor"])

    formulario_header_map = {}
    if leads_meta_formulario_ws is not None:
        formulario_header_map = ensure_headers_exist(leads_meta_formulario_ws, ["qualificado"])

    smark_email_map = build_smark_email_map(smark_records)
    smark_phone_map = build_smark_phone_map(smark_records)

    opportunities_by_code = {}
    generated_user_ids_site = set()
    processed_match_emails = set()
    site_matches = 0
    instagram_matches = 0
    formulario_matches = 0
    qualified_sim = 0
    qualified_duplicate = 0
    instagram_qualified_sim = 0
    formulario_qualified_sim = 0
    site_opportunities_added = 0
    instagram_opportunities_added = 0
    formulario_opportunities_added = 0
    opportunities_skipped = 0

    leads_headers = leads_ws.row_values(1)
    qualified_col_idx = leads_headers.index(BASE_QUALIFIED_COLUMN) + 1
    qualified_values = []

    for row in leads_records:
        email_norm = normalize_email(row.get(base_email_col)) if base_email_col else None
        current_qualified_value = normalize_text(row.get(BASE_QUALIFIED_COLUMN))
        target_value = current_qualified_value

        if email_norm and email_norm in smark_email_map:
            site_matches += 1
            smark_row = smark_email_map[email_norm]

            if email_norm not in processed_match_emails:
                processed_match_emails.add(email_norm)
                if not has_qualified_marker(current_qualified_value):
                    target_value = "SIM"
                    qualified_sim += 1

                opportunity_code = normalize_text(smark_row.get("Cod. Oportunidade"))
                user_id_value = normalize_text(row.get("user_id_email"))
                user_id_key = user_id_value.lower()

                if not opportunity_code or not user_id_value or user_id_key in generated_user_ids_site:
                    opportunities_skipped += 1
                elif opportunity_code not in opportunities_by_code:
                    opportunities_by_code[opportunity_code] = build_opportunity_record(
                        row.get("data_hora"),
                        "Site",
                        user_id_value,
                        row.get("origem"),
                        row.get("utm_campaign"),
                        smark_row,
                    )
                    generated_user_ids_site.add(user_id_key)
                    site_opportunities_added += 1
                else:
                    opportunities_skipped += 1
            else:
                if not has_qualified_marker(current_qualified_value):
                    target_value = "Duplicado"
                    qualified_duplicate += 1

        qualified_values.append([target_value])

    if qualified_values:
        qualified_col_letter = gspread.utils.rowcol_to_a1(1, qualified_col_idx)[:-1]
        qualified_range = f"{qualified_col_letter}2:{qualified_col_letter}{len(qualified_values) + 1}"
        leads_ws.update(qualified_range, qualified_values, value_input_option="USER_ENTERED")

    formulario_qualified_values = []

    for row in formulario_records:
        email_norm = normalize_email(row.get("email"))
        current_qualified_value = normalize_text(row.get("qualificado"))
        target_value = current_qualified_value

        if email_norm and email_norm in smark_email_map:
            formulario_matches += 1
            smark_row = smark_email_map[email_norm]
            target_value = "SIM"
            formulario_qualified_sim += 1

            opportunity_code = normalize_text(smark_row.get("Cod. Oportunidade"))
            if opportunity_code and opportunity_code not in opportunities_by_code:
                opportunities_by_code[opportunity_code] = build_opportunity_record(
                    row.get("data_hora"),
                    "Meta - Formulário",
                    row.get("user_id_email"),
                    row.get("origem"),
                    row.get("campanha"),
                    smark_row,
                )
                formulario_opportunities_added += 1
            else:
                opportunities_skipped += 1

        formulario_qualified_values.append([target_value])

    if leads_meta_formulario_ws is not None and formulario_records:
        qual_col_idx = formulario_header_map["qualificado"]
        qual_col_letter = gspread.utils.rowcol_to_a1(1, qual_col_idx)[:-1]
        qual_range = f"{qual_col_letter}2:{qual_col_letter}{len(formulario_qualified_values) + 1}"
        leads_meta_formulario_ws.update(qual_range, formulario_qualified_values, value_input_option="USER_ENTERED")

    instagram_qualified_values = []
    instagram_consultor_values = []

    for row in instagram_records:
        phone_norm = normalize_phone_for_lookup(row.get("telefone"))
        target_value = ""
        consultor_value = ""

        if phone_norm and phone_norm in smark_phone_map:
            instagram_matches += 1
            smark_row = smark_phone_map[phone_norm]
            target_value = "SIM"
            consultor_value = normalize_text(smark_row.get("Nome Colaborador Responsável"))
            instagram_qualified_sim += 1

            opportunity_code = normalize_text(smark_row.get("Cod. Oportunidade"))
            if opportunity_code and opportunity_code not in opportunities_by_code:
                opportunities_by_code[opportunity_code] = build_opportunity_record(
                    row.get("data"),
                    "Meta - Whatsapp",
                    row.get("user_id_cel"),
                    row.get("origem"),
                    row.get("campanha"),
                    smark_row,
                )
                instagram_opportunities_added += 1
            else:
                opportunities_skipped += 1

        instagram_qualified_values.append([target_value])
        instagram_consultor_values.append([consultor_value])

    if leads_meta_whatsapp_ws is not None and instagram_records:
        qual_col_idx = instagram_header_map["qualificado"]
        consultor_col_idx = instagram_header_map["consultor"]

        qual_col_letter = gspread.utils.rowcol_to_a1(1, qual_col_idx)[:-1]
        qual_range = f"{qual_col_letter}2:{qual_col_letter}{len(instagram_qualified_values) + 1}"
        leads_meta_whatsapp_ws.update(qual_range, instagram_qualified_values, value_input_option="USER_ENTERED")

        consultor_col_letter = gspread.utils.rowcol_to_a1(1, consultor_col_idx)[:-1]
        consultor_range = f"{consultor_col_letter}2:{consultor_col_letter}{len(instagram_consultor_values) + 1}"
        leads_meta_whatsapp_ws.update(consultor_range, instagram_consultor_values, value_input_option="USER_ENTERED")

    opportunities_headers = opp_required_headers.copy()

    opportunities_records = sorted(
        opportunities_by_code.values(),
        key=lambda record: parse_date_for_sort(record.get("data_oportunidade")),
    )
    opportunities_rows = [
        [record.get(header, "") for header in opportunities_headers]
        for record in opportunities_records
    ]
    opportunities_payload = [opportunities_headers] + opportunities_rows
    opportunities_ws.clear()
    opportunities_ws.update(
        f"A1:{gspread.utils.rowcol_to_a1(len(opportunities_payload), len(opportunities_headers))}",
        opportunities_payload,
        value_input_option="USER_ENTERED",
    )

    return {
        "matches": site_matches + instagram_matches + formulario_matches,
        "site_matches": site_matches,
        "instagram_matches": instagram_matches,
        "formulario_matches": formulario_matches,
        "qualified_sim": qualified_sim,
        "qualified_duplicate": qualified_duplicate,
        "instagram_qualified_sim": instagram_qualified_sim,
        "formulario_qualified_sim": formulario_qualified_sim,
        "opportunities_added": site_opportunities_added + instagram_opportunities_added + formulario_opportunities_added,
        "site_opportunities_added": site_opportunities_added,
        "instagram_opportunities_added": instagram_opportunities_added,
        "formulario_opportunities_added": formulario_opportunities_added,
        "opportunities_skipped": opportunities_skipped,
        "leads_rows": len(df_leads),
        "instagram_rows": len(df_instagram),
        "formulario_rows": len(df_formulario),
        "smark_rows": len(df_smark),
        "base_email_col": base_email_col,
    }


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

    out = pd.to_datetime(
        s,
        format="%d/%m/%Y - %H:%M:%S",
        dayfirst=True,
        errors="coerce",
    )
    if out.isna().all():
        out = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return out


def _standardize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in ["origem", "dispositivo", "utm_campaign", "utm_term", "ip_address", "evento", "user_id_email"]:
        if col not in df.columns:
            df[col] = None

    for col in ["origem", "dispositivo", "utm_campaign", "utm_term", "ip_address", "evento", "user_id_email"]:
        df[col] = df[col].apply(normalize_empty)

    df["utm_campaign"] = df["utm_campaign"].replace(["{campaignname}", "(not set)", "(notset)"], None)
    df["origem"] = df["origem"].fillna("Origem não identificada")
    df["dispositivo"] = df["dispositivo"].fillna("Dispositivo não identificado")
    df["dispositivo"] = df["dispositivo"].apply(
        lambda x: x if x == "Dispositivo não identificado" else str(x).capitalize()
    )
    df["dispositivo"] = df["dispositivo"].replace({"Mobile - ios": "Mobile - iOS"})
    df["ip_address"] = df["ip_address"].fillna("IP não identificado")
    df["utm_campaign"] = df["utm_campaign"].fillna("Campanha não identificada")
    df["utm_term"] = df["utm_term"].fillna("Palavra-chave não identificada")

    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_sheet(company_slug: str, sheet_name: str) -> pd.DataFrame:
    try:
        ws = get_base_spreadsheet(company_slug).worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        return pd.DataFrame()

    data = ws.get_all_records()
    df = pd.DataFrame(data)

    if df.empty:
        return df

    if sheet_name == "meta_campanhas":
        return prepare_meta_ads_dataframe(df)

    if sheet_name == "oportunidades":
        if "data_oportunidade" in df.columns:
            df["data_hora"] = pd.to_datetime(df["data_oportunidade"], dayfirst=True, errors="coerce")
        elif "data_lead" in df.columns:
            df["data_hora"] = pd.to_datetime(df["data_lead"], dayfirst=True, errors="coerce")
        else:
            df["data_hora"] = pd.NaT

        # Carrega data de encerramento como coluna de data para filtro de Negócios Efetuados
        data_encerramento_col = None
        if "data encerramento" in df.columns:
            data_encerramento_col = "data encerramento"
        elif "data_encerramento" in df.columns:
            data_encerramento_col = "data_encerramento"

        if data_encerramento_col:
            df["data_encerramento_parsed"] = pd.to_datetime(df[data_encerramento_col], dayfirst=True, errors="coerce")
        else:
            df["data_encerramento_parsed"] = pd.NaT
    else:
        df = _standardize_common_columns(df)

        if "data_hora" not in df.columns:
            for alt in ["dataHora", "datetime", "timestamp", "data"]:
                if alt in df.columns:
                    df["data_hora"] = df[alt]
                    break
            if "data_hora" not in df.columns:
                df["data_hora"] = None

        df["data_hora"] = _parse_datetime_series(df["data_hora"])

    df = df.dropna(subset=["data_hora"])

    if df.empty:
        return df

    df["data"] = df["data_hora"].dt.date
    df["ano"] = df["data_hora"].dt.year.astype(int)
    df["mes"] = df["data_hora"].dt.month.astype(int)
    df["hora"] = df["data_hora"].dt.hour
    df["dia_semana"] = df["data_hora"].dt.dayofweek.map(DIA_SEMANA_LABEL)

    if sheet_name != "oportunidades":
        df = _standardize_common_columns(df)

    return df


def get_period_filtered_df(df_src: pd.DataFrame, periodo_sel: str, hoje: date, ontem: date) -> pd.DataFrame:
    if df_src.empty or "data" not in df_src.columns:
        return df_src

    if periodo_sel == "Hoje":
        return df_src[df_src["data"] == hoje].copy()
    if periodo_sel == "Ontem":
        return df_src[df_src["data"] == ontem].copy()
    if periodo_sel == "Últimos 30 dias":
        start = hoje - timedelta(days=29)
        end = hoje
        return df_src[(df_src["data"] >= start) & (df_src["data"] <= end)].copy()
    if periodo_sel == "Este mês":
        start = date(hoje.year, hoje.month, 1)
        end = hoje
        return df_src[(df_src["data"] >= start) & (df_src["data"] <= end)].copy()
    if periodo_sel == "Este ano":
        start = date(hoje.year, 1, 1)
        end = hoje
        return df_src[(df_src["data"] >= start) & (df_src["data"] <= end)].copy()
    return df_src.copy()


def get_effective_period_filtered_df(df_src: pd.DataFrame, periodo_sel: str, hoje: date, ontem: date) -> pd.DataFrame:
    if periodo_sel == "Personalizado":
        if st.session_state.get("custom_aplicado", False):
            return filter_by_custom_selection(
                df_src,
                int(st.session_state.get("custom_ano", hoje.year)),
                st.session_state.get("custom_mes_label", "Todo o ano"),
            )
        return get_period_filtered_df(df_src, "Últimos 30 dias", hoje, ontem)
    return get_period_filtered_df(df_src, periodo_sel, hoje, ontem)


def format_period_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def last_day_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def get_period_label(periodo_sel: str, hoje: date, ontem: date) -> str:
    if periodo_sel == "Hoje":
        return "Período: Hoje"
    if periodo_sel == "Ontem":
        return "Período: Ontem"
    if periodo_sel == "Últimos 30 dias":
        start = hoje - timedelta(days=29)
        return f"Período: {format_period_date(start)} a {format_period_date(hoje)}"
    if periodo_sel == "Este mês":
        start = date(hoje.year, hoje.month, 1)
        return f"Período: {format_period_date(start)} a {format_period_date(hoje)}"
    if periodo_sel == "Este ano":
        start = date(hoje.year, 1, 1)
        return f"Período: {format_period_date(start)} a {format_period_date(hoje)}"
    if periodo_sel == "Personalizado" and st.session_state.get("custom_aplicado", False):
        custom_ano = int(st.session_state.get("custom_ano", hoje.year))
        custom_mes_label = st.session_state.get("custom_mes_label", "Todo o ano")
        if custom_mes_label == "Todo o ano":
            start = date(custom_ano, 1, 1)
            end = hoje if custom_ano == hoje.year else date(custom_ano, 12, 31)
        else:
            mes_num = month_label_to_num(custom_mes_label)
            start = date(custom_ano, mes_num, 1)
            end = hoje if (custom_ano == hoje.year and mes_num == hoje.month) else last_day_of_month(custom_ano, mes_num)
        return f"Período: {format_period_date(start)} a {format_period_date(end)}"
    if periodo_sel == "Personalizado":
        start = hoje - timedelta(days=29)
        return f"Período: {format_period_date(start)} a {format_period_date(hoje)}"
    return "Período: Todos os dados"


def render_period_label(periodo_sel: str, hoje: date, ontem: date):
    st.markdown(
        f"""
        <div style="
            margin: 12px 0 2px 0;
            font-size: 14px;
            font-weight: 700;
            opacity: 0.82;">
            {html.escape(get_period_label(periodo_sel, hoje, ontem))}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compare_period_label(ano: int, m1_label: str, m2_label: str):
    def _range(month_label: str) -> str:
        month = month_label_to_num(month_label)
        start = date(int(ano), month, 1)
        end = last_day_of_month(int(ano), month)
        return f"{format_period_date(start)} a {format_period_date(end)}"

    st.markdown(
        f"""
        <div style="
            margin: 12px 0 2px 0;
            font-size: 14px;
            font-weight: 700;
            opacity: 0.82;">
            {html.escape(f"Período: {_range(m1_label)} vs {_range(m2_label)}")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_extra_filters_leads(df_base: pd.DataFrame, eventos_sel, origens_sel, dispositivos_sel) -> pd.DataFrame:
    if df_base.empty:
        return df_base.copy()
    df_out = df_base[
        (df_base["evento"].isin(eventos_sel))
        & (df_base["origem"].isin(origens_sel))
        & (df_base["dispositivo"].isin(dispositivos_sel))
    ].copy()
    return df_out.sort_values("data_hora")


def apply_common_filters(df_base: pd.DataFrame, origens_sel, dispositivos_sel) -> pd.DataFrame:
    if df_base.empty:
        return df_base.copy()
    df_out = df_base.copy()
    if "origem" in df_out.columns:
        df_out = df_out[df_out["origem"].isin(origens_sel)]
    if "dispositivo" in df_out.columns:
        df_out = df_out[df_out["dispositivo"].isin(dispositivos_sel)]
    if "data_hora" in df_out.columns:
        df_out = df_out.sort_values("data_hora")
    return df_out


def concat_non_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [df for df in frames if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def is_meta_origin(value) -> bool:
    text = normalize_text(value).lower()
    return any(token in text for token in ["meta", "instagram", "facebook", "fb ads"])


def is_google_ads_origin(value) -> bool:
    text = normalize_text(value).lower()
    return any(token in text for token in ["google ads", "adwords", "google / cpc", "google-cpc", "google cpc"])


def count_unique_leads(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    prepared = prepare_leads_for_reporting(df)
    if prepared.empty or "lead_key" not in prepared.columns:
        return 0
    return int(prepared["lead_key"].nunique())


def count_unique_leads_by(df: pd.DataFrame, group_cols: list[str], count_name: str = "leads") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=group_cols + [count_name])

    prepared = prepare_leads_for_reporting(df)
    if prepared.empty:
        return pd.DataFrame(columns=group_cols + [count_name])

    for col in group_cols:
        if col not in prepared.columns:
            prepared[col] = None

    prepared = prepared.dropna(subset=group_cols).copy()
    if prepared.empty:
        return pd.DataFrame(columns=group_cols + [count_name])

    return (
        prepared.drop_duplicates(subset=group_cols + ["lead_key"])
        .groupby(group_cols)["lead_key"]
        .nunique()
        .reset_index(name=count_name)
    )


def build_combined_leads(
    company_slug: str,
    df_leads_site: pd.DataFrame,
    df_leads_meta_whatsapp: pd.DataFrame | None = None,
    df_leads_meta_formulario: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [df_leads_site]
    if company_slug == "nextqs":
        frames.extend([df_leads_meta_whatsapp, df_leads_meta_formulario])
    return concat_non_empty(frames)


def get_highlight_kpis(
    company_slug: str,
    df_leads_site: pd.DataFrame,
    df_leads_meta_whatsapp: pd.DataFrame | None = None,
    df_leads_meta_formulario: pd.DataFrame | None = None,
) -> tuple[int, int, int]:
    source_frames = []

    if df_leads_site is not None and not df_leads_site.empty:
        df_site = df_leads_site.copy()
        df_site["_lead_source"] = "leads_site"
        source_frames.append(df_site)

    if company_slug == "nextqs":
        if df_leads_meta_whatsapp is not None and not df_leads_meta_whatsapp.empty:
            df_whatsapp = df_leads_meta_whatsapp.copy()
            df_whatsapp["_lead_source"] = "leads_meta_whatsapp"
            source_frames.append(df_whatsapp)
        if df_leads_meta_formulario is not None and not df_leads_meta_formulario.empty:
            df_formulario = df_leads_meta_formulario.copy()
            df_formulario["_lead_source"] = "leads_meta_formulario"
            source_frames.append(df_formulario)

    combined = concat_non_empty(source_frames)
    if combined.empty:
        return 0, 0, 0

    leads_totais = count_unique_leads(combined)

    meta_mask = combined["_lead_source"].isin(["leads_meta_whatsapp", "leads_meta_formulario"])
    if "origem" in combined.columns:
        meta_mask = meta_mask | combined["origem"].apply(is_meta_origin)
        google_mask = combined["origem"].apply(is_google_ads_origin)
    else:
        google_mask = pd.Series(False, index=combined.index)

    leads_meta = count_unique_leads(combined[meta_mask].copy())
    leads_google_ads = count_unique_leads(combined[google_mask].copy())

    return leads_totais, leads_meta, leads_google_ads


def is_negocio_efetuado(value) -> bool:
    return strip_accents(value) == "negocio efetuado"


def is_negocio_nao_efetuado(value) -> bool:
    return strip_accents(value) == "negocio nao efetuado"


def first_matching_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    normalized_candidates = {strip_accents(c).replace(" ", "_") for c in candidates}
    for col in df.columns:
        normalized_col = strip_accents(col).replace(" ", "_")
        if normalized_col in normalized_candidates:
            return col
    return None


def mode_text(df: pd.DataFrame, col: str | None) -> str:
    if df is None or df.empty or not col or col not in df.columns:
        return "—"
    values = df[col].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "—"
    return values.value_counts().index[0]


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.1f}%".replace(".", ",")


def format_avg_days(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = max(0.0, float(value))
    if abs(value - 1.0) < 0.05:
        return "1 dia"
    return f"{value:.1f} dias".replace(".", ",")


def average_days_between(df: pd.DataFrame, start_col: str, end_col: str) -> float | None:
    if df is None or df.empty or start_col not in df.columns or end_col not in df.columns:
        return None
    start = pd.to_datetime(df[start_col], dayfirst=True, errors="coerce")
    end = pd.to_datetime(df[end_col], dayfirst=True, errors="coerce")
    days = (end - start).dt.total_seconds() / 86400
    days = days.dropna()
    days = days[days >= 0]
    if days.empty:
        return None
    return float(days.mean())


def classify_origin(value, lead_source: str | None = None) -> str:
    if lead_source in {"leads_meta_whatsapp", "leads_meta_formulario"} or is_meta_origin(value):
        return "Meta Ads"

    text = strip_accents(value)
    if is_google_ads_origin(value):
        return "Google Ads"
    if "organic" in text or "organica" in text or "organico" in text or "busca organica" in text:
        return "Busca Orgânica"

    origin = normalize_origin(value)
    return origin if origin else "Origem não identificada"


def get_common_origin(
    company_slug: str,
    df_leads_site: pd.DataFrame,
    df_leads_meta_whatsapp: pd.DataFrame | None,
    df_leads_meta_formulario: pd.DataFrame | None,
) -> str:
    frames = []
    for source, df in [
        ("leads_site", df_leads_site),
        ("leads_meta_whatsapp", df_leads_meta_whatsapp),
        ("leads_meta_formulario", df_leads_meta_formulario),
    ]:
        if df is None or df.empty:
            continue
        if source != "leads_site" and company_slug != "nextqs":
            continue
        item = df.copy()
        item["_lead_source"] = source
        frames.append(item)

    combined = concat_non_empty(frames)
    if combined.empty:
        return "—"

    prepared = prepare_leads_for_reporting(combined)
    prepared["_origin_group"] = prepared.apply(
        lambda row: classify_origin(row.get("origem"), row.get("_lead_source")),
        axis=1,
    )
    unique = prepared.drop_duplicates(subset=["_origin_group", "lead_key"])
    if unique.empty:
        return "—"
    return unique["_origin_group"].value_counts().index[0]


def get_opportunity_metrics(df_opportunities: pd.DataFrame) -> dict[str, str]:
    if df_opportunities is None:
        df_opportunities = pd.DataFrame()

    opp_count = len(df_opportunities) if not df_opportunities.empty else 0
    negocio_count = count_negocios_efetuados_in_opportunities(df_opportunities)
    negocio_rate = (negocio_count / opp_count) if opp_count else None

    segmento_col = first_matching_col(df_opportunities, ["area_atuação", "area_atuacao", "Área de Atuação"])

    return {
        "oportunidades": str(opp_count),
        "segmento": mode_text(df_opportunities, segmento_col),
        "taxa_negocios": format_percent(negocio_rate),
        "tempo_oportunidade": format_avg_days(average_days_between(df_opportunities, "data_lead", "data_oportunidade")),
        "tempo_encerramento": format_avg_days(average_days_between(df_opportunities, "data_oportunidade", "data_encerramento")),
    }


def render_highlight_card(label: str, value: str, color: str = GREEN_COLOR):
    value_html = (
        f"<div style='font-size: clamp(20px, 2.2vw, 32px); font-weight: 800; color: {color}; "
        f"line-height: 1.05; overflow-wrap: anywhere;'>{html.escape(str(value))}</div>"
    )

    st.markdown(
        f"""
        <div style="
            min-height: 96px;
            padding: 10px 12px;
            border-radius: 10px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            display: flex;
            flex-direction: column;
            justify-content: space-between;">
            <div style="font-size: 14px; opacity: 0.85; line-height: 1.2;">{html.escape(str(label))}</div>
            {value_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_highlight_cards(
    company_slug: str,
    df_leads_site: pd.DataFrame,
    df_leads_meta_whatsapp: pd.DataFrame,
    df_leads_meta_formulario: pd.DataFrame,
    df_opportunities_periodo: pd.DataFrame,
) -> list[tuple[str, str]]:
    df_leads_all = build_combined_leads(
        company_slug,
        df_leads_site,
        df_leads_meta_whatsapp,
        df_leads_meta_formulario,
    )
    opp_metrics = get_opportunity_metrics(df_opportunities_periodo)
    leads_count = count_unique_leads(df_leads_all)
    oportunidades_count = int(opp_metrics["oportunidades"])
    taxa_oportunidades = (oportunidades_count / leads_count) if leads_count else None

    return [
        ("Leads", str(leads_count)),
        ("Oportunidades", opp_metrics["oportunidades"]),
        ("Origem principal", get_common_origin(company_slug, df_leads_site, df_leads_meta_whatsapp, df_leads_meta_formulario)),
        ("Segmento principal", opp_metrics["segmento"]),
        ("Taxa de oportunidades", format_percent(taxa_oportunidades)),
        ("Taxa de negócios efetuados", opp_metrics["taxa_negocios"]),
        ("Tempo médio oportunidade", opp_metrics["tempo_oportunidade"]),
        ("Tempo médio encerramento", opp_metrics["tempo_encerramento"]),
    ]


def render_highlights(
    company_slug: str,
    df_leads_site: pd.DataFrame,
    df_leads_meta_whatsapp: pd.DataFrame,
    df_leads_meta_formulario: pd.DataFrame,
    df_opportunities_periodo: pd.DataFrame,
):
    cards = get_highlight_cards(
        company_slug,
        df_leads_site,
        df_leads_meta_whatsapp,
        df_leads_meta_formulario,
        df_opportunities_periodo,
    )

    for start in (0, 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, cards[start:start + 4]):
            with col:
                render_highlight_card(label, value, GREEN_COLOR)
        if start == 0:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


def render_compare_highlight_card(label: str, value_1: str, value_2: str, color_1: str, color_2: str):
    value_1_html = f"<div style='font-size: clamp(18px, 1.7vw, 28px); font-weight: 800; color: {color_1}; line-height: 1.1; overflow-wrap: anywhere;'>{html.escape(str(value_1))}</div>"
    value_2_html = f"<div style='font-size: clamp(18px, 1.7vw, 28px); font-weight: 800; color: {color_2}; line-height: 1.1; overflow-wrap: anywhere;'>{html.escape(str(value_2))}</div>"

    st.markdown(
        f"""
        <div style="
            min-height: 112px;
            padding: 10px 12px;
            border-radius: 10px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);">
            <div style="font-size: 14px; opacity: 0.85; line-height: 1.2; margin-bottom: 8px;">{html.escape(str(label))}</div>
            {value_1_html}
            {value_2_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compare_highlights(
    company_slug: str,
    df_m1_leads: pd.DataFrame,
    df_m1_meta_whatsapp: pd.DataFrame,
    df_m1_meta_formulario: pd.DataFrame,
    df_m1_opportunities: pd.DataFrame,
    df_m2_leads: pd.DataFrame,
    df_m2_meta_whatsapp: pd.DataFrame,
    df_m2_meta_formulario: pd.DataFrame,
    df_m2_opportunities: pd.DataFrame,
):
    cards_1 = get_highlight_cards(company_slug, df_m1_leads, df_m1_meta_whatsapp, df_m1_meta_formulario, df_m1_opportunities)
    cards_2 = get_highlight_cards(company_slug, df_m2_leads, df_m2_meta_whatsapp, df_m2_meta_formulario, df_m2_opportunities)

    for start in (0, 4):
        cols = st.columns(4)
        for idx, col in enumerate(cols, start=start):
            label = cards_1[idx][0]
            with col:
                render_compare_highlight_card(label, cards_1[idx][1], cards_2[idx][1], COMPARE_COLOR_1, COMPARE_COLOR_2)
        if start == 0:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


def render_leads_by_day_chart(df_leads: pd.DataFrame, y_label: str = "Leads"):
    conv_por_dia = count_unique_leads_by(df_leads, ["data"])
    if conv_por_dia.empty:
        st.info("Sem dados para o gráfico.")
        return

    conv_por_dia = conv_por_dia.sort_values("data")
    fig = px.line(conv_por_dia, x="data", y="leads", markers=True, template="plotly_dark")
    fig.update_traces(line=dict(color="#5B6CFF", width=2), marker=dict(size=6, color="#5B6CFF"))
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Data",
        yaxis_title=y_label,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def get_kpis(df_kpi: pd.DataFrame):
    conv_total = len(df_kpi)
    usuarios_unicos = df_kpi["user_id_email"].nunique() if (not df_kpi.empty and "user_id_email" in df_kpi.columns) else 0
    origem_top = df_kpi["origem"].value_counts().idxmax() if (not df_kpi.empty and "origem" in df_kpi.columns) else "-"

    if conv_total > 0 and "dispositivo" in df_kpi.columns:
        dist_dispositivos = df_kpi["dispositivo"].value_counts(normalize=True) * 100
        dispositivo_top = dist_dispositivos.index[0]
        pct_top = float(dist_dispositivos.loc[dispositivo_top])
    else:
        dispositivo_top = "-"
        pct_top = 0.0

    return conv_total, usuarios_unicos, origem_top, dispositivo_top, pct_top


def kpi_value_html(value: str, color: str, size_px: int = 32) -> str:
    return f"<div style='font-size:{size_px}px; font-weight:bold; color:{color}; line-height:1.1'>{value}</div>"


def render_kpi_single(title: str, value: str, color: str):
    st.text(title)
    st.markdown(kpi_value_html(value, color), unsafe_allow_html=True)


def render_kpi_dual(title: str, value_1: str, value_2: str, color_1: str, color_2: str):
    st.text(title)
    st.markdown(
        kpi_value_html(value_1, color_1) + kpi_value_html(value_2, color_2),
        unsafe_allow_html=True,
    )


def build_funnel_figure(title: str, steps: list[tuple[str, int]], base_color: str, min_ratio: float = 0.18) -> go.Figure:
    labels = [s[0] for s in steps]
    values_real = [max(0, int(s[1])) for s in steps]

    maxv = max(values_real) if values_real else 1
    minv = max(1, int(round(maxv * float(min_ratio))))

    scaled = []
    prev_scaled = maxv
    for v in values_real:
        target = max(v, minv)
        sv = min(prev_scaled, target)
        scaled.append(sv)
        prev_scaled = sv

    pct_initial = [(v / maxv) if maxv else 0.0 for v in values_real]
    pct_prev = []
    prev_real = None
    for v in values_real:
        if prev_real in (None, 0):
            pct_prev.append(1.0 if v else 0.0)
        else:
            pct_prev.append(v / prev_real)
        prev_real = v

    text_inside = [
        f"<b style='color:white'>{v}</b><br><b style='color:white'>{p:.1%}</b>"
        for v, p in zip(values_real, pct_initial)
    ]
    customdata = [[v, p_i, p_p] for v, p_i, p_p in zip(values_real, pct_initial, pct_prev)]

    fig = go.Figure()
    fig.add_trace(
        go.Funnel(
            name=title,
            y=labels,
            x=scaled,
            text=text_inside,
            customdata=customdata,
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Quantidade: %{customdata[0]}<br>"
                "% do início: %{customdata[1]:.1%}<br>"
                "% do passo anterior: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
            marker=dict(color=base_color, line=dict(color="rgba(0,0,0,0.28)", width=2)),
            connector=dict(line=dict(color="rgba(0,0,0,0.22)", width=1)),
            opacity=0.95,
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=55, b=10),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14, color="white"),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return fig


# ---------------------------------------------------------------------------
# Funções de contagem para o novo funil: Leads, Oportunidades, Negócios Efetuados
# ---------------------------------------------------------------------------

def get_leads_count_for_funnel(
    company_slug: str,
    dfs: dict,
    df_periodo_leads_site: pd.DataFrame,
    df_periodo_leads_meta_whatsapp: pd.DataFrame | None,
    df_periodo_leads_meta_formulario: pd.DataFrame | None = None,
) -> int:
    """
    NextQS: leads_site + leads_meta_whatsapp + leads_meta_formulario sem duplicações.
    StarLed: apenas leads_site sem duplicações.
    """
    combined = build_combined_leads(
        company_slug,
        df_periodo_leads_site,
        df_periodo_leads_meta_whatsapp,
        df_periodo_leads_meta_formulario,
    )
    return count_unique_leads(combined)


def count_negocios_efetuados_in_opportunities(df_opportunities: pd.DataFrame) -> int:
    if df_opportunities is None or df_opportunities.empty or "status_funil" not in df_opportunities.columns:
        return 0
    return int(df_opportunities["status_funil"].apply(is_negocio_efetuado).sum())


def get_negocios_efetuados_count(df_opp_full: pd.DataFrame, periodo_sel: str, hoje: date, ontem: date) -> int:
    """
    Conta negócios efetuados filtrando por Data Encerramento no período selecionado
    e status_funil == 'Negócio efetuado'.
    """
    if df_opp_full.empty:
        return 0
    if "status_funil" not in df_opp_full.columns:
        return 0

    df_neg = df_opp_full[df_opp_full["status_funil"].apply(is_negocio_efetuado)].copy()

    if df_neg.empty:
        return 0

    if "data_encerramento_parsed" not in df_neg.columns:
        return len(df_neg)

    df_neg = df_neg.dropna(subset=["data_encerramento_parsed"])
    df_neg["data_enc_date"] = df_neg["data_encerramento_parsed"].dt.date

    if periodo_sel == "Hoje":
        return len(df_neg[df_neg["data_enc_date"] == hoje])
    if periodo_sel == "Ontem":
        return len(df_neg[df_neg["data_enc_date"] == ontem])
    if periodo_sel == "Últimos 30 dias":
        start = hoje - timedelta(days=29)
        return len(df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)])
    if periodo_sel == "Este mês":
        start = date(hoje.year, hoje.month, 1)
        return len(df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)])
    if periodo_sel == "Este ano":
        start = date(hoje.year, 1, 1)
        return len(df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)])
    # Personalizado ou outros: usa todos os que estão no df_opp já filtrado
    return len(df_neg)


def get_negocios_efetuados_count_by_month(df_opp_full: pd.DataFrame, ano: int, mes_num: int) -> int:
    """Conta negócios efetuados por mês (usando data_encerramento_parsed)."""
    if df_opp_full.empty or "status_funil" not in df_opp_full.columns:
        return 0
    df_neg = df_opp_full[df_opp_full["status_funil"].apply(is_negocio_efetuado)].copy()
    if df_neg.empty or "data_encerramento_parsed" not in df_neg.columns:
        return len(df_neg)
    df_neg = df_neg.dropna(subset=["data_encerramento_parsed"])
    df_neg = df_neg[
        (df_neg["data_encerramento_parsed"].dt.year == ano)
        & (df_neg["data_encerramento_parsed"].dt.month == mes_num)
    ]
    return len(df_neg)


def render_central_funnel(
    company_slug: str,
    dfs: dict,
    df_leads_site_periodo: pd.DataFrame,
    df_leads_meta_whatsapp_periodo: pd.DataFrame | None,
    df_leads_meta_formulario_periodo: pd.DataFrame | None,
    df_opportunities_periodo: pd.DataFrame,
    df_opp_full: pd.DataFrame,
    periodo_sel: str,
    hoje: date,
    ontem: date,
    title: str = "Funil",
):
    leads_count = get_leads_count_for_funnel(
        company_slug,
        dfs,
        df_leads_site_periodo,
        df_leads_meta_whatsapp_periodo,
        df_leads_meta_formulario_periodo,
    )
    opp_count = len(df_opportunities_periodo) if not df_opportunities_periodo.empty else 0
    negocios_count = count_negocios_efetuados_in_opportunities(df_opportunities_periodo)

    steps = [
        ("Leads", leads_count),
        ("Oportunidades", opp_count),
        ("Negócios Efetuados", negocios_count),
    ]
    st.subheader(title)
    fig = build_funnel_figure("", steps, base_color=FUNNEL_COLOR)
    st.plotly_chart(fig, use_container_width=True)


def render_central_funnel_compare(
    company_slug: str,
    dfs: dict,
    df_leads_site_periodo: pd.DataFrame,
    df_leads_meta_whatsapp_periodo: pd.DataFrame | None,
    df_leads_meta_formulario_periodo: pd.DataFrame | None,
    df_opportunities_periodo: pd.DataFrame,
    df_opp_full: pd.DataFrame,
    ano: int,
    mes_num: int,
    title: str = "Funil",
):
    leads_count = get_leads_count_for_funnel(
        company_slug,
        dfs,
        df_leads_site_periodo,
        df_leads_meta_whatsapp_periodo,
        df_leads_meta_formulario_periodo,
    )
    opp_count = len(df_opportunities_periodo) if not df_opportunities_periodo.empty else 0
    negocios_count = count_negocios_efetuados_in_opportunities(df_opportunities_periodo)

    steps = [
        ("Leads", leads_count),
        ("Oportunidades", opp_count),
        ("Negócios Efetuados", negocios_count),
    ]
    st.subheader(title)
    fig = build_funnel_figure("", steps, base_color=FUNNEL_COLOR)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers de tabelas de origem, campanhas e termos
# ---------------------------------------------------------------------------

def build_unique_lead_key_from_row(row) -> str:
    for col in ["user_id_email", "email", "user_id_cel", "telefone", "user_id"]:
        value = normalize_text(row.get(col))
        if value:
            return value.lower()
    return ""


def prepare_leads_for_reporting(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["origem", "lead_key", "utm_campaign", "utm_term"])

    out = df.copy()
    if "origem" not in out.columns:
        out["origem"] = None
    if "utm_campaign" not in out.columns:
        out["utm_campaign"] = None
    if "utm_term" not in out.columns:
        out["utm_term"] = None

    out["origem"] = out["origem"].apply(normalize_origin)
    out["lead_key"] = out.apply(build_unique_lead_key_from_row, axis=1)

    missing_mask = out["lead_key"].eq("")
    if missing_mask.any():
        out.loc[missing_mask, "lead_key"] = [f"sem_id_{idx}" for idx in out[missing_mask].index]

    return out


def get_opportunity_lead_keys(df: pd.DataFrame) -> set[str]:
    if df is None or df.empty or "user_id" not in df.columns:
        return set()
    return {
        normalize_text(value).lower()
        for value in df["user_id"].dropna()
        if normalize_text(value)
    }


def count_opportunities_by_column(df: pd.DataFrame, source_col: str, output_col: str) -> pd.DataFrame:
    if df is None or df.empty or source_col not in df.columns:
        return pd.DataFrame(columns=[output_col, "count"])

    out = df.copy()
    out[output_col] = out[source_col].apply(normalize_text)
    out = out[out[output_col] != ""].copy()
    if out.empty:
        return pd.DataFrame(columns=[output_col, "count"])

    opportunity_col = "oportunidade" if "oportunidade" in out.columns else None
    if opportunity_col:
        out["_opp_key"] = out[opportunity_col].apply(normalize_text)
        empty_mask = out["_opp_key"].eq("")
        if empty_mask.any():
            out.loc[empty_mask, "_opp_key"] = [f"sem_codigo_{idx}" for idx in out[empty_mask].index]
        counted = out.groupby(output_col)["_opp_key"].nunique()
    else:
        counted = out.groupby(output_col).size()

    return counted.reset_index(name="count")


def count_opportunities_by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "campanha" not in df.columns:
        return pd.DataFrame(columns=["Campanha", "count"])

    out = df.copy()
    out["Campanha"] = out["campanha"].apply(normalize_campaign)
    out = out[out["Campanha"] != "Campanha não identificada"].copy()
    if out.empty:
        return pd.DataFrame(columns=["Campanha", "count"])

    opportunity_col = "oportunidade" if "oportunidade" in out.columns else None
    if opportunity_col:
        out["_opp_key"] = out[opportunity_col].apply(normalize_text)
        empty_mask = out["_opp_key"].eq("")
        if empty_mask.any():
            out.loc[empty_mask, "_opp_key"] = [f"sem_codigo_{idx}" for idx in out[empty_mask].index]
        counted = out.groupby("Campanha")["_opp_key"].nunique()
    else:
        counted = out.groupby("Campanha").size()

    return counted.reset_index(name="count")


def get_negocios_efetuados_df(df_opp_full: pd.DataFrame, periodo_sel: str, hoje: date, ontem: date) -> pd.DataFrame:
    if df_opp_full.empty or "status_funil" not in df_opp_full.columns:
        return pd.DataFrame(columns=df_opp_full.columns if not df_opp_full.empty else [])

    df_neg = df_opp_full[df_opp_full["status_funil"].apply(is_negocio_efetuado)].copy()

    if df_neg.empty:
        return df_neg

    if "data_encerramento_parsed" not in df_neg.columns:
        return df_neg

    df_neg = df_neg.dropna(subset=["data_encerramento_parsed"]).copy()
    df_neg["data_enc_date"] = df_neg["data_encerramento_parsed"].dt.date

    if periodo_sel == "Hoje":
        return df_neg[df_neg["data_enc_date"] == hoje].copy()
    if periodo_sel == "Ontem":
        return df_neg[df_neg["data_enc_date"] == ontem].copy()
    if periodo_sel == "Últimos 30 dias":
        start = hoje - timedelta(days=29)
        return df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)].copy()
    if periodo_sel == "Este mês":
        start = date(hoje.year, hoje.month, 1)
        return df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)].copy()
    if periodo_sel == "Este ano":
        start = date(hoje.year, 1, 1)
        return df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)].copy()
    return df_neg.copy()


def build_origin_table(
    company_slug: str,
    df_leads_site: pd.DataFrame,
    df_leads_meta_whatsapp: pd.DataFrame | None,
    df_leads_meta_formulario: pd.DataFrame | None,
    df_opportunities_periodo: pd.DataFrame,
    df_negocios_periodo: pd.DataFrame,
) -> pd.DataFrame:
    leads_frames = []

    if not df_leads_site.empty:
        leads_frames.append(df_leads_site)

    if company_slug == "nextqs" and df_leads_meta_whatsapp is not None and not df_leads_meta_whatsapp.empty:
        leads_frames.append(df_leads_meta_whatsapp)

    if company_slug == "nextqs" and df_leads_meta_formulario is not None and not df_leads_meta_formulario.empty:
        leads_frames.append(df_leads_meta_formulario)

    if leads_frames:
        df_leads_combined = prepare_leads_for_reporting(pd.concat(leads_frames, ignore_index=True))
        df_leads_counts = (
            df_leads_combined.drop_duplicates(subset=["origem", "lead_key"])
            .groupby("origem")["lead_key"]
            .nunique()
            .reset_index(name="Leads")
            .rename(columns={"origem": "Origem"})
        )
    else:
        df_leads_counts = pd.DataFrame(columns=["Origem", "Leads"])

    df_opp_counts = (
        count_opportunities_by_column(df_opportunities_periodo, "origem", "Origem")
        .rename(columns={"count": "Oportunidades"})
    )
    df_neg_counts = (
        count_opportunities_by_column(df_negocios_periodo, "origem", "Origem")
        .rename(columns={"count": "Negócios"})
    )

    df_out = df_leads_counts.merge(df_opp_counts, on="Origem", how="outer")
    df_out = df_out.merge(df_neg_counts, on="Origem", how="outer")
    if df_out.empty:
        return pd.DataFrame(columns=["Origem", "Leads", "Oportunidades", "Negócios"])

    for col in ["Leads", "Oportunidades", "Negócios"]:
        df_out[col] = df_out[col].fillna(0).astype(int)
    return df_out.sort_values(["Leads", "Oportunidades", "Negócios"], ascending=False).reset_index(drop=True)


def get_nextqs_email_campaign(row) -> str | None:
    medium = normalize_text(row.get("utm_medium")).casefold()
    country = normalize_text(row.get("country")).upper()
    if medium != "email":
        return None
    if country == "BR":
        return "[NextQSBR]-Email"
    if country == "PT":
        return "[NextQSPT]-Email"
    return None


def standardize_nextqs_email_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "utm_medium" not in df.columns or "country" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    if "utm_campaign" not in out.columns:
        out["utm_campaign"] = None

    standardized_campaigns = out.apply(get_nextqs_email_campaign, axis=1)
    mask = standardized_campaigns.notna()
    out.loc[mask, "utm_campaign"] = standardized_campaigns[mask]
    return out


def build_nextqs_email_campaign_map(df_leads_site: pd.DataFrame) -> dict[str, str]:
    if df_leads_site is None or df_leads_site.empty:
        return {}

    out = standardize_nextqs_email_campaigns(df_leads_site)
    out["_email_campaign"] = out.apply(get_nextqs_email_campaign, axis=1)
    out["_lead_key"] = out.apply(build_unique_lead_key_from_row, axis=1)
    out = out[(out["_email_campaign"].notna()) & (out["_lead_key"] != "")].copy()
    if out.empty:
        return {}

    if "data_hora" in out.columns:
        out = out.sort_values("data_hora")
    out = out.drop_duplicates(subset=["_lead_key"], keep="last")
    return dict(zip(out["_lead_key"], out["_email_campaign"]))


def apply_email_campaign_map_to_opportunities(
    df_opportunities: pd.DataFrame,
    email_campaign_map: dict[str, str],
) -> pd.DataFrame:
    if (
        df_opportunities is None
        or df_opportunities.empty
        or not email_campaign_map
        or "user_id" not in df_opportunities.columns
    ):
        return df_opportunities.copy() if df_opportunities is not None else pd.DataFrame()

    out = df_opportunities.copy()
    if "campanha" not in out.columns:
        out["campanha"] = None

    lead_keys = out["user_id"].apply(lambda value: normalize_text(value).lower())
    standardized_campaigns = lead_keys.map(email_campaign_map)
    mask = standardized_campaigns.notna()
    out.loc[mask, "campanha"] = standardized_campaigns[mask]
    return out


def prepare_leads_for_campaign_reporting(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Campanha", "lead_key"])

    out = df.copy()
    if "utm_campaign" not in out.columns:
        out["utm_campaign"] = None
    if "campanha" not in out.columns:
        out["campanha"] = None

    utm_campaign = out["utm_campaign"].apply(normalize_campaign)
    campanha = out["campanha"].apply(normalize_campaign)
    out["Campanha"] = utm_campaign.where(utm_campaign != "Campanha não identificada", campanha)
    out["lead_key"] = out.apply(build_unique_lead_key_from_row, axis=1)

    missing_mask = out["lead_key"].eq("")
    if missing_mask.any():
        out.loc[missing_mask, "lead_key"] = [f"sem_id_{idx}" for idx in out[missing_mask].index]

    return out[["Campanha", "lead_key"]]


def build_campaign_counts_from_opportunities(df: pd.DataFrame, count_col: str) -> pd.DataFrame:
    if df.empty or "campanha" not in df.columns:
        return pd.DataFrame(columns=["Campanha", count_col])

    out = df.copy()
    out["Campanha"] = out["campanha"].apply(normalize_campaign)
    out = out[out["Campanha"] != "Campanha não identificada"].copy()

    if out.empty:
        return pd.DataFrame(columns=["Campanha", count_col])

    return out.groupby("Campanha").size().reset_index(name=count_col)


def build_campaign_table(
    df_leads_filtrado: pd.DataFrame,
    df_opp_full: pd.DataFrame,
    extra_leads_dfs: list[pd.DataFrame] | None = None,
    df_opportunities_periodo: pd.DataFrame | None = None,
    df_negocios_periodo: pd.DataFrame | None = None,
    company_slug: str = "nextqs",
    df_leads_site_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Cria tabela de campanhas com colunas: Campanha, Leads, Oportunidades, Negócios.
    """
    opp_source = df_opportunities_periodo if df_opportunities_periodo is not None else df_opp_full
    neg_source = df_negocios_periodo
    if neg_source is None:
        neg_source = (
            df_opp_full[df_opp_full["status_funil"].apply(is_negocio_efetuado)]
            if (not df_opp_full.empty and "status_funil" in df_opp_full.columns)
            else pd.DataFrame()
        )

    site_leads_for_campaign = df_leads_filtrado
    if company_slug == "nextqs":
        site_leads_for_campaign = standardize_nextqs_email_campaigns(df_leads_filtrado)
        email_campaign_map = build_nextqs_email_campaign_map(
            df_leads_site_lookup if df_leads_site_lookup is not None else df_leads_filtrado
        )
        opp_source = apply_email_campaign_map_to_opportunities(opp_source, email_campaign_map)
        neg_source = apply_email_campaign_map_to_opportunities(neg_source, email_campaign_map)

    lead_frames = [
        df
        for df in [site_leads_for_campaign] + (extra_leads_dfs or [])
        if df is not None and not df.empty
    ]

    if lead_frames:
        df_c_unique = prepare_leads_for_campaign_reporting(pd.concat(lead_frames, ignore_index=True))
        df_c_unique = df_c_unique[df_c_unique["Campanha"] != "Campanha não identificada"].copy()
        df_leads_counts = (
            df_c_unique.drop_duplicates(subset=["Campanha", "lead_key"])
            .groupby("Campanha")["lead_key"]
            .nunique()
            .reset_index(name="Leads")
            if not df_c_unique.empty
            else pd.DataFrame(columns=["Campanha", "Leads"])
        )
    else:
        df_leads_counts = pd.DataFrame(columns=["Campanha", "Leads"])

    df_opp_counts = count_opportunities_by_campaign(opp_source).rename(columns={"count": "Oportunidades"})
    df_neg_counts = count_opportunities_by_campaign(neg_source).rename(columns={"count": "Negócios"})

    df_campaign_out = df_leads_counts.merge(df_opp_counts, on="Campanha", how="outer")
    df_campaign_out = df_campaign_out.merge(df_neg_counts, on="Campanha", how="outer")
    if df_campaign_out.empty:
        return pd.DataFrame(columns=["Campanha", "Leads", "Oportunidades", "Negócios"])

    for col in ["Leads", "Oportunidades", "Negócios"]:
        df_campaign_out[col] = df_campaign_out[col].fillna(0).astype(int)
    df_campaign_out = df_campaign_out[
        (df_campaign_out["Leads"] > 0)
        | (df_campaign_out["Oportunidades"] > 0)
        | (df_campaign_out["Negócios"] > 0)
    ].copy()
    if df_campaign_out.empty:
        return pd.DataFrame(columns=["Campanha", "Leads", "Oportunidades", "Negócios"])
    return (
        df_campaign_out
        .sort_values(["Leads", "Oportunidades", "Negócios"], ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data
def image_as_data_uri(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        return ""
    mime_types = {
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def campaign_badge_html(campaign_name: str) -> str:
    campaign_key = campaign_name.casefold()
    badge_rules = [
        ("[nextqsbr]-metaads", "br", "meta"),
        ("[nextqsbr]-instagrambio", "br", "meta"),
        ("[nextqspt]-metaads", "pt", "meta"),
        ("[nextqsbr]-googleads", "br", "google"),
        ("[nextqspt]-googleads", "pt", "google"),
        ("[nextqsbr]-email", "br", "email"),
        ("[nextqspt]-email", "pt", "email"),
    ]
    for marker, country_code, platform_class in badge_rules:
        if marker in campaign_key:
            return (
                "<span class='campaign-badge'>"
                f"<span class='campaign-flag campaign-flag-{country_code}' "
                f"role='img' aria-label='{country_code.upper()}'></span>"
                f"<span class='campaign-platform campaign-platform-{platform_class}'></span>"
                "</span>"
            )
    return ""


def render_campaign_table(df_campaigns: pd.DataFrame, height: int = 400):
    assets_dir = Path(__file__).resolve().parent / "assets"
    meta_icon = image_as_data_uri(str(assets_dir / "meta-ads.png"))
    google_icon = image_as_data_uri(str(assets_dir / "google-ads.png"))
    email_icon = image_as_data_uri(str(assets_dir / "email.png"))
    br_flag = image_as_data_uri(str(assets_dir / "flag-br.svg"))
    pt_flag = image_as_data_uri(str(assets_dir / "flag-pt.svg"))

    header_html = "".join(
        f"<th class='{'campaign-name-col' if col == 'Campanha' else 'campaign-number-col'}'>{html.escape(str(col))}</th>"
        for col in df_campaigns.columns
    )

    rows_html = []
    for _, row in df_campaigns.iterrows():
        cells = []
        for col in df_campaigns.columns:
            value = row[col]
            if col == "Campanha":
                campaign_name = normalize_text(value)
                badge = campaign_badge_html(campaign_name)
                cells.append(
                    "<td class='campaign-name-col'>"
                    f"{badge}<span class='campaign-name'>{html.escape(campaign_name)}</span>"
                    "</td>"
                )
            else:
                display_value = "" if pd.isna(value) else str(int(value)) if isinstance(value, (int, float)) else str(value)
                cells.append(f"<td class='campaign-number-col'>{html.escape(display_value)}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <style>
        .campaign-table-wrap {{
            max-height: {int(height)}px;
            overflow: auto;
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 6px;
        }}
        .campaign-table {{
            width: 100%;
            border-collapse: collapse;
            color: inherit;
            font-size: 0.875rem;
        }}
        .campaign-table th {{
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 0.65rem 0.75rem;
            background: var(--background-color, #0e1117);
            border-bottom: 1px solid rgba(128, 128, 128, 0.35);
            font-weight: 500;
            text-align: left;
            white-space: nowrap;
        }}
        .campaign-table td {{
            padding: 0.58rem 0.75rem;
            border-bottom: 1px solid rgba(128, 128, 128, 0.22);
            vertical-align: middle;
        }}
        .campaign-table tr:last-child td {{ border-bottom: 0; }}
        .campaign-name-col {{ min-width: 320px; }}
        .campaign-number-col {{ width: 1%; min-width: 110px; text-align: right; }}
        .campaign-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.2em;
            margin-right: 0.4em;
            vertical-align: middle;
            white-space: nowrap;
        }}
        .campaign-flag {{
            display: inline-block;
            width: 1.1em;
            height: 1.1em;
            background-position: center;
            background-repeat: no-repeat;
            background-size: contain;
            flex: 0 0 1.1em;
        }}
        .campaign-flag-br {{ background-image: url('{br_flag}'); }}
        .campaign-flag-pt {{ background-image: url('{pt_flag}'); }}
        .campaign-platform {{
            display: inline-block;
            width: 1.05em;
            height: 1.05em;
            background-position: center;
            background-repeat: no-repeat;
            background-size: contain;
            flex: 0 0 1.05em;
        }}
        .campaign-platform-meta {{ background-image: url('{meta_icon}'); }}
        .campaign-platform-google {{ background-image: url('{google_icon}'); }}
        .campaign-platform-email {{ background-image: url('{email_icon}'); }}
        .campaign-name {{ vertical-align: middle; }}
    </style>
    <div class="campaign-table-wrap">
        <table class="campaign-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def build_terms_table(
    df_leads_filtrado: pd.DataFrame,
    df_opportunities_periodo: pd.DataFrame,
    df_leads_lookup: pd.DataFrame | None = None,
    df_negocios_periodo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Cria tabela de termos de pesquisa com colunas: Palavra-chave, Leads, Oportunidades, Negócios.
    """
    df_t_unique = prepare_leads_for_reporting(df_leads_filtrado)
    if df_t_unique.empty or "utm_term" not in df_t_unique.columns:
        leads_por_term = pd.DataFrame(columns=["Palavra-chave", "Leads"])
    else:
        df_t_unique = df_t_unique[df_t_unique["utm_term"] != "Palavra-chave não identificada"].copy()
        leads_por_term = (
            df_t_unique.drop_duplicates(subset=["utm_term", "lead_key"])
            .groupby("utm_term")["lead_key"]
            .nunique()
            .reset_index(name="Leads")
            .rename(columns={"utm_term": "Palavra-chave"})
            if not df_t_unique.empty
            else pd.DataFrame(columns=["Palavra-chave", "Leads"])
        )

    lookup_source = df_leads_lookup if df_leads_lookup is not None else df_leads_filtrado
    df_lookup = prepare_leads_for_reporting(lookup_source)
    if df_lookup.empty or "utm_term" not in df_lookup.columns:
        df_opp_counts = pd.DataFrame(columns=["Palavra-chave", "Oportunidades"])
        df_neg_counts = pd.DataFrame(columns=["Palavra-chave", "Negócios"])
    else:
        df_lookup = df_lookup[df_lookup["utm_term"] != "Palavra-chave não identificada"].copy()
        df_lookup = df_lookup.drop_duplicates(subset=["lead_key", "utm_term"])

        def _count_by_term(df_opp: pd.DataFrame, count_col: str) -> pd.DataFrame:
            if df_opp is None or df_opp.empty or df_lookup.empty or "user_id" not in df_opp.columns:
                return pd.DataFrame(columns=["Palavra-chave", count_col])

            opp = df_opp.copy()
            opp["lead_key"] = opp["user_id"].apply(lambda value: normalize_text(value).lower())
            opp = opp[opp["lead_key"] != ""].copy()
            if opp.empty:
                return pd.DataFrame(columns=["Palavra-chave", count_col])

            if "oportunidade" in opp.columns:
                opp["_opp_key"] = opp["oportunidade"].apply(normalize_text)
                empty_mask = opp["_opp_key"].eq("")
                if empty_mask.any():
                    opp.loc[empty_mask, "_opp_key"] = [f"sem_codigo_{idx}" for idx in opp[empty_mask].index]
            else:
                opp["_opp_key"] = [f"linha_{idx}" for idx in opp.index]

            joined = opp[["lead_key", "_opp_key"]].merge(
                df_lookup[["lead_key", "utm_term"]],
                on="lead_key",
                how="inner",
            )
            if joined.empty:
                return pd.DataFrame(columns=["Palavra-chave", count_col])

            return (
                joined.drop_duplicates(subset=["utm_term", "_opp_key"])
                .groupby("utm_term")["_opp_key"]
                .nunique()
                .reset_index(name=count_col)
                .rename(columns={"utm_term": "Palavra-chave"})
            )

        df_opp_counts = _count_by_term(df_opportunities_periodo, "Oportunidades")
        if df_negocios_periodo is None:
            df_negocios_periodo = (
                df_opportunities_periodo[df_opportunities_periodo["status_funil"].apply(is_negocio_efetuado)]
                if (
                    df_opportunities_periodo is not None
                    and not df_opportunities_periodo.empty
                    and "status_funil" in df_opportunities_periodo.columns
                )
                else pd.DataFrame()
            )
        df_neg_counts = _count_by_term(df_negocios_periodo, "Negócios")

    df_out = leads_por_term.merge(df_opp_counts, on="Palavra-chave", how="outer")
    df_out = df_out.merge(df_neg_counts, on="Palavra-chave", how="outer")
    if df_out.empty:
        return pd.DataFrame(columns=["Palavra-chave", "Leads", "Oportunidades", "Negócios"])

    for col in ["Leads", "Oportunidades", "Negócios"]:
        df_out[col] = df_out[col].fillna(0).astype(int)
    df_out = df_out[
        (df_out["Leads"] > 0)
        | (df_out["Oportunidades"] > 0)
        | (df_out["Negócios"] > 0)
    ].copy()
    if df_out.empty:
        return pd.DataFrame(columns=["Palavra-chave", "Leads", "Oportunidades", "Negócios"])
    return df_out.sort_values(["Leads", "Oportunidades", "Negócios"], ascending=False).reset_index(drop=True)


def normalize_conversion_page(value) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    if re.match(r"^https?://", text, flags=re.IGNORECASE):
        parsed = urlparse(text)
        path = parsed.path
    elif text.startswith("www.") or text.startswith("nextqs.com") or text.startswith("starled.com"):
        parsed = urlparse(f"https://{text}")
        path = parsed.path
    else:
        path = text.split("?", 1)[0]

    page = unquote(path).strip("/")
    return "Home" if not page else page


def build_pages_conversion_table(df_leads_site: pd.DataFrame) -> pd.DataFrame:
    if df_leads_site is None or df_leads_site.empty or "page_conversao" not in df_leads_site.columns:
        return pd.DataFrame(columns=["Páginas", "Conversões"])

    out = df_leads_site.copy()
    out["Páginas"] = out["page_conversao"].apply(normalize_conversion_page)
    out = out[out["Páginas"] != ""].copy()
    if out.empty:
        return pd.DataFrame(columns=["Páginas", "Conversões"])

    return (
        out.groupby("Páginas")
        .size()
        .reset_index(name="Conversões")
        .sort_values(["Conversões", "Páginas"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_consultor_table(df_opportunities: pd.DataFrame) -> pd.DataFrame:
    columns = ["Nome do consultor", "Oportunidades", "Negócios não efetuados", "Negócios efetuados"]
    if df_opportunities is None or df_opportunities.empty:
        return pd.DataFrame(columns=columns)

    consultor_col = first_matching_col(df_opportunities, ["consultor", "Nome Colaborador Responsável"])
    oportunidade_col = first_matching_col(df_opportunities, ["oportunidade", "Cod. Oportunidade"])
    status_col = first_matching_col(df_opportunities, ["status_funil", "Funil de Venda"])
    if not consultor_col:
        return pd.DataFrame(columns=columns)

    out = df_opportunities.copy()
    out["Nome do consultor"] = out[consultor_col].apply(normalize_text)
    out = out[out["Nome do consultor"] != ""].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)

    if oportunidade_col:
        out["_opp_key"] = out[oportunidade_col].apply(normalize_text)
        empty_mask = out["_opp_key"].eq("")
        if empty_mask.any():
            out.loc[empty_mask, "_opp_key"] = [f"sem_codigo_{idx}" for idx in out[empty_mask].index]
    else:
        out["_opp_key"] = [f"linha_{idx}" for idx in out.index]

    if status_col:
        out["_negocio_nao_efetuado"] = out[status_col].apply(is_negocio_nao_efetuado).astype(int)
        out["_negocio_efetuado"] = out[status_col].apply(is_negocio_efetuado).astype(int)
    else:
        out["_negocio_nao_efetuado"] = 0
        out["_negocio_efetuado"] = 0

    return (
        out.groupby("Nome do consultor")
        .agg(
            Oportunidades=("_opp_key", "nunique"),
            **{
                "Negócios não efetuados": ("_negocio_nao_efetuado", "sum"),
                "Negócios efetuados": ("_negocio_efetuado", "sum"),
            },
        )
        .reset_index()
        .sort_values(["Oportunidades", "Negócios efetuados", "Nome do consultor"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Render modes
# ---------------------------------------------------------------------------

def normalize_opportunity_key(value) -> str:
    text = normalize_text(value).lstrip("#").strip()
    if not text:
        return ""
    numeric_match = re.fullmatch(r"(\d+)(?:\.0+)?", text)
    return numeric_match.group(1) if numeric_match else text.casefold()


def normalize_identity_value(value) -> str:
    text = normalize_text(value).strip().casefold()
    invalid_values = {
        "",
        "false",
        "none",
        "null",
        "undefined",
        "nan",
        "ip nao identificado",
        "dispositivo nao identificado",
    }
    return "" if strip_accents(text) in invalid_values else text


def identity_values_from_rows(df: pd.DataFrame, columns: list[str]) -> set[str]:
    values = set()
    if df is None or df.empty:
        return values
    for col in columns:
        if col not in df.columns:
            continue
        values.update(
            normalized
            for normalized in df[col].apply(normalize_identity_value)
            if normalized
        )
    return values


def identity_match_mask(df: pd.DataFrame, columns: list[str], values: set[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if df.empty or not values:
        return mask
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].apply(normalize_identity_value).isin(values)
    return mask


def format_opportunity_summary_date(value) -> str:
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    return parsed.strftime("%d/%m/%Y") if not pd.isna(parsed) else "-"


def build_opportunity_summary(company_slug: str, opportunity_query: str, dfs: dict) -> dict:
    opportunity_key = normalize_opportunity_key(opportunity_query)
    df_opportunities = dfs.get("oportunidades", pd.DataFrame())
    if not opportunity_key or df_opportunities.empty or "oportunidade" not in df_opportunities.columns:
        return {"found": False}

    opportunity_mask = df_opportunities["oportunidade"].apply(normalize_opportunity_key).eq(opportunity_key)
    matched_opportunities = df_opportunities[opportunity_mask].copy()
    if matched_opportunities.empty:
        return {"found": False}

    opportunity = matched_opportunities.sort_values("data_hora").iloc[-1]
    user_ids = identity_values_from_rows(matched_opportunities, ["user_id"])

    df_site = dfs.get("leads_site", pd.DataFrame())
    df_meta_form = dfs.get("leads_meta_formulario", pd.DataFrame())
    df_meta_whatsapp = dfs.get("leads_meta_whatsapp", pd.DataFrame())

    site_mask = identity_match_mask(df_site, ["user_id_email"], user_ids)
    meta_form_mask = identity_match_mask(df_meta_form, ["user_id_email"], user_ids)
    meta_whatsapp_mask = identity_match_mask(df_meta_whatsapp, ["user_id_cel"], user_ids)

    matched_site = df_site[site_mask].copy()
    matched_meta_form = df_meta_form[meta_form_mask].copy()
    matched_meta_whatsapp = df_meta_whatsapp[meta_whatsapp_mask].copy()

    browser_ids = identity_values_from_rows(matched_site, ["nextqs_anon_id", "_ga"])
    first_origin = ""
    df_pageview = load_sheet(company_slug, "pageview")
    if not df_pageview.empty:
        pageview_mask = identity_match_mask(df_pageview, ["nextqs_anon_id", "_ga"], browser_ids)
        matched_pageviews = df_pageview[pageview_mask].sort_values("data_hora")
        if not matched_pageviews.empty and "primeira_origem" in matched_pageviews.columns:
            first_origin = normalize_text(matched_pageviews.iloc[0].get("primeira_origem"))

    if not first_origin and not matched_site.empty and "primeira_origem" in matched_site.columns:
        first_origin = normalize_text(matched_site.sort_values("data_hora").iloc[0].get("primeira_origem"))

    df_downloads = load_sheet(company_slug, "downloads")
    download_mask = identity_match_mask(df_downloads, ["user_id_email"], user_ids)
    download_mask = download_mask | identity_match_mask(df_downloads, ["user_id_stape"], browser_ids)

    df_newsletter = load_sheet(company_slug, "newsletter")
    newsletter_mask = identity_match_mask(df_newsletter, ["user_id_email"], user_ids)
    newsletter_mask = newsletter_mask | identity_match_mask(df_newsletter, ["user_id", "_ga"], browser_ids)

    conversions = []
    conversion_sources = [
        (matched_site, "Site"),
        (matched_meta_form, "Formulário da Meta"),
        (matched_meta_whatsapp, "WhatsApp"),
    ]
    for conversion_df, source_label in conversion_sources:
        if conversion_df.empty:
            continue
        for event_date in conversion_df["data_hora"].dropna().sort_values():
            conversions.append((event_date, f"{event_date.strftime('%d/%m/%Y')} no {source_label}"))

    conversion_labels = []
    seen_conversions = set()
    for _, label in sorted(conversions, key=lambda item: item[0]):
        if label not in seen_conversions:
            seen_conversions.add(label)
            conversion_labels.append(label)

    return {
        "found": True,
        "opportunity_key": opportunity_key,
        "first_origin": first_origin or "-",
        "downloaded": "Sim" if bool(download_mask.any()) else "Não",
        "newsletter": "Sim" if bool(newsletter_mask.any()) else "Não",
        "converted": conversion_labels or ["-"],
        "opportunity_date": format_opportunity_summary_date(opportunity.get("data_oportunidade")),
        "consultant": normalize_text(opportunity.get("consultor")) or "-",
        "status": normalize_text(opportunity.get("status_funil")) or "-",
    }


def render_opportunity_summary_sidebar(summary: dict):
    if not summary.get("found"):
        st.sidebar.warning("Oportunidade não encontrada.")
        return

    converted_html = "<br>".join(html.escape(value) for value in summary["converted"])
    rows = [
        ("Primeiro acesso ao site", html.escape(summary["first_origin"])),
        ("Baixou material", summary["downloaded"]),
        ("Assinou Newsletter", summary["newsletter"]),
        ("Converteu", converted_html),
        ("Oportunidade", summary["opportunity_date"]),
        ("Consultor", html.escape(summary["consultant"])),
        ("Status atual", html.escape(summary["status"])),
    ]
    rows_html = "".join(
        f"""
        <div class="opportunity-summary-row">
            <div class="opportunity-summary-label">{label}</div>
            <div class="opportunity-summary-value">{value}</div>
        </div>
        """
        for label, value in rows
    )
    st.sidebar.markdown(
        f"""
        <style>
            .opportunity-summary {{
                margin-top: 0.7rem;
                padding-top: 0.7rem;
                border-top: 1px solid rgba(128, 128, 128, 0.3);
            }}
            .opportunity-summary-title {{
                margin-bottom: 0.55rem;
                font-size: 0.9rem;
                font-weight: 650;
            }}
            .opportunity-summary-row {{
                margin-bottom: 0.65rem;
            }}
            .opportunity-summary-label {{
                color: rgba(170, 170, 170, 0.95);
                font-size: 0.72rem;
                line-height: 1.25;
            }}
            .opportunity-summary-value {{
                margin-top: 0.08rem;
                font-size: 0.84rem;
                font-weight: 550;
                line-height: 1.4;
                overflow-wrap: anywhere;
            }}
        </style>
        <div class="opportunity-summary">
            <div class="opportunity-summary-title">
                Oportunidade #{html.escape(summary["opportunity_key"])}
            </div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_meta_ads_number(value) -> float:
    text = normalize_text(value)
    if not text:
        return 0.0
    text = (
        text.replace("R$", "")
        .replace("%", "")
        .replace("\xa0", " ")
        .strip()
    )
    text = re.sub(r"[^\d,.\-]", "", text)
    if not text:
        return 0.0
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def format_brl(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "R$ 0,00"
    text = f"R$ {float(value):,.2f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_int_br(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(round(float(value))):,}".replace(",", ".")


def format_percent_br(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%".replace(".", ",")


def get_meta_ads_source_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    normalized_candidates = {
        strip_accents(candidate).replace(" ", "_").replace("-", "_")
        for candidate in candidates
    }
    for col in df.columns:
        normalized_col = strip_accents(col).replace(" ", "_").replace("-", "_")
        if normalized_col in normalized_candidates:
            return col
    return None


def prepare_meta_ads_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    source_cols = {
        "mes_ano": ["mês_ano", "mes_ano", "mês ano", "mes ano", "month"],
        "plataforma": ["plataforma", "platform"],
        "destino": ["destino", "destination"],
        "objetivo": ["objetivo", "objective"],
        "criativo_anuncio": ["criativo_anúncio", "criativo_anuncio", "criativo anúncio", "criativo anuncio", "anuncio", "ad"],
        "valor_usado": ["valor_usado", "valor usado", "valor utilizado", "gasto", "investimento", "amount spent"],
        "alcance": ["alcance", "reach"],
        "impressoes": ["impressões", "impressoes", "impressions"],
        "resultados": ["resultados", "results"],
        "custo_por_resultado": ["custo por resultados", "custo_por_resultados", "custo por resultado", "cost per result"],
        "cliques": ["cliques", "clicks"],
        "ctr": ["ctr"],
        "cpm": ["cpm"],
        "visitas_perfil": ["visitas ao perfil do instagram", "visitas_perfil", "profile visits"],
    }

    out = pd.DataFrame(index=df.index)
    for canonical, candidates in source_cols.items():
        source_col = get_meta_ads_source_col(df, candidates)
        out[canonical] = df[source_col] if source_col else None

    month_text = out["mes_ano"].fillna("").astype(str).str.strip()
    out["data_hora"] = pd.to_datetime(month_text + "-01", format="%Y-%m-%d", errors="coerce")
    missing_dates = out["data_hora"].isna()
    if missing_dates.any():
        out.loc[missing_dates, "data_hora"] = pd.to_datetime(
            month_text[missing_dates],
            dayfirst=True,
            errors="coerce",
        )

    out = out.dropna(subset=["data_hora"]).copy()
    if out.empty:
        return out

    text_cols = ["plataforma", "destino", "objetivo", "criativo_anuncio"]
    for col in text_cols:
        out[col] = out[col].apply(normalize_empty).fillna("Não informado")

    numeric_cols = [
        "valor_usado",
        "alcance",
        "impressoes",
        "resultados",
        "custo_por_resultado",
        "cliques",
        "ctr",
        "cpm",
        "visitas_perfil",
    ]
    for col in numeric_cols:
        out[col] = out[col].apply(parse_meta_ads_number)

    out["data"] = out["data_hora"].dt.date
    out["ano"] = out["data_hora"].dt.year.astype(int)
    out["mes"] = out["data_hora"].dt.month.astype(int)
    out["mes_label"] = out["mes"].map(MESES_LABEL) + "/" + out["ano"].astype(str)
    out["mes_ano"] = out["data_hora"].dt.strftime("%Y-%m")
    return out.sort_values("data_hora")


def filter_meta_ads_by_selection(df_meta: pd.DataFrame, ano: int, mes_label: str) -> pd.DataFrame:
    if df_meta.empty:
        return df_meta.copy()
    out = df_meta[df_meta["ano"] == int(ano)].copy()
    if mes_label != "Todo o ano":
        out = out[out["mes"] == month_label_to_num(mes_label)].copy()
    return out


def render_meta_ads_filters(df_meta: pd.DataFrame) -> tuple[pd.DataFrame, int | None, str]:
    st.sidebar.markdown("<h2 style='margin-bottom: 0.25rem;'>Filtros</h2>", unsafe_allow_html=True)
    if df_meta.empty or "mes_label" not in df_meta.columns:
        return df_meta.copy(), None, "Todo o ano"

    meses_df = (
        df_meta[["data_hora", "mes_label"]]
        .drop_duplicates()
        .sort_values("data_hora")
    )
    opcoes_meses = ["Todo o ano"] + meses_df["mes_label"].tolist()
    if st.session_state.get("meta_ads_mes_label") not in opcoes_meses:
        st.session_state["meta_ads_mes_label"] = "Todo o ano"
    mes_label = st.sidebar.selectbox("Mês", options=opcoes_meses, key="meta_ads_mes_label")
    if mes_label == "Todo o ano":
        return df_meta.copy(), None, mes_label
    return df_meta[df_meta["mes_label"] == mes_label].copy(), None, mes_label


def aggregate_meta_ads(df_meta: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df_meta.empty:
        return pd.DataFrame(columns=group_cols)
    grouped = (
        df_meta.groupby(group_cols, dropna=False)
        .agg(
            valor_usado=("valor_usado", "sum"),
            alcance=("alcance", "sum"),
            impressoes=("impressoes", "sum"),
            resultados=("resultados", "sum"),
            cliques=("cliques", "sum"),
            visitas_perfil=("visitas_perfil", "sum"),
        )
        .reset_index()
    )
    grouped["ctr_calc"] = grouped.apply(
        lambda row: (row["cliques"] / row["impressoes"]) if row["impressoes"] else None,
        axis=1,
    )
    grouped["cpm_calc"] = grouped.apply(
        lambda row: (row["valor_usado"] / row["impressoes"] * 1000) if row["impressoes"] else None,
        axis=1,
    )
    grouped["custo_por_resultado_calc"] = grouped.apply(
        lambda row: (row["valor_usado"] / row["resultados"]) if row["resultados"] else None,
        axis=1,
    )
    return grouped


def render_meta_ads_dashboard(df_meta: pd.DataFrame, ano_sel: int | None, mes_label: str):
    st.title("Meta ADS NextQS")

    if df_meta.empty:
        st.warning("Nenhum dado encontrado na aba 'meta_campanhas' para o filtro selecionado.")
        return

    periodo = mes_label if mes_label != "Todo o ano" else "Todo o ano"
    st.markdown(
        f"""
        <div style="
            margin: 6px 0 18px 0;
            font-size: 14px;
            font-weight: 700;
            opacity: 0.82;">
            {html.escape(periodo)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    investimento = float(df_meta["valor_usado"].sum())
    alcance = float(df_meta["alcance"].sum())
    impressoes = float(df_meta["impressoes"].sum())
    resultados = float(df_meta["resultados"].sum())
    cliques = float(df_meta["cliques"].sum())
    visitas_perfil = float(df_meta["visitas_perfil"].sum())
    ctr = (cliques / impressoes) if impressoes else None
    cpm = (investimento / impressoes * 1000) if impressoes else None
    custo_resultado = (investimento / resultados) if resultados else None

    cols_top = st.columns(4)
    with cols_top[0]:
        render_highlight_card("Investimento", format_brl(investimento))
    with cols_top[1]:
        render_highlight_card("Resultados", format_int_br(resultados))
    with cols_top[2]:
        render_highlight_card("Custo por resultado", format_brl(custo_resultado))
    with cols_top[3]:
        render_highlight_card("CTR", format_percent_br(ctr))

    cols_bottom = st.columns(4)
    with cols_bottom[0]:
        render_highlight_card("Alcance", format_int_br(alcance))
    with cols_bottom[1]:
        render_highlight_card("Impressões", format_int_br(impressoes))
    with cols_bottom[2]:
        render_highlight_card("Cliques", format_int_br(cliques))
    with cols_bottom[3]:
        render_highlight_card("Visitas ao perfil", format_int_br(visitas_perfil))

    st.markdown("---")

    monthly = aggregate_meta_ads(df_meta, ["ano", "mes", "mes_label"]).sort_values(["ano", "mes"])
    destino = aggregate_meta_ads(df_meta, ["destino"]).sort_values("valor_usado", ascending=False)
    objetivo = aggregate_meta_ads(df_meta, ["objetivo"]).sort_values("resultados", ascending=False)

    if mes_label == "Todo o ano" and len(monthly) > 1:
        st.subheader("Investimento e resultados por mês")
        fig_month = go.Figure()
        fig_month.add_trace(
            go.Bar(
                x=monthly["mes_label"],
                y=monthly["valor_usado"],
                name="Investimento",
                marker_color=COMPARE_COLOR_1,
            )
        )
        fig_month.add_trace(
            go.Scatter(
                x=monthly["mes_label"],
                y=monthly["resultados"],
                name="Resultados",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color=COMPARE_COLOR_2, width=3),
            )
        )
        fig_month.update_layout(
            xaxis_title="Mês",
            yaxis_title="Investimento",
            yaxis2=dict(title="Resultados", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_month, use_container_width=True)

    col_chart_1, col_chart_2 = st.columns(2)
    with col_chart_1:
        st.subheader("Resultados por destino")
        if destino.empty:
            st.info("Nenhum destino encontrado no período.")
        else:
            fig_destino = px.bar(destino, x="destino", y="resultados", color="destino")
            fig_destino.update_layout(xaxis_title="Destino", yaxis_title="Resultados", showlegend=False)
            st.plotly_chart(fig_destino, use_container_width=True)

    with col_chart_2:
        st.subheader("Investimento por objetivo")
        if objetivo.empty:
            st.info("Nenhum objetivo encontrado no período.")
        else:
            fig_objetivo = px.bar(objetivo, x="objetivo", y="valor_usado", color="objetivo")
            fig_objetivo.update_layout(xaxis_title="Objetivo", yaxis_title="Investimento", showlegend=False)
            st.plotly_chart(fig_objetivo, use_container_width=True)

    st.markdown("---")
    st.subheader("Criativos por resultado e custo")
    creative = aggregate_meta_ads(df_meta, ["criativo_anuncio", "destino", "objetivo"])
    creative = creative[creative["valor_usado"] > 0].copy()
    if creative.empty:
        st.info("Nenhum criativo encontrado no período.")
    else:
        creative["criativo_curto"] = creative["criativo_anuncio"].apply(
            lambda value: unquote(urlparse(normalize_text(value)).path.rstrip("/").split("/")[-1])
            if normalize_text(value).startswith("http")
            else normalize_text(value)
        )
        fig_creative = px.scatter(
            creative,
            x="custo_por_resultado_calc",
            y="resultados",
            size="valor_usado",
            color="destino",
            hover_name="criativo_curto",
            hover_data={
                "objetivo": True,
                "valor_usado": ":.2f",
                "cliques": ":.0f",
                "impressoes": ":.0f",
                "custo_por_resultado_calc": ":.2f",
            },
        )
        fig_creative.update_layout(
            xaxis_title="Custo por resultado",
            yaxis_title="Resultados",
        )
        st.plotly_chart(fig_creative, use_container_width=True)

    st.markdown("### Informações por destino")
    destino_table = destino.copy()
    if destino_table.empty:
        st.info("Nenhuma informação por destino encontrada.")
    else:
        destino_table = destino_table.rename(columns={"destino": "Destino"})
        destino_table["Investimento"] = destino_table["valor_usado"].apply(format_brl)
        destino_table["Alcance"] = destino_table["alcance"].apply(format_int_br)
        destino_table["Impressões"] = destino_table["impressoes"].apply(format_int_br)
        destino_table["Resultados"] = destino_table["resultados"].apply(format_int_br)
        destino_table["Custo por resultado"] = destino_table["custo_por_resultado_calc"].apply(format_brl)
        destino_table["Cliques"] = destino_table["cliques"].apply(format_int_br)
        destino_table["CTR"] = destino_table["ctr_calc"].apply(format_percent_br)
        destino_table["CPM"] = destino_table["cpm_calc"].apply(format_brl)
        destino_table["Visitas ao perfil"] = destino_table["visitas_perfil"].apply(format_int_br)
        st.dataframe(
            destino_table[
                [
                    "Destino",
                    "Investimento",
                    "Alcance",
                    "Impressões",
                    "Resultados",
                    "Custo por resultado",
                    "Cliques",
                    "CTR",
                    "CPM",
                    "Visitas ao perfil",
                ]
            ],
            use_container_width=True,
            height=260,
        )

    st.markdown("### Informações por criativo")
    creative_table = creative.sort_values(["resultados", "valor_usado"], ascending=False).copy()
    if creative_table.empty:
        st.info("Nenhuma informação por criativo encontrada.")
    else:
        creative_table["Investimento"] = creative_table["valor_usado"].apply(format_brl)
        creative_table["Resultados"] = creative_table["resultados"].apply(format_int_br)
        creative_table["Custo por resultado"] = creative_table["custo_por_resultado_calc"].apply(format_brl)
        creative_table["Cliques"] = creative_table["cliques"].apply(format_int_br)
        creative_table["CTR"] = creative_table["ctr_calc"].apply(format_percent_br)
        creative_table["CPM"] = creative_table["cpm_calc"].apply(format_brl)
        st.dataframe(
            creative_table.rename(
                columns={
                    "criativo_anuncio": "Criativo",
                    "destino": "Destino",
                    "objetivo": "Objetivo",
                }
            )[
                [
                    "Criativo",
                    "Destino",
                    "Objetivo",
                    "Investimento",
                    "Resultados",
                    "Custo por resultado",
                    "Cliques",
                    "CTR",
                    "CPM",
                ]
            ],
            use_container_width=True,
            height=420,
        )


def render_normal_mode(
    company_slug: str,
    dfs: dict,
    df_periodo_leads,
    df_periodo_opportunities,
    df_opp_full,
    eventos_sel,
    origens_sel,
    dispositivos_sel,
    periodo_sel,
    hoje,
    ontem,
):
    df_filtrado = apply_extra_filters_leads(df_periodo_leads, eventos_sel, origens_sel, dispositivos_sel)
    df_leads_meta_whatsapp_periodo = get_effective_period_filtered_df(dfs.get("leads_meta_whatsapp", pd.DataFrame()), periodo_sel, hoje, ontem)
    df_leads_meta_whatsapp_filtrado = apply_common_filters(df_leads_meta_whatsapp_periodo, origens_sel, dispositivos_sel)
    df_leads_meta_formulario_periodo = get_effective_period_filtered_df(dfs.get("leads_meta_formulario", pd.DataFrame()), periodo_sel, hoje, ontem)
    df_leads_meta_formulario_filtrado = apply_common_filters(df_leads_meta_formulario_periodo, origens_sel, dispositivos_sel)
    df_leads_all_filtrado = build_combined_leads(
        company_slug,
        df_filtrado,
        df_leads_meta_whatsapp_filtrado,
        df_leads_meta_formulario_filtrado,
    )
    df_leads_with_time_filtrado = concat_non_empty(
        [df_filtrado, df_leads_meta_formulario_filtrado if company_slug == "nextqs" else pd.DataFrame()]
    )

    if df_filtrado.empty and (
        company_slug != "nextqs"
        or (df_leads_meta_whatsapp_filtrado.empty and df_leads_meta_formulario_filtrado.empty)
    ):
        st.warning("Nenhum Lead encontrado para o período selecionado (após filtros).")
        st.stop()

    render_highlights(
        company_slug,
        df_filtrado,
        df_leads_meta_whatsapp_filtrado,
        df_leads_meta_formulario_filtrado,
        df_periodo_opportunities,
    )
    render_period_label(periodo_sel, hoje, ontem)

    st.markdown("---")

    if periodo_sel in ["Hoje", "Ontem"]:
        st.subheader("Leads por Hora")
        conv_por_hora = count_unique_leads_by(df_leads_with_time_filtrado, ["hora"]).sort_values("hora")
        fig_hora = px.bar(conv_por_hora, x="hora", y="leads")
        fig_hora.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads", xaxis=dict(dtick=1))
        st.plotly_chart(fig_hora, use_container_width=True)
    else:
        st.subheader("Leads por Dia")
        render_leads_by_day_chart(df_leads_all_filtrado)

    st.markdown("---")

    render_central_funnel(
        company_slug,
        dfs,
        df_filtrado,
        df_leads_meta_whatsapp_filtrado if not df_leads_meta_whatsapp_filtrado.empty else None,
        df_leads_meta_formulario_filtrado if not df_leads_meta_formulario_filtrado.empty else None,
        df_periodo_opportunities,
        df_opp_full,
        periodo_sel,
        hoje,
        ontem,
    )
    st.markdown("---")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Leads por Origem")
        conv_origem = count_unique_leads_by(df_leads_all_filtrado, ["origem"]).sort_values("leads", ascending=False)
        fig_origem = px.bar(conv_origem, x="origem", y="leads")
        fig_origem.update_layout(xaxis_title="Origem", yaxis_title="Leads")
        st.plotly_chart(fig_origem, use_container_width=True)

    with col_g2:
        st.subheader("Formulários do Site")
        conv_evento = count_unique_leads_by(df_leads_all_filtrado, ["evento"])
        conv_evento["evento_legenda"] = conv_evento["evento"].apply(label_evento)
        fig_evento = px.bar(conv_evento, x="evento_legenda", y="leads")
        fig_evento.update_layout(xaxis_title="Evento", yaxis_title="Leads")
        st.plotly_chart(fig_evento, use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.subheader("Leads por Dispositivo")
        conv_disp = count_unique_leads_by(df_filtrado, ["dispositivo"])
        fig_disp = px.bar(conv_disp, x="dispositivo", y="leads")
        fig_disp.update_layout(xaxis_title="Dispositivo", yaxis_title="Leads")
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário dos Leads")
        conv_hora = count_unique_leads_by(df_leads_with_time_filtrado, ["hora"]).sort_values("hora")
        fig_hora2 = px.bar(conv_hora, x="hora", y="leads")
        fig_hora2.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads")
        st.plotly_chart(fig_hora2, use_container_width=True)

    df_negocios_periodo = (
        df_periodo_opportunities[df_periodo_opportunities["status_funil"].apply(is_negocio_efetuado)].copy()
        if (not df_periodo_opportunities.empty and "status_funil" in df_periodo_opportunities.columns)
        else pd.DataFrame()
    )
    df_origem_table = build_origin_table(
        company_slug,
        df_filtrado,
        df_leads_meta_whatsapp_filtrado if not df_leads_meta_whatsapp_filtrado.empty else None,
        df_leads_meta_formulario_filtrado if not df_leads_meta_formulario_filtrado.empty else None,
        df_periodo_opportunities,
        df_negocios_periodo,
    )

    st.markdown("### Informações por Origem")
    if df_origem_table.empty:
        st.info("Nenhuma origem encontrada no período filtrado.")
    else:
        st.dataframe(df_origem_table, use_container_width=True, height=320)

    st.markdown("### Informações por Campanhas")
    extra_campaign_leads = []
    if not df_leads_meta_whatsapp_filtrado.empty:
        extra_campaign_leads.append(df_leads_meta_whatsapp_filtrado)
    if not df_leads_meta_formulario_filtrado.empty:
        extra_campaign_leads.append(df_leads_meta_formulario_filtrado)

    df_camp_table = build_campaign_table(
        df_filtrado,
        df_opp_full,
        extra_leads_dfs=extra_campaign_leads,
        df_opportunities_periodo=df_periodo_opportunities,
        df_negocios_periodo=df_negocios_periodo,
        company_slug=company_slug,
        df_leads_site_lookup=dfs.get("leads_site", pd.DataFrame()),
    )
    if df_camp_table.empty:
        st.info("Nenhuma campanha válida encontrada no período filtrado.")
    else:
        render_campaign_table(df_camp_table, height=400)

    st.markdown("### Informações por Termos de Pesquisa")
    df_leads_site_lookup = apply_extra_filters_leads(dfs.get("leads_site", pd.DataFrame()), eventos_sel, origens_sel, dispositivos_sel)
    df_leads_meta_whatsapp_lookup = apply_common_filters(dfs.get("leads_meta_whatsapp", pd.DataFrame()), origens_sel, dispositivos_sel)
    df_leads_meta_formulario_lookup = apply_common_filters(dfs.get("leads_meta_formulario", pd.DataFrame()), origens_sel, dispositivos_sel)
    df_leads_all_lookup = build_combined_leads(
        company_slug,
        df_leads_site_lookup,
        df_leads_meta_whatsapp_lookup,
        df_leads_meta_formulario_lookup,
    )
    df_terms_table = build_terms_table(
        df_leads_all_filtrado,
        df_periodo_opportunities,
        df_leads_lookup=df_leads_all_lookup,
        df_negocios_periodo=df_negocios_periodo,
    )
    if df_terms_table.empty:
        st.info("Nenhuma palavra-chave válida encontrada no período filtrado.")
    else:
        st.dataframe(df_terms_table, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("Páginas do site com mais conversões")
    df_pages_table = build_pages_conversion_table(df_filtrado)
    if df_pages_table.empty:
        st.info("Nenhuma página de conversão encontrada no período filtrado.")
    else:
        st.dataframe(df_pages_table, use_container_width=True, height=360)

    st.subheader("Informações por consultor")
    df_consultor_table = build_consultor_table(df_periodo_opportunities)
    if df_consultor_table.empty:
        st.info("Nenhuma informação de consultor encontrada no período filtrado.")
    else:
        st.dataframe(df_consultor_table, use_container_width=True, height=360)



def render_compare_mode(
    company_slug: str,
    dfs,
    df_leads,
    df_opportunities,
    df_opp_full,
    eventos_sel,
    origens_sel,
    dispositivos_sel,
    ano_sel,
    m1_label,
    m2_label,
    hoje,
    ontem,
):
    m1_num = month_label_to_num(m1_label)
    m2_num = month_label_to_num(m2_label)

    df_m1_leads_base = filter_by_year_month(df_leads, ano_sel, m1_num)
    df_m2_leads_base = filter_by_year_month(df_leads, ano_sel, m2_num)
    df_m1_opp = filter_by_year_month(df_opportunities, ano_sel, m1_num)
    df_m2_opp = filter_by_year_month(df_opportunities, ano_sel, m2_num)

    df_m1 = apply_extra_filters_leads(df_m1_leads_base, eventos_sel, origens_sel, dispositivos_sel)
    df_m2 = apply_extra_filters_leads(df_m2_leads_base, eventos_sel, origens_sel, dispositivos_sel)

    df_instagram_full = dfs.get("leads_meta_whatsapp", pd.DataFrame())
    df_formulario_full = dfs.get("leads_meta_formulario", pd.DataFrame())
    df_ig_m1_base = filter_by_year_month(df_instagram_full, ano_sel, m1_num) if not df_instagram_full.empty else pd.DataFrame()
    df_ig_m2_base = filter_by_year_month(df_instagram_full, ano_sel, m2_num) if not df_instagram_full.empty else pd.DataFrame()
    df_form_m1_base = filter_by_year_month(df_formulario_full, ano_sel, m1_num) if not df_formulario_full.empty else pd.DataFrame()
    df_form_m2_base = filter_by_year_month(df_formulario_full, ano_sel, m2_num) if not df_formulario_full.empty else pd.DataFrame()

    df_ig_m1 = apply_common_filters(df_ig_m1_base, origens_sel, dispositivos_sel)
    df_ig_m2 = apply_common_filters(df_ig_m2_base, origens_sel, dispositivos_sel)
    df_form_m1 = apply_common_filters(df_form_m1_base, origens_sel, dispositivos_sel)
    df_form_m2 = apply_common_filters(df_form_m2_base, origens_sel, dispositivos_sel)

    df_m1_all = build_combined_leads(company_slug, df_m1, df_ig_m1, df_form_m1)
    df_m2_all = build_combined_leads(company_slug, df_m2, df_ig_m2, df_form_m2)

    if df_m1_all.empty and df_m2_all.empty:
        st.warning("Nenhum Lead encontrado para os meses selecionados (após filtros).")
        st.stop()

    min1, max1 = (df_m1["data"].min(), df_m1["data"].max()) if not df_m1.empty else ("-", "-")
    min2, max2 = (df_m2["data"].min(), df_m2["data"].max()) if not df_m2.empty else ("-", "-")
    st.caption(f"Comparação: {m1_label}/{ano_sel} ({min1} a {max1}) vs {m2_label}/{ano_sel} ({min2} a {max2})")

    render_compare_highlights(
        company_slug,
        df_m1,
        df_ig_m1,
        df_form_m1,
        df_m1_opp,
        df_m2,
        df_ig_m2,
        df_form_m2,
        df_m2_opp,
    )
    render_compare_period_label(ano_sel, m1_label, m2_label)

    st.markdown("---")
    st.subheader("Leads por Dia")
    conv_por_dia_1 = count_unique_leads_by(df_m1_all, ["data"])
    conv_por_dia_2 = count_unique_leads_by(df_m2_all, ["data"])
    fig = go.Figure()
    if not conv_por_dia_1.empty:
        fig.add_trace(go.Scatter(x=conv_por_dia_1["data"], y=conv_por_dia_1["leads"], mode="lines+markers", name=f"{m1_label}/{ano_sel}", line=dict(color=COMPARE_COLOR_1), marker=dict(size=6)))
    if not conv_por_dia_2.empty:
        fig.add_trace(go.Scatter(x=conv_por_dia_2["data"], y=conv_por_dia_2["leads"], mode="lines+markers", name=f"{m2_label}/{ano_sel}", line=dict(color=COMPARE_COLOR_2), marker=dict(size=6)))
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Leads",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Funil")
    tab_f1, tab_f2 = st.tabs([f"{m1_label}/{ano_sel}", f"{m2_label}/{ano_sel}"])

    with tab_f1:
        render_central_funnel_compare(
            company_slug,
            dfs,
            df_m1,
            df_ig_m1 if not df_ig_m1.empty else None,
            df_form_m1 if not df_form_m1.empty else None,
            df_m1_opp,
            df_opp_full,
            ano_sel,
            m1_num,
            title=f"Funil - {m1_label}/{ano_sel}",
        )

    with tab_f2:
        render_central_funnel_compare(
            company_slug,
            dfs,
            df_m2,
            df_ig_m2 if not df_ig_m2.empty else None,
            df_form_m2 if not df_form_m2.empty else None,
            df_m2_opp,
            df_opp_full,
            ano_sel,
            m2_num,
            title=f"Funil - {m2_label}/{ano_sel}",
        )

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Leads por Origem")
        o1 = count_unique_leads_by(df_m1_all, ["origem"])
        o1["mes"] = f"{m1_label}/{ano_sel}"
        o2 = count_unique_leads_by(df_m2_all, ["origem"])
        o2["mes"] = f"{m2_label}/{ano_sel}"
        conv_origem = pd.concat([o1, o2], ignore_index=True)
        fig_origem = px.bar(conv_origem, x="origem", y="leads", color="mes", barmode="group", color_discrete_map={f"{m1_label}/{ano_sel}": COMPARE_COLOR_1, f"{m2_label}/{ano_sel}": COMPARE_COLOR_2})
        st.plotly_chart(fig_origem, use_container_width=True)

    with col_g2:
        st.subheader("Formulários do Site")
        e1 = count_unique_leads_by(df_m1_all, ["evento"])
        e1["evento_legenda"] = e1["evento"].apply(label_evento)
        e1["mes"] = f"{m1_label}/{ano_sel}"
        e2 = count_unique_leads_by(df_m2_all, ["evento"])
        e2["evento_legenda"] = e2["evento"].apply(label_evento)
        e2["mes"] = f"{m2_label}/{ano_sel}"
        conv_evento = pd.concat([e1, e2], ignore_index=True)
        fig_evento = px.bar(conv_evento, x="evento_legenda", y="leads", color="mes", barmode="group", color_discrete_map={f"{m1_label}/{ano_sel}": COMPARE_COLOR_1, f"{m2_label}/{ano_sel}": COMPARE_COLOR_2})
        st.plotly_chart(fig_evento, use_container_width=True)

    # Dispositivo e Horário (abaixo de Origem e Evento)
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.subheader("Leads por Dispositivo")
        d1 = count_unique_leads_by(df_m1, ["dispositivo"])
        d1["mes"] = f"{m1_label}/{ano_sel}"
        d2 = count_unique_leads_by(df_m2, ["dispositivo"])
        d2["mes"] = f"{m2_label}/{ano_sel}"
        conv_disp = pd.concat([d1, d2], ignore_index=True)
        fig_disp = px.bar(conv_disp, x="dispositivo", y="leads", color="mes", barmode="group", color_discrete_map={f"{m1_label}/{ano_sel}": COMPARE_COLOR_1, f"{m2_label}/{ano_sel}": COMPARE_COLOR_2})
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário dos Leads")
        df_m1_time = concat_non_empty([df_m1, df_form_m1 if company_slug == "nextqs" else pd.DataFrame()])
        df_m2_time = concat_non_empty([df_m2, df_form_m2 if company_slug == "nextqs" else pd.DataFrame()])
        h1 = count_unique_leads_by(df_m1_time, ["hora"])
        h1["mes"] = f"{m1_label}/{ano_sel}"
        h2 = count_unique_leads_by(df_m2_time, ["hora"])
        h2["mes"] = f"{m2_label}/{ano_sel}"
        conv_hora = pd.concat([h1, h2], ignore_index=True)
        fig_hora = px.bar(conv_hora, x="hora", y="leads", color="mes", barmode="group", color_discrete_map={f"{m1_label}/{ano_sel}": COMPARE_COLOR_1, f"{m2_label}/{ano_sel}": COMPARE_COLOR_2})
        fig_hora.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig_hora, use_container_width=True)


# ===========================================================================
# MAIN
# ===========================================================================

company = get_selected_company()
company_slug = company["slug"]

logo_path = Path(company["logo_path"])
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)

# ---------------------------------------------------------------------------
# Ações do SMARK na sidebar
# ---------------------------------------------------------------------------
st.session_state.setdefault("sync_message", None)
st.session_state.setdefault("open_upload_panel", False)
st.session_state.setdefault("dashboard_view", "leads")

st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] > label {
        display: none;
    }
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
        padding: 0.35rem 0 0 0;
        border: 0;
        background: transparent;
    }
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
        padding: 0.75rem;
        border-radius: 12px;
    }
    section[data-testid="stSidebar"] .smark-last-update {
        font-size: 11px;
        line-height: 1.15;
        white-space: nowrap;
        margin: 0.35rem 0 0 0;
        opacity: 0.9;
    }
    section[data-testid="stSidebar"] .smark-upload-hint {
        font-size: 11px;
        opacity: 0.78;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Exibe a última data de atualização com SMARK
ultima_data_smark = st.session_state.get("ultima_data_smark")
if ultima_data_smark is None:
    ultima_data_smark = get_smark_ultima_data_from_sheet()
    if ultima_data_smark:
        st.session_state["ultima_data_smark"] = ultima_data_smark

smark_csv_file = None
send_smark_clicked = False

with st.sidebar:
    upload_container = st.container()
    with upload_container:
        if hasattr(st, "popover"):
            with st.popover("Enviar Dados SMARK", use_container_width=True):
                st.markdown(
                    "<div class='smark-upload-hint'>Selecione o CSV exportado do SMARK para atualizar os dados.</div>",
                    unsafe_allow_html=True,
                )
                with st.form("smark_upload_form_popover", clear_on_submit=False):
                    smark_csv_file = st.file_uploader(
                        "Selecionar arquivo CSV do SMARK",
                        type=["csv"],
                        key="smark_csv_uploader",
                        label_visibility="collapsed",
                    )
                    send_smark_clicked = st.form_submit_button("Enviar", use_container_width=True)
        else:
            if st.button("Enviar Dados SMARK", use_container_width=True, type="primary"):
                st.session_state["open_upload_panel"] = not st.session_state["open_upload_panel"]

            if st.session_state.get("open_upload_panel", False):
                with st.expander("Selecionar arquivo CSV", expanded=True):
                    st.markdown(
                        "<div class='smark-upload-hint'>Selecione o CSV exportado do SMARK para atualizar os dados.</div>",
                        unsafe_allow_html=True,
                    )
                    with st.form("smark_upload_form_expander", clear_on_submit=False):
                        smark_csv_file = st.file_uploader(
                            "Selecionar arquivo CSV do SMARK",
                            type=["csv"],
                            key="smark_csv_uploader",
                            label_visibility="collapsed",
                        )
                        send_smark_clicked = st.form_submit_button("Enviar", use_container_width=True)

    refresh_clicked = st.button("Atualizar Dashboard", use_container_width=True)
    render_company_selector(company_slug)
    if st.button(
        "Meta ADS NextQS",
        use_container_width=True,
        disabled=st.session_state.get("dashboard_view") == "meta_ads_nextqs" and company_slug == "nextqs",
    ):
        st.session_state["dashboard_view"] = "meta_ads_nextqs"
        st.query_params["empresa"] = "nextqs"
        st.rerun()
    if st.session_state.get("dashboard_view") == "meta_ads_nextqs":
        if st.button("Dashboard de Leads", use_container_width=True):
            st.session_state["dashboard_view"] = "leads"
            st.rerun()

    if ultima_data_smark:
        st.markdown(
            f"<div class='smark-last-update'>Última atualização com SMARK: <b>{ultima_data_smark}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='smark-last-update'><i>Nenhuma atualização SMARK registrada.</i></div>",
            unsafe_allow_html=True,
        )

    if st.session_state.get("sync_message"):
        st.success(st.session_state["sync_message"])
        st.session_state["sync_message"] = None

# Processa o CSV enviado
if send_smark_clicked:
    if smark_csv_file is None:
        st.sidebar.warning("Selecione um arquivo CSV antes de clicar em Enviar.")
    else:
        with st.spinner("Enviando dados do SMARK para o Google Sheets..."):
            try:
                upload_result = upload_csv_to_smark_sheet(smark_csv_file)

                if upload_result["divergent_cols"]:
                    st.sidebar.warning(
                        "⚠️ Colunas do CSV divergentes do esperado: "
                        + ", ".join(upload_result["divergent_cols"])
                    )

                if upload_result["ultima_data"]:
                    st.session_state["ultima_data_smark"] = upload_result["ultima_data"]

                with st.spinner("Sincronizando oportunidades da NextQS e StarLed..."):
                    sync_results = {
                        slug: sync_opportunities_with_smark(slug)
                        for slug in COMPANIES.keys()
                    }
                    load_sheet.clear()
                    sync_summary = " | ".join(
                        (
                            f"{COMPANIES[slug]['nome']}: "
                            f"site {result['site_matches']}, "
                            f"Meta WhatsApp {result['instagram_matches']}, "
                            f"Meta Formulário {result['formulario_matches']}, "
                            f"oportunidades {result['opportunities_added']}"
                        )
                        for slug, result in sync_results.items()
                    )
                    st.session_state["sync_message"] = (
                        f"CSV enviado com sucesso ({upload_result['rows']} linhas). "
                        f"Sincronização concluída nas duas empresas. {sync_summary}."
                    )
                    st.session_state["open_upload_panel"] = False
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Erro ao enviar dados do SMARK: {e}")

if refresh_clicked:
    ultima_data_refresh = get_smark_ultima_data_from_sheet()
    if ultima_data_refresh:
        st.session_state["ultima_data_smark"] = ultima_data_refresh
    trigger_sheet_reload()

if st.session_state.get("dashboard_view") == "meta_ads_nextqs":
    df_meta_ads = load_sheet("nextqs", "meta_campanhas")
    if df_meta_ads.empty:
        st.warning("Nenhum dado encontrado na aba 'meta_campanhas' da planilha NextQS.")
        st.stop()

    df_meta_ads_filtrado, meta_ads_ano, meta_ads_mes_label = render_meta_ads_filters(df_meta_ads)
    render_meta_ads_dashboard(df_meta_ads_filtrado, meta_ads_ano, meta_ads_mes_label)
    st.stop()

dfs = {name: load_sheet(company_slug, name) for name in SHEETS_REQUIRED}
for name in OPTIONAL_SHEETS:
    dfs[name] = load_sheet(company_slug, name)

df_leads = dfs["leads_site"]
df_opportunities = dfs["oportunidades"]

if df_leads.empty:
    st.warning(f"Nenhum dado encontrado na aba 'leads_site' da planilha {company['nome']}.")
    st.stop()

PERIODOS = ["Hoje", "Ontem", "Últimos 30 dias", "Este mês", "Este ano", "Personalizado", "Comparar meses"]
if st.session_state.get("periodo_sel") == "Últimos 7 dias":
    st.session_state["periodo_sel"] = "Últimos 30 dias"
st.session_state.setdefault("periodo_sel", "Últimos 30 dias")
st.session_state.setdefault("periodo_sel_prev", st.session_state["periodo_sel"])

st.sidebar.markdown("<h2 style='margin-bottom: 0.25rem;'>Filtros</h2>", unsafe_allow_html=True)
periodo_sel = st.sidebar.radio(label="Período", options=PERIODOS, key="periodo_sel", label_visibility="collapsed")
if periodo_sel != st.session_state.get("periodo_sel_prev"):
    st.session_state["periodo_sel_prev"] = periodo_sel

hoje = get_today_local()
ontem = hoje - timedelta(days=1)
compare_mode = False

df_periodo_leads = get_period_filtered_df(df_leads, periodo_sel, hoje, ontem)
df_periodo_opportunities = get_period_filtered_df(df_opportunities, periodo_sel, hoje, ontem)

if periodo_sel == "Personalizado":
    with st.sidebar.expander("Personalizado", expanded=True):
        anos_disponiveis = sorted(df_leads["ano"].unique())
        ano_default = anos_disponiveis[-1] if anos_disponiveis else hoje.year

        st.session_state.setdefault("custom_ano", ano_default)
        st.session_state.setdefault("custom_mes_label", "Todo o ano")

        custom_ano = st.selectbox("Ano", options=anos_disponiveis, key="custom_ano")
        meses_disponiveis = sorted(df_leads[df_leads["ano"] == custom_ano]["mes"].unique())
        opcoes_meses = ["Todo o ano"] + [MESES_LABEL[m] for m in meses_disponiveis]
        custom_mes_label = st.selectbox("Mês", options=opcoes_meses, key="custom_mes_label")
        aplicar = st.button("Aplicar")

    st.session_state.setdefault("custom_aplicado", False)
    if aplicar:
        st.session_state["custom_aplicado"] = True

    if st.session_state.get("custom_aplicado", False):
        if custom_mes_label == "Todo o ano":
            df_periodo_leads = df_leads[df_leads["ano"] == custom_ano].copy()
            df_periodo_opportunities = df_opportunities[df_opportunities["ano"] == custom_ano].copy() if not df_opportunities.empty else df_opportunities
        else:
            mes_num_sel = month_label_to_num(custom_mes_label)
            df_periodo_leads = df_leads[(df_leads["ano"] == custom_ano) & (df_leads["mes"] == mes_num_sel)].copy()
            if not df_opportunities.empty:
                df_periodo_opportunities = df_opportunities[(df_opportunities["ano"] == custom_ano) & (df_opportunities["mes"] == mes_num_sel)].copy()
            else:
                df_periodo_opportunities = df_opportunities
    else:
        df_periodo_leads = get_period_filtered_df(df_leads, "Últimos 30 dias", hoje, ontem)
        df_periodo_opportunities = get_period_filtered_df(df_opportunities, "Últimos 30 dias", hoje, ontem)

elif periodo_sel == "Comparar meses":
    compare_mode = True
    with st.sidebar.expander("Comparar meses", expanded=True):
        anos_disponiveis = sorted(df_leads["ano"].unique())
        ano_default = anos_disponiveis[-1] if anos_disponiveis else hoje.year
        st.session_state.setdefault("compare_ano", ano_default)

        compare_ano = st.selectbox("Ano", options=anos_disponiveis, key="compare_ano")
        meses_disponiveis = sorted(df_leads[df_leads["ano"] == compare_ano]["mes"].unique())
        if not meses_disponiveis:
            st.warning("Não há meses disponíveis para o ano selecionado.")
            st.stop()

        opcoes_meses = [MESES_LABEL[m] for m in meses_disponiveis]
        st.session_state.setdefault("compare_mes1_label", opcoes_meses[-1])
        st.session_state.setdefault("compare_mes2_label", opcoes_meses[-2] if len(opcoes_meses) >= 2 else opcoes_meses[-1])

        st.selectbox("Mês 1", options=opcoes_meses, key="compare_mes1_label")
        st.selectbox("Mês 2", options=opcoes_meses, key="compare_mes2_label")
        aplicar_compare = st.button("Aplicar")

    st.session_state.setdefault("compare_aplicado", False)
    if aplicar_compare:
        st.session_state["compare_aplicado"] = True

st.sidebar.header("Filtros adicionais")
if not compare_mode:
    df_periodo_leads_meta_whatsapp_empty_check = get_effective_period_filtered_df(dfs.get("leads_meta_whatsapp", pd.DataFrame()), periodo_sel, hoje, ontem)
    df_periodo_leads_meta_formulario_empty_check = get_effective_period_filtered_df(dfs.get("leads_meta_formulario", pd.DataFrame()), periodo_sel, hoje, ontem)
    if df_periodo_leads.empty and (
        company_slug != "nextqs"
        or (df_periodo_leads_meta_whatsapp_empty_check.empty and df_periodo_leads_meta_formulario_empty_check.empty)
    ):
        st.warning("Nenhum Lead encontrado para o período selecionado.")
        st.stop()

if compare_mode and st.session_state.get("compare_aplicado", False):
    m1_num = month_label_to_num(st.session_state["compare_mes1_label"])
    m2_num = month_label_to_num(st.session_state["compare_mes2_label"])
    df_filter_frames = [
        filter_by_year_month(df_leads, st.session_state["compare_ano"], m1_num),
        filter_by_year_month(df_leads, st.session_state["compare_ano"], m2_num),
    ]
    df_instagram_full = dfs.get("leads_meta_whatsapp", pd.DataFrame())
    if company_slug == "nextqs" and not df_instagram_full.empty:
        df_filter_frames.extend(
            [
                filter_by_year_month(df_instagram_full, st.session_state["compare_ano"], m1_num),
                filter_by_year_month(df_instagram_full, st.session_state["compare_ano"], m2_num),
            ]
        )
    df_formulario_full = dfs.get("leads_meta_formulario", pd.DataFrame())
    if company_slug == "nextqs" and not df_formulario_full.empty:
        df_filter_frames.extend(
            [
                filter_by_year_month(df_formulario_full, st.session_state["compare_ano"], m1_num),
                filter_by_year_month(df_formulario_full, st.session_state["compare_ano"], m2_num),
            ]
        )
    df_for_filters = pd.concat(df_filter_frames, ignore_index=True).sort_values("data_hora")
else:
    df_filter_frames = [df_periodo_leads]
    df_periodo_leads_meta_whatsapp_sidebar = get_effective_period_filtered_df(dfs.get("leads_meta_whatsapp", pd.DataFrame()), periodo_sel, hoje, ontem)
    if company_slug == "nextqs" and not df_periodo_leads_meta_whatsapp_sidebar.empty:
        df_filter_frames.append(df_periodo_leads_meta_whatsapp_sidebar)
    df_periodo_leads_meta_formulario_sidebar = get_effective_period_filtered_df(dfs.get("leads_meta_formulario", pd.DataFrame()), periodo_sel, hoje, ontem)
    if company_slug == "nextqs" and not df_periodo_leads_meta_formulario_sidebar.empty:
        df_filter_frames.append(df_periodo_leads_meta_formulario_sidebar)
    df_for_filters = pd.concat(df_filter_frames, ignore_index=True).sort_values("data_hora")

eventos = sorted(df_for_filters["evento"].dropna().unique().tolist()) if "evento" in df_for_filters.columns else []
eventos_sel = st.sidebar.multiselect("Tipo de evento", options=eventos, default=eventos)
origens = sorted(df_for_filters["origem"].dropna().unique().tolist()) if "origem" in df_for_filters.columns else []
origens_sel = st.sidebar.multiselect("Origem", options=origens, default=origens)
dispositivos = sorted(df_for_filters["dispositivo"].dropna().unique().tolist()) if "dispositivo" in df_for_filters.columns else []
dispositivos_sel = st.sidebar.multiselect("Dispositivo", options=dispositivos, default=dispositivos)

with st.sidebar.form("opportunity_search_form"):
    opportunity_search_input = st.text_input(
        "Pesquise sobre uma oportunidade:",
        placeholder="Número da oportunidade",
    )
    opportunity_search_submitted = st.form_submit_button("Pesquisar", use_container_width=True)

if opportunity_search_submitted:
    st.session_state["searched_opportunity"] = normalize_opportunity_key(opportunity_search_input)

searched_opportunity = st.session_state.get("searched_opportunity", "")
if searched_opportunity:
    with st.sidebar:
        with st.spinner("Pesquisando..."):
            opportunity_summary = build_opportunity_summary(company_slug, searched_opportunity, dfs)
        render_opportunity_summary_sidebar(opportunity_summary)

if compare_mode and st.session_state.get("compare_aplicado", False):
    render_compare_mode(
        company_slug,
        dfs,
        df_leads,
        df_opportunities,
        df_opportunities,  # df_opp_full para negócios efetuados
        eventos_sel,
        origens_sel,
        dispositivos_sel,
        int(st.session_state["compare_ano"]),
        st.session_state["compare_mes1_label"],
        st.session_state["compare_mes2_label"],
        hoje,
        ontem,
    )
else:
    render_normal_mode(
        company_slug,
        dfs,
        df_periodo_leads,
        df_periodo_opportunities,
        df_opportunities,  # df_opp_full para negócios efetuados
        eventos_sel,
        origens_sel,
        dispositivos_sel,
        periodo_sel,
        hoje,
        ontem,
    )
