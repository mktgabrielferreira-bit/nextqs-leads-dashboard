
import io
import re
from datetime import date, datetime, timedelta
from pathlib import Path
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
    "sessions",
]

OPTIONAL_SHEETS = [
    "oportunidades",
    "leads_instagram",
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

GREEN_COLOR = "#22c55e"
COMPARE_COLOR_1 = "#2563EB"
COMPARE_COLOR_2 = "#F97316"
FUNNEL_COLOR = "#B32157"


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


def render_company_switcher(current_slug: str):
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "NextQS",
            use_container_width=True,
            disabled=current_slug == "nextqs",
            type="primary" if current_slug == "nextqs" else "secondary",
        ):
            st.query_params["empresa"] = "nextqs"
            load_sheet.clear()
            st.rerun()

    with col2:
        if st.button(
            "StarLed",
            use_container_width=True,
            disabled=current_slug == "starled",
            type="primary" if current_slug == "starled" else "secondary",
        ):
            st.query_params["empresa"] = "starled"
            load_sheet.clear()
            st.rerun()


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


def normalize_origin(value) -> str:
    text = normalize_text(value)
    return text if text else "Origem não identificada"


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


def build_opportunity_row(data_lead, canal: str, user_id, origem, smark_row: dict) -> list[str]:
    return [
        format_date_br_from_any(data_lead),
        canal,
        normalize_text(user_id),
        normalize_text(smark_row.get("Cod. Oportunidade")),
        format_smark_swapped_date_to_br(smark_row.get("Data Oportunidade")),
        normalize_text(smark_row.get("Área de Atuação")),
        normalize_text(smark_row.get("Nome Colaborador Responsável")),
        clean_status_funil(smark_row.get("Funil de Venda")),
        format_smark_swapped_date_to_br(smark_row.get("Data Encerramento")),
        normalize_origin(origem),
    ]


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
        leads_instagram_ws = base_spreadsheet.worksheet("leads_instagram")
        instagram_records = leads_instagram_ws.get_all_records()
    except gspread.exceptions.WorksheetNotFound:
        leads_instagram_ws = None
        instagram_records = []

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
        "data_lead",
        "canal",
        "user_id",
        "oportunidade",
        "data_oportunidade",
        "area_atuação",
        "consultor",
        "status_funil",
        "data_encerramento",
        "origem",
    ]

    df_leads = pd.DataFrame(leads_records)
    df_smark = pd.DataFrame(smark_records)
    df_instagram = pd.DataFrame(instagram_records)

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

    if leads_instagram_ws is not None and not df_instagram.empty:
        required_instagram_columns = ["telefone", "user_id_cel", "data"]
        missing_instagram = [col for col in required_instagram_columns if col not in df_instagram.columns]
        if missing_instagram:
            raise ValueError("A aba 'leads_instagram' não possui as colunas obrigatórias: " + ", ".join(missing_instagram))

    ensure_column_exists(leads_ws, BASE_QUALIFIED_COLUMN)
    ensure_headers_exist(opportunities_ws, opp_required_headers)

    instagram_header_map = {}
    if leads_instagram_ws is not None:
        instagram_header_map = ensure_headers_exist(leads_instagram_ws, ["qualificado", "consultor"])

    smark_email_map = build_smark_email_map(smark_records)
    smark_phone_map = build_smark_phone_map(smark_records)

    opportunities_by_code = {}
    generated_user_ids_site = set()
    processed_match_emails = set()
    site_matches = 0
    instagram_matches = 0
    qualified_sim = 0
    qualified_duplicate = 0
    instagram_qualified_sim = 0
    site_opportunities_added = 0
    instagram_opportunities_added = 0
    opportunities_skipped = 0

    leads_headers = leads_ws.row_values(1)
    qualified_col_idx = leads_headers.index(BASE_QUALIFIED_COLUMN) + 1
    qualified_values = []

    for row in leads_records:
        email_norm = normalize_email(row.get(base_email_col)) if base_email_col else None
        target_value = ""

        if email_norm and email_norm in smark_email_map:
            site_matches += 1
            smark_row = smark_email_map[email_norm]

            if email_norm not in processed_match_emails:
                target_value = "SIM"
                processed_match_emails.add(email_norm)
                qualified_sim += 1

                opportunity_code = normalize_text(smark_row.get("Cod. Oportunidade"))
                user_id_value = normalize_text(row.get("user_id_email"))
                user_id_key = user_id_value.lower()

                if not opportunity_code or not user_id_value or user_id_key in generated_user_ids_site:
                    opportunities_skipped += 1
                elif opportunity_code not in opportunities_by_code:
                    opportunities_by_code[opportunity_code] = build_opportunity_row(
                        row.get("data_hora"),
                        "Site",
                        user_id_value,
                        row.get("origem"),
                        smark_row,
                    )
                    generated_user_ids_site.add(user_id_key)
                    site_opportunities_added += 1
                else:
                    opportunities_skipped += 1
            else:
                target_value = "Duplicado"
                qualified_duplicate += 1

        qualified_values.append([target_value])

    if qualified_values:
        qualified_col_letter = gspread.utils.rowcol_to_a1(1, qualified_col_idx)[:-1]
        qualified_range = f"{qualified_col_letter}2:{qualified_col_letter}{len(qualified_values) + 1}"
        leads_ws.update(qualified_range, qualified_values, value_input_option="USER_ENTERED")

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
                opportunities_by_code[opportunity_code] = build_opportunity_row(
                    row.get("data"),
                    "Instagram",
                    row.get("user_id_cel"),
                    row.get("origem"),
                    smark_row,
                )
                instagram_opportunities_added += 1
            else:
                opportunities_skipped += 1

        instagram_qualified_values.append([target_value])
        instagram_consultor_values.append([consultor_value])

    if leads_instagram_ws is not None and instagram_records:
        qual_col_idx = instagram_header_map["qualificado"]
        consultor_col_idx = instagram_header_map["consultor"]

        qual_col_letter = gspread.utils.rowcol_to_a1(1, qual_col_idx)[:-1]
        qual_range = f"{qual_col_letter}2:{qual_col_letter}{len(instagram_qualified_values) + 1}"
        leads_instagram_ws.update(qual_range, instagram_qualified_values, value_input_option="USER_ENTERED")

        consultor_col_letter = gspread.utils.rowcol_to_a1(1, consultor_col_idx)[:-1]
        consultor_range = f"{consultor_col_letter}2:{consultor_col_letter}{len(instagram_consultor_values) + 1}"
        leads_instagram_ws.update(consultor_range, instagram_consultor_values, value_input_option="USER_ENTERED")

    opportunities_payload = [opp_required_headers] + list(opportunities_by_code.values())
    opportunities_ws.clear()
    opportunities_ws.update(
        f"A1:{gspread.utils.rowcol_to_a1(len(opportunities_payload), len(opp_required_headers))}",
        opportunities_payload,
        value_input_option="USER_ENTERED",
    )

    return {
        "matches": site_matches + instagram_matches,
        "site_matches": site_matches,
        "instagram_matches": instagram_matches,
        "qualified_sim": qualified_sim,
        "qualified_duplicate": qualified_duplicate,
        "instagram_qualified_sim": instagram_qualified_sim,
        "opportunities_added": site_opportunities_added + instagram_opportunities_added,
        "site_opportunities_added": site_opportunities_added,
        "instagram_opportunities_added": instagram_opportunities_added,
        "opportunities_skipped": opportunities_skipped,
        "leads_rows": len(df_leads),
        "instagram_rows": len(df_instagram),
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


@st.cache_data
def load_sheet(company_slug: str, sheet_name: str) -> pd.DataFrame:
    try:
        ws = get_base_spreadsheet(company_slug).worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        return pd.DataFrame()

    data = ws.get_all_records()
    df = pd.DataFrame(data)

    if df.empty:
        return df

    if sheet_name == "oportunidades":
        if "data_oportunidade" in df.columns:
            df["data_hora"] = pd.to_datetime(df["data_oportunidade"], dayfirst=True, errors="coerce")
        elif "data_lead" in df.columns:
            df["data_hora"] = pd.to_datetime(df["data_lead"], dayfirst=True, errors="coerce")
        else:
            df["data_hora"] = pd.NaT

        # Carrega data_encerramento como coluna de data para filtro de Negócios Efetuados
        if "data_encerramento" in df.columns:
            df["data_encerramento_parsed"] = pd.to_datetime(df["data_encerramento"], dayfirst=True, errors="coerce")
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
    if periodo_sel == "Últimos 7 dias":
        start = hoje - timedelta(days=7)
        end = ontem
        return df_src[(df_src["data"] >= start) & (df_src["data"] <= end)].copy()
    if periodo_sel == "Este mês":
        start = date(hoje.year, hoje.month, 1)
        end = ontem
        return df_src[(df_src["data"] >= start) & (df_src["data"] <= end)].copy()
    if periodo_sel == "Este ano":
        start = date(hoje.year, 1, 1)
        end = hoje
        return df_src[(df_src["data"] >= start) & (df_src["data"] <= end)].copy()
    return df_src.copy()


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
    df_periodo_leads_instagram: pd.DataFrame | None,
) -> int:
    """
    NextQS: leads_site + leads_instagram sem duplicações (por email/user_id_email).
    StarLed: apenas leads_site sem duplicações.
    """
    if company_slug == "nextqs":
        frames = []
        if not df_periodo_leads_site.empty:
            frames.append(df_periodo_leads_site)
        if df_periodo_leads_instagram is not None and not df_periodo_leads_instagram.empty:
            frames.append(df_periodo_leads_instagram)
        if not frames:
            return 0
        combined = pd.concat(frames, ignore_index=True)
        # Deduplicação por user_id_email quando disponível
        if "user_id_email" in combined.columns:
            unique_emails = combined["user_id_email"].dropna().nunique()
            sem_email = combined["user_id_email"].isna().sum()
            return unique_emails + sem_email
        return len(combined)
    else:
        # StarLed: apenas leads_site
        if df_periodo_leads_site.empty:
            return 0
        if "user_id_email" in df_periodo_leads_site.columns:
            unique_emails = df_periodo_leads_site["user_id_email"].dropna().nunique()
            sem_email = df_periodo_leads_site["user_id_email"].isna().sum()
            return unique_emails + sem_email
        return len(df_periodo_leads_site)


def get_negocios_efetuados_count(df_opp_full: pd.DataFrame, periodo_sel: str, hoje: date, ontem: date) -> int:
    """
    Conta negócios efetuados filtrando por Data Encerramento no período selecionado
    e status_funil == 'Negócio efetuado'.
    """
    if df_opp_full.empty:
        return 0
    if "status_funil" not in df_opp_full.columns:
        return 0

    df_neg = df_opp_full[
        df_opp_full["status_funil"].str.strip().str.lower() == "negócio efetuado"
    ].copy()

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
    if periodo_sel == "Últimos 7 dias":
        start = hoje - timedelta(days=7)
        return len(df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= ontem)])
    if periodo_sel == "Este mês":
        start = date(hoje.year, hoje.month, 1)
        return len(df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= ontem)])
    if periodo_sel == "Este ano":
        start = date(hoje.year, 1, 1)
        return len(df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)])
    # Personalizado ou outros: usa todos os que estão no df_opp já filtrado
    return len(df_neg)


def get_negocios_efetuados_count_by_month(df_opp_full: pd.DataFrame, ano: int, mes_num: int) -> int:
    """Conta negócios efetuados por mês (usando data_encerramento_parsed)."""
    if df_opp_full.empty or "status_funil" not in df_opp_full.columns:
        return 0
    df_neg = df_opp_full[
        df_opp_full["status_funil"].str.strip().str.lower() == "negócio efetuado"
    ].copy()
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
    df_leads_instagram_periodo: pd.DataFrame | None,
    df_opportunities_periodo: pd.DataFrame,
    df_opp_full: pd.DataFrame,
    periodo_sel: str,
    hoje: date,
    ontem: date,
    title: str = "Funil",
):
    leads_count = get_leads_count_for_funnel(company_slug, dfs, df_leads_site_periodo, df_leads_instagram_periodo)
    opp_count = len(df_opportunities_periodo) if not df_opportunities_periodo.empty else 0
    negocios_count = get_negocios_efetuados_count(df_opp_full, periodo_sel, hoje, ontem)

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
    df_leads_instagram_periodo: pd.DataFrame | None,
    df_opportunities_periodo: pd.DataFrame,
    df_opp_full: pd.DataFrame,
    ano: int,
    mes_num: int,
    title: str = "Funil",
):
    leads_count = get_leads_count_for_funnel(company_slug, dfs, df_leads_site_periodo, df_leads_instagram_periodo)
    opp_count = len(df_opportunities_periodo) if not df_opportunities_periodo.empty else 0
    negocios_count = get_negocios_efetuados_count_by_month(df_opp_full, ano, mes_num)

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


def get_negocios_efetuados_df(df_opp_full: pd.DataFrame, periodo_sel: str, hoje: date, ontem: date) -> pd.DataFrame:
    if df_opp_full.empty or "status_funil" not in df_opp_full.columns:
        return pd.DataFrame(columns=df_opp_full.columns if not df_opp_full.empty else [])

    df_neg = df_opp_full[
        df_opp_full["status_funil"].astype(str).str.strip().str.lower() == "negócio efetuado"
    ].copy()

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
    if periodo_sel == "Últimos 7 dias":
        start = hoje - timedelta(days=7)
        return df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= ontem)].copy()
    if periodo_sel == "Este mês":
        start = date(hoje.year, hoje.month, 1)
        return df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= ontem)].copy()
    if periodo_sel == "Este ano":
        start = date(hoje.year, 1, 1)
        return df_neg[(df_neg["data_enc_date"] >= start) & (df_neg["data_enc_date"] <= hoje)].copy()
    return df_neg.copy()


def build_origin_table(
    company_slug: str,
    df_leads_site: pd.DataFrame,
    df_leads_instagram: pd.DataFrame | None,
    df_opportunities_periodo: pd.DataFrame,
    df_negocios_periodo: pd.DataFrame,
) -> pd.DataFrame:
    leads_frames = []

    if not df_leads_site.empty:
        leads_frames.append(prepare_leads_for_reporting(df_leads_site))

    if company_slug == "nextqs" and df_leads_instagram is not None and not df_leads_instagram.empty:
        leads_frames.append(prepare_leads_for_reporting(df_leads_instagram))

    if leads_frames:
        df_leads_combined = pd.concat(leads_frames, ignore_index=True)
        leads_por_origem = (
            df_leads_combined.drop_duplicates(subset=["origem", "lead_key"])
            .groupby("origem")["lead_key"]
            .nunique()
            .reset_index(name="Leads")
        )
    else:
        leads_por_origem = pd.DataFrame(columns=["origem", "Leads"])

    if not df_opportunities_periodo.empty:
        df_opp = df_opportunities_periodo.copy()
        df_opp["origem"] = df_opp.get("origem", pd.Series(index=df_opp.index, dtype="object")).apply(normalize_origin)
        opp_por_origem = df_opp.groupby("origem").size().reset_index(name="Oportunidades")
    else:
        opp_por_origem = pd.DataFrame(columns=["origem", "Oportunidades"])

    if not df_negocios_periodo.empty:
        df_neg = df_negocios_periodo.copy()
        df_neg["origem"] = df_neg.get("origem", pd.Series(index=df_neg.index, dtype="object")).apply(normalize_origin)
        neg_por_origem = df_neg.groupby("origem").size().reset_index(name="Negócios")
    else:
        neg_por_origem = pd.DataFrame(columns=["origem", "Negócios"])

    df_out = leads_por_origem.rename(columns={"origem": "Origem"})

    if df_out.empty:
        df_out = pd.DataFrame(columns=["Origem", "Leads"])

    if not opp_por_origem.empty:
        df_out = df_out.merge(opp_por_origem.rename(columns={"origem": "Origem"}), on="Origem", how="outer")
    else:
        df_out["Oportunidades"] = 0

    if not neg_por_origem.empty:
        df_out = df_out.merge(neg_por_origem.rename(columns={"origem": "Origem"}), on="Origem", how="outer")
    else:
        df_out["Negócios"] = 0

    for col in ["Leads", "Oportunidades", "Negócios"]:
        if col not in df_out.columns:
            df_out[col] = 0
        df_out[col] = df_out[col].fillna(0).astype(int)

    return df_out.sort_values(["Leads", "Oportunidades", "Negócios"], ascending=False).reset_index(drop=True)


def build_campaign_table(
    df_leads_filtrado: pd.DataFrame,
    df_opp_full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria tabela de campanhas com colunas: Campanha, Leads, Oportunidades, Negócios.
    """
    df_c = df_leads_filtrado[df_leads_filtrado["utm_campaign"] != "Campanha não identificada"].copy()
    if df_c.empty:
        return pd.DataFrame()

    leads_por_camp = df_c.groupby("utm_campaign").size().reset_index(name="Leads")
    leads_por_camp = leads_por_camp.rename(columns={"utm_campaign": "Campanha"})

    # Oportunidades por campanha: join via user_id_email
    if not df_opp_full.empty and "user_id" in df_opp_full.columns and "user_id_email" in df_c.columns:
        opp_user_ids = set(df_opp_full["user_id"].dropna().astype(str).str.strip().str.lower())
        neg_df = df_opp_full[df_opp_full["status_funil"].str.strip().str.lower() == "negócio efetuado"] if "status_funil" in df_opp_full.columns else pd.DataFrame()
        neg_user_ids = set(neg_df["user_id"].dropna().astype(str).str.strip().str.lower()) if not neg_df.empty else set()

        opp_counts = []
        neg_counts = []
        for _, row in leads_por_camp.iterrows():
            camp_leads = df_c[df_c["utm_campaign"] == row["Campanha"]]
            emails = set(camp_leads["user_id_email"].dropna().astype(str).str.strip().str.lower())
            opp_counts.append(len(emails & opp_user_ids))
            neg_counts.append(len(emails & neg_user_ids))
        leads_por_camp["Oportunidades"] = opp_counts
        leads_por_camp["Negócios"] = neg_counts
    else:
        leads_por_camp["Oportunidades"] = 0
        leads_por_camp["Negócios"] = 0

    return leads_por_camp.sort_values("Leads", ascending=False).reset_index(drop=True)


def build_terms_table(
    df_leads_filtrado: pd.DataFrame,
    df_opp_full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria tabela de termos de pesquisa com colunas: Palavra-chave, Leads, Oportunidades, Negócios.
    """
    df_t = df_leads_filtrado[df_leads_filtrado["utm_term"] != "Palavra-chave não identificada"].copy()
    if df_t.empty:
        return pd.DataFrame()

    leads_por_term = df_t.groupby("utm_term").size().reset_index(name="Leads")
    leads_por_term = leads_por_term.rename(columns={"utm_term": "Palavra-chave"})

    if not df_opp_full.empty and "user_id" in df_opp_full.columns and "user_id_email" in df_t.columns:
        opp_user_ids = set(df_opp_full["user_id"].dropna().astype(str).str.strip().str.lower())
        neg_df = df_opp_full[df_opp_full["status_funil"].str.strip().str.lower() == "negócio efetuado"] if "status_funil" in df_opp_full.columns else pd.DataFrame()
        neg_user_ids = set(neg_df["user_id"].dropna().astype(str).str.strip().str.lower()) if not neg_df.empty else set()

        opp_counts = []
        neg_counts = []
        for _, row in leads_por_term.iterrows():
            term_leads = df_t[df_t["utm_term"] == row["Palavra-chave"]]
            emails = set(term_leads["user_id_email"].dropna().astype(str).str.strip().str.lower())
            opp_counts.append(len(emails & opp_user_ids))
            neg_counts.append(len(emails & neg_user_ids))
        leads_por_term["Oportunidades"] = opp_counts
        leads_por_term["Negócios"] = neg_counts
    else:
        leads_por_term["Oportunidades"] = 0
        leads_por_term["Negócios"] = 0

    return leads_por_term.sort_values("Leads", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Render modes
# ---------------------------------------------------------------------------

def render_normal_mode(
    company_slug: str,
    dfs: dict,
    df_periodo_leads,
    df_periodo_sessions,
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
    df_leads_instagram_periodo = get_period_filtered_df(dfs.get("leads_instagram", pd.DataFrame()), periodo_sel, hoje, ontem)
    df_leads_instagram_filtrado = apply_common_filters(df_leads_instagram_periodo, origens_sel, dispositivos_sel)

    if df_filtrado.empty and (company_slug != "nextqs" or df_leads_instagram_filtrado.empty):
        st.warning("Nenhum Lead encontrado para o período selecionado (após filtros).")
        st.stop()

    conv_total, usuarios_unicos, origem_top, dispositivo_top, pct_top = get_kpis(df_filtrado)

    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        render_kpi_single("Leads no período", f"{conv_total}", GREEN_COLOR)
    with col2:
        render_kpi_single("Leads únicos", f"{usuarios_unicos}", GREEN_COLOR)
    with col3:
        render_kpi_single("Origem mais comum", f"{origem_top}", GREEN_COLOR)
    with col4:
        st.text(f"{dispositivo_top} (%)")
        st.markdown(kpi_value_html(f"{pct_top:.1f}%", GREEN_COLOR), unsafe_allow_html=True)

    st.markdown("---")

    if periodo_sel in ["Hoje", "Ontem"]:
        st.subheader("Leads por Hora")
        conv_por_hora = df_filtrado.groupby("hora").size().reset_index(name="leads").sort_values("hora")
        fig_hora = px.bar(conv_por_hora, x="hora", y="leads")
        fig_hora.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads", xaxis=dict(dtick=1))
        st.plotly_chart(fig_hora, use_container_width=True)
    else:
        st.subheader("Leads por Dia")
        conv_por_dia = df_filtrado.groupby("data").size().reset_index(name="leads")
        fig_dia = px.line(conv_por_dia, x="data", y="leads")
        fig_dia.update_layout(xaxis_title="Data", yaxis_title="Leads")
        st.plotly_chart(fig_dia, use_container_width=True)

    st.markdown("---")

    render_central_funnel(
        company_slug,
        dfs,
        df_periodo_leads,
        df_leads_instagram_periodo if not df_leads_instagram_periodo.empty else None,
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
        conv_origem = df_filtrado.groupby("origem").size().reset_index(name="leads").sort_values("leads", ascending=False)
        fig_origem = px.bar(conv_origem, x="origem", y="leads")
        fig_origem.update_layout(xaxis_title="Origem", yaxis_title="Leads")
        st.plotly_chart(fig_origem, use_container_width=True)

    with col_g2:
        st.subheader("Leads por Evento")
        conv_evento = df_filtrado.groupby("evento").size().reset_index(name="leads")
        conv_evento["evento_legenda"] = conv_evento["evento"].apply(label_evento)
        fig_evento = px.bar(conv_evento, x="evento_legenda", y="leads")
        fig_evento.update_layout(xaxis_title="Evento", yaxis_title="Leads")
        st.plotly_chart(fig_evento, use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.subheader("Leads por Dispositivo")
        conv_disp = df_filtrado.groupby("dispositivo").size().reset_index(name="leads")
        fig_disp = px.bar(conv_disp, x="dispositivo", y="leads")
        fig_disp.update_layout(xaxis_title="Dispositivo", yaxis_title="Leads")
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário dos Leads")
        conv_hora = df_filtrado.groupby("hora").size().reset_index(name="leads").sort_values("hora")
        fig_hora2 = px.bar(conv_hora, x="hora", y="leads")
        fig_hora2.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads")
        st.plotly_chart(fig_hora2, use_container_width=True)

    df_negocios_periodo = get_negocios_efetuados_df(df_opp_full, periodo_sel, hoje, ontem)
    df_origem_table = build_origin_table(
        company_slug,
        df_filtrado,
        df_leads_instagram_filtrado if not df_leads_instagram_filtrado.empty else None,
        df_periodo_opportunities,
        df_negocios_periodo,
    )

    st.markdown("### Informações por Origem")
    if df_origem_table.empty:
        st.info("Nenhuma origem encontrada no período filtrado.")
    else:
        st.dataframe(df_origem_table, use_container_width=True, height=320)

    st.markdown("### Informações por Campanhas")
    df_camp_table = build_campaign_table(df_filtrado, df_opp_full)
    if df_camp_table.empty:
        st.info("Nenhuma campanha válida encontrada no período filtrado.")
    else:
        st.dataframe(df_camp_table, use_container_width=True, height=400)

    st.markdown("### Informações por Termos de Pesquisa")
    df_terms_table = build_terms_table(df_filtrado, df_opp_full)
    if df_terms_table.empty:
        st.info("Nenhuma palavra-chave válida encontrada no período filtrado.")
    else:
        st.dataframe(df_terms_table, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("Dados detalhados (após filtros)")
    detail_columns = [c for c in ["data_hora", "evento", "dispositivo", "origem", "user_id_email", "ip_address"] if c in df_filtrado.columns]
    st.dataframe(df_filtrado[detail_columns], use_container_width=True)



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

    if df_m1.empty and df_m2.empty:
        st.warning("Nenhum Lead encontrado para os meses selecionados (após filtros).")
        st.stop()

    min1, max1 = (df_m1["data"].min(), df_m1["data"].max()) if not df_m1.empty else ("-", "-")
    min2, max2 = (df_m2["data"].min(), df_m2["data"].max()) if not df_m2.empty else ("-", "-")
    st.caption(f"Comparação: {m1_label}/{ano_sel} ({min1} a {max1}) vs {m2_label}/{ano_sel} ({min2} a {max2})")

    conv_total_1, usuarios_unicos_1, origem_top_1, disp_top_1, pct_top_1 = get_kpis(df_m1)
    conv_total_2, usuarios_unicos_2, origem_top_2, disp_top_2, pct_top_2 = get_kpis(df_m2)

    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        render_kpi_dual("Leads no período", f"{conv_total_1}", f"{conv_total_2}", COMPARE_COLOR_1, COMPARE_COLOR_2)
    with col2:
        render_kpi_dual("Leads únicos", f"{usuarios_unicos_1}", f"{usuarios_unicos_2}", COMPARE_COLOR_1, COMPARE_COLOR_2)
    with col3:
        render_kpi_dual("Origem mais comum", f"{origem_top_1}", f"{origem_top_2}", COMPARE_COLOR_1, COMPARE_COLOR_2)
    with col4:
        st.text("Dispositivo (%)")
        st.markdown(
            kpi_value_html(f"{disp_top_1} {pct_top_1:.1f}%", COMPARE_COLOR_1)
            + kpi_value_html(f"{disp_top_2} {pct_top_2:.1f}%", COMPARE_COLOR_2),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Leads por Dia")
    conv_por_dia_1 = df_m1.groupby("data").size().reset_index(name="leads")
    conv_por_dia_2 = df_m2.groupby("data").size().reset_index(name="leads")
    fig = go.Figure()
    if not conv_por_dia_1.empty:
        fig.add_trace(go.Scatter(x=conv_por_dia_1["data"], y=conv_por_dia_1["leads"], mode="lines", name=f"{m1_label}/{ano_sel}", line=dict(color=COMPARE_COLOR_1)))
    if not conv_por_dia_2.empty:
        fig.add_trace(go.Scatter(x=conv_por_dia_2["data"], y=conv_por_dia_2["leads"], mode="lines", name=f"{m2_label}/{ano_sel}", line=dict(color=COMPARE_COLOR_2)))
    fig.update_layout(xaxis_title="Data", yaxis_title="Leads")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Funil")
    tab_f1, tab_f2 = st.tabs([f"{m1_label}/{ano_sel}", f"{m2_label}/{ano_sel}"])

    # Instagram para compare mode
    df_instagram_full = dfs.get("leads_instagram", pd.DataFrame())
    df_ig_m1 = filter_by_year_month(df_instagram_full, ano_sel, m1_num) if not df_instagram_full.empty else pd.DataFrame()
    df_ig_m2 = filter_by_year_month(df_instagram_full, ano_sel, m2_num) if not df_instagram_full.empty else pd.DataFrame()

    with tab_f1:
        render_central_funnel_compare(
            company_slug,
            dfs,
            df_m1_leads_base,
            df_ig_m1 if not df_ig_m1.empty else None,
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
            df_m2_leads_base,
            df_ig_m2 if not df_ig_m2.empty else None,
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
        o1 = df_m1.groupby("origem").size().reset_index(name="leads")
        o1["mes"] = f"{m1_label}/{ano_sel}"
        o2 = df_m2.groupby("origem").size().reset_index(name="leads")
        o2["mes"] = f"{m2_label}/{ano_sel}"
        conv_origem = pd.concat([o1, o2], ignore_index=True)
        fig_origem = px.bar(conv_origem, x="origem", y="leads", color="mes", barmode="group", color_discrete_map={f"{m1_label}/{ano_sel}": COMPARE_COLOR_1, f"{m2_label}/{ano_sel}": COMPARE_COLOR_2})
        st.plotly_chart(fig_origem, use_container_width=True)

    with col_g2:
        st.subheader("Leads por Evento")
        e1 = df_m1.groupby("evento").size().reset_index(name="leads")
        e1["evento_legenda"] = e1["evento"].apply(label_evento)
        e1["mes"] = f"{m1_label}/{ano_sel}"
        e2 = df_m2.groupby("evento").size().reset_index(name="leads")
        e2["evento_legenda"] = e2["evento"].apply(label_evento)
        e2["mes"] = f"{m2_label}/{ano_sel}"
        conv_evento = pd.concat([e1, e2], ignore_index=True)
        fig_evento = px.bar(conv_evento, x="evento_legenda", y="leads", color="mes", barmode="group", color_discrete_map={f"{m1_label}/{ano_sel}": COMPARE_COLOR_1, f"{m2_label}/{ano_sel}": COMPARE_COLOR_2})
        st.plotly_chart(fig_evento, use_container_width=True)

    # Dispositivo e Horário (abaixo de Origem e Evento)
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.subheader("Leads por Dispositivo")
        d1 = df_m1.groupby("dispositivo").size().reset_index(name="leads")
        d1["mes"] = f"{m1_label}/{ano_sel}"
        d2 = df_m2.groupby("dispositivo").size().reset_index(name="leads")
        d2["mes"] = f"{m2_label}/{ano_sel}"
        conv_disp = pd.concat([d1, d2], ignore_index=True)
        fig_disp = px.bar(conv_disp, x="dispositivo", y="leads", color="mes", barmode="group", color_discrete_map={f"{m1_label}/{ano_sel}": COMPARE_COLOR_1, f"{m2_label}/{ano_sel}": COMPARE_COLOR_2})
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário dos Leads")
        h1 = df_m1.groupby("hora").size().reset_index(name="leads")
        h1["mes"] = f"{m1_label}/{ano_sel}"
        h2 = df_m2.groupby("hora").size().reset_index(name="leads")
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

                with st.spinner(f"Sincronizando oportunidades da {company['nome']}..."):
                    sync_result = sync_opportunities_with_smark(company_slug)
                    load_sheet.clear()
                    st.session_state["sync_message"] = (
                        f"CSV enviado com sucesso ({upload_result['rows']} linhas) para {company['nome']}. "
                        f"Matches site: {sync_result['site_matches']}. "
                        f"Matches Instagram: {sync_result['instagram_matches']}. "
                        f"Oportunidades regravadas: {sync_result['opportunities_added']}."
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

st.sidebar.markdown("## Filtros")

dfs = {name: load_sheet(company_slug, name) for name in SHEETS_REQUIRED}
for name in OPTIONAL_SHEETS:
    dfs[name] = load_sheet(company_slug, name)

df_leads = dfs["leads_site"]
df_opportunities = dfs["oportunidades"]

render_company_switcher(company_slug)

if df_leads.empty:
    st.warning(f"Nenhum dado encontrado na aba 'leads_site' da planilha {company['nome']}.")
    st.stop()

PERIODOS = ["Hoje", "Ontem", "Últimos 7 dias", "Este mês", "Este ano", "Personalizado", "Comparar meses"]
st.session_state.setdefault("periodo_sel", "Últimos 7 dias")
st.session_state.setdefault("periodo_sel_prev", st.session_state["periodo_sel"])

periodo_sel = st.sidebar.radio(label="", options=PERIODOS, key="periodo_sel")
if periodo_sel != st.session_state.get("periodo_sel_prev"):
    st.session_state["periodo_sel_prev"] = periodo_sel
    trigger_sheet_reload()

hoje = get_today_local()
ontem = hoje - timedelta(days=1)
compare_mode = False

df_periodo_leads = get_period_filtered_df(df_leads, periodo_sel, hoje, ontem)
df_periodo_sessions = get_period_filtered_df(dfs["sessions"], periodo_sel, hoje, ontem)
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
        trigger_sheet_reload()

    if st.session_state.get("custom_aplicado", False):
        if custom_mes_label == "Todo o ano":
            df_periodo_leads = df_leads[df_leads["ano"] == custom_ano].copy()
            df_periodo_sessions = dfs["sessions"][dfs["sessions"]["ano"] == custom_ano].copy()
            df_periodo_opportunities = df_opportunities[df_opportunities["ano"] == custom_ano].copy() if not df_opportunities.empty else df_opportunities
        else:
            mes_num_sel = month_label_to_num(custom_mes_label)
            df_periodo_leads = df_leads[(df_leads["ano"] == custom_ano) & (df_leads["mes"] == mes_num_sel)].copy()
            df_periodo_sessions = dfs["sessions"][(dfs["sessions"]["ano"] == custom_ano) & (dfs["sessions"]["mes"] == mes_num_sel)].copy()
            if not df_opportunities.empty:
                df_periodo_opportunities = df_opportunities[(df_opportunities["ano"] == custom_ano) & (df_opportunities["mes"] == mes_num_sel)].copy()
            else:
                df_periodo_opportunities = df_opportunities
    else:
        df_periodo_leads = get_period_filtered_df(df_leads, "Últimos 7 dias", hoje, ontem)
        df_periodo_sessions = get_period_filtered_df(dfs["sessions"], "Últimos 7 dias", hoje, ontem)
        df_periodo_opportunities = get_period_filtered_df(df_opportunities, "Últimos 7 dias", hoje, ontem)

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
        trigger_sheet_reload()

if not compare_mode:
    if df_periodo_leads.empty:
        st.warning("Nenhum Lead encontrado para o período selecionado.")
        st.stop()

st.sidebar.header("Filtros adicionais")
if compare_mode and st.session_state.get("compare_aplicado", False):
    m1_num = month_label_to_num(st.session_state["compare_mes1_label"])
    m2_num = month_label_to_num(st.session_state["compare_mes2_label"])
    df_filter_frames = [
        filter_by_year_month(df_leads, st.session_state["compare_ano"], m1_num),
        filter_by_year_month(df_leads, st.session_state["compare_ano"], m2_num),
    ]
    df_instagram_full = dfs.get("leads_instagram", pd.DataFrame())
    if company_slug == "nextqs" and not df_instagram_full.empty:
        df_filter_frames.extend(
            [
                filter_by_year_month(df_instagram_full, st.session_state["compare_ano"], m1_num),
                filter_by_year_month(df_instagram_full, st.session_state["compare_ano"], m2_num),
            ]
        )
    df_for_filters = pd.concat(df_filter_frames, ignore_index=True).sort_values("data_hora")
else:
    df_filter_frames = [df_periodo_leads]
    df_periodo_leads_instagram_sidebar = get_period_filtered_df(dfs.get("leads_instagram", pd.DataFrame()), periodo_sel, hoje, ontem)
    if company_slug == "nextqs" and not df_periodo_leads_instagram_sidebar.empty:
        df_filter_frames.append(df_periodo_leads_instagram_sidebar)
    df_for_filters = pd.concat(df_filter_frames, ignore_index=True).sort_values("data_hora")

eventos = sorted(df_for_filters["evento"].dropna().unique().tolist()) if "evento" in df_for_filters.columns else []
eventos_sel = st.sidebar.multiselect("Tipo de evento", options=eventos, default=eventos)
origens = sorted(df_for_filters["origem"].dropna().unique().tolist()) if "origem" in df_for_filters.columns else []
origens_sel = st.sidebar.multiselect("Origem", options=origens, default=origens)
dispositivos = sorted(df_for_filters["dispositivo"].dropna().unique().tolist()) if "dispositivo" in df_for_filters.columns else []
dispositivos_sel = st.sidebar.multiselect("Dispositivo", options=dispositivos, default=dispositivos)

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
        df_periodo_sessions,
        df_periodo_opportunities,
        df_opportunities,  # df_opp_full para negócios efetuados
        eventos_sel,
        origens_sel,
        dispositivos_sel,
        periodo_sel,
        hoje,
        ontem,
    )
