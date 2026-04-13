
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

SHEETS_REQUIRED = [
    "leads_site",
    "sessions",
]

OPTIONAL_SHEETS = [
    "oportunidades",
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
    text = normalize_text(value)
    if not text:
        return ""

    match = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:\s+.*)?$", text)
    if match:
        first, second, year = match.groups()
        return f"{int(second):02d}/{int(first):02d}/{year}"
    return text


def clean_status_funil(value) -> str:
    text = normalize_text(value)
    return re.sub(r"^\s*\d+\s*-\s*", "", text)


def sync_opportunities_with_smark(company_slug: str) -> dict:
    client = get_gspread_client()

    base_spreadsheet = client.open_by_key(COMPANIES[company_slug]["base_spreadsheet_id"])
    leads_ws = base_spreadsheet.worksheet("leads_site")
    opportunities_ws = get_or_create_worksheet(base_spreadsheet, OPPORTUNITIES_SHEET_NAME)

    smark_spreadsheet = client.open_by_key(SMARK_SPREADSHEET_ID)
    smark_ws = get_worksheet_by_gid(smark_spreadsheet, SMARK_WORKSHEET_GID)

    leads_records = leads_ws.get_all_records()
    smark_records = smark_ws.get_all_records()

    opp_required_headers = [
        "data_lead",
        "canal",
        "user_id",
        "oportunidade",
        "data_oportunidade",
        "consultor",
        "status_funil",
        "data_encerramento",
    ]

    if not leads_records:
        opportunities_ws.clear()
        opportunities_ws.update("A1:H1", [opp_required_headers])
        return {
            "matches": 0,
            "qualified_sim": 0,
            "qualified_duplicate": 0,
            "opportunities_added": 0,
            "opportunities_skipped": 0,
            "leads_rows": 0,
            "smark_rows": len(smark_records),
        }

    df_leads = pd.DataFrame(leads_records)
    df_smark = pd.DataFrame(smark_records)

    base_email_col = "email" if "email" in df_leads.columns else "user_id_email" if "user_id_email" in df_leads.columns else None
    if base_email_col is None:
        raise ValueError("A planilha base precisa ter a coluna 'email' ou 'user_id_email'.")

    if "user_id_email" not in df_leads.columns:
        raise ValueError("A aba 'leads_site' precisa ter a coluna 'user_id_email' para preencher 'user_id' em oportunidades.")

    required_smark_columns = [
        SMARK_EMAIL_COLUMN,
        "Cod. Oportunidade",
        "Data Oportunidade",
        "Nome Colaborador Responsável",
        "Funil de Venda",
        "Data Encerramento",
    ]
    missing_smark = [col for col in required_smark_columns if col not in df_smark.columns]
    if missing_smark:
        raise ValueError("A planilha do SMARK não possui as colunas obrigatórias: " + ", ".join(missing_smark))

    ensure_column_exists(leads_ws, BASE_QUALIFIED_COLUMN)
    ensure_headers_exist(opportunities_ws, opp_required_headers)

    smark_map = {}
    for row in smark_records:
        email_norm = normalize_email(row.get(SMARK_EMAIL_COLUMN))
        if email_norm and email_norm not in smark_map:
            smark_map[email_norm] = row

    leads_headers = leads_ws.row_values(1)
    qualified_col_idx = leads_headers.index(BASE_QUALIFIED_COLUMN) + 1

    qualified_values = []
    opportunities_rows = []
    qualified_sim = 0
    qualified_duplicate = 0
    opportunities_added = 0
    opportunities_skipped = 0
    matches = 0
    processed_match_emails = set()
    generated_user_ids = set()

    for row in leads_records:
        email_norm = normalize_email(row.get(base_email_col))
        target_value = ""

        if email_norm and email_norm in smark_map:
            matches += 1
            if email_norm not in processed_match_emails:
                target_value = "SIM"
                processed_match_emails.add(email_norm)
            else:
                target_value = "Duplicado"

            if target_value == "SIM":
                qualified_sim += 1
                user_id_value = normalize_text(row.get("user_id_email"))
                user_id_key = user_id_value.lower()

                if not user_id_value or user_id_key in generated_user_ids:
                    opportunities_skipped += 1
                else:
                    smark_row = smark_map[email_norm]
                    opportunities_rows.append([
                        format_date_br_from_any(row.get("data_hora")),
                        "Site",
                        user_id_value,
                        normalize_text(smark_row.get("Cod. Oportunidade")),
                        format_smark_swapped_date_to_br(smark_row.get("Data Oportunidade")),
                        normalize_text(smark_row.get("Nome Colaborador Responsável")),
                        clean_status_funil(smark_row.get("Funil de Venda")),
                        format_smark_swapped_date_to_br(smark_row.get("Data Encerramento")),
                    ])
                    generated_user_ids.add(user_id_key)
                    opportunities_added += 1
            else:
                qualified_duplicate += 1

        qualified_values.append([target_value])

    if qualified_values:
        qualified_col_letter = gspread.utils.rowcol_to_a1(1, qualified_col_idx)[:-1]
        qualified_range = f"{qualified_col_letter}2:{qualified_col_letter}{len(qualified_values) + 1}"
        leads_ws.update(qualified_range, qualified_values, value_input_option="USER_ENTERED")

    opportunities_ws.clear()
    opportunities_payload = [opp_required_headers] + opportunities_rows
    opportunities_ws.update(
        f"A1:{gspread.utils.rowcol_to_a1(len(opportunities_payload), len(opp_required_headers))}",
        opportunities_payload,
        value_input_option="USER_ENTERED",
    )

    return {
        "matches": matches,
        "qualified_sim": qualified_sim,
        "qualified_duplicate": qualified_duplicate,
        "opportunities_added": opportunities_added,
        "opportunities_skipped": opportunities_skipped,
        "leads_rows": len(df_leads),
        "smark_rows": len(df_smark),
        "worksheet_smark": smark_ws.title,
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

    text_inside = [f"{v}<br>{p:.1%}" for v, p in zip(values_real, pct_initial)]
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
        font=dict(size=14),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return fig


def get_opportunities_count(df_opportunities: pd.DataFrame, origens_sel, dispositivos_sel, leads_base: pd.DataFrame) -> int:
    if df_opportunities.empty:
        return 0

    leads_filtered = apply_common_filters(leads_base, origens_sel, dispositivos_sel)
    if leads_filtered.empty or "user_id_email" not in leads_filtered.columns or "user_id" not in df_opportunities.columns:
        return len(df_opportunities)

    valid_user_ids = set(
        leads_filtered["user_id_email"].dropna().astype(str).str.strip().str.lower().tolist()
    )
    opp_df = df_opportunities.copy()
    opp_df["user_id_normalizado"] = opp_df["user_id"].astype(str).str.strip().str.lower()
    opp_df = opp_df[opp_df["user_id_normalizado"].isin(valid_user_ids)]
    return len(opp_df)


def compute_central_funnel_counts(df_sessions, df_leads, df_opportunities, origens_sel, dispositivos_sel):
    sessions_filtered = apply_common_filters(df_sessions, origens_sel, dispositivos_sel)
    conversions_filtered = apply_common_filters(df_leads, origens_sel, dispositivos_sel)
    opportunities_count = get_opportunities_count(df_opportunities, origens_sel, dispositivos_sel, df_leads)
    return len(sessions_filtered), len(conversions_filtered), opportunities_count


def render_central_funnel(df_sessions, df_leads, df_opportunities, origens_sel, dispositivos_sel, title="Funil Central"):
    sessions_count, conversions_count, opportunities_count = compute_central_funnel_counts(
        df_sessions,
        df_leads,
        df_opportunities,
        origens_sel,
        dispositivos_sel,
    )

    steps = [
        ("Sessões", sessions_count),
        ("Conversões", conversions_count),
        ("Oportunidades", opportunities_count),
    ]
    st.subheader(title)
    fig = build_funnel_figure("Sessões → Conversões → Oportunidades", steps, base_color="#22c55e")
    st.plotly_chart(fig, use_container_width=True)


def render_normal_mode(df_periodo_leads, df_periodo_sessions, df_periodo_opportunities, eventos_sel, origens_sel, dispositivos_sel, periodo_sel):
    df_filtrado = apply_extra_filters_leads(df_periodo_leads, eventos_sel, origens_sel, dispositivos_sel)

    if df_filtrado.empty:
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
    render_central_funnel(df_periodo_sessions, df_periodo_leads, df_periodo_opportunities, origens_sel, dispositivos_sel)
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

    col_rank1, col_rank2 = st.columns(2)
    with col_rank1:
        st.markdown("### Campanhas")
        df_c = df_filtrado[df_filtrado["utm_campaign"] != "Campanha não identificada"]
        if df_c.empty:
            st.info("Nenhuma campanha válida encontrada no período filtrado.")
        else:
            ranking_campanhas = (
                df_c.groupby("utm_campaign")
                .size()
                .reset_index(name="Conversões")
                .sort_values("Conversões", ascending=False)
                .rename(columns={"utm_campaign": "Campanha"})
            )
            st.dataframe(ranking_campanhas, use_container_width=True, height=400)

    with col_rank2:
        st.markdown("### Termos de Pesquisa")
        df_t = df_filtrado[df_filtrado["utm_term"] != "Palavra-chave não identificada"]
        if df_t.empty:
            st.info("Nenhuma palavra-chave válida encontrada no período filtrado.")
        else:
            ranking_terms = (
                df_t.groupby("utm_term")
                .size()
                .reset_index(name="Conversões")
                .sort_values("Conversões", ascending=False)
                .rename(columns={"utm_term": "Palavra-Chave"})
            )
            st.dataframe(ranking_terms, use_container_width=True, height=400)

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
        fig_hora = px.bar(conv_hora, x="hora", y="leads")
        fig_hora.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads")
        st.plotly_chart(fig_hora, use_container_width=True)

    st.markdown("---")
    st.subheader("Dados detalhados (após filtros)")
    detail_columns = [c for c in ["data_hora", "evento", "dispositivo", "origem", "user_id_email", "ip_address"] if c in df_filtrado.columns]
    st.dataframe(df_filtrado[detail_columns], use_container_width=True)


def render_compare_mode(dfs, df_leads, df_opportunities, eventos_sel, origens_sel, dispositivos_sel, ano_sel, m1_label, m2_label):
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
    st.subheader("Funil Central")
    tab_f1, tab_f2 = st.tabs([f"{m1_label}/{ano_sel}", f"{m2_label}/{ano_sel}"])

    with tab_f1:
        render_central_funnel(
            filter_by_year_month(dfs["sessions"], ano_sel, m1_num),
            df_m1_leads_base,
            df_m1_opp,
            origens_sel,
            dispositivos_sel,
            title=f"Funil Central - {m1_label}/{ano_sel}",
        )

    with tab_f2:
        render_central_funnel(
            filter_by_year_month(dfs["sessions"], ano_sel, m2_num),
            df_m2_leads_base,
            df_m2_opp,
            origens_sel,
            dispositivos_sel,
            title=f"Funil Central - {m2_label}/{ano_sel}",
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


company = get_selected_company()
company_slug = company["slug"]

logo_path = Path(company["logo_path"])
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)

sync_clicked = st.sidebar.button("Atualizar Oportunidades", use_container_width=True)
refresh_clicked = st.sidebar.button("Atualizar Dashboard", use_container_width=True)

if refresh_clicked:
    trigger_sheet_reload()

if sync_clicked:
    with st.spinner(f"Atualizando oportunidades da {company['nome']} a partir do SMARK..."):
        try:
            sync_result = sync_opportunities_with_smark(company_slug)
            load_sheet.clear()
            st.session_state["sync_message"] = (
                f"Atualização concluída para {company['nome']}. Matches: {sync_result['matches']}. "
                f"Qualificados com SIM: {sync_result['qualified_sim']}. "
                f"Marcados como Duplicado: {sync_result['qualified_duplicate']}. "
                f"Oportunidades regravadas: {sync_result['opportunities_added']}. "
                f"Ignoradas por user_id duplicado no processamento: {sync_result['opportunities_skipped']}."
            )
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Erro na atualização de oportunidades: {e}")

if st.session_state.get("sync_message"):
    st.sidebar.success(st.session_state["sync_message"])
    st.session_state["sync_message"] = None

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
    df_for_filters = pd.concat(
        [
            filter_by_year_month(df_leads, st.session_state["compare_ano"], m1_num),
            filter_by_year_month(df_leads, st.session_state["compare_ano"], m2_num),
        ],
        ignore_index=True,
    ).sort_values("data_hora")
else:
    df_for_filters = df_periodo_leads

eventos = sorted(df_for_filters["evento"].dropna().unique().tolist()) if "evento" in df_for_filters.columns else []
eventos_sel = st.sidebar.multiselect("Tipo de evento", options=eventos, default=eventos)
origens = sorted(df_for_filters["origem"].dropna().unique().tolist()) if "origem" in df_for_filters.columns else []
origens_sel = st.sidebar.multiselect("Origem", options=origens, default=origens)
dispositivos = sorted(df_for_filters["dispositivo"].dropna().unique().tolist()) if "dispositivo" in df_for_filters.columns else []
dispositivos_sel = st.sidebar.multiselect("Dispositivo", options=dispositivos, default=dispositivos)

if compare_mode and st.session_state.get("compare_aplicado", False):
    render_compare_mode(
        dfs,
        df_leads,
        df_opportunities,
        eventos_sel,
        origens_sel,
        dispositivos_sel,
        int(st.session_state["compare_ano"]),
        st.session_state["compare_mes1_label"],
        st.session_state["compare_mes2_label"],
    )
else:
    render_normal_mode(
        df_periodo_leads,
        df_periodo_sessions,
        df_periodo_opportunities,
        eventos_sel,
        origens_sel,
        dispositivos_sel,
        periodo_sel,
    )
