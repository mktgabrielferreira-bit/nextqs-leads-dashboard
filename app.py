import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import re

# =============================================================================
# PROTEÇÃO POR SENHA
# =============================================================================
def login():
    st.title("🔒 Acesso restrito")
    senha = st.text_input("Digite a senha:", type="password")

    if st.button("Entrar"):
        if senha == st.secrets["SENHA_DASH"]:
            st.session_state["autenticado"] = True
        else:
            st.error("Senha incorreta")


if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

if not st.session_state["autenticado"]:
    login()
    st.stop()

st.set_page_config(
    page_title="Relatório de Leads NextQS",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# MAPAS AUXILIARES
# =============================================================================
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

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================
def normalize_empty(value):
    """Normaliza valores vazios/false/undefined/etc sem alterar o texto válido (maiúsc./minúsc.)."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    low = text.lower()
    if low in ["false", "none", "null", "undefined", "", "nan"]:
        return None
    return text  # devolve o texto ORIGINAL


def get_today_local() -> date:
    """Retorna a data de hoje considerando fuso BR (se disponível); senão usa data local do servidor."""
    try:
        return datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    except Exception:
        return date.today()


def trigger_sheet_reload():
    """Força releitura da planilha (mesmo efeito do botão 'Atualizar Informações')."""
    load_sheet.clear()
    st.rerun()


def month_label_to_num(mes_label: str) -> int:
    return [k for k, v in MESES_LABEL.items() if v == mes_label][0]


def filter_by_year_month(df: pd.DataFrame, ano: int, mes_num: int) -> pd.DataFrame:
    return df[(df["ano"] == int(ano)) & (df["mes"] == int(mes_num))].copy()


def apply_extra_filters_leads(df_base: pd.DataFrame, eventos_sel, origens_sel, dispositivos_sel) -> pd.DataFrame:
    """Filtros do relatório de leads."""
    df_out = df_base[
        (df_base["evento"].isin(eventos_sel))
        & (df_base["origem"].isin(origens_sel))
        & (df_base["dispositivo"].isin(dispositivos_sel))
    ].copy()
    return df_out.sort_values("data_hora")


def apply_common_filters(df_base: pd.DataFrame, origens_sel, dispositivos_sel) -> pd.DataFrame:
    """Filtros comuns (origem/dispositivo) para abas de funil."""
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


def render_kpi_dual(title: str, v1: str, v2: str, c1: str, c2: str):
    st.text(title)
    st.markdown(
        kpi_value_html(v1, c1) + kpi_value_html(v2, c2),
        unsafe_allow_html=True,
    )


def render_kpi_single(title: str, v: str, color: str):
    st.text(title)
    st.markdown(
        kpi_value_html(v, color),
        unsafe_allow_html=True,
    )


def label_evento(v):
    v_str = str(v).strip().lower()
    if "whats" in v_str:
        return "WhatsApp"
    if "form" in v_str:
        return "Formulário"
    return str(v).title()


def is_whatsapp_event(v) -> bool:
    return "whats" in str(v).strip().lower()


def is_form_event(v) -> bool:
    s = str(v).strip().lower()
    # "form" cobre "formulario", "formulário", etc.
    return "form" in s


# =============================================================================
# CARREGAR DADOS DO GOOGLE SHEETS (PRIVADO)
# =============================================================================
SPREADSHEET_ID = "1dw5ssrZu9UfzymB7GLs0rqZf0LggvKAnC5Tek3go1cM"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SMARK_SPREADSHEET_ID = "1sYMJcxzmDQ1r-KPAZ38TNlrQeg7NQMrk-Yk26pyzMBg"
SMARK_WORKSHEET_GID = 841055934
SMARK_EMAIL_COLUMN = "E-mail Contato"
BASE_QUALIFIED_COLUMN = "qualificado"
OPPORTUNITIES_SHEET_NAME = "oportunidades"

SHEETS_REQUIRED = [
    "leads_site",
    "sessions",
    "click_whatsapp",
    "click_formulario",
    "form_start_whatsapp",
    "form_start_formulario",
]


def _get_creds():
    # usa secrets no Streamlit Cloud; usa arquivo local se estiver rodando na máquina
    if "gcp_service_account" in st.secrets:
        service_info = dict(st.secrets["gcp_service_account"])
        return Credentials.from_service_account_info(service_info, scopes=SCOPES)
    return Credentials.from_service_account_file("credenciais_sheets.json", scopes=SCOPES)


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


def get_worksheet_by_gid(spreadsheet, gid: int):
    for ws in spreadsheet.worksheets():
        if getattr(ws, "id", None) == gid:
            return ws
    raise ValueError(f"Não foi possível localizar a aba com gid={gid} na planilha do SMARK.")


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

    for fmt in ["%d/%m/%Y - %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(text, fmt).strftime("%d/%m/%Y")
        except Exception:
            pass

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%d/%m/%Y")

    return text


def clean_status_funil(value) -> str:
    text = normalize_text(value)
    return re.sub(r"^\s*\d+\s*-\s*", "", text)


def sync_opportunities_with_smark() -> dict:
    creds = _get_creds()
    client = gspread.authorize(creds)

    base_spreadsheet = client.open_by_key(SPREADSHEET_ID)
    leads_ws = base_spreadsheet.worksheet("leads_site")
    opportunities_ws = base_spreadsheet.worksheet(OPPORTUNITIES_SHEET_NAME)

    smark_spreadsheet = client.open_by_key(SMARK_SPREADSHEET_ID)
    smark_ws = get_worksheet_by_gid(smark_spreadsheet, SMARK_WORKSHEET_GID)

    leads_records = leads_ws.get_all_records()
    smark_records = smark_ws.get_all_records()

    if not leads_records:
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

    if df_smark.empty:
        return {
            "matches": 0,
            "qualified_sim": 0,
            "qualified_duplicate": 0,
            "opportunities_added": 0,
            "opportunities_skipped": 0,
            "leads_rows": len(df_leads),
            "smark_rows": 0,
        }

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

    if BASE_QUALIFIED_COLUMN not in df_leads.columns:
        df_leads[BASE_QUALIFIED_COLUMN] = ""
    ensure_column_exists(leads_ws, BASE_QUALIFIED_COLUMN)

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
    ensure_headers_exist(opportunities_ws, opp_required_headers)

    opportunities_records = opportunities_ws.get_all_records()
    existing_user_ids = {
        normalize_text(row.get("user_id")).lower()
        for row in opportunities_records
        if normalize_text(row.get("user_id"))
    }

    smark_map = {}
    for row in smark_records:
        email_norm = normalize_email(row.get(SMARK_EMAIL_COLUMN))
        if email_norm and email_norm not in smark_map:
            smark_map[email_norm] = row

    leads_headers = leads_ws.row_values(1)
    qualified_col_idx = leads_headers.index(BASE_QUALIFIED_COLUMN) + 1

    leads_updates = []
    opportunities_to_append = []
    qualified_sim = 0
    qualified_duplicate = 0
    opportunities_added = 0
    opportunities_skipped = 0
    matches = 0
    processed_match_emails = set()

    for row_number, row in enumerate(leads_records, start=2):
        email_norm = normalize_email(row.get(base_email_col))
        if not email_norm or email_norm not in smark_map:
            continue

        matches += 1
        current_value = normalize_text(row.get(BASE_QUALIFIED_COLUMN)).upper()
        if email_norm not in processed_match_emails:
            target_value = "SIM"
            processed_match_emails.add(email_norm)
        else:
            target_value = "Duplicado"

        if current_value != target_value.upper():
            a1_ref = gspread.utils.rowcol_to_a1(row_number, qualified_col_idx)
            leads_updates.append({"range": a1_ref, "values": [[target_value]]})

        if target_value == "SIM":
            qualified_sim += 1
        else:
            qualified_duplicate += 1

        user_id_value = normalize_text(row.get("user_id_email"))
        user_id_key = user_id_value.lower()

        if not user_id_value or user_id_key in existing_user_ids:
            opportunities_skipped += 1
            continue

        smark_row = smark_map[email_norm]
        opportunities_to_append.append([
            format_date_br_from_any(row.get("data_hora")),
            "Site",
            user_id_value,
            normalize_text(smark_row.get("Cod. Oportunidade")),
            format_date_br_from_any(smark_row.get("Data Oportunidade")),
            normalize_text(smark_row.get("Nome Colaborador Responsável")),
            clean_status_funil(smark_row.get("Funil de Venda")),
            format_date_br_from_any(smark_row.get("Data Encerramento")),
        ])
        existing_user_ids.add(user_id_key)
        opportunities_added += 1

    if leads_updates:
        leads_ws.batch_update(leads_updates, value_input_option="USER_ENTERED")

    if opportunities_to_append:
        opportunities_ws.append_rows(opportunities_to_append, value_input_option="USER_ENTERED")

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
    """
    Tenta parsear 'data_hora' nos formatos:
    - 04/12/2025 - 14:45:58
    - 26/01/2026 - 10:12:13
    """
    if s is None or s.empty:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

    # tenta formato padrão do relatório
    out = pd.to_datetime(
        s,
        format="%d/%m/%Y - %H:%M:%S",
        dayfirst=True,
        errors="coerce",
    )
    # fallback: deixa o pandas tentar (mantém dayfirst)
    if out.isna().all():
        out = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return out


def _standardize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas comuns e normalização básica (sem 'forçar' maiúsculas/minúsculas)."""
    if df.empty:
        return df

    for col in ["origem", "dispositivo", "utm_campaign", "utm_term", "ip_address", "evento", "user_id_email"]:
        if col not in df.columns:
            df[col] = None

    # normaliza textos "sujos" (SEM mudar formatação original boa)
    for col in ["origem", "dispositivo", "utm_campaign", "utm_term", "ip_address", "evento", "user_id_email"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_empty)

    # remove lixos específicos
    if "utm_campaign" in df.columns:
        df["utm_campaign"] = df["utm_campaign"].replace(["{campaignname}", "(not set)", "(notset)"], None)

    # aplica valores substitutos (apenas quando col existe)
    if "origem" in df.columns:
        df["origem"] = df["origem"].fillna("Origem não identificada")
    if "dispositivo" in df.columns:
        df["dispositivo"] = df["dispositivo"].fillna("Dispositivo não identificado")
        df["dispositivo"] = df["dispositivo"].apply(
            lambda x: x if x == "Dispositivo não identificado" else str(x).capitalize()
        )
        df["dispositivo"] = df["dispositivo"].replace({"Mobile - ios": "Mobile - iOS"})
    if "ip_address" in df.columns:
        df["ip_address"] = df["ip_address"].fillna("IP não identificado")
    if "utm_campaign" in df.columns:
        df["utm_campaign"] = df["utm_campaign"].fillna("Campanha não identificada")
    if "utm_term" in df.columns:
        df["utm_term"] = df["utm_term"].fillna("Palavra-chave não identificada")

    return df


@st.cache_data
def load_sheet(sheet_name: str) -> pd.DataFrame:
    """
    Carrega uma aba do Google Sheets e padroniza:
    - data_hora -> datetime
    - data/ano/mes/hora/dia_semana
    - normalização de origem/dispositivo/etc quando existir
    """
    creds = _get_creds()
    client = gspread.authorize(creds)
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)

    data = ws.get_all_records()
    df = pd.DataFrame(data)

    if df.empty:
        return df

    # padroniza colunas comuns
    df = _standardize_common_columns(df)

    # garante data_hora
    if "data_hora" not in df.columns:
        # tenta alternativas comuns
        for alt in ["dataHora", "datetime", "timestamp", "data"]:
            if alt in df.columns:
                df["data_hora"] = df[alt]
                break
        if "data_hora" not in df.columns:
            df["data_hora"] = None

    df["data_hora"] = _parse_datetime_series(df["data_hora"])
    df = df.dropna(subset=["data_hora"])

    # colunas derivadas
    df["data"] = df["data_hora"].dt.date
    df["ano"] = df["data_hora"].dt.year.astype(int)
    df["mes"] = df["data_hora"].dt.month.astype(int)
    df["hora"] = df["data_hora"].dt.hour
    df["dia_semana"] = df["data_hora"].dt.dayofweek.map(DIA_SEMANA_LABEL)

    return df


def get_period_filtered_df(df_src: pd.DataFrame, periodo_sel: str, hoje: date, ontem: date) -> pd.DataFrame:
    """Aplica o filtro de período (normal) em qualquer DF que tenha coluna 'data'."""
    if df_src.empty:
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

    # Personalizado / Comparar meses são tratados fora
    return df_src.copy()


def build_funnel_figure(title: str, steps: list[tuple[str, int]], base_color: str, min_ratio: float = 0.18) -> go.Figure:
    """
    Funil visual (Plotly) com uma sensação "3D" (sombra/contorno).

    Importante: usamos uma escala visual com largura mínima (min_ratio) para
    manter as etapas pequenas visíveis, MAS os percentuais exibidos são sempre
    calculados a partir dos valores reais (e não da largura visual).
    """
    labels = [s[0] for s in steps]
    values_real = [max(0, int(s[1])) for s in steps]

    # -------------------------------------------------------------------------
    # 1) Escala visual (apenas para desenho)
    # -------------------------------------------------------------------------
    maxv = max(values_real) if values_real else 1
    minv = max(1, int(round(maxv * float(min_ratio))))

    scaled = []
    prev_scaled = maxv
    for v in values_real:
        # garante uma largura mínima, mas nunca "aumenta" depois que afunila
        target = max(v, minv)
        sv = min(prev_scaled, target)
        scaled.append(sv)
        prev_scaled = sv

    # -------------------------------------------------------------------------
    # 2) Percentuais reais (para texto e tooltip)
    # -------------------------------------------------------------------------
    pct_initial = [(v / maxv) if maxv else 0.0 for v in values_real]

    pct_prev = []
    prev_real = None
    for v in values_real:
        if prev_real in (None, 0):
            pct_prev.append(1.0 if v else 0.0)
        else:
            pct_prev.append(v / prev_real)
        prev_real = v

    # Texto dentro do funil (compacto)
    text_inside = [f"{v}<br>{p:.1%}" for v, p in zip(values_real, pct_initial)]

    customdata = [[v, p_i, p_p] for v, p_i, p_p in zip(values_real, pct_initial, pct_prev)]

    fig = go.Figure()
    fig.add_trace(
        go.Funnel(
            name=title,
            y=labels,
            x=scaled,                 # largura visual
            text=text_inside,         # exibição (usa valores reais)
            customdata=customdata,    # para tooltip (valores reais)
            texttemplate="%{text}",
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Quantidade: %{customdata[0]}<br>"
                "% do início: %{customdata[1]:.1%}<br>"
                "% do passo anterior: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
            marker=dict(
                color=base_color,
                line=dict(color="rgba(0,0,0,0.28)", width=2),
            ),
            connector=dict(line=dict(color="rgba(0,0,0,0.22)", width=1)),
            opacity=0.95,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=55, b=10),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
        uniformtext=dict(minsize=10, mode="hide"),  # evita texto ilegível em etapas pequenas
    )
    return fig

def compute_funnel_counts(
    df_sessions: pd.DataFrame,
    df_click: pd.DataFrame,
    df_form_start: pd.DataFrame,
    df_leads: pd.DataFrame,
    lead_event_filter_fn,
    origens_sel,
    dispositivos_sel,
) -> tuple[int, int, int, int]:
    s = apply_common_filters(df_sessions, origens_sel, dispositivos_sel)
    c = apply_common_filters(df_click, origens_sel, dispositivos_sel)
    f = apply_common_filters(df_form_start, origens_sel, dispositivos_sel)

    # conversões: linhas na aba leads_site cujo evento bate
    leads_base = apply_common_filters(df_leads, origens_sel, dispositivos_sel)
    if "evento" in leads_base.columns and not leads_base.empty:
        leads_conv = leads_base[leads_base["evento"].apply(lead_event_filter_fn)].copy()
    else:
        leads_conv = leads_base.iloc[0:0].copy()

    return len(s), len(c), len(f), len(leads_conv)


# =============================================================================
# SIDEBAR: FILTROS
# =============================================================================
sync_col_1, sync_col_2 = st.sidebar.columns([1, 1])
with sync_col_1:
    sync_clicked = st.button("Atualizar Oportunidades", use_container_width=True)
with sync_col_2:
    refresh_clicked = st.button("Atualizar Dados", use_container_width=True)

if refresh_clicked:
    trigger_sheet_reload()

if sync_clicked:
    with st.spinner("Atualizando oportunidades a partir do SMARK..."):
        try:
            sync_result = sync_opportunities_with_smark()
            load_sheet.clear()
            st.sidebar.success(
                f"Atualização concluída. Matches: {sync_result['matches']}. "
                f"Qualificados com SIM: {sync_result['qualified_sim']}. "
                f"Marcados como Duplicado: {sync_result['qualified_duplicate']}. "
                f"Novas oportunidades: {sync_result['opportunities_added']}. "
                f"Ignoradas por user_id já existente: {sync_result['opportunities_skipped']}."
            )
        except Exception as e:
            st.sidebar.error(f"Erro na atualização de oportunidades: {e}")

st.sidebar.markdown("## Filtros")

# Carrega abas necessárias
dfs = {name: load_sheet(name) for name in SHEETS_REQUIRED}
df_leads = dfs["leads_site"]

st.title("📊 Relatório de Leads no Site NextQS")

if df_leads.empty:
    st.warning("Nenhum dado encontrado na aba 'leads_site' do Google Sheets.")
    st.stop()

# Período disponível (baseado em leads)
st.caption(
    f"Período disponível (Leads): {df_leads['data'].min()} até {df_leads['data'].max()} "
    f"({df_leads['ano'].min()} - {df_leads['ano'].max()})"
)

# Defaults / estado
PERIODOS = ["Hoje", "Ontem", "Últimos 7 dias", "Este mês", "Este ano", "Personalizado", "Comparar meses"]

st.session_state.setdefault("periodo_sel", "Últimos 7 dias")
st.session_state.setdefault("periodo_sel_prev", st.session_state["periodo_sel"])

periodo_sel = st.sidebar.radio(
    label="",
    options=PERIODOS,
    key="periodo_sel",
)

# quando mudar qualquer opção, recarrega (para refletir dados mais recentes)
if periodo_sel != st.session_state.get("periodo_sel_prev"):
    st.session_state["periodo_sel_prev"] = periodo_sel
    trigger_sheet_reload()

# =============================================================================
# APLICAR FILTRO DE PERÍODO (LEADS + FUNIS)
# =============================================================================
hoje = get_today_local()
ontem = hoje - timedelta(days=1)

compare_mode = False

# Base leads do período
df_periodo_leads = get_period_filtered_df(df_leads, periodo_sel, hoje, ontem)

# Para funis: também filtramos cada aba pelo período selecionado (exceto comparar/personalizado)
df_periodo_sessions = get_period_filtered_df(dfs["sessions"], periodo_sel, hoje, ontem)
df_periodo_click_whatsapp = get_period_filtered_df(dfs["click_whatsapp"], periodo_sel, hoje, ontem)
df_periodo_click_formulario = get_period_filtered_df(dfs["click_formulario"], periodo_sel, hoje, ontem)
df_periodo_form_start_whatsapp = get_period_filtered_df(dfs["form_start_whatsapp"], periodo_sel, hoje, ontem)
df_periodo_form_start_formulario = get_period_filtered_df(dfs["form_start_formulario"], periodo_sel, hoje, ontem)

# Personalizado / Comparar meses: tratado abaixo
if periodo_sel == "Personalizado":
    with st.sidebar.expander("Personalizado", expanded=True):
        anos_disponiveis = sorted(df_leads["ano"].unique())
        ano_default = anos_disponiveis[-1] if anos_disponiveis else hoje.year

        st.session_state.setdefault("custom_ano", ano_default)
        st.session_state.setdefault("custom_mes_label", "Todo o ano")

        custom_ano = st.selectbox(
            "Ano",
            options=anos_disponiveis,
            index=anos_disponiveis.index(st.session_state["custom_ano"])
            if st.session_state["custom_ano"] in anos_disponiveis
            else len(anos_disponiveis) - 1,
            key="custom_ano",
        )

        meses_disponiveis = sorted(df_leads[df_leads["ano"] == custom_ano]["mes"].unique())
        opcoes_meses = ["Todo o ano"] + [MESES_LABEL[m] for m in meses_disponiveis]

        custom_mes_label = st.selectbox(
            "Mês",
            options=opcoes_meses,
            index=opcoes_meses.index(st.session_state["custom_mes_label"])
            if st.session_state["custom_mes_label"] in opcoes_meses
            else 0,
            key="custom_mes_label",
        )

        aplicar = st.button("Aplicar")

    st.session_state.setdefault("custom_aplicado", False)

    if aplicar:
        st.session_state["custom_aplicado"] = True
        trigger_sheet_reload()

    if st.session_state.get("custom_aplicado", False):
        if custom_mes_label == "Todo o ano":
            df_periodo_leads = df_leads[df_leads["ano"] == custom_ano].copy()
            # mesma regra para abas do funil
            df_periodo_sessions = dfs["sessions"][dfs["sessions"]["ano"] == custom_ano].copy()
            df_periodo_click_whatsapp = dfs["click_whatsapp"][dfs["click_whatsapp"]["ano"] == custom_ano].copy()
            df_periodo_click_formulario = dfs["click_formulario"][dfs["click_formulario"]["ano"] == custom_ano].copy()
            df_periodo_form_start_whatsapp = dfs["form_start_whatsapp"][dfs["form_start_whatsapp"]["ano"] == custom_ano].copy()
            df_periodo_form_start_formulario = dfs["form_start_formulario"][dfs["form_start_formulario"]["ano"] == custom_ano].copy()
        else:
            mes_num_sel = month_label_to_num(custom_mes_label)
            df_periodo_leads = df_leads[(df_leads["ano"] == custom_ano) & (df_leads["mes"] == mes_num_sel)].copy()

            def ym(df_):
                if df_.empty:
                    return df_
                return df_[(df_["ano"] == custom_ano) & (df_["mes"] == mes_num_sel)].copy()

            df_periodo_sessions = ym(dfs["sessions"])
            df_periodo_click_whatsapp = ym(dfs["click_whatsapp"])
            df_periodo_click_formulario = ym(dfs["click_formulario"])
            df_periodo_form_start_whatsapp = ym(dfs["form_start_whatsapp"])
            df_periodo_form_start_formulario = ym(dfs["form_start_formulario"])
    else:
        # antes do primeiro aplicar, mantém padrão (Últimos 7 dias)
        df_periodo_leads = get_period_filtered_df(df_leads, "Últimos 7 dias", hoje, ontem)
        df_periodo_sessions = get_period_filtered_df(dfs["sessions"], "Últimos 7 dias", hoje, ontem)
        df_periodo_click_whatsapp = get_period_filtered_df(dfs["click_whatsapp"], "Últimos 7 dias", hoje, ontem)
        df_periodo_click_formulario = get_period_filtered_df(dfs["click_formulario"], "Últimos 7 dias", hoje, ontem)
        df_periodo_form_start_whatsapp = get_period_filtered_df(dfs["form_start_whatsapp"], "Últimos 7 dias", hoje, ontem)
        df_periodo_form_start_formulario = get_period_filtered_df(dfs["form_start_formulario"], "Últimos 7 dias", hoje, ontem)

elif periodo_sel == "Comparar meses":
    compare_mode = True

    with st.sidebar.expander("Comparar meses", expanded=True):
        anos_disponiveis = sorted(df_leads["ano"].unique())
        ano_default = anos_disponiveis[-1] if anos_disponiveis else hoje.year
        st.session_state.setdefault("compare_ano", ano_default)

        compare_ano = st.selectbox(
            "Ano",
            options=anos_disponiveis,
            index=anos_disponiveis.index(st.session_state["compare_ano"])
            if st.session_state["compare_ano"] in anos_disponiveis
            else len(anos_disponiveis) - 1,
            key="compare_ano",
        )

        meses_disponiveis = sorted(df_leads[df_leads["ano"] == compare_ano]["mes"].unique())
        if not meses_disponiveis:
            st.warning("Não há meses disponíveis para o ano selecionado.")
            st.stop()

        opcoes_meses = [MESES_LABEL[m] for m in meses_disponiveis]

        st.session_state.setdefault("compare_mes1_label", opcoes_meses[-1])
        st.session_state.setdefault("compare_mes2_label", opcoes_meses[-2] if len(opcoes_meses) >= 2 else opcoes_meses[-1])

        M1_COLOR = "#2563EB"  # azul
        M2_COLOR = "#F97316"  # laranja

        col_label_1, col_select_1 = st.columns([1, 3])
        with col_label_1:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;'>"
                f"<strong>Mês 1</strong>"
                f"<span style='width:10px;height:10px;border-radius:50%;background:{M1_COLOR};display:inline-block;'></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_select_1:
            compare_mes1_label = st.selectbox(
                "",
                options=opcoes_meses,
                index=opcoes_meses.index(st.session_state["compare_mes1_label"])
                if st.session_state["compare_mes1_label"] in opcoes_meses
                else len(opcoes_meses) - 1,
                key="compare_mes1_label",
                label_visibility="collapsed",
            )

        col_label_2, col_select_2 = st.columns([1, 3])
        with col_label_2:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;'>"
                f"<strong>Mês 2</strong>"
                f"<span style='width:10px;height:10px;border-radius:50%;background:{M2_COLOR};display:inline-block;'></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_select_2:
            compare_mes2_label = st.selectbox(
                "",
                options=opcoes_meses,
                index=opcoes_meses.index(st.session_state["compare_mes2_label"])
                if st.session_state["compare_mes2_label"] in opcoes_meses
                else max(len(opcoes_meses) - 2, 0),
                key="compare_mes2_label",
                label_visibility="collapsed",
            )

        aplicar_compare = st.button("Aplicar")

    st.session_state.setdefault("compare_aplicado", False)

    if aplicar_compare:
        st.session_state["compare_aplicado"] = True
        trigger_sheet_reload()

    if not st.session_state.get("compare_aplicado", False):
        # Antes do primeiro "Aplicar", mantém o padrão (Últimos 7 dias)
        df_periodo_leads = get_period_filtered_df(df_leads, "Últimos 7 dias", hoje, ontem)
        df_periodo_sessions = get_period_filtered_df(dfs["sessions"], "Últimos 7 dias", hoje, ontem)
        df_periodo_click_whatsapp = get_period_filtered_df(dfs["click_whatsapp"], "Últimos 7 dias", hoje, ontem)
        df_periodo_click_formulario = get_period_filtered_df(dfs["click_formulario"], "Últimos 7 dias", hoje, ontem)
        df_periodo_form_start_whatsapp = get_period_filtered_df(dfs["form_start_whatsapp"], "Últimos 7 dias", hoje, ontem)
        df_periodo_form_start_formulario = get_period_filtered_df(dfs["form_start_formulario"], "Últimos 7 dias", hoje, ontem)

# Proteção para casos sem dados no período (modo normal)
if not compare_mode:
    if df_periodo_leads.empty:
        st.warning("Nenhum Lead encontrado para o período selecionado.")
        st.stop()

    st.caption(f"Período filtrado: {df_periodo_leads['data'].min()} até {df_periodo_leads['data'].max()}")

# =============================================================================
# FILTROS EXTRAS (EVENTO / ORIGEM / DISPOSITIVO)
# =============================================================================
st.sidebar.header("Filtros adicionais")

if compare_mode and st.session_state.get("compare_aplicado", False):
    m1_num = month_label_to_num(st.session_state["compare_mes1_label"])
    m2_num = month_label_to_num(st.session_state["compare_mes2_label"])
    df_m1_base = filter_by_year_month(df_leads, st.session_state["compare_ano"], m1_num)
    df_m2_base = filter_by_year_month(df_leads, st.session_state["compare_ano"], m2_num)
    df_union = pd.concat([df_m1_base, df_m2_base], ignore_index=True).sort_values("data_hora")
    df_for_filters = df_union
else:
    df_for_filters = df_periodo_leads

eventos = sorted(df_for_filters["evento"].dropna().unique().tolist()) if "evento" in df_for_filters.columns else []
eventos_sel = st.sidebar.multiselect("Tipo de evento", options=eventos, default=eventos)

origens = sorted(df_for_filters["origem"].dropna().unique().tolist()) if "origem" in df_for_filters.columns else []
origens_sel = st.sidebar.multiselect("Origem", options=origens, default=origens)

dispositivos = sorted(df_for_filters["dispositivo"].dropna().unique().tolist()) if "dispositivo" in df_for_filters.columns else []
dispositivos_sel = st.sidebar.multiselect("Dispositivo", options=dispositivos, default=dispositivos)

# =============================================================================
# DASH
# =============================================================================
GREEN_COLOR = "#22c55e"

def render_funnels_section(
    df_sessions_base: pd.DataFrame,
    df_click_whatsapp_base: pd.DataFrame,
    df_form_start_whatsapp_base: pd.DataFrame,
    df_click_formulario_base: pd.DataFrame,
    df_form_start_formulario_base: pd.DataFrame,
    df_leads_base: pd.DataFrame,
    origens_sel,
    dispositivos_sel,
    show_title: bool = True,
):
    if show_title:
        st.subheader("Funis de Conversão no Site")

    # WhatsApp
    w_sessions, w_clicks, w_starts, w_convs = compute_funnel_counts(
        df_sessions_base,
        df_click_whatsapp_base,
        df_form_start_whatsapp_base,
        df_leads_base,
        is_whatsapp_event,
        origens_sel,
        dispositivos_sel,
    )

    # Formulário
    f_sessions, f_clicks, f_starts, f_convs = compute_funnel_counts(
        df_sessions_base,
        df_click_formulario_base,
        df_form_start_formulario_base,
        df_leads_base,
        is_form_event,
        origens_sel,
        dispositivos_sel,
    )

    col_w, col_f = st.columns(2)

    with col_w:
        steps_w = [
            ("Sessões", w_sessions),
            ("Cliques no botão", w_clicks),
            ("Começaram a preencher", w_starts),
            ("Converteram", w_convs),
        ]
        fig_w = build_funnel_figure("WhatsApp", steps_w, base_color="#25D366")
        st.plotly_chart(fig_w, use_container_width=True)

    with col_f:
        steps_f = [
            ("Sessões", f_sessions),
            ("Cliques no botão", f_clicks),
            ("Começaram a preencher", f_starts),
            ("Converteram", f_convs),
        ]
        fig_f = build_funnel_figure("Formulário", steps_f, base_color="#F97316")
        st.plotly_chart(fig_f, use_container_width=True)


if compare_mode and st.session_state.get("compare_aplicado", False):
    # Cores fixas
    M1_COLOR = "#2563EB"
    M2_COLOR = "#F97316"

    ano_sel = int(st.session_state["compare_ano"])
    m1_label = st.session_state["compare_mes1_label"]
    m2_label = st.session_state["compare_mes2_label"]
    m1_num = month_label_to_num(m1_label)
    m2_num = month_label_to_num(m2_label)

    # Leads por mês (base)
    df_m1_leads_base = filter_by_year_month(df_leads, ano_sel, m1_num)
    df_m2_leads_base = filter_by_year_month(df_leads, ano_sel, m2_num)

    # Aplica filtros extras nos leads (inclui evento)
    df_m1 = apply_extra_filters_leads(df_m1_leads_base, eventos_sel, origens_sel, dispositivos_sel)
    df_m2 = apply_extra_filters_leads(df_m2_leads_base, eventos_sel, origens_sel, dispositivos_sel)

    if df_m1.empty and df_m2.empty:
        st.warning("Nenhum Lead encontrado para os meses selecionados (após filtros).")
        st.stop()

    min1, max1 = (df_m1["data"].min(), df_m1["data"].max()) if not df_m1.empty else ("-", "-")
    min2, max2 = (df_m2["data"].min(), df_m2["data"].max()) if not df_m2.empty else ("-", "-")
    st.caption(
        f"Comparação: {m1_label}/{ano_sel} ({min1} a {max1})  vs  {m2_label}/{ano_sel} ({min2} a {max2})"
    )

    conv_total_1, usuarios_unicos_1, origem_top_1, disp_top_1, pct_top_1 = get_kpis(df_m1)
    conv_total_2, usuarios_unicos_2, origem_top_2, disp_top_2, pct_top_2 = get_kpis(df_m2)

    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        render_kpi_dual("Leads no período", f"{conv_total_1}", f"{conv_total_2}", M1_COLOR, M2_COLOR)
    with col2:
        render_kpi_dual("Leads únicos", f"{usuarios_unicos_1}", f"{usuarios_unicos_2}", M1_COLOR, M2_COLOR)
    with col3:
        render_kpi_dual("Origem mais comum", f"{origem_top_1}", f"{origem_top_2}", M1_COLOR, M2_COLOR)
    with col4:
        st.text("Dispositivo (%)")
        st.markdown(
            kpi_value_html(f"{disp_top_1} {pct_top_1:.1f}%", M1_COLOR)
            + kpi_value_html(f"{disp_top_2} {pct_top_2:.1f}%", M2_COLOR),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # GRÁFICO: LEADS POR DIA ou POR HORA (2 LINHAS)
    if periodo_sel in ["Hoje", "Ontem"]:
        st.subheader("Leads por Hora")

        conv_hora_1 = df_m1.groupby("hora").size().reset_index(name="leads")
        conv_hora_2 = df_m2.groupby("hora").size().reset_index(name="leads")

        fig = go.Figure()
        if not conv_hora_1.empty:
            fig.add_trace(go.Scatter(
                x=conv_hora_1["hora"],
                y=conv_hora_1["leads"],
                mode="lines+markers",
                name=f"{m1_label}/{ano_sel}",
                line=dict(color=M1_COLOR),
            ))
        if not conv_hora_2.empty:
            fig.add_trace(go.Scatter(
                x=conv_hora_2["hora"],
                y=conv_hora_2["leads"],
                mode="lines+markers",
                name=f"{m2_label}/{ano_sel}",
                line=dict(color=M2_COLOR),
            ))

        fig.update_layout(
            xaxis_title="Hora do dia",
            yaxis_title="Leads",
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("Leads por Dia")

        conv_por_dia_1 = df_m1.groupby("data").size().reset_index(name="leads")
        conv_por_dia_2 = df_m2.groupby("data").size().reset_index(name="leads")

        fig = go.Figure()
        if not conv_por_dia_1.empty:
            fig.add_trace(go.Scatter(
                x=conv_por_dia_1["data"],
                y=conv_por_dia_1["leads"],
                mode="lines",
                name=f"{m1_label}/{ano_sel}",
                line=dict(color=M1_COLOR),
            ))
        if not conv_por_dia_2.empty:
            fig.add_trace(go.Scatter(
                x=conv_por_dia_2["data"],
                y=conv_por_dia_2["leads"],
                mode="lines",
                name=f"{m2_label}/{ano_sel}",
                line=dict(color=M2_COLOR),
            ))

        fig.update_layout(xaxis_title="Data", yaxis_title="Leads")
        st.plotly_chart(fig, use_container_width=True)

    # Funis: em tabs por mês, sempre usando leads do mês (MAS ignorando filtro de evento para conversão)
    st.subheader("Funis de Conversão no Site")
    st.markdown("### Funis (por mês)")
    tab_f1, tab_f2 = st.tabs([f"{m1_label}/{ano_sel}", f"{m2_label}/{ano_sel}"])

    def ym(df_, mes_num):
        if df_.empty:
            return df_
        return df_[(df_["ano"] == ano_sel) & (df_["mes"] == mes_num)].copy()

    with tab_f1:
        render_funnels_section(
            ym(dfs["sessions"], m1_num),
            ym(dfs["click_whatsapp"], m1_num),
            ym(dfs["form_start_whatsapp"], m1_num),
            ym(dfs["click_formulario"], m1_num),
            ym(dfs["form_start_formulario"], m1_num),
            df_m1_leads_base,  # base (sem filtro de evento)
            origens_sel,
            dispositivos_sel,
            show_title=False,
        )

    with tab_f2:
        render_funnels_section(
            ym(dfs["sessions"], m2_num),
            ym(dfs["click_whatsapp"], m2_num),
            ym(dfs["form_start_whatsapp"], m2_num),
            ym(dfs["click_formulario"], m2_num),
            ym(dfs["form_start_formulario"], m2_num),
            df_m2_leads_base,  # base (sem filtro de evento)
            origens_sel,
            dispositivos_sel,
            show_title=False,
        )

    st.markdown("---")

    
# LINHA 2: ORIGEM x EVENTO (COMPARAÇÃO)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Leads por Origem")

        o1 = df_m1.groupby("origem").size().reset_index(name="leads")
        o1["mes"] = f"{m1_label}/{ano_sel}"
        o2 = df_m2.groupby("origem").size().reset_index(name="leads")
        o2["mes"] = f"{m2_label}/{ano_sel}"

        conv_origem = pd.concat([o1, o2], ignore_index=True).sort_values("leads", ascending=False)

        fig_origem = px.bar(
            conv_origem,
            x="origem",
            y="leads",
            color="mes",
            barmode="group",
            color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR},
        )
        fig_origem.update_layout(xaxis_title="Origem", yaxis_title="Leads", legend_title_text="")
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

        fig_evento = px.bar(
            conv_evento,
            x="evento_legenda",
            y="leads",
            color="mes",
            barmode="group",
            color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR},
        )
        fig_evento.update_layout(xaxis_title="Evento", yaxis_title="Leads", legend_title_text="")
        st.plotly_chart(fig_evento, use_container_width=True)

    # RANKING: CAMPANHAS + PALAVRAS-CHAVE (LISTAS) - COMPARAÇÃO
    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("### Campanhas")

        def build_rank_campaigns(df_src: pd.DataFrame, label: str):
            df_c = df_src[df_src["utm_campaign"] != "Campanha não identificada"]
            if df_c.empty:
                return pd.DataFrame(columns=["Campanha", "Conversões", "Mês"])
            out = (
                df_c.groupby("utm_campaign")
                .size()
                .reset_index(name="Conversões")
                .sort_values("Conversões", ascending=False)
                .rename(columns={"utm_campaign": "Campanha"})
            )
            out["Mês"] = label
            return out

        rank_c1 = build_rank_campaigns(df_m1, f"{m1_label}/{ano_sel}")
        rank_c2 = build_rank_campaigns(df_m2, f"{m2_label}/{ano_sel}")
        ranking_campanhas = pd.concat([rank_c1, rank_c2], ignore_index=True)

        if ranking_campanhas.empty:
            st.info("Nenhuma campanha válida encontrada nos meses selecionados.")
        else:
            st.dataframe(ranking_campanhas, use_container_width=True, height=400)

    with col_rank2:
        st.markdown("### Termos de Pesquisa")

        def build_rank_terms(df_src: pd.DataFrame, label: str):
            df_t = df_src[df_src["utm_term"] != "Palavra-chave não identificada"]
            if df_t.empty:
                return pd.DataFrame(columns=["Palavra-Chave", "Conversões", "Mês"])
            out = (
                df_t.groupby("utm_term")
                .size()
                .reset_index(name="Conversões")
                .sort_values("Conversões", ascending=False)
                .rename(columns={"utm_term": "Palavra-Chave"})
            )
            out["Mês"] = label
            return out

        rank_t1 = build_rank_terms(df_m1, f"{m1_label}/{ano_sel}")
        rank_t2 = build_rank_terms(df_m2, f"{m2_label}/{ano_sel}")
        ranking_terms = pd.concat([rank_t1, rank_t2], ignore_index=True)

        if ranking_terms.empty:
            st.info("Nenhuma palavra-chave válida encontrada nos meses selecionados.")
        else:
            st.dataframe(ranking_terms, use_container_width=True, height=400)

    # LINHA 3: DISPOSITIVO x HORA (COMPARAÇÃO)
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.subheader("Leads por Dispositivo")

        d1 = df_m1.groupby("dispositivo").size().reset_index(name="leads")
        d1["mes"] = f"{m1_label}/{ano_sel}"
        d2 = df_m2.groupby("dispositivo").size().reset_index(name="leads")
        d2["mes"] = f"{m2_label}/{ano_sel}"

        conv_disp = pd.concat([d1, d2], ignore_index=True)
        fig_disp = px.bar(
            conv_disp,
            x="dispositivo",
            y="leads",
            color="mes",
            barmode="group",
            color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR},
        )
        fig_disp.update_layout(xaxis_title="Dispositivo", yaxis_title="Leads", legend_title_text="")
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário dos Leads")

        h1 = df_m1.groupby("hora").size().reset_index(name="leads")
        h1["mes"] = f"{m1_label}/{ano_sel}"
        h2 = df_m2.groupby("hora").size().reset_index(name="leads")
        h2["mes"] = f"{m2_label}/{ano_sel}"

        conv_hora = pd.concat([h1, h2], ignore_index=True).sort_values("hora")

        fig_hora = px.bar(
            conv_hora,
            x="hora",
            y="leads",
            color="mes",
            barmode="group",
            color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR},
        )
        fig_hora.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads", legend_title_text="")
        st.plotly_chart(fig_hora, use_container_width=True)

    # TABELA DETALHADA (COMPARAÇÃO)
    st.markdown("---")
    st.subheader("Dados detalhados (após filtros)")

    tab1, tab2 = st.tabs([f"{m1_label}/{ano_sel}", f"{m2_label}/{ano_sel}"])

    with tab1:
        st.dataframe(
            df_m1[["data_hora", "evento", "dispositivo", "origem", "user_id_email", "ip_address"]],
            use_container_width=True,
        )

    with tab2:
        st.dataframe(
            df_m2[["data_hora", "evento", "dispositivo", "origem", "user_id_email", "ip_address"]],
            use_container_width=True,
        )

else:
    # =============================================================================
    # MODO NORMAL (SEM COMPARAÇÃO)
    # =============================================================================
    df_filtrado = apply_extra_filters_leads(df_periodo_leads, eventos_sel, origens_sel, dispositivos_sel)

    if df_filtrado.empty:
        st.warning("Nenhum Lead encontrado para o período selecionado (após filtros).")
        st.stop()

    # KPIs
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


    # GRÁFICO: LEADS POR DIA ou POR HORA (quando Hoje/Ontem)
    if periodo_sel in ["Hoje", "Ontem"]:
        st.subheader("Leads por Hora")
        conv_por_hora_dia = df_filtrado.groupby("hora").size().reset_index(name="leads").sort_values("hora")
        fig_hora_dia = px.bar(conv_por_hora_dia, x="hora", y="leads")
        fig_hora_dia.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads", xaxis=dict(dtick=1))
        st.plotly_chart(fig_hora_dia, use_container_width=True)
    else:
        st.subheader("Leads por Dia")
        conv_por_dia = df_filtrado.groupby("data").size().reset_index(name="leads")
        fig_dia = px.line(conv_por_dia, x="data", y="leads")
        fig_dia.update_layout(xaxis_title="Data", yaxis_title="Leads")
        st.plotly_chart(fig_dia, use_container_width=True)

        st.markdown("---")

    # FUNIS (Sessões -> Conversões) — respeita Origem/Dispositivo e Período (conversão via leads por evento)
    render_funnels_section(
        df_periodo_sessions,
        df_periodo_click_whatsapp,
        df_periodo_form_start_whatsapp,
        df_periodo_click_formulario,
        df_periodo_form_start_formulario,
        df_periodo_leads,  # base sem filtro de evento (mas com período)
        origens_sel,
        dispositivos_sel,
        show_title=True,
    )

    st.markdown("---")

# RANKING DE MESES (APENAS QUANDO O PERÍODO FILTRADO FOR "ANO INTEIRO")
    show_ranking_meses = False
    if periodo_sel == "Personalizado":
        if st.session_state.get("custom_aplicado", False) and st.session_state.get("custom_mes_label") == "Todo o ano":
            show_ranking_meses = True
    else:
        if periodo_sel in ["Este ano"]:
            show_ranking_meses = True

    if show_ranking_meses:
        st.subheader("Ordem dos Meses com mais Conversões (Leads únicos)")
        ranking_meses = df_filtrado.groupby("mes")["user_id_email"].nunique().reset_index(name="leads_unicos")

        if ranking_meses.empty:
            st.info("Nenhuma informação de leads únicos para montar o ranking de meses.")
        else:
            ranking_meses["mes_nome"] = ranking_meses["mes"].map(MESES_LABEL)
            ranking_meses = ranking_meses.sort_values("leads_unicos", ascending=False)
            fig_meses = px.bar(ranking_meses, x="mes_nome", y="leads_unicos")
            fig_meses.update_layout(xaxis_title="Mês", yaxis_title="Leads únicos")
            st.plotly_chart(fig_meses, use_container_width=True)

    # LINHA 2: ORIGEM x EVENTO
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Leads por Origem")
        conv_por_origem = df_filtrado.groupby("origem").size().reset_index(name="leads").sort_values("leads", ascending=False)
        fig_origem = px.bar(conv_por_origem, x="origem", y="leads")
        fig_origem.update_layout(xaxis_title="Origem", yaxis_title="Leads")
        st.plotly_chart(fig_origem, use_container_width=True)

    with col_g2:
        st.subheader("Leads por Evento")
        conv_por_evento = df_filtrado.groupby("evento").size().reset_index(name="leads")
        conv_por_evento["evento_legenda"] = conv_por_evento["evento"].apply(label_evento)

        fig_evento = px.bar(
            conv_por_evento,
            x="evento_legenda",
            y="leads",
            color="evento_legenda",
            color_discrete_map={"WhatsApp": "#25D366", "Formulário": "#FFA726"},
        )
        fig_evento.update_layout(xaxis_title="Evento", yaxis_title="Leads", showlegend=False)
        st.plotly_chart(fig_evento, use_container_width=True)

    # RANKING: CAMPANHAS + PALAVRAS-CHAVE (LISTAS)
    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("### Campanhas")
        df_campanhas = df_filtrado[df_filtrado["utm_campaign"] != "Campanha não identificada"]

        if not df_campanhas.empty:
            ranking_campanhas = (
                df_campanhas.groupby("utm_campaign")
                .size()
                .reset_index(name="Conversões")
                .sort_values("Conversões", ascending=False)
                .rename(columns={"utm_campaign": "Campanha"})
            )
            st.dataframe(ranking_campanhas, use_container_width=True, height=400)
        else:
            st.info("Nenhuma campanha válida encontrada no período.")

    with col_rank2:
        st.markdown("### Termos de Pesquisa")
        df_terms = df_filtrado[df_filtrado["utm_term"] != "Palavra-chave não identificada"]

        if not df_terms.empty:
            ranking_terms = (
                df_terms.groupby("utm_term")
                .size()
                .reset_index(name="Conversões")
                .sort_values("Conversões", ascending=False)
                .rename(columns={"utm_term": "Palavra-Chave"})
            )
            st.dataframe(ranking_terms, use_container_width=True, height=400)
        else:
            st.info("Nenhuma palavra-chave válida encontrada no período.")

    # LINHA 3: DISPOSITIVO x HORA
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.subheader("Leads por Dispositivo")
        conv_por_disp = df_filtrado.groupby("dispositivo").size().reset_index(name="leads").sort_values("leads", ascending=False)
        fig_disp = px.bar(conv_por_disp, x="dispositivo", y="leads")
        fig_disp.update_layout(xaxis_title="Dispositivo", yaxis_title="Leads")
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário das Leads")
        conv_por_hora = df_filtrado.groupby("hora").size().reset_index(name="leads").sort_values("hora")
        fig_hora = px.bar(conv_por_hora, x="hora", y="leads")
        fig_hora.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads")
        st.plotly_chart(fig_hora, use_container_width=True)

    # TABELA DETALHADA
    st.markdown("---")
    st.subheader("Dados detalhados (após filtros)")
    st.dataframe(
        df_filtrado[["data_hora", "evento", "dispositivo", "origem", "user_id_email", "ip_address"]],
        use_container_width=True,
    )
