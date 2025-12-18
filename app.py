import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------------
# PROTEÇÃO POR SENHA
# -----------------------------
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
    layout="wide"
)

# -----------------------------
# MAPAS AUXILIARES
# -----------------------------
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

# -----------------------------
# FUNÇÕES UTILITÁRIAS
# -----------------------------
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
    load_all_data.clear()
    st.rerun()


def month_label_to_num(mes_label: str) -> int:
    return [k for k, v in MESES_LABEL.items() if v == mes_label][0]


def filter_by_year_month(df: pd.DataFrame, ano: int, mes_num: int) -> pd.DataFrame:
    return df[(df["ano"] == int(ano)) & (df["mes"] == int(mes_num))].copy()


def apply_extra_filters(df_base: pd.DataFrame, eventos_sel, origens_sel, dispositivos_sel) -> pd.DataFrame:
    df_out = df_base[
        (df_base["evento"].isin(eventos_sel))
        & (df_base["origem"].isin(origens_sel))
        & (df_base["dispositivo"].isin(dispositivos_sel))
    ].copy()
    return df_out.sort_values("data_hora")


def get_kpis(df_kpi: pd.DataFrame):
    conv_total = len(df_kpi)
    usuarios_unicos = df_kpi["user_id_email"].nunique() if not df_kpi.empty else 0

    origem_top = (
        df_kpi["origem"].value_counts().idxmax()
        if not df_kpi.empty
        else "-"
    )

    if conv_total > 0:
        dist_dispositivos = df_kpi["dispositivo"].value_counts(normalize=True) * 100
        dispositivo_top = dist_dispositivos.index[0]  # já vem em ordem desc
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


# -----------------------------
# CARREGAR DADOS DO GOOGLE SHEETS (PRIVADO)
# -----------------------------
@st.cache_data
def load_all_data():
    # ID da sua planilha e nome da aba
    SPREADSHEET_ID = "1M_yYBJxtwbzdleT2VDNcQfe0lSXxDX0hNe7bGm7xKUQ"
    SHEET_NAME = "eventos"

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    # usa secrets no Streamlit Cloud; usa arquivo local se estiver rodando na máquina
    if "gcp_service_account" in st.secrets:
        service_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(
            service_info,
            scopes=SCOPES,
        )
    else:
        creds = Credentials.from_service_account_file(
            "credenciais_sheets.json",
            scopes=SCOPES,
        )

    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    if df.empty:
        return df

    # garante colunas importantes
    for col in ["origem", "dispositivo", "ip_address", "utm_campaign", "utm_term"]:
        if col not in df.columns:
            df[col] = None

    # normaliza textos "sujos" (SEM mudar formatação original boa)
    df["origem"] = df["origem"].apply(normalize_empty)
    df["dispositivo"] = df["dispositivo"].apply(normalize_empty)
    df["ip_address"] = df["ip_address"].apply(normalize_empty)
    df["utm_campaign"] = df["utm_campaign"].apply(normalize_empty)
    df["utm_term"] = df["utm_term"].apply(normalize_empty)

    # remove lixos específicos em utm_campaign
    df["utm_campaign"] = df["utm_campaign"].replace(
        ["{campaignname}", "(not set)", "(notset)"],
        None
    )

    # aplica valores substitutos
    df["origem"] = df["origem"].fillna("Origem não identificada")
    df["dispositivo"] = df["dispositivo"].fillna("Dispositivo não identificado")
    df["ip_address"] = df["ip_address"].fillna("IP não identificado")
    df["utm_campaign"] = df["utm_campaign"].fillna("Campanha não identificada")
    df["utm_term"] = df["utm_term"].fillna("Palavra-chave não identificada")

    # DISPOSITIVO: aqui sim padronizamos inicial maiúscula
    df["dispositivo"] = df["dispositivo"].apply(
        lambda x: x if x == "Dispositivo não identificado" else str(x).capitalize()
    )

    # Ajuste para manter "iOS" em maiúsculas
    df["dispositivo"] = df["dispositivo"].replace({"Mobile - ios": "Mobile - iOS"})

    # converte data_hora (formato: 04/12/2025 - 14:45:58)
    df["data_hora"] = pd.to_datetime(
        df["data_hora"],
        format="%d/%m/%Y - %H:%M:%S",
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(subset=["data_hora"])

    # colunas derivadas de data
    df["data"] = df["data_hora"].dt.date
    df["ano"] = df["data_hora"].dt.year.astype(int)
    df["mes"] = df["data_hora"].dt.month.astype(int)
    df["hora"] = df["data_hora"].dt.hour
    df["dia_semana"] = df["data_hora"].dt.dayofweek.map(DIA_SEMANA_LABEL)

    # organiza colunas finais
    colunas_base = [
        "data_hora",
        "data",
        "ano",
        "mes",
        "hora",
        "dia_semana",
        "evento",
        "ip_address",
        "dispositivo",
        "origem",
        "user_id_email",
        "utm_campaign",
        "utm_term",
    ]
    for c in colunas_base:
        if c not in df.columns:
            df[c] = None

    df = df[colunas_base]

    return df


# -----------------------------
# SIDEBAR: FILTROS
# -----------------------------
st.sidebar.markdown("## Filtros")

# Carrega dados
df = load_all_data()

st.title("📊 Relatório de Leads no Site NextQS")

if df.empty:
    st.warning("Nenhum dado encontrado na planilha do Google Sheets.")
    st.stop()

# Período disponível
st.caption(
    f"Período disponível: {df['data'].min()} até {df['data'].max()} "
    f"({df['ano'].min()} - {df['ano'].max()})"
)

# Defaults / estado
PERIODOS = ["Hoje", "Ontem", "Últimos 7 dias", "Este mês", "Este ano", "Personalizado", "Comparar meses"]

st.session_state.setdefault("periodo_sel", "Últimos 7 dias")
st.session_state.setdefault("periodo_sel_prev", st.session_state["periodo_sel"])

# Rádio (bolinhas)
periodo_sel = st.sidebar.radio(
    label="",
    options=PERIODOS,
    key="periodo_sel",
)

# Quando mudar qualquer opção (inclusive entrar em Personalizado/Comparar meses),
# faz a releitura da planilha (mesmo do botão).
if periodo_sel != st.session_state.get("periodo_sel_prev"):
    st.session_state["periodo_sel_prev"] = periodo_sel
    trigger_sheet_reload()

# -----------------------------
# APLICAR FILTRO DE PERÍODO
# -----------------------------
hoje = get_today_local()
ontem = hoje - timedelta(days=1)

df_periodo = df.copy()
compare_mode = False

if periodo_sel == "Hoje":
    df_periodo = df[df["data"] == hoje].copy()

elif periodo_sel == "Ontem":
    df_periodo = df[df["data"] == ontem].copy()

elif periodo_sel == "Últimos 7 dias":
    # EXEMPLO do usuário: 12/12 -> 05/12 a 11/12 (exclui hoje)
    start = hoje - timedelta(days=7)
    end = ontem
    df_periodo = df[(df["data"] >= start) & (df["data"] <= end)].copy()

elif periodo_sel == "Este mês":
    start = date(hoje.year, hoje.month, 1)
    end = ontem
    df_periodo = df[(df["data"] >= start) & (df["data"] <= end)].copy()

elif periodo_sel == "Este ano":
    start = date(hoje.year, 1, 1)
    end = hoje
    df_periodo = df[(df["data"] >= start) & (df["data"] <= end)].copy()

elif periodo_sel == "Personalizado":
    # "setinha": expander
    with st.sidebar.expander("Personalizado", expanded=True):
        anos_disponiveis = sorted(df["ano"].unique())
        ano_default = anos_disponiveis[-1] if anos_disponiveis else hoje.year

        if "custom_ano" not in st.session_state:
            st.session_state["custom_ano"] = ano_default
        if "custom_mes_label" not in st.session_state:
            st.session_state["custom_mes_label"] = "Todo o ano"

        custom_ano = st.selectbox(
            "Ano",
            options=anos_disponiveis,
            index=anos_disponiveis.index(st.session_state["custom_ano"])
            if st.session_state["custom_ano"] in anos_disponiveis
            else len(anos_disponiveis) - 1,
            key="custom_ano",
        )

        meses_disponiveis = sorted(df[df["ano"] == custom_ano]["mes"].unique())
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

    # Só aplica quando clicar "Aplicar"
    if "custom_aplicado" not in st.session_state:
        st.session_state["custom_aplicado"] = False

    if aplicar:
        st.session_state["custom_aplicado"] = True
        trigger_sheet_reload()

    if st.session_state.get("custom_aplicado", False):
        if custom_mes_label == "Todo o ano":
            df_periodo = df[df["ano"] == custom_ano].copy()
        else:
            mes_num_sel = month_label_to_num(custom_mes_label)
            df_periodo = df[(df["ano"] == custom_ano) & (df["mes"] == mes_num_sel)].copy()
    else:
        # Antes do primeiro "Aplicar", mantém o padrão (Últimos 7 dias)
        start = hoje - timedelta(days=7)
        end = ontem
        df_periodo = df[(df["data"] >= start) & (df["data"] <= end)].copy()

elif periodo_sel == "Comparar meses":
    compare_mode = True

    with st.sidebar.expander("Comparar meses", expanded=True):
        anos_disponiveis = sorted(df["ano"].unique())
        ano_default = anos_disponiveis[-1] if anos_disponiveis else hoje.year

        if "compare_ano" not in st.session_state:
            st.session_state["compare_ano"] = ano_default

        compare_ano = st.selectbox(
            "Ano",
            options=anos_disponiveis,
            index=anos_disponiveis.index(st.session_state["compare_ano"])
            if st.session_state["compare_ano"] in anos_disponiveis
            else len(anos_disponiveis) - 1,
            key="compare_ano",
        )

        meses_disponiveis = sorted(df[df["ano"] == compare_ano]["mes"].unique())
        if not meses_disponiveis:
            st.warning("Não há meses disponíveis para o ano selecionado.")
            st.stop()

        opcoes_meses = [MESES_LABEL[m] for m in meses_disponiveis]

        # Defaults (últimos 2 meses disponíveis)
        if "compare_mes1_label" not in st.session_state:
            st.session_state["compare_mes1_label"] = opcoes_meses[-1]
        if "compare_mes2_label" not in st.session_state:
            st.session_state["compare_mes2_label"] = opcoes_meses[-2] if len(opcoes_meses) >= 2 else opcoes_meses[-1]

        # Cores fixas (boas para contraste em fundo claro)
        M1_COLOR = "#2563EB"  # azul
        M2_COLOR = "#F97316"  # laranja

        # Mês 1 (label + cor + seletor na mesma linha)
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

        # Mês 2 (label + cor + seletor na mesma linha)
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

    if "compare_aplicado" not in st.session_state:
        st.session_state["compare_aplicado"] = False

    if aplicar_compare:
        st.session_state["compare_aplicado"] = True
        trigger_sheet_reload()

    if not st.session_state.get("compare_aplicado", False):
        # Antes do primeiro "Aplicar", mantém o padrão (Últimos 7 dias)
        start = hoje - timedelta(days=7)
        end = ontem
        df_periodo = df[(df["data"] >= start) & (df["data"] <= end)].copy()

# Proteção para casos sem dados no período (modo normal)
if not compare_mode:
    if df_periodo.empty:
        st.warning("Nenhum Lead encontrado para o período selecionado.")
        st.stop()

    st.caption(
        f"Período filtrado: {df_periodo['data'].min()} até {df_periodo['data'].max()}"
    )

# -----------------------------
# FILTROS EXTRAS (EVENTO / ORIGEM / DISPOSITIVO)
# -----------------------------
st.sidebar.header("Filtros adicionais")

# Em modo comparar, os filtros adicionais precisam considerar os 2 meses juntos
if compare_mode and st.session_state.get("compare_aplicado", False):
    m1_num = month_label_to_num(st.session_state["compare_mes1_label"])
    m2_num = month_label_to_num(st.session_state["compare_mes2_label"])
    df_m1_base = filter_by_year_month(df, st.session_state["compare_ano"], m1_num)
    df_m2_base = filter_by_year_month(df, st.session_state["compare_ano"], m2_num)
    df_union = pd.concat([df_m1_base, df_m2_base], ignore_index=True)
    df_union = df_union.sort_values("data_hora")
    df_for_filters = df_union
else:
    df_for_filters = df_periodo

eventos = sorted(df_for_filters["evento"].dropna().unique().tolist())
eventos_sel = st.sidebar.multiselect(
    "Tipo de evento", options=eventos, default=eventos
)

origens = sorted(df_for_filters["origem"].dropna().unique().tolist())
origens_sel = st.sidebar.multiselect(
    "Origem", options=origens, default=origens
)

dispositivos = sorted(df_for_filters["dispositivo"].dropna().unique().tolist())
dispositivos_sel = st.sidebar.multiselect(
    "Dispositivo", options=dispositivos, default=dispositivos
)

# -----------------------------
# CONSTRUÇÃO DO DATASET FINAL (NORMAL x COMPARAR)
# -----------------------------
GREEN_COLOR = "#22c55e"

if compare_mode and st.session_state.get("compare_aplicado", False):
    # Cores fixas (mantém igual ao sidebar)
    M1_COLOR = "#2563EB"  # azul
    M2_COLOR = "#F97316"  # laranja

    ano_sel = int(st.session_state["compare_ano"])
    m1_label = st.session_state["compare_mes1_label"]
    m2_label = st.session_state["compare_mes2_label"]
    m1_num = month_label_to_num(m1_label)
    m2_num = month_label_to_num(m2_label)

    df_m1 = apply_extra_filters(df_m1_base, eventos_sel, origens_sel, dispositivos_sel)
    df_m2 = apply_extra_filters(df_m2_base, eventos_sel, origens_sel, dispositivos_sel)

    if df_m1.empty and df_m2.empty:
        st.warning("Nenhum Lead encontrado para os meses selecionados (após filtros).")
        st.stop()

    # Cabeçalho de período (comparação)
    min1, max1 = (df_m1["data"].min(), df_m1["data"].max()) if not df_m1.empty else ("-", "-")
    min2, max2 = (df_m2["data"].min(), df_m2["data"].max()) if not df_m2.empty else ("-", "-")
    st.caption(
        f"Comparação: {m1_label}/{ano_sel} ({min1} a {max1})  vs  {m2_label}/{ano_sel} ({min2} a {max2})"
    )

    # -----------------------------
    # KPIs (comparação)
    # -----------------------------
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

    # -----------------------------
    # GRÁFICO: LEADS POR DIA ou POR HORA (2 LINHAS)
    # -----------------------------
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

    # -----------------------------
    # LINHA 2: ORIGEM x EVENTO (COMPARAÇÃO)
    # -----------------------------
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Leads por Origem")

        o1 = df_m1.groupby("origem").size().reset_index(name="leads")
        o1["mes"] = f"{m1_label}/{ano_sel}"
        o2 = df_m2.groupby("origem").size().reset_index(name="leads")
        o2["mes"] = f"{m2_label}/{ano_sel}"

        conv_origem = pd.concat([o1, o2], ignore_index=True)
        conv_origem = conv_origem.sort_values("leads", ascending=False)

        fig_origem = px.bar(conv_origem, x="origem", y="leads", color="mes", barmode="group",
                            color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR})
        fig_origem.update_layout(xaxis_title="Origem", yaxis_title="Leads", legend_title_text="")
        st.plotly_chart(fig_origem, use_container_width=True)

    with col_g2:
        st.subheader("Leads por Evento")

        def label_evento(v):
            v_str = str(v).strip().lower()
            if "whats" in v_str:
                return "WhatsApp"
            if "form" in v_str:
                return "Formulário"
            return str(v).title()

        e1 = df_m1.groupby("evento").size().reset_index(name="leads")
        e1["evento_legenda"] = e1["evento"].apply(label_evento)
        e1["mes"] = f"{m1_label}/{ano_sel}"

        e2 = df_m2.groupby("evento").size().reset_index(name="leads")
        e2["evento_legenda"] = e2["evento"].apply(label_evento)
        e2["mes"] = f"{m2_label}/{ano_sel}"

        conv_evento = pd.concat([e1, e2], ignore_index=True)

        fig_evento = px.bar(conv_evento, x="evento_legenda", y="leads", color="mes", barmode="group",
                            color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR})
        fig_evento.update_layout(xaxis_title="Evento", yaxis_title="Leads", legend_title_text="")
        st.plotly_chart(fig_evento, use_container_width=True)

    # -----------------------------
    # RANKING: CAMPANHAS + PALAVRAS-CHAVE (LISTAS) - COMPARAÇÃO
    # -----------------------------
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

    # -----------------------------
    # LINHA 3: DISPOSITIVO x HORA (COMPARAÇÃO)
    # -----------------------------
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.subheader("Leads por Dispositivo")

        d1 = df_m1.groupby("dispositivo").size().reset_index(name="leads")
        d1["mes"] = f"{m1_label}/{ano_sel}"
        d2 = df_m2.groupby("dispositivo").size().reset_index(name="leads")
        d2["mes"] = f"{m2_label}/{ano_sel}"

        conv_disp = pd.concat([d1, d2], ignore_index=True)
        fig_disp = px.bar(conv_disp, x="dispositivo", y="leads", color="mes", barmode="group",
                          color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR})
        fig_disp.update_layout(xaxis_title="Dispositivo", yaxis_title="Leads", legend_title_text="")
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário dos Leads")

        h1 = df_m1.groupby("hora").size().reset_index(name="leads")
        h1["mes"] = f"{m1_label}/{ano_sel}"
        h2 = df_m2.groupby("hora").size().reset_index(name="leads")
        h2["mes"] = f"{m2_label}/{ano_sel}"

        conv_hora = pd.concat([h1, h2], ignore_index=True)
        conv_hora = conv_hora.sort_values("hora")

        fig_hora = px.bar(conv_hora, x="hora", y="leads", color="mes", barmode="group",
                          color_discrete_map={f"{m1_label}/{ano_sel}": M1_COLOR, f"{m2_label}/{ano_sel}": M2_COLOR})
        fig_hora.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads", legend_title_text="")
        st.plotly_chart(fig_hora, use_container_width=True)

    # -----------------------------
    # TABELA DETALHADA (COMPARAÇÃO)
    # -----------------------------
    st.markdown("---")
    st.subheader("Dados detalhados (após filtros)")

    tab1, tab2 = st.tabs([f"{m1_label}/{ano_sel}", f"{m2_label}/{ano_sel}"])

    with tab1:
        st.dataframe(
            df_m1[
                [
                    "data_hora",
                    "evento",
                    "dispositivo",
                    "origem",
                    "user_id_email",
                    "ip_address",
                ]
            ],
            use_container_width=True,
        )

    with tab2:
        st.dataframe(
            df_m2[
                [
                    "data_hora",
                    "evento",
                    "dispositivo",
                    "origem",
                    "user_id_email",
                    "ip_address",
                ]
            ],
            use_container_width=True,
        )

else:
    # -----------------------------
    # MODO NORMAL (SEM COMPARAÇÃO)
    # -----------------------------
    df_filtrado = apply_extra_filters(df_periodo, eventos_sel, origens_sel, dispositivos_sel)

    if df_filtrado.empty:
        st.warning("Nenhum Lead encontrado para o período selecionado (após filtros).")
        st.stop()

    # -----------------------------
    # KPIs
    # -----------------------------
    conv_total, usuarios_unicos, origem_top, dispositivo_top, pct_top = get_kpis(df_filtrado)

    # KPIs em colunas com a coluna 3 mais larga
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        render_kpi_single("Leads no período", f"{conv_total}", GREEN_COLOR)

    with col2:
        render_kpi_single("Leads únicos", f"{usuarios_unicos}", GREEN_COLOR)

    with col3:
        render_kpi_single("Origem mais comum", f"{origem_top}", GREEN_COLOR)

    with col4:
        st.text(f"{dispositivo_top} (%)")
        st.markdown(
            kpi_value_html(f"{pct_top:.1f}%", GREEN_COLOR),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -----------------------------
    # GRÁFICO: LEADS POR DIA ou POR HORA (quando Hoje/Ontem)
    # -----------------------------
    if periodo_sel in ["Hoje", "Ontem"]:
        st.subheader("Leads por Hora")

        conv_por_hora_dia = df_filtrado.groupby("hora").size().reset_index(name="leads")
        conv_por_hora_dia = conv_por_hora_dia.sort_values("hora")

        fig_hora_dia = px.bar(conv_por_hora_dia, x="hora", y="leads")
        fig_hora_dia.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads", xaxis=dict(dtick=1))
        st.plotly_chart(fig_hora_dia, use_container_width=True)

    else:
        st.subheader("Leads por Dia")

        conv_por_dia = df_filtrado.groupby("data").size().reset_index(name="leads")
        fig_dia = px.line(conv_por_dia, x="data", y="leads")
        fig_dia.update_layout(xaxis_title="Data", yaxis_title="Leads")
        st.plotly_chart(fig_dia, use_container_width=True)

    # -----------------------------
    # RANKING DE MESES (APENAS QUANDO O PERÍODO FILTRADO FOR "ANO INTEIRO")
    # -----------------------------
    show_ranking_meses = False
    if periodo_sel == "Personalizado":
        if st.session_state.get("custom_aplicado", False) and st.session_state.get("custom_mes_label") == "Todo o ano":
            show_ranking_meses = True
    else:
        if periodo_sel in ["Este ano"]:
            show_ranking_meses = True

    if show_ranking_meses:
        st.subheader("Ordem dos Meses com mais Conversões (Leads únicos)")

        ranking_meses = (
            df_filtrado
            .groupby("mes")["user_id_email"]
            .nunique()
            .reset_index(name="leads_unicos")
        )

        if ranking_meses.empty:
            st.info("Nenhuma informação de leads únicos para montar o ranking de meses.")
        else:
            ranking_meses["mes_nome"] = ranking_meses["mes"].map(MESES_LABEL)
            ranking_meses = ranking_meses.sort_values("leads_unicos", ascending=False)

            fig_meses = px.bar(
                ranking_meses,
                x="mes_nome",
                y="leads_unicos",
            )
            fig_meses.update_layout(
                xaxis_title="Mês",
                yaxis_title="Leads únicos",
            )
            st.plotly_chart(fig_meses, use_container_width=True)

    # -----------------------------
    # LINHA 2: ORIGEM x EVENTO
    # -----------------------------
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Leads por Origem")
        conv_por_origem = (
            df_filtrado.groupby("origem").size().reset_index(name="leads")
        )
        conv_por_origem = conv_por_origem.sort_values("leads", ascending=False)
        fig_origem = px.bar(conv_por_origem, x="origem", y="leads")
        fig_origem.update_layout(xaxis_title="Origem", yaxis_title="Leads")
        st.plotly_chart(fig_origem, use_container_width=True)

    with col_g2:
        st.subheader("Leads por Evento")

        conv_por_evento = df_filtrado.groupby("evento").size().reset_index(name="leads")

        def label_evento(v):
            v_str = str(v).strip().lower()
            if "whats" in v_str:
                return "WhatsApp"
            if "form" in v_str:
                return "Formulário"
            return str(v).title()

        conv_por_evento["evento_legenda"] = conv_por_evento["evento"].apply(label_evento)

        fig_evento = px.bar(
            conv_por_evento,
            x="evento_legenda",
            y="leads",
            color="evento_legenda",
            color_discrete_map={
                "WhatsApp": "#25D366",
                "Formulário": "#FFA726",
            },
        )
        fig_evento.update_layout(xaxis_title="Evento", yaxis_title="Leads", showlegend=False)
        st.plotly_chart(fig_evento, use_container_width=True)

    # -----------------------------
    # RANKING: CAMPANHAS + PALAVRAS-CHAVE (LISTAS)
    # -----------------------------
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
            )
            ranking_campanhas = ranking_campanhas.rename(columns={"utm_campaign": "Campanha"})
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
            )
            ranking_terms = ranking_terms.rename(columns={"utm_term": "Palavra-Chave"})
            st.dataframe(ranking_terms, use_container_width=True, height=400)
        else:
            st.info("Nenhuma palavra-chave válida encontrada no período.")

    # -----------------------------
    # LINHA 3: DISPOSITIVO x HORA
    # -----------------------------
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.subheader("Leads por Dispositivo")
        conv_por_disp = (
            df_filtrado.groupby("dispositivo").size().reset_index(name="leads")
        )
        conv_por_disp = conv_por_disp.sort_values("leads", ascending=False)
        fig_disp = px.bar(conv_por_disp, x="dispositivo", y="leads")
        fig_disp.update_layout(xaxis_title="Dispositivo", yaxis_title="Leads")
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_g4:
        st.subheader("Horário das Leads")
        conv_por_hora = df_filtrado.groupby("hora").size().reset_index(name="leads")
        conv_por_hora = conv_por_hora.sort_values("hora")
        fig_hora = px.bar(conv_por_hora, x="hora", y="leads")
        fig_hora.update_layout(xaxis_title="Hora do dia", yaxis_title="Leads")
        st.plotly_chart(fig_hora, use_container_width=True)

    # -----------------------------
    # TABELA DETALHADA
    # -----------------------------
    st.markdown("---")
    st.subheader("Dados detalhados (após filtros)")

    st.dataframe(
        df_filtrado[
            [
                "data_hora",
                "evento",
                "dispositivo",
                "origem",
                "user_id_email",
                "ip_address",
            ]
        ],
        use_container_width=True,
    )
