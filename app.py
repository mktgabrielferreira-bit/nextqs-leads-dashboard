import streamlit as st
import pandas as pd
import plotly.express as px
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
# SIDEBAR: ATUALIZAR + FILTROS DE PERÍODO (NOVO)
# -----------------------------
# Botão "Atualizar Informações" (como já está)
if st.sidebar.button("🔄 Atualizar Informações"):
    trigger_sheet_reload()

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
PERIODOS = ["Hoje", "Ontem", "Últimos 7 dias", "Este mês", "Este ano", "Personalizado"]
if "periodo_sel" not in st.session_state:
    st.session_state["periodo_sel"] = "Últimos 7 dias"  # padrão ao abrir
if "periodo_sel_prev" not in st.session_state:
    st.session_state["periodo_sel_prev"] = st.session_state["periodo_sel"]

# Rádio (bolinhas)
periodo_sel = st.sidebar.radio(
    label="",
    options=PERIODOS,
    index=PERIODOS.index(st.session_state["periodo_sel"]),
    key="periodo_sel",
)

# Quando mudar qualquer opção (inclusive entrar em Personalizado),
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
    end = ontem
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
            mes_num_sel = [k for k, v in MESES_LABEL.items() if v == custom_mes_label][0]
            df_periodo = df[(df["ano"] == custom_ano) & (df["mes"] == mes_num_sel)].copy()
    else:
        # Antes do primeiro "Aplicar", mantém o padrão (Últimos 7 dias)
        start = hoje - timedelta(days=7)
        end = ontem
        df_periodo = df[(df["data"] >= start) & (df["data"] <= end)].copy()

# Proteção para casos sem dados no período
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

eventos = sorted(df_periodo["evento"].dropna().unique().tolist())
eventos_sel = st.sidebar.multiselect(
    "Tipo de evento", options=eventos, default=eventos
)

origens = sorted(df_periodo["origem"].dropna().unique().tolist())
origens_sel = st.sidebar.multiselect(
    "Origem", options=origens, default=origens
)

dispositivos = sorted(df_periodo["dispositivo"].dropna().unique().tolist())
dispositivos_sel = st.sidebar.multiselect(
    "Dispositivo", options=dispositivos, default=dispositivos
)

df_filtrado = df_periodo[
    (df_periodo["evento"].isin(eventos_sel))
    & (df_periodo["origem"].isin(origens_sel))
    & (df_periodo["dispositivo"].isin(dispositivos_sel))
].copy()

df_filtrado = df_filtrado.sort_values("data_hora")

# -----------------------------
# KPIs
# -----------------------------
conv_total = len(df_filtrado)
usuarios_unicos = df_filtrado["user_id_email"].nunique()
origem_top = (
    df_filtrado["origem"].value_counts().idxmax()
    if not df_filtrado.empty
    else "-"
)

if conv_total > 0:
    dist_dispositivos = df_filtrado["dispositivo"].value_counts(normalize=True) * 100
    top_disp = dist_dispositivos.index[0]
    pct_top = float(dist_dispositivos.iloc[0])
else:
    top_disp = "Dispositivo"
    pct_top = 0.0

# KPIs em colunas com a coluna 3 mais larga
col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

GREEN_COLOR = "#22c55e"

with col1:
    st.text("Leads no período")
    st.markdown(
        f"<span style='font-size:32px; font-weight:bold; color:{GREEN_COLOR}'>{conv_total}</span>",
        unsafe_allow_html=True,
    )

with col2:
    st.text("Leads únicos")
    st.markdown(
        f"<span style='font-size:32px; font-weight:bold; color:{GREEN_COLOR}'>{usuarios_unicos}</span>",
        unsafe_allow_html=True,
    )

with col3:
    st.text("Origem mais comum")
    st.markdown(
        f"<span style='font-size:32px; font-weight:bold; color:{GREEN_COLOR}'>{origem_top}</span>",
        unsafe_allow_html=True,
    )

with col4:
    st.text(f"{top_disp} (%)")
    st.markdown(
        f"<span style='font-size:32px; font-weight:bold; color:{GREEN_COLOR}'>{pct_top:.1f}%</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# -----------------------------
# GRÁFICO: LEADS POR DIA
# -----------------------------
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
