import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

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
# FUNÇÃO PARA NORMALIZAR false/undefined/etc
# (SEM ALTERAR MAIÚSCULA/MINÚSCULA ORIGINAL)
# -----------------------------
def normalize_empty(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    low = text.lower()
    if low in ["false", "none", "null", "undefined", "", "nan"]:
        return None
    return text  # devolve o texto ORIGINAL


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
        # st.secrets["gcp_service_account"] vem do [gcp_service_account] no secrets.toml
        service_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(
            service_info,
            scopes=SCOPES,
        )
    else:
        # opcional: só funciona se você tiver o JSON local na máquina
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
    for col in ["origem", "dispositivo", "ip_address", "utm_campaign"]:
        if col not in df.columns:
            df[col] = None

    # normaliza textos "sujos" (SEM mudar formatação original boa)
    df["origem"] = df["origem"].apply(normalize_empty)
    df["dispositivo"] = df["dispositivo"].apply(normalize_empty)
    df["ip_address"] = df["ip_address"].apply(normalize_empty)
    df["utm_campaign"] = df["utm_campaign"].apply(normalize_empty)

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
    ]
    for c in colunas_base:
        if c not in df.columns:
            df[c] = None

    df = df[colunas_base]

    return df


# -----------------------------
# DASHBOARD
# -----------------------------
df = load_all_data()

st.title("📊 Relatório de Leads no Site NextQS")

if df.empty:
    st.warning("Nenhum dado encontrado na planilha do Google Sheets.")
    st.stop()

st.caption(
    f"Período disponível: {df['data'].min()} até {df['data'].max()} "
    f"({df['ano'].min()} - {df['ano'].max()})"
)

# -----------------------------
# SIDEBAR: FILTROS DE ANO / MÊS
# -----------------------------
st.sidebar.header("Filtros de período")

anos_disponiveis = sorted(df["ano"].unique())
ano_sel = st.sidebar.selectbox("Ano", options=anos_disponiveis, index=len(anos_disponiveis) - 1)

meses_disponiveis = sorted(df[df["ano"] == ano_sel]["mes"].unique())
opcoes_meses = ["Todo o ano"] + [MESES_LABEL[m] for m in meses_disponiveis]
mes_label_sel = st.sidebar.selectbox("Mês", options=opcoes_meses, index=0)

if mes_label_sel == "Todo o ano":
    df_periodo = df[df["ano"] == ano_sel].copy()
else:
    mes_num_sel = [k for k, v in MESES_LABEL.items() if v == mes_label_sel][0]
    df_periodo = df[(df["ano"] == ano_sel) & (df["mes"] == mes_num_sel)].copy()

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
    dispositivo_top = dist_dispositivos.idxmax()
    pct_top = dist_dispositivos.iloc[0]
else:
    dispositivo_top = "-"
    pct_top = 0.0

# KPIs em colunas com a coluna 3 mais larga
col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

with col1:
    st.metric("Leads no período", conv_total)

with col2:
    st.metric("Leads únicos", usuarios_unicos)

with col3:
    st.metric("Origem mais comum", origem_top)

with col4:
    st.metric("Desktop (%)", f"{pct_top:.1f}%", delta=f"{dispositivo_top} mais comum")

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
# RANKING: CAMPANHAS COM MAIS CONVERSÕES
# -----------------------------
st.subheader("Ranking de Campanhas (utm_campaign)")

df_campanhas = df_filtrado[df_filtrado["utm_campaign"] != "Campanha não identificada"]

if not df_campanhas.empty:
    ranking_campanhas = (
        df_campanhas.groupby("utm_campaign")
        .size()
        .reset_index(name="leads")
        .sort_values("leads", ascending=False)
    )

    fig_campanhas = px.bar(
        ranking_campanhas,
        x="utm_campaign",
        y="leads",
        title="Campanhas com mais conversões",
    )
    fig_campanhas.update_layout(
        xaxis_title="Campanha",
        yaxis_title="Leads",
    )
    st.plotly_chart(fig_campanhas, use_container_width=True)
else:
    st.info("Nenhuma campanha válida encontrada no período.")

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
