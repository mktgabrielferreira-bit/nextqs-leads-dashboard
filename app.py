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
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}

DIA_SEMANA_LABEL = {
    0: "Segunda", 1: "Terça", 2: "Quarta",
    3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo",
}

def normalize_empty(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    low = text.lower()
    if low in ["false", "none", "null", "undefined", "", "nan"]:
        return None
    return text


# -----------------------------
# CARREGAR DADOS DO GOOGLE SHEETS
# -----------------------------
@st.cache_data
def load_all_data():
    SPREADSHEET_ID = "1M_yYBJxtwbzdleT2VDNcQfe0lSXxDX0hNe7bGm7xKUQ"
    SHEET_NAME = "eventos"
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    if "gcp_service_account" in st.secrets:
        service_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(service_info, scopes=SCOPES)
    else:
        creds = Credentials.from_service_account_file("credenciais_sheets.json", scopes=SCOPES)

    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    df = pd.DataFrame(sheet.get_all_records())

    for col in ["origem", "dispositivo", "ip_address", "utm_campaign", "utm_term"]:
        if col not in df.columns:
            df[col] = None

    df["utm_campaign"] = df["utm_campaign"].apply(normalize_empty).fillna("Campanha não identificada")
    df["utm_term"] = df["utm_term"].apply(normalize_empty).fillna("Palavra-chave não identificada")

    df["data_hora"] = pd.to_datetime(df["data_hora"], format="%d/%m/%Y - %H:%M:%S", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["data_hora"])

    df["data"] = df["data_hora"].dt.date
    df["ano"] = df["data_hora"].dt.year.astype(int)
    df["mes"] = df["data_hora"].dt.month.astype(int)
    df["hora"] = df["data_hora"].dt.hour
    df["dia_semana"] = df["data_hora"].dt.dayofweek.map(DIA_SEMANA_LABEL)

    return df


if st.sidebar.button("🔄 Atualizar Informações"):
    load_all_data.clear()
    st.rerun()

df = load_all_data()

st.title("📊 Relatório de Leads no Site NextQS")

st.caption(f"Período disponível: {df['data'].min()} até {df['data'].max()}")

# -----------------------------
# ✅ NOVO BLOCO: LISTAS UTM
# -----------------------------
st.markdown("---")
st.subheader("Ranking de Campanhas e Palavras-Chaves")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📌 Ranking de Campanhas (utm_campaign)")
    ranking_campaign = (
        df[df["utm_campaign"] != "Campanha não identificada"]
        .groupby("utm_campaign")
        .size()
        .reset_index(name="Leads")
        .sort_values("Leads", ascending=False)
    )
    st.dataframe(ranking_campaign, use_container_width=True, height=400)

with col2:
    st.markdown("### 🔑 Ranking de Palavras-Chaves (utm_term)")
    ranking_term = (
        df[df["utm_term"] != "Palavra-chave não identificada"]
        .groupby("utm_term")
        .size()
        .reset_index(name="Leads")
        .sort_values("Leads", ascending=False)
    )
    st.dataframe(ranking_term, use_container_width=True, height=400)
