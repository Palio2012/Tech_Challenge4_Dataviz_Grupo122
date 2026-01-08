import streamlit as st
import pandas as pd
import joblib
import os


# ===============================
# Configuração da página
# ===============================
st.set_page_config(
    page_title="Avaliador de Obesidade",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Avaliador de Obesidade")
st.caption("Aplicação de Machine Learning para classificação do nível de obesidade")
st.markdown("---")


# ===============================
# Entradas do Usuário
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🧍 Dados Pessoais")

    gender = st.radio("Gênero", ["male", "female"])
    age = st.number_input("Idade", 10, 100)

    family_history = st.radio(
        "Histórico familiar de obesidade?",
        ["yes", "no"]
    )

with col2:
    st.subheader("🍽️ Alimentação")

    favc = st.radio(
        "Consome alimentos calóricos frequentemente?",
        ["yes", "no"]
    )

    fcvc = st.slider(
        "Consome vegetais regularmente?",
        0.0, 3.0, step=0.5
    )

    ncp = st.slider(
        "Refeições principais por dia",
        1.0, 4.0, step=0.5
    )

    caec = st.radio(
        "Consome lanches entre as refeições?",
        ["no", "Sometimes", "Frequently", "Always"]
    )

with col3:
    st.subheader("🏃 Estilo de Vida")

    smoke = st.radio("Fumante?", ["yes", "no"])

    ch2o = st.slider(
        "Consumo diário de água",
        1.0, 3.0, step=0.5
    )

    faf = st.slider(
        "Atividade física",
        0.0, 3.0, step=0.5
    )


with st.expander("🧬 Outros hábitos"):
    col4, col5, col6 = st.columns(3)

    with col4:
        scc = st.radio("Monitora ingestão calórica?", ["yes", "no"])

    with col5:
        tue = st.slider(
            "Tempo em eletrônicos",
            0.0, 2.0, step=0.5
        )

    with col6:
        calc = st.radio(
            "Consumo de álcool",
            ["no", "Sometimes", "Frequently", "Always"]
        )


st.subheader("🚗 Transporte")
mtrans = st.selectbox(
    "Meio de transporte habitual",
    [
        "Walking",
        "Bike",
        "Public_Transportation",
        "Motorbike",
        "Automobile"
    ]
)


# ===============================
# Carregar modelo
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "modelo", "model_obesity.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


# ===============================
# DataFrame EXACTAMENTE como no treino
# ===============================
usuario_predict_df = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "family_history": family_history,
    "favc": favc,
    "fcvc": fcvc,
    "ncp": ncp,
    "caec": caec,
    "smoke": smoke,
    "ch2o": ch2o,
    "scc": scc,
    "faf": faf,
    "tue": tue,
    "calc": calc,
    "mtrans": mtrans
}])


# ===============================
# Predição
# ===============================
st.markdown("---")
avaliar = st.button("🔍 Avaliar nível de obesidade", use_container_width=True)

if avaliar:
    pred = model.predict(usuario_predict_df)
    classe = pred[0]

    proba = model.predict_proba(usuario_predict_df)
    conf = proba.max() * 100

    with st.container(border=True):
        st.subheader("Resultado da Avaliação")
        st.write(f"Classificação prevista: **{classe}**")
        st.progress(conf / 100)
        st.caption(f"Confiança estimada: {conf:.2f}%")


st.caption("⚠️ Aplicação educacional — não substitui avaliação médica.")
