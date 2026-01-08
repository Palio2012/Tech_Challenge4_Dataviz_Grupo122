import streamlit as st
import pandas as pd
import joblib
import os

# Configuração da página

st.set_page_config(
    page_title="Avaliador de Obesidade",
    page_icon="⚖️",
    layout="wide"
)


# CSS Simples para melhora do visual

st.markdown("""
<style>
    div.stButton > button {
        height: 3em;
        font-size: 18px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Título

st.title("⚖️ Avaliador de Obesidade")
st.caption("Aplicação de Machine Learning para classificação do nível de obesidade")

st.markdown("---")


# Entradas do Usuário

col1, col2, col3 = st.columns(3)


# Dados Pessoais

with col1:
    st.subheader("🧍 Dados Pessoais")
    Gender = st.radio("Gênero", ["Masculino", "Feminino"])
    Age = st.number_input("Idade", 10, 100)
    family_history_with_overweight = st.radio(
        "Histórico familiar de obesidade?", ["Sim", "Não"]
    )

# Alimentação

with col2:
    st.subheader("🍽️ Alimentação")
    FAVC = st.radio(
        "Consome alimentos calóricos frequentemente?", ["Sim", "Não"]
    )
    FCVC = st.radio(
        "Consome vegetais regularmente?", ["Sim", "Não"]
    )
    NCP = st.number_input("Refeições principais por dia", 0, 10)

    CAEC = st.radio(
        "Consome lanches entre as refeições?",
        ["Não", "Ás vezes", "Frequentemente", "Sempre"]
    )


# Estilo de Vida

with col3:
    st.subheader("🏃 Estilo de Vida")
    SMOKE = st.radio("Fumante?", ["Sim", "Não"])

    CH2O = st.radio(
        "Consumo diário de água",
        ["1 litro ou menos", "1,5 litros", "2 litros ou mais"]
    )

    FAF = st.radio(
        "Atividade física",
        [
            "Nenhuma",
            "1 ou 2 vezes na semana",
            "3 ou 4 vezes na semana",
            "5 vezes na semana ou mais"
        ]
    )


# Hábitos Adicionais

with st.expander("🧬 Outros hábitos"):
    col4, col5, col6 = st.columns(3)

    with col4:
        SCC = st.radio("Monitora ingestão calórica?", ["Sim", "Não"])

    with col5:
        TUE = st.radio(
            "Tempo em eletrônicos",
            ["0-2h por dia", "3-5h por dia", "5h por dia ou mais"]
        )

    with col6:
        CALC = st.radio(
            "Consumo de álcool",
            ["Não bebe", "Ás vezes", "Frequentemente", "Sempre"]
        )

# Transporte

st.subheader("🚗 Transporte")
MTRANS = st.selectbox(
    "Meio de transporte habitual",
    ["Caminhando", "Bicicleta", "Transporte Público", "Motocicleta", "Automóvel"]
)

# Carregando o Modelo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "modelo", "model_obesity.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Novo Usuário
usuario_predict_df = pd.DataFrame([{
    "Gender": Gender,
    "Age": Age,
    "family_history_with_overweight": family_history_with_overweight,
    "FAVC": FAVC,
    "FCVC": FCVC,
    "NCP": NCP,
    "CAEC": CAEC,
    "SMOKE": SMOKE,
    "CH2O": CH2O,
    "SCC": SCC,
    "FAF": FAF,
    "TUE": TUE,
    "CALC": CALC,
    "MTRANS": MTRANS
}])

# Botão Central

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    avaliar = st.button("🔍 Avaliar nível de obesidade", use_container_width=True)


# Resultado

if avaliar:
    pred = model.predict(usuario_predict_df)
    classe = pred[0]

    labels = {
        0: "Peso insuficiente",
        1: "Peso normal",
        2: "Sobrepeso nível I",
        3: "Sobrepeso nível II",
        4: "Obesidade tipo I",
        5: "Obesidade tipo II",
        6: "Obesidade tipo III"
    }

    proba = model.predict_proba(usuario_predict_df)
    conf = proba.max() * 100

    with st.container(border=True):
        st.subheader("Resultado da Avaliação")
        st.markdown(f"### Classificação: **{labels.get(classe)}**")
        st.progress(conf / 100)

        if conf > 90:
            st.success("Confiança estimada: Alta")
        elif conf > 70:
            st.warning("Confiança estimada: Média")
        else:
            st.error("Confiança estimada: Baixa")

# Rodapé

st.caption("⚠️ Este aplicativo tem finalidade educacional e não substitui avaliação médica.")




