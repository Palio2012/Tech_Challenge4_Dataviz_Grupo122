import streamlit as st
import pandas as pd
import joblib


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

    input_gender = st.radio("Gênero", ["Masculino", "Feminino"])
    gender_dict = {"Masculino": 1, "Feminino": 2}
    gender = gender_dict.get(input_gender)

    age = st.number_input("Idade", 10, 100)

    input_family_history = st.radio("Histórico familiar de obesidade?", ["Sim", "Não"])
    family_history_dict = {"Sim": 1, "Não": 0}
    family_history = family_history_dict.get(input_family_history)

# Alimentação

with col2:
    st.subheader("🍽️ Alimentação")

    input_favc = st.radio("Consome alimentos calóricos frequentemente?", ["Sim", "Não"])
    favc_dict = {"Sim": 1, "Não": 0}
    favc = favc_dict.get(input_favc)

    input_fcvc = st.radio("Consome vegetais regularmente?", ["Sim", "Não"])
    fcvc_dict = {"Sim": 1, "Não": 0}
    fcvc = fcvc_dict.get(input_fcvc)

    ncp = st.number_input("Refeições principais por dia", 0, 10)

    input_caec = st.radio(
        "Consome lanches entre as refeições?",
        ["Não", "Ás vezes", "Frequentemente", "Sempre"]
    )
    caec_dict = {"Não": 0, "Ás vezes": 1, "Frequentemente": 2, "Sempre": 3}
    caec = caec_dict.get(input_caec)


# Estilo de Vida

with col3:
    st.subheader("🏃 Estilo de Vida")

    input_smoke = st.radio("Fumante?", ["Sim", "Não"])
    smoke_dict = {"Sim": 1, "Não": 0}
    smoke = smoke_dict.get(input_smoke)

    input_ch2o = st.radio(
        "Consumo diário de água",
        ["1 litro ou menos", "1,5 litros", "2 litros ou mais"]
    )
    ch2o_dict = {
        "1 litro ou menos": 1,
        "1,5 litros": 2,
        "2 litros ou mais": 3
    }
    ch2o = ch2o_dict.get(input_ch2o)

    input_faf = st.radio(
        "Atividade física",
        [
            "Nenhuma",
            "1 ou 2 vezes na semana",
            "3 ou 4 vezes na semana",
            "5 vezes na semana ou mais"
        ]
    )
    faf_dict = {
        "Nenhuma": 0,
        "1 ou 2 vezes na semana": 1,
        "3 ou 4 vezes na semana": 2,
        "5 vezes na semana ou mais": 3
    }
    faf = faf_dict.get(input_faf)


# Hábitos Adicionais

with st.expander("🧬 Outros hábitos"):
    col4, col5, col6 = st.columns(3)

    with col4:
        input_scc = st.radio("Monitora ingestão calórica?", ["Sim", "Não"])
        scc_dict = {"Sim": 1, "Não": 0}
        scc = scc_dict.get(input_scc)

    with col5:
        input_tue = st.radio(
            "Tempo em eletrônicos",
            ["0-2h por dia", "3-5h por dia", "5h por dia ou mais"]
        )
        tue_dict = {
            "0-2h por dia": 0,
            "3-5h por dia": 1,
            "5h por dia ou mais": 2
        }
        tue = tue_dict.get(input_tue)

    with col6:
        input_calc = st.radio(
            "Consumo de álcool",
            ["Não bebe", "Ás vezes", "Frequentemente", "Sempre"]
        )
        calc_dict = {
            "Não bebe": 0,
            "Ás vezes": 1,
            "Frequentemente": 2,
            "Sempre": 3
        }
        calc = calc_dict.get(input_calc)


# Transporte

st.subheader("🚗 Transporte")
input_mtrans = st.selectbox(
    "Meio de transporte habitual",
    ["Caminhando", "Bicicleta", "Transporte Público", "Motocicleta", "Automóvel"]
)
mtrans_dict = {
    "Caminhando": 1,
    "Bicicleta": 2,
    "Transporte Público": 3,
    "Motocicleta": 4,
    "Automóvel": 5
}
mtrans = mtrans_dict.get(input_mtrans)

# Carregando o Modelo

model = joblib.load("modelo/model_obesityv2.pkl")


usuario_predict_df = pd.DataFrame([{
    # Dados Pessoais
    "gender": input_gender, 
    "age": age,
    "family_history_with_overweight": input_family_history, 
    
    # Alimentação
    "favc": input_favc,
    "fcvc": input_fcvc, 
    "ncp": ncp,
    "caec": input_caec,

    # Estilo de Vida
    "smoke": input_smoke,
    "ch2o": input_ch2o, 
    "scc": input_scc,
    "faf": input_faf, 
    "tue": input_tue, 
    
    # Outros
    "calc": input_calc,
    "mtrans": input_mtrans
}])

usuario_predict_df = usuario_predict_df[model.feature_names_in_]

# Botão Central

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    avaliar = st.button("🔍 Avaliar nível de obesidade", use_container_width=True)

# --- ÁREA DE DIAGNÓSTICO (Apagar depois de resolver) ---
st.write("### Diagnóstico de Colunas")
cols_modelo = list(model.feature_names_in_)
cols_dataframe = list(usuario_predict_df.columns)

st.write("**Colunas que o MODELO espera:**", cols_modelo)
st.write("**Colunas que o DATAFRAME possui:**", cols_dataframe)

# Achar a diferença
diferenca = set(cols_modelo) - set(cols_dataframe)
st.error(f"⚠️ Colunas faltando no DataFrame: {diferenca}")
# -------------------------------------------------------

# Sua linha original que dá erro:
usuario_predict_df = usuario_predict_df[model.feature_names_in_]


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



