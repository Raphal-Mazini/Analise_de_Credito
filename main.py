import streamlit as st
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# Configuração inicial
st.title("Solicitação de Empréstimo")
st.write("### Preencha para simular sua solicitação de empréstimo!")

# Mapeamentos
mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 1, 'Semi Urban': 1, 'Rural': 0}
}

def user_input_features():
    """Função para coleta dos dados de entrada via menu lateral."""
    gender = mappings['Gender'][st.sidebar.selectbox("Gender", ("Male", "Female"))]
    married = mappings['Married'][st.sidebar.selectbox("Is Married?", ("Yes", "No"))]
    dependents = mappings['Dependents'][st.sidebar.selectbox("Number of Dependents", ("0", "1", "2", "3+"))]
    education = mappings['Education'][st.sidebar.selectbox("Education Level", ("Graduate", "Not Graduate"))]
    self_employed = mappings['Self_Employed'][st.sidebar.selectbox("Is Self Employed?", ("Yes", "No"))]
    property_area = mappings['Property_Area'][st.sidebar.selectbox("Property Area?", ("Urban", "Semi Urban", "Rural"))]

    applicant_income = st.sidebar.slider("Applicant Income?", 5000, 10000, 8000) / 1000
    coapplicant_income = st.sidebar.slider("Co Applicant Income?", 0, 10000, 4000) / 1000
    loan_amount = st.sidebar.slider("Loan Amount", 10, 400, 200)
    loan_amount_term = st.sidebar.slider("Loan Amount Term", 12, 480, 300)
    credit_history = st.sidebar.slider("Credit History", 0, 1, 1)

    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
    }
    
    return pd.DataFrame(data, index=[0])
@st.cache_resource
def load_model(filename='Analise_de_Credito_BANK.pkl'):
    """Função para carregar o modelo salvo."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def predict_loan_approval(model, features):
    """Função para realizar a previsão e obter probabilidades."""
    prediction = model.predict(features)
    prediction_probability = model.predict_proba(features)
    return prediction, prediction_probability

if __name__ == "__main__":

    # Fluxo principal
    input_df = user_input_features()

    # Carregando o dataset de teste e concatenando com os dados do usuário
    promotion_test = pd.read_csv('x_teste.csv')
    df = pd.concat([input_df, promotion_test], axis=0).head(1)

    # Carregando o modelo e realizando a previsão
    model = load_model()
    prediction, prediction_probability = predict_loan_approval(model, df)

    # Resultados
    st.subheader('Resultado')
    result_message = ['Desculpe, não podemos lhe conceder um empréstimo!', 'Parabéns, sua solicitação foi aprovada!']
    st.write(result_message[prediction[0]])

    st.subheader('Probabilidade de Aprovação')
    st.write(f'Baseado nos dados de entrada, você tem {prediction_probability[0][1] * 100:.2f}% de chances de aprovação de empréstimo.')

    # Exibindo imagem de acordo com a previsão
    image = r'imagem_bank.jfif'
    st.image(
        image, 
        caption='Aprovação de Empréstimo' if prediction[0] == 1 else 'Reprovação de Empréstimo')