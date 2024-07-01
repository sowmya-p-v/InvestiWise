import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
import tensorflow as tf
import transformers as ts
st.set_page_config(page_title="InvestiWise",
                   layout="wide",
                   page_icon=" ")
placeholder=st.image(r"C:\Users\sowmy\Downloads\istockphoto-1297492947-612x612.jpg")
st.write(f'tensorflow: {tf.__version__}')
st.write(f'streamlit: {st.__version__}')
st.write(f'transformers: {ts.__version__}')

working_dir = os.path.dirname(os.path.abspath(__file__))

# st.image(r"C:\Users\sowmy\Downloads\istockphoto-1297492947-612x612.jpg")
# st.sidebar.title("Enter the credentials")
# st.sidebar.text_input("Enter user name")
# st.sidebar.text_input("Password")
# st.sidebar.button("login")
credit_rate =  pickle.load(open(f'{working_dir}/models/Credit_rating _model.pkl', 'rb'))
# encoder=  pickle.load(open(f'{working_dir}//models/encoder.pkl', 'rb'))
scaler = pickle.load(open(f'{working_dir}/models/min_max_scaler.pkl', 'rb'))
model_path = model_path = os.path.join(working_dir, 'models/saved_model')  
tokenizer_path =model_path = os.path.join(working_dir, 'models/DistilBert_Tokenizer')
invest = pickle.load(open(f'{working_dir}/models/investment_risk_model.pkl', 'rb'))
with st.sidebar:
    selected = option_menu("Comprehensive Investment Risk Analysis",
                           [ 'InvestiWise:',
                                'Investment Risk Prediction',
                                     'View Company Details',
                                     'Performance Analysis'],
                                  icons=['','graph-up-arrow','file-text','bar-chart'],
                                     default_index=0
                           )
if selected == 'InvestiWise':
    st.write("hbfjfkgbfgb")



if selected == 'Investment Risk Prediction':
    placeholder.empty()
    st.title('Investment Risk Prediction using ML')
    col1,col2,col3=st.columns([1,2,1])
    with col1:
        Market=st.selectbox("Select Market",['Emerging Markets','Developed Markets'])
    with col3:
        Region = st.selectbox("Select Region", ['Americas', 'Asia', 'CEEMEA', 'Europe'])
    if Market == 'Emerging Markets':
        if Region =='Americas':
            Country_risk_market_return = st.slider('Country Risk Market Return',0.0,30.00)
            Country_risk_premium = st.slider('Country Risk Premium',0.0,15.0)
            Country_risk_rfr= st.slider('Country Risk free Rate',0.0,10.00)
            Risk_Premium=st.slider('Risk Premium',0.00,35.00)
        elif Region =='Asia':
            Country_risk_market_return = st.slider('Country Risk Market Return',0.0,15.00)
            Country_risk_premium = st.slider('Country Risk Premium',0.0,15.0)
            Country_risk_rfr= st.slider('Country Risk free Rate',0.0,10.00)
            Risk_Premium = st.slider('Risk Premium', -1.00, 20.00)
        elif Region =='CEEMEA':
            Country_risk_market_return = st.slider('Country Risk Market Return',0.0,20.00)
            Country_risk_premium = st.slider('Country Risk Premium',0.0,15.0)
            Country_risk_rfr= st.slider('Country Risk free Rate',0.0,15.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 15.00)
        else:
            st.warning('Change the Market type to Developed Markets')
    elif Market== 'Developed Markets':
        if Region =='Americas':
            Country_risk_market_return = st.slider('Country Risk Market Return',0.0,15.00)
            Country_risk_premium = st.slider('Country Risk Premium',0.0,15.0)
            Country_risk_rfr= st.slider('Country Risk free Rate',0.0,10.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 20.00)
        elif Region =='Asia':
            Country_risk_market_return = st.slider('Country Risk Market Return',0.0,15.00)
            Country_risk_premium = st.slider('Country Risk Premium',0.0,15.00)
            Country_risk_rfr= st.slider('Country Risk free Rate',-1.00,3.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 20.00)
        elif Region =='Europe':
            Country_risk_market_return = st.slider('Country Risk Market Return',0.0,20.00)
            Country_risk_premium = st.slider('Country Risk Premium',0.0,15.0)
            Country_risk_rfr= st.slider('Country Risk free Rate',-1.00,5.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 30.00)
        else:
            st.warning('Change the Market type to Emerging Markets')

    Gross_Margin= st.slider('Gross Margin',0.00,100.00)
    Is_int_EXP = st.slider('Interest Expense', 0.00,8500.00,)
    Oper_Margin = st.slider('Operating Margin',-50.00,100.00)
    Unlevered_Beta = st.slider('Unlevered Beta',-2.00,3.50)
    WACC = st.slider('Weighted Average Cost of Capital(WACC)',0.00,25.00)
    WACC_COST_DEBT=st.slider('WACC Cost Debt',0.00,10.00)
    WACC_COST_Equity =st.slider('WACC Cost Equity',0.00,20.00)
    EPS_Growth= st.slider('Earning Per Share Growth',-600,900)
    col1,col2,col3=st.columns([1,2,1])
    with col1:
        Total_E = st.number_input('Environment Score',0,20)
    with col2:
        Total_S = st.number_input('Social Score', 0, 20)
    with col3:
        Total_G = st.number_input('Governance Score', 0, 20)
    news=st.text_input('News')
    invest_pred=''
    col1,col2,col3,col4 = st.columns([1,2,3,1])
    with col3:
        predictions=st.button('Predict')

    if predictions:
        # st.warning("Work under progress...!")
        if Market == 'Emerging Market':
            market_input = 1
        else:
            market_input = 0

        # User input for Region (assuming only one region can be selected)
        region_input = {
            'Asia': 1 if Region == 'Asia' else 0,
            'CEEMEA': 1 if Region == 'CEEMEA' else 0,
            'Europe': 1 if Region == 'Europe' else 0
        }
        numerical_inputs =[Country_risk_market_return,Country_risk_premium,
                     Country_risk_rfr,EPS_Growth,Gross_Margin,
                     Is_int_EXP,Oper_Margin,Risk_Premium,
                     Unlevered_Beta,WACC,WACC_COST_DEBT,WACC_COST_Equity, Total_E,Total_S,Total_G]
        # categorical_data=[Market,Region]
        # Assuming Market and Region are categorical variables
        market_input_array = np.array([market_input]).reshape(1, -1)
        region_input_array = np.array([list(region_input.values())]).reshape(1, -1)
        # numerical_inputs_array = np.array(numerical_inputs).reshape(1, -1)
        numerical_inputs_array = np.array(numerical_inputs).reshape(1, -1)
        # st.writ(
        input_vector = np.concatenate((numerical_inputs_array,market_input_array,region_input_array), axis=1)
        scaled_inputs = scaler.transform(input_vector)
      
        # st.warning(input_vector)
        # Predict using the model
        credit_rating_impact = credit_rate.predict(scaled_inputs)
       
        news_input =[news]
        

        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)


        model = DistilBertForSequenceClassification.from_pretrained(model_path, from_tf=True)
        inputs = tokenizer(news, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Perform the prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Map class to sentiment
        sentiment_map = {0: -1, 1: 0, 2: 1}
        predicted_sentiment =sentiment_map[predicted_class]

        extra_features = np.array([[credit_rating_impact, predicted_sentiment]])
        scaled_inputs_trimmed = scaled_inputs[:, :-4]
        

        input_with_predictions = np.concatenate((extra_features[0], scaled_inputs_trimmed, extra_features[1]), axis=1)

        # input_with_predictions = np.concatenate((input_vector, np.array([[credit_rating_impact, predicted_sentiment]])), axis=1)

        invesment_risk = invest(input_with_predictions)
        descriptions = {
              0: "Stable Financials, ESG Positive with Positive News Impact - Low to Moderate Investment Risk",
              1: "Stable Financials with Negative News Impact - Low to Moderate Investment Risk",
              2: "High Risk, ESG Positive with Positive News Impact - High Investment Risk considering Financial Health",
              3: "Stable Financials, ESG neutral with Positive News Impact - Low to Moderate Investment Risk",
              4: "High Risk, ESG neutral with Negative News Impact - High Investment Risk considering Financial Health"
        }
        try:
            predicted_description = descriptions[investment_risk]
        except KeyError:
            predicted_description = "Unknown Investment Risk"

        st.write(predicted_description)

    st.success(invest_pred)

