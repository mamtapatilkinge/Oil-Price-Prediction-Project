import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle 
from pickle import dump
from pickle import load
import joblib
from tensorflow.keras.models import load_model



st.title('Model Deployment: Oil Price Forecasting')

st.sidebar.header('User Input')

def user_input_features():
    #CLMSEX = st.sidebar.selectbox('Gender',('1','0'))
    #CLMINSUR = st.sidebar.selectbox('Insurance',('1','0'))
    #SEATBELT = st.sidebar.selectbox('SeatBelt',('1','0'))
    Timestamp = st.sidebar.number_input("Insert the days",min_value = 1, max_value = 365)
    #LOSS = st.sidebar.number_input("Insert Loss")
   # data = {#'CLMSEX':CLMSEX,
            #'CLMINSUR':CLMINSUR,
            #'SEATBELT':SEATBELT,
            #'Timestamp':Timestamp,
            #'LOSS':LOSS}
    #values == 'data'
    
    return Timestamp
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
uploaded_file = load(open('model.sav','rb'))
#uploaded_fil_1= pickle.load(uploaded_file)

forecasting = uploaded_file.forecast(df)
#prediction_proba = loaded_model.forecast(start=df.index[0], end=df.index[-1])

st.subheader('Predicted Result')
st.write(forecasting)

#st.subheader('Prediction Probability')
#st.write(forecasting)