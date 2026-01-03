import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Walmart Sales Predictor", page_icon="🛒", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e1e4e8;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛒 Walmart Weekly Sales Predictor")
st.markdown("Predict your store's performance with precision using our on-the-fly Linear Regression model.")


@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv('Walmart_Sales.csv')
    
        df['Date_dt'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Day'] = df['Date_dt'].dt.day
        df['Month'] = df['Date_dt'].dt.month
        df['Year'] = df['Date_dt'].dt.year
        
        X = df.drop(['Weekly_Sales', 'Date', 'Date_dt'], axis=1)
        y = df['Weekly_Sales']
        
        model = LinearRegression()
        model.fit(X, y)
        
        return df, model
    except Exception as e:
        st.error(f"Error loading data or training model: {e}")
        return None, None

df, model = load_and_train()

st.sidebar.header("Input Features")

def user_input_features():
    store = st.sidebar.slider("Store Number", 1, 45, 1)
    date = st.sidebar.date_input("Date", datetime(2010, 2, 5))
    holiday_flag = st.sidebar.selectbox("Is it a Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    temperature = st.sidebar.slider("Temperature (F)", -2.0, 100.0, 42.0)
    fuel_price = st.sidebar.slider("Fuel Price", 2.0, 5.0, 2.57)
    cpi = st.sidebar.slider("CPI", 120.0, 230.0, 211.0)
    unemployment = st.sidebar.slider("Unemployment Rate", 3.0, 15.0, 8.1)
    
    data = {
        'Store': store,
        'Holiday_Flag': holiday_flag,
        'Temperature': temperature,
        'Fuel_Price': fuel_price,
        'CPI': cpi,
        'Unemployment': unemployment,
        'Day': date.day,
        'Month': date.month,
        'Year': date.year
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Input Parameters")
    st.write(input_df)

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

with col2:
    st.subheader("Sales Prediction")
    
    if model is not None:
        prediction = model.predict(input_df)
        
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style="margin-bottom: 0;">Predicted Weekly Sales</h3>
                <h1 style="color: #28a745; font-size: 3em; margin-top: 10px;">${prediction[0]:,.2f}</h1>
                <p style="color: #6c757d;">Based on Linear Regression analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Model trained successfully on the current dataset!")
    else:
        st.error("❌ Model training failed. Please check the dataset.")

    if df is not None:
        st.subheader("Historical Sales Trend (Selected Store)")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_store = df[df['Store'] == input_df['Store'][0]].sort_values('Date_dt')
        sns.lineplot(data=df_store, x='Date_dt', y='Weekly_Sales', ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Weekly Sales")
        st.pyplot(fig)

st.markdown("---")

