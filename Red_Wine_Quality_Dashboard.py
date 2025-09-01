import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Red Wine Quality Predictor", layout="wide")

# Load model and data
def model():
    return joblib.load('Outputs/ML Model for Red Wine Prediction.pkl')

def feature():
    return joblib.load('Outputs/feature_columns.pkl')

def dt_model():
    return joblib.load('Outputs/Decision Tree for Quality of Red Wine.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('Data/redwine.csv')


features = feature()
rf_model = model()
data = load_data()
data_df = pd.DataFrame(data)

# Title and description
st.title(" :red[Red] W🍷ne Quality Prediction Dashboard")
st.markdown("""
This dashboard predicts the quality of red wine based on physicochemical properties.
Input the wine characteristics below and click **Predict** to see the result!
""")
st.markdown('<style>div.block-container{padding-top: 2rem;}</style>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Data Overview
with st.expander("📝**Data Overview**"):
    st.subheader("Dataset head(5)")
    st.dataframe(data_df.head().drop('Unnamed: 0', axis=1))
    
    # Independent variables
    st.write("**Independent Variables (X)**")
    X = data_df.drop(['quality', 'quality bins', 'Unnamed: 0'], axis=1)
    st.dataframe(X)
# Dependent Variables
    st.write("**Dependent Variables (y)**")
    y = data_df[['quality', 'quality bins']]
    st.dataframe(y)


# Sidebar for user inputs
with st.sidebar.header("**Input Wine Features**"):
    volatile_acidity = st.sidebar.slider('Volatile Acidity (g/L)', 0.0, 2.0, 0.5, step=0.01)
    citric_acid = st.sidebar.slider('Citric Acid (g/L)', 0.0, 1.0, 0.3, step=0.01)
    chlorides = st.sidebar.slider('Chlorides (g/L)', 0.0, 0.7, 0.08, step=0.01)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide (mg/L)', 0.0, 290.0, 40.0, step=1.0)
    density = st.sidebar.slider('Density (g/cm³)', 0.98, 1.01, 0.995, step=0.001)
    pH = st.sidebar.slider('pH', 2.0, 4.0, 3.3, step=0.01)
    sulphates = st.sidebar.slider('Sulphates (g/L)', 0.0, 2.0, 0.6, step=0.01)
    alcohol = st.sidebar.slider('Alcohol (% vol)', 8.0, 15.0, 10.0, step=0.1)
    
# Create a dataframe for inputs
    input_data = pd.DataFrame({
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'chlorides': chlorides,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }, index=[0])

# Main content
data = data.rename(columns={'quality bins': 'Quality Category'})
data['Quality Category'] = data['Quality Category'].astype('category')
data['Quality Category'] = data['Quality Category'].map({0: 'Low Quality', 1: 'Mid Quality', 2: 'High Quality'})
scatter_feature = data_df[['volatile acidity', 'citric acid', 'alcohol', 'sulphates']]

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("**Prediction**")
    if st.button("Predict Quality"):
        prediction = rf_model.predict(input_data)
        proba = rf_model.predict_proba(input_data).round(1)
        df_proba = pd.DataFrame(proba)
        quality = np.array(['Low Quality', 'Mid Quality', 'High Quality'])
        for i in prediction:
            if i == 0:
                st.warning('The new prediction for the quality of the Red Wine is: Low Quality')
            elif i == 1:
                st.info("The new prediction for the quality of the Red Wine is: Mid Quality")
            else:
                st.success("The new prediction for the quality of the Red Wine is: High Quality!!")
            df_proba.columns = ['Low Quality', 'Mid Quality', 'High Quality']
            df_proba.rename(columns={0: 'Low Qaulity', 1: 'Mid Quality', 2: 'High Quality'})
        st.write("**Prediction Probability**")
        st.dataframe(df_proba, column_config={
            'Low Quality': st.column_config.ProgressColumn('Low Quality', format='percent', width='medium', min_value=0, max_value=1),
            'Mid Quality': st.column_config.ProgressColumn('Mid Quality', format='percent', width='medium', min_value=0, max_value=1),
            'High Quality': st.column_config.ProgressColumn('High Quality', format='percent', width='medium', min_value=0, max_value=1)
        }, hide_index=True, use_container_width=True)
        st.write("Input Features Contributions")
        value = input_data.iloc[0]
        fig = px.pie(input_data, names=input_data.columns, values=input_data.iloc[0], color_discrete_sequence=px.colors.sequential.RdBu, hover_data=[value])
        st.plotly_chart(fig, use_container_width=True)

# Feature Importance
with col2:
    st.subheader("**Feature Importance**")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    fig = px.bar(importance, x='Importance', y='Feature', title="Feature Importance in Prediction", color=importance['Importance'], color_continuous_scale='Bluered_r', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("**About Features**"):
        st.write("""
        - **Volatile Acidity**: Higher values can lead to an unpleasant taste.
        - **Citric Acid**: Contributes to the freshness and flavor of the wine.
        - **Chlorides**: High levels can indicate poor quality.
        - **Total Sulfur Dioxide**: Used as a preservative; excessive amounts can affect taste.
        - **Density**: Related to the sugar content; affects mouthfeel.
        - **pH**: Indicates acidity; affects taste and stability.
        - **Sulphates**: Can enhance flavor and act as a preservative.
        - **Alcohol**: Higher alcohol content can increase body and warmth of the wine.
        """)

# Data Visualization
st.title(" :bar_chart: **Explore the Dataset**", width="stretch")
st.write("This visualization contains various physicochemical properties of red wine and their corresponding quality ratings.")
feature_to_plot = st.selectbox("Select a feature to visualize distribution", data.columns.drop(['quality', "Unnamed: 0", "Quality Category"]))
fig_dist = px.histogram(data, x=feature_to_plot, title=f"Distribution of {feature_to_plot}", nbins=30, color='Quality Category', barmode='relative')
st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("Correlations between Important Features")
with st.expander("Correlations"):
    correlations = data[features].corr().round(1)
    st.dataframe(correlations)
## HeatMap
    st.write("The HeatMap between Correlated Features:")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("""
---
Red Wine Quality Prediction by [Ireoluwawolemi Jeremiah Akindipe].  
LinkedIn: [Ireoluwawolemi Jeremiah Akindipe](https://www.linkedin.com/in/ireoluwawolemi-akindipe-16b711373?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app).  
GitHub: [Ireoluwawolemi Jeremiah Akindipe](https://github.com/Dhela456).  
Dataset: [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality).  
Source code: [GitHub](#).
""")