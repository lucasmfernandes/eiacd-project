"""
Streamlit App - California Housing ML Prediction System
Imports functions from existing modules.

Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import from existing modules
from load_dataset import load_california_housing
from prediction_pipeline import prepare_data, train_model, make_predictions, process_text_input
from model_confidence import (
    calculate_regression_metrics,
    calculate_prediction_intervals,
    get_feature_importance,
    get_model_confidence_summary,
)

# Page config
st.set_page_config(page_title="California Housing ML", page_icon="🏠", layout="wide")

# Dark theme CSS
st.markdown("""
<style>
    .metric-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 1.5rem; border-radius: 10px; border: 1px solid #0f3460; margin: 0.5rem 0; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #a3a3a3; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)


# Cached data loading and training
@st.cache_data
def load_data():
    return load_california_housing()


@st.cache_resource
def get_trained_model(_df):
    X, y, features = prepare_data(_df, 'target', True)
    pipeline, X_train, X_test, y_train, y_test = train_model(X, y)
    return {'pipeline': pipeline, 'features': features, 
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


# Load and train
df, info = load_data()
model = get_trained_model(df)


# Sidebar
st.sidebar.title("Navegacao")
page = st.sidebar.radio("Pagina:", ["Overview", "Tool 1: Predicao", "Tool 2: Confianca", "Tool 3: Texto"])
st.sidebar.markdown("---")
st.sidebar.metric("Amostras", f"{info['n_samples']:,}")
st.sidebar.metric("Features", info['n_features'])


# Overview Page
if page == "Overview":
    st.title("California Housing ML Dashboard")
    
    y_pred = make_predictions(model['pipeline'], model['X_test'])
    metrics = calculate_regression_metrics(model['y_test'], y_pred, "Test")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2 Score", f"{metrics['r2_score']:.3f}")
    c2.metric("RMSE", f"${metrics['rmse']*100000:,.0f}")
    c3.metric("MAE", f"${metrics['mae']*100000:,.0f}")
    c4.metric("MAPE", f"{metrics['mape']:.1f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(x=model['y_test'], y=y_pred, opacity=0.5,
                        labels={'x': 'Real ($100k)', 'y': 'Predito ($100k)'})
        fig.add_scatter(x=[0,5], y=[0,5], mode='lines', name='Ideal', line=dict(dash='dash', color='red'))
        fig.update_layout(template='plotly_dark', height=400, title='Predicao vs Real')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        imp = get_feature_importance(model['pipeline'], model['features'])
        fig = px.bar(imp, x='importance_pct', y='feature', orientation='h', title='Importancia das Features')
        fig.update_layout(template='plotly_dark', height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Dataset")
    st.dataframe(df.head(50), use_container_width=True)


# Tool 1: Table Prediction
elif page == "Tool 1: Predicao":
    st.title("Tool 1: Predicao para Dados em Tabela")
    
    st.subheader("Dataset California Housing (Carregado Automaticamente)")
    
    # Slider para selecionar quantas amostras usar
    n_samples = st.slider("Numero de amostras para predicao:", min_value=10, max_value=len(df), value=100, step=10)
    
    # Seleciona amostras aleatorias do dataset
    sample_df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    st.write(f"Amostra de {n_samples} registros:")
    st.dataframe(sample_df.head(20), use_container_width=True)
    
    if st.button("Prever Valores", key="auto"):
        X, _, _ = prepare_data(sample_df.drop(columns=['target']), is_training=False)
        preds = make_predictions(model['pipeline'], X)
        
        result_df = sample_df.copy()
        result_df['Valor_Predito'] = preds
        result_df['Preco_Predito_USD'] = preds * 100000
        result_df['Preco_Real_USD'] = result_df['target'] * 100000
        result_df['Erro_USD'] = abs(result_df['Preco_Predito_USD'] - result_df['Preco_Real_USD'])
        
        st.subheader("Resultados das Predicoes")
        st.dataframe(result_df, use_container_width=True)
        
        # Metricas resumidas
        c1, c2, c3 = st.columns(3)
        c1.metric("Erro Medio", f"${result_df['Erro_USD'].mean():,.2f}")
        c2.metric("Erro Mediano", f"${result_df['Erro_USD'].median():,.2f}")
        c3.metric("Erro Maximo", f"${result_df['Erro_USD'].max():,.2f}")
        
        # Grafico comparativo
        fig = px.scatter(result_df, x='Preco_Real_USD', y='Preco_Predito_USD', 
                        opacity=0.6, title='Preco Real vs Predito')
        fig.add_scatter(x=[0, result_df['Preco_Real_USD'].max()], 
                       y=[0, result_df['Preco_Real_USD'].max()], 
                       mode='lines', name='Ideal', line=dict(dash='dash', color='red'))
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button("Download CSV", result_df.to_csv(index=False), "predicoes.csv")


# Tool 2: Confidence
elif page == "Tool 2: Confianca":
    st.title("Tool 2: Metricas de Confianca")
    
    y_train_pred = make_predictions(model['pipeline'], model['X_train'])
    y_test_pred = make_predictions(model['pipeline'], model['X_test'])
    
    train_m = calculate_regression_metrics(model['y_train'], y_train_pred, "Train")
    test_m = calculate_regression_metrics(model['y_test'], y_test_pred, "Test")
    intervals = calculate_prediction_intervals(model['pipeline'], model['X_test'])
    conf = get_model_confidence_summary(train_m, test_m, intervals)
    
    score = conf['overall_confidence_score']
    color = "#22c55e" if score >= 70 else "#eab308" if score >= 50 else "#ef4444"
    
    st.markdown(f"""
    <div style="text-align:center; padding:2rem; background:#1a1a2e; border-radius:1rem; border:2px solid {color};">
        <h1 style="font-size:4rem; color:{color}; margin:0;">{score:.1f}</h1>
        <p style="color:#a3a3a3;">Score de Confianca ({conf['confidence_level']})</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Treino")
        st.metric("R2", f"{train_m['r2_score']:.4f}")
        st.metric("RMSE", f"${train_m['rmse']*100000:,.2f}")
    with c2:
        st.subheader("Teste")
        st.metric("R2", f"{test_m['r2_score']:.4f}")
        st.metric("RMSE", f"${test_m['rmse']*100000:,.2f}")
    
    st.subheader("Distribuicao dos Residuos")
    residuals = model['y_test'].values - y_test_pred
    fig = px.histogram(residuals, nbins=50)
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)


# Tool 3: Text Input
elif page == "Tool 3: Texto":
    st.title("Tool 3: Predicao via Texto")
    
    example = "MedInc=8.5, HouseAge=25, AveRooms=7.2, AveBedrms=1.5, Population=1200, AveOccup=2.8, Latitude=34.05, Longitude=-118.25"
    text = st.text_area("Descreva o imovel:", value=example, height=100)
    
    if st.button("Processar"):
        try:
            parsed = process_text_input(text, model['features'])
            st.write("Features extraidas:", parsed)
            
            pred = make_predictions(model['pipeline'], parsed)[0]
            st.success(f"Valor Predito: **${pred * 100000:,.2f}**")
            
            intervals = calculate_prediction_intervals(model['pipeline'], parsed)
            st.info(f"Intervalo 95%: ${intervals['lower_bound'][0]*100000:,.2f} - ${intervals['upper_bound'][0]*100000:,.2f}")
        except Exception as e:
            st.error(f"Erro: {e}")
