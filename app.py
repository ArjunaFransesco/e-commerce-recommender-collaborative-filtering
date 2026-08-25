import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hybrid Collaborative Filtering & Content-Based Recommender System",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Hybrid Collaborative Filtering & Content-Based Recommender System")
st.markdown("**Domain**: `E-Commerce / Recommendation Engines` | **Tech Stack**: `Matrix Factorization (SVD), TF-IDF, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    user_tenure_days = st.slider("User Tenure Days", float(10.0), float(1200.0), float(300.0))
    avg_basket_value = st.slider("Avg Basket Value", float(15.0), float(350.0), float(85.0))
    purchase_frequency = st.slider("Purchase Frequency", float(0.5), float(20.0), float(4.5))
    category_affinity_score = st.slider("Category Affinity Score", float(0.05), float(0.99), float(0.65))
    loyalty_tier = st.slider("Loyalty Tier", int(1), int(5), int(2))
    review_sentiment_polarity = st.slider("Review Sentiment Polarity", float(-1.0), float(1.0), float(0.4))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"user_tenure_days": user_tenure_days, "avg_basket_value": avg_basket_value, "purchase_frequency": purchase_frequency, "category_affinity_score": category_affinity_score, "loyalty_tier": loyalty_tier, "review_sentiment_polarity": review_sentiment_polarity}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `is_high_propensity`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
