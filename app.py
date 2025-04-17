import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("impression_predictor.pkl")

model = load_model()

st.title("📣 Social Post Impressions Simulator")
st.markdown("Use the sliders and dropdowns to simulate your next high-performing post.")

# Input fields
sentiment = st.slider("Post Sentiment (Polarity)", -1.0, 1.0, 0.2, 0.01)
word_count = st.slider("Post Copy Word Count", 10, 500, 150, 10)
cluster = st.selectbox("Post Theme Cluster", [0, 1, 2, 3, 4])
asset_type = st.selectbox("Asset Type", ["Video", "Text-only", "Single Image", "Link Share Media", "Poll"])

# Create input DataFrame
user_input = pd.DataFrame([{
    "post_copy_sentiment": sentiment,
    "post_copy_word_count": word_count,
    "copy_cluster": cluster,
    "asset_type": asset_type
}])

# Predict and show result
if st.button("Predict Impressions"):
    predicted = model.predict(user_input)[0]
    st.success(f"Predicted Impressions: {int(predicted):,}")
