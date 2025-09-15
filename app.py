import streamlit as st
import pandas as pd
import pickle

st.title("Diamond CTR Prediction")

best_model = pickle.load(open("best_ctr_model.pkl", "rb"))

inputs = {}
for col in best_model.named_steps['preprocessor'].transformers_[0][2]:  # categorical columns
    options = best_model.named_steps['preprocessor'].transformers_[0][1].categories_[0]
    inputs[col] = st.selectbox(col, options)

for col in best_model.named_steps['preprocessor'].transformers_[1][2]:  # passthrough numeric columns
    inputs[col] = st.number_input(col, 0.0, 10000.0, 0.0)

input_df = pd.DataFrame([inputs])

if st.button("Predict CTR"):
    prediction = best_model.predict(input_df)
    st.success(f"Predicted CTR: {prediction[0]:.2f}")