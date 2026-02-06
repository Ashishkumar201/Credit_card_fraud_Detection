import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details to predict fraud risk.")



st.write("Provide transaction details below:")

# ---- Amount Slider ----
amt = st.slider("Transaction Amount", 0, 30000, 100)

# ---- Gender ----
gender_option = st.radio("Gender", ["Male", "Female"])
gender = 1 if gender_option == "Male" else 0

# ---- City Population ----
city_pop = st.slider("City Population", 20, 2906700, 50000)


# st.subheader("Transaction Category")

# categories = [
#     "category_food_dining",
#     "category_gas_transport",
#     "category_grocery_net",
#     "category_grocery_pos",
#     "category_health_fitness",
#     "category_home",
#     "category_kids_pets",
#     "category_misc_net",
#     "category_misc_pos",
#     "category_personal_care",
#     "category_shopping_net",
#     "category_shopping_pos",
#     "category_travel",
# ]

# category_inputs = {}

# for cat in categories:
#     value = st.radio(
#         cat.replace("_", " ").title(),
#         ["False", "True"],
#         horizontal=True,
#         key=cat
#     )
#     category_inputs[cat] = 1 if value == "True" else 0

st.subheader("Transaction Category")

categories = [
    "category_food_dining",
    "category_gas_transport",
    "category_grocery_net",
    "category_grocery_pos",
    "category_health_fitness",
    "category_home",
    "category_kids_pets",
    "category_misc_net",
    "category_misc_pos",
    "category_personal_care",
    "category_shopping_net",
    "category_shopping_pos",
    "category_travel",
]

category_inputs = {}

# Create 3 columns
cols = st.columns(3)

for i, cat in enumerate(categories):
    col = cols[i % 3]  # distribute across 3 columns
    
    with col:
        value = st.radio(
            cat.replace("_", " ").title(),
            ["False", "True"],
            horizontal=True,
            key=cat
        )
        category_inputs[cat] = 1 if value == "True" else 0

# ---- Distance ----
distance = st.slider("Distance from Merchant (km)", 0.0, 500.0, 5.0)

# ---- Merchant Fraud Rate ----
merchant_fraud_rate = 0  # always zero

# ---- Collect Inputs ----
input_data = {
    "amt": amt,
    "gender": gender,
    "city_pop": city_pop,
    **category_inputs,
    "merchant_fraud_rate": merchant_fraud_rate,
    "distance": distance,
}

input_df = pd.DataFrame([input_data])

st.subheader("Model Input Preview")
st.dataframe(input_df)

# ---- Prediction Button Placeholder ----
# ---- Prediction ----
if st.button("Predict Fraud"):

    # Make prediction using full input dataframe
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Fraud Probability: {prob:.2%}")

    if prob > 0.5:
        st.error("⚠️ High Fraud Risk Transaction")
    else:
        st.success("✅ Transaction Looks Safe")




# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load trained model
# model = joblib.load("fraud_model.pkl")

# st.title("💳 Credit Card Fraud Detection")

# st.write("Enter transaction details to predict fraud risk.")



# st.write("Provide transaction details below:")

# # ---- Amount Slider ----
# amt = st.slider("Transaction Amount", 0, 30000, 100)

# # ---- Gender ----
# gender_option = st.radio("Gender", ["Male", "Female"])
# gender = 1 if gender_option == "Male" else 0

# # ---- City Population ----
# city_pop = st.slider("City Population", 20, 2906700, 50000)

# st.subheader("Transaction Category")

# categories = [
#     "category_food_dining",
#     "category_gas_transport",
#     "category_grocery_net",
#     "category_grocery_pos",
#     "category_health_fitness",
#     "category_home",
#     "category_kids_pets",
#     "category_misc_net",
#     "category_misc_pos",
#     "category_personal_care",
#     "category_shopping_net",
#     "category_shopping_pos",
#     "category_travel",
# ]

# category_inputs = {}

# for cat in categories:
#     value = st.radio(
#         cat.replace("_", " ").title(),
#         ["False", "True"],
#         horizontal=True,
#         key=cat
#     )
#     category_inputs[cat] = 1 if value == "True" else 0

# # ---- Distance ----
# distance = st.slider("Distance from Merchant (km)", 0.0, 500.0, 5.0)

# # ---- Merchant Fraud Rate ----
# merchant_fraud_rate = 0  # always zero

# # ---- Collect Inputs ----
# input_data = {
#     "amt": amt,
#     "gender": gender,
#     "city_pop": city_pop,
#     **category_inputs,
#     "merchant_fraud_rate": merchant_fraud_rate,
#     "distance": distance,
# }

# input_df = pd.DataFrame([input_data])

# st.subheader("Model Input Preview")
# st.dataframe(input_df)

# # ---- Prediction Button Placeholder ----
# # ---- Prediction ----
# if st.button("Predict Fraud"):

#     # Make prediction using full input dataframe
#     prob = model.predict_proba(input_df)[0][1]

#     st.subheader(f"Fraud Probability: {prob:.2%}")

#     if prob > 0.5:
#         st.error("⚠️ High Fraud Risk Transaction")
#     else:
#         st.success("✅ Transaction Looks Safe")



