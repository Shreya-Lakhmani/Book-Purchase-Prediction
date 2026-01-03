
import streamlit as st
import numpy as np
import joblib

#Load the trained model
model = joblib.load("purchase_predictor_.joblib")

# Set up the app
st.set_page_config(page_title="Book Purchase Predictor", page_icon="📚")
st.title("🤖 Smart Purchase Prediction System")
st.markdown("Predict whether a customer will purchase a book based on product and user behavior.")
st.markdown(
    """
    This application uses **Machine Learning** to predict whether a customer is likely to buy a book.
    """
)
st.markdown("---")

st.sidebar.header("🔧 Input Features")

#Input fields
price = st.sidebar.number_input("Book Price (£)", min_value=0.0, max_value=200.0, value=25.0)
rating = st.sidebar.selectbox("Rating", [1, 2, 3, 4, 5], index=2)

col1, col2 = st.columns(2)

with col1:
    st.info("📘 Lower price + higher rating increases purchase probability.")

with col2:
    st.info("🛒 Model trained using Random Forest classifier.")

#Preparing the feature array
input_data = np.array([[price, rating]])

# making prediction
if st.button("🔍 Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ This customer is likely to purchase the book! (Confidence: {probability:.2f})")
    else:
        st.warning(f"❌ This customer is unlikely to make a purchase. (Confidence: {probability:.2f})")

    # Showing the prediction probability for both classes
    st.subheader("📊 Prediction Probabilities:")
    st.write({
        "Not Purchased (0)": round(model.predict_proba(input_data)[0][0], 2),
        "Purchased (1)": round(model.predict_proba(input_data)[0][1], 2)
    })
    st.bar_chart({
    "Not Purchased": model.predict_proba(input_data)[0][0],
    "Purchased": model.predict_proba(input_data)[0][1]
})

# footer
st.markdown("---")
st.caption("ML-powered Book Purchase Predictor | Built with Streamlit & Scikit-learn")
