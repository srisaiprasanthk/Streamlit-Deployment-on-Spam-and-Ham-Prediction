import streamlit as st
import pickle
from preprocessing import clean, setup_nltk

# Download NLTK resources (runs once)
setup_nltk()

# Load the trained model
with open("spam_email_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📧 Spam Email Classifier")
st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

email_text = st.text_area("Email Text", height=150)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned_text = clean(email_text)
        prediction = model.predict([cleaned_text])[0]

        if prediction == "spam":
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is HAM (Not Spam)")
