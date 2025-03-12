import streamlit as st
import requests

st.title("German2French Translation")

text = st.text_area("Enter your German text to translate", "")

if st.button("Translate"):
    if text:
        response = requests.post("http://localhost:8000/translate", json={"text": text})
        if response.status_code == 200:
            translation = response.json()["translation"]
            st.subheader("Translation:")
            st.write(translation)
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please enter text to translate.")
