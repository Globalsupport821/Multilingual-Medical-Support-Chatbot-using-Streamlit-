import streamlit as st
import torch
from transformers import pipeline
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🩺 Medical Chatbot", layout="wide")
st.title("🩺 Multilingual Medical Support Chatbot")
st.markdown("---")
st.error("⚠️ Not a substitute for professional medical advice")

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Describe symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 Analyzing..."):
            response = model(f"medical advice: {prompt}", max_length=100)[0]['generated_text']
            st.success(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("🗑️ Clear Chat"): 
    st.session_state.messages = []
    st.rerun()
