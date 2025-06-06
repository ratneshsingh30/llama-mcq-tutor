
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

st.set_page_config(page_title="BDSS MCQ Explainer", layout="centered")
st.title("🧠 BDSS Tutor – Powered by LLaMA 3.2")

@st.cache_resource
def load_model():
    model_path = "./llama32-3b-mcq-adapter"  # your local model folder
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    return tokenizer, model

tokenizer, model = load_model()

mcq_input = st.text_area("Paste your MCQ with options below:", height=300)

if st.button("Explain & Answer"):
    with st.spinner("Thinking..."):
        prompt = f"Explain and answer the following MCQ with reasoning:\n\n{mcq_input}\n\nEnd with 'Correct Option: X'"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=300)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown(result.replace("\n", "<br>"), unsafe_allow_html=True)
