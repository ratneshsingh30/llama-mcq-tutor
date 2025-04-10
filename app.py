
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="BDSS Explainer", layout="wide")
st.title("BDSS Tutor 🧠📘 (LLaMA 3.2 Fine-Tuned)")

@st.cache_resource
def load_model():
    model_path = "./llama32-3b-mcq-adapter"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("📥 Enter a BDSS-style MCQ with options (or paste question text):", height=300)

if st.button("🧪 Generate Explanation + Answer"):
    with st.spinner("Generating..."):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=300)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        st.markdown("### 📌 Explanation + Answer")
        st.success(result)
