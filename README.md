
# 🧠 LLaMA 3.2 BDSS MCQ Tutor (Streamlit)

This app fine-tunes Meta's LLaMA 3.2 on 10,000 BDSS-style MCQs and explains them.

## ✅ How to Deploy on Streamlit Cloud

1. Clone or upload this repo to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and deploy the repo.
3. Make sure this directory includes:
   - app.py
   - requirements.txt
   - llama32-3b-mcq-adapter/ (with adapter_model.safetensors + tokenizer)

Enjoy smart MCQ explanations with minimal infra!

## Sample Usage
> Paste:
```
Which heuristic is being used when individuals rely on immediate examples that come to mind?

A. Representativeness  
B. Anchoring  
C. Availability  
D. Framing  
```

> Output:
Full explanation followed by **Correct Option: C**
