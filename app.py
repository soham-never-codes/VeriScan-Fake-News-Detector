import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

st.set_page_config(page_title="VeriScan | AI News Detector", page_icon="📰", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #2C3E50; color: white; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: 800; font-size: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .real-news { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
    .fake-news { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    model_path = "/content/drive/MyDrive/FakeNewsProject/final_model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_ai_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model from Google Drive. Ensure the path is correct. Error: {e}")
    model_loaded = False

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2905/2905080.png", width=100)
    st.title("VeriScan Engine")
    st.info("**Architecture:** DistilBERT-base-uncased\n**Task:** Binary Sequence Classification\n**Token Limit:** 512 Max Length")
    st.divider()
    st.write("Built for the Advanced NLP Mini-Project.")

st.title("🔍 Fake News Detection AI")
st.write("Enter a news headline or full article below to evaluate its authenticity using our fine-tuned Transformer model.")

user_text = st.text_area("News Content", placeholder="Paste the news article here...", height=200)

if st.button("Analyze Authenticity") and model_loaded:
    if len(user_text.strip()) < 10:
        st.warning("⚠️ Please enter a longer text snippet for accurate analysis.")
    else:
        with st.spinner('Neural network is analyzing semantic patterns...'):
            inputs = tokenizer(user_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                confidence, prediction = torch.max(probabilities, dim=-1)
            
            is_real = prediction.item() == 1
            conf_score = confidence.item() * 100
            
            st.divider()
            if is_real:
                st.markdown('<div class="result-box real-news">✅ This article appears to be REAL NEWS.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box fake-news">🚨 This article appears to be FAKE NEWS.</div>', unsafe_allow_html=True)
            
            st.write("") 
            st.write(f"**AI Confidence Score:** {conf_score:.2f}%")
            st.progress(confidence.item())
