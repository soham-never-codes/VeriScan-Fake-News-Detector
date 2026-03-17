# VeriScan-Fake-News-Detector
An end-to-end NLP pipeline using DistilBERT and Streamlit to classify fake news.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## 📌 Project Overview
The rapid spread of misinformation requires automated, scalable, and highly accurate detection systems. Developed as a 10-day mini-project, this repository contains an end-to-end AI engineering and UI/UX pipeline that classifies news articles as 'Real' or 'Fake' using state-of-the-art Transformer architecture. 

## 🚀 The Engineering Approach
1. **Data Processing:** Ingested the Kaggle Fake News dataset. Engineered a custom feature combining the `Title` and `Text` columns to maximize contextual understanding, ensuring critical information wasn't lost due to token limits.
2. **Tokenization & Batching:** Utilized the HuggingFace `AutoTokenizer` with a hard truncation limit of 512 tokens to match BERT's maximum sequence length. Implemented `DataCollatorWithPadding` for dynamic, memory-efficient batching.
3. **Model Fine-Tuning:** Fine-tuned `distilbert-base-uncased` using the HuggingFace `Trainer` API with mixed-precision (`fp16`) for accelerated GPU training.
4. **Evaluation Strategy:** Prioritized F1-score and Recall to specifically monitor and minimize False Positives (flagging real news as fake), which is critical for user trust.
5. **UI/UX Deployment:** Built an interactive, user-friendly web application using Streamlit. Deployed securely via Cloudflare Tunnels to ensure stable WebSocket connections and instant inference feedback.

## 📊 Model Performance
After fine-tuning on a T4 GPU, the model achieved the following on the unseen test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 100% |
| **F1-Score** | [100 %] |
| **Precision** | [100 %] |
| **Recall** | [100 %] |

*(See `Fake_News_Training_Pipeline.ipynb` for the full classification report and confusion matrix).*

## 🧠 Error Analysis & Key Learnings
During model evaluation, I extracted the misclassified samples and identified a key failure pattern:
* **The Sarcasm Blindspot:** The model occasionally struggles with high-quality satire. Because DistilBERT maps semantic structure, it was tricked by the professional formatting and grammar of satirical articles. This proves that while Transformers excel at syntax, they still have blind spots with deep, real-world irony.

## 💻 How to Run the App Locally
1. Clone the repository: 
   `git clone https://github.com/soham-never-codes/VeriScan-Fake-News-Detector.git`
2. Install the required dependencies: 
   `pip install -r requirements.txt`
3. Launch the Streamlit interface: 
   `streamlit run app.py`

---
*Built for the Advanced NLP & UI/UX Engineering Mentorship.*
