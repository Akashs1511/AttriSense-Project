# ✅ Full Project Essentials

# 🧩 AttriSense – Employee Attrition Risk Predictor with GenAI

AttriSense is an **intelligent HR analytics tool** that predicts employee attrition risk and uses OpenAI GPT to provide actionable HR strategies.

---

## 🚀 Features

✅ Secure Login  

✅ Upload HR CSV Data  

✅ Predict Attrition Risk (High/Low) 

✅ Separate Analysis Report of each type   

✅ Download CSV & PDF Reports  

✅ Send Report via Email 

✅ Email Report in Tabular List of each High Rish emp. in PDF    

✅ Report for Improvment Suggestions  

✅ Simplified clear Report Access  

✅ GenAI HR Advisor for improvement suggestions  


---

## 📂 Project Structure

AttriSense/
│
├── app.py
├── attrisense_rf_model.pkl
├── .env
├── requirements.txt
├── example.csv
└── README.md

---

## ⚙️ Requirements

- Python 3.9+
- OpenAI API Key
- Gmail App Password (optional, for yagmail)

---

## 🗂️ Installation

```bash
git clone <your-repo-url>
cd AttriSense

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt


# 🧩 Setup

### 1️⃣ Create .env

# .env
OPENAI_API_KEY=sk-xxxxxx
Add your OpenAI API Key.
Never commit .env to Git!

2️⃣ Put your ML model file

Place attrisense_rf_model.pkl in the same directory.

3️⃣ Prepare your example.csv


## Requirements.txt

streamlit
pandas
scikit-learn
joblib
fpdf
yagmail
python-dotenv
langchain-openai
langchain-community
langchain-core
langchain-text-splitters


