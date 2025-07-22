import os
import streamlit as st
import requests
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import yagmail
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENAI_API_KEY not set. Please configure it in your .env file.")
    st.stop()
 
# if not OPENAI_API_KEY:
#     st.error("OPENAI_API_KEY not set. Please configure it in your .env file.")
#     st.stop()
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# if not ANTHROPIC_API_KEY:
#     st.error("🚨 ANTHROPIC_API_KEY not set! Please set it as an environment variable.")
#     st.stop()

# Simple user credentials (plaintext for demo)
valid_users = {
    "akash": "test123",
    "john": "johnpass",
    "emma": "emma456"
}

st.set_page_config(page_title="AttriSense", layout="wide")

# Initialize session state for login
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Login UI
if not st.session_state["auth"]:
    st.title("AttriSense Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in valid_users and password == valid_users[username]:
            st.session_state["auth"] = True
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# Main app for authenticated users
if st.session_state["auth"]:
    st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state["auth"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.title("AttriSense: Employee Attrition Risk Predictor")

    st.markdown("""
    ### About

    AttriSense predicts attrition risk and uses OpenAI GPT for actionable HR suggestions.

    Developed by Akash Singh.
    """)

    # Email config (edit as needed)
    HR_EMAIL = "info.andisoftwaresolutions@gmail.com"
    SENDER_EMAIL = "akash02155@gmail.com"
    APP_PASSWORD = "app password"
    
    # for single employee check
    with st.form("Single Employee Check"):
        age = st.number_input("Age", 18, 100)
        satisfaction = st.slider("Job Satisfaction", 1, 5)
        distance = st.number_input("Distance From Home", 0, 100)
        # other fields...

        submitted = st.form_submit_button("Check Risk")

        if submitted:
            payload = {
                "Age": age,
                "JobSatisfaction": satisfaction,
                "DistanceFromHome": distance,
                # ...
            }
            response = requests.post("http://127.0.0.1:8000/check_employee", json=payload)
            risk = response.json()["attrition_risk"]
            st.success(f"✅ Predicted Attrition Risk: {risk}")

    # Upload CSV
    uploaded_file = st.file_uploader("Upload your HR data CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df.head())

        # Load trained model
        model = joblib.load('attrisense_rf_model.pkl')

        # Prepare data: Encode categorical columns
        df_encoded = df.copy()
        le = LabelEncoder()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns

        if 'Attrition' in categorical_cols:
            categorical_cols = categorical_cols.drop('Attrition')
        if 'Attrition' in df_encoded.columns:
            df_encoded = df_encoded.drop('Attrition', axis=1)

        for col in categorical_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col])

        # Make predictions
        predictions = model.predict(df_encoded)
        df['Attrition Risk'] = ['High' if p == 1 else 'Low' for p in predictions]

        st.write("Attrition Predictions Summary:")
        st.dataframe(df[['Attrition Risk']].value_counts().reset_index())
        
        # Some More Analysis:
        if st.checkbox("🚦 Show Key Drivers of Attrition"):
            high_risk = df[df['Attrition Risk'] == 'High']
            low_risk = df[df['Attrition Risk'] == 'Low']

            st.write(f"🔹 High Risk Employees: {len(high_risk)}")
            st.write(f"🔹 Low Risk Employees: {len(low_risk)}")

            st.write("**Average values for High Risk group:**")
            st.write(high_risk.describe())

            st.write("**Compare with Low Risk group:**")
            st.write(low_risk.describe())

        # Download results
        st.download_button(
            label="Download Results CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='attrisense_predictions.csv',
            mime='text/csv'
        )

        # Generate PDF report
        
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="AttriSense - Employee Attrition Report", ln=True, align='C')
            pdf.set_font("Arial", "I", 12)
            pdf.cell(200, 10, txt="ANDi Software Solutions | Hackathon", ln=True, align='C')
            pdf.ln(10)

            # Filter high risk employees
            high_risk_df = df[df['Attrition Risk'] == 'High']

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt="List of Employees at High Attrition Risk:", ln=True)
            pdf.set_font("Arial", "", 12)

            if not high_risk_df.empty:
                max_rows = 300
                table_columns = ["EmployeeNumber", "Age", "YearsSinceLastPromotion", "YearsAtCompany", "DistanceFromHome"]

                # ✅ Use manual widths for each column (in mm)
                col_widths = [34, 20, 49, 35, 40]  # adjust as needed to fit your page

                # Table header
                pdf.set_fill_color(200, 200, 200)  # light gray background
                pdf.set_font("Arial", "B", 10)
                for i, col in enumerate(table_columns):
                    pdf.cell(col_widths[i], 10, col, border=1, align='C', fill=True)
                pdf.ln()

                # Table rows
                pdf.set_font("Arial", "", 10)
                for idx, (_, row) in enumerate(high_risk_df.iterrows()):
                    if max_rows <= 0:
                        pdf.cell(0, 10, txt="... (truncated for length)", ln=True)
                        break
                    for i, col in enumerate(table_columns):
                        value = str(row[col]) if col in row else "N/A"
                        pdf.cell(col_widths[i], 10, value, border=1, align='C')
                    pdf.ln()
                    max_rows -= 1
            else:
                pdf.cell(0, 10, txt="No employees at High Attrition Risk.", ln=True)

            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt="Summary:", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, txt=f"Total Records: {len(df)}", ln=True)
            pdf.cell(0, 10, txt=f"High Risk: {sum(df['Attrition Risk'] == 'High')}", ln=True)
            pdf.cell(0, 10, txt=f"Low Risk: {sum(df['Attrition Risk'] == 'Low')}", ln=True)

            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt="Suggestions to Manage High Risk Employees:", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, txt=(
                "- Schedule regular 1:1 check-ins\n"
                "- Offer flexible work arrangements\n"
                "- Recognize & reward contributions\n"
                "- Provide clear career growth paths\n"
                "- Foster open communication and trust"
            ))

            pdf_output = "AttriSense_Report.pdf"
            pdf.output(pdf_output)

            st.session_state['pdf_generated'] = True
            st.session_state['pdf_path'] = pdf_output

            with open(pdf_output, "rb") as f:
                st.download_button("Download PDF Report", f, file_name=pdf_output)

            st.success("✅ Detailed PDF Report with Table generated!")


        # Send report via email
        if st.session_state.get('pdf_generated', False):
            if st.button("Send Report to HR"):
                try:
                    yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
                    yag.send(
                        to=HR_EMAIL,
                        subject="AttriSense - Employee Attrition Report",
                        contents="Attached is the latest attrition report.",
                        attachments=st.session_state['pdf_path']
                    )
                    st.success(f"Report sent to {HR_EMAIL}")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

        # OpenAI GPT HR recommendations
        st.header("Smart HR Recommendations")

        context_text = df.head(20).to_csv(index=False)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_text(context_text)
        doc_objs = [Document(page_content=d) for d in docs]

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(doc_objs, embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(model="qwen/qwen3-32b:free")
        # llm = ChatAnthropic(model_name="claude-3-sonnet-20240229")

        prompt = ChatPromptTemplate.from_template("""
        You are an HR strategy advisor.
        Analyze this HR attrition data and provide actionable recommendations:
        - Identify weaknesses in engagement
        - Suggest HR policy improvements
        - Recommend fun team-building ideas
        - Tips for better collaboration and retention

        Question: {question}
        Context: {context}
        Answer in 200+ words.
        """)

        user_query = st.text_area("Ask the HR Advisor")

        if st.button("Generate GPT Recommendations"):
            relevant_docs = retriever.get_relevant_documents(user_query)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            final_prompt = prompt.format_messages(question=user_query, context=context)
            response = llm.invoke(final_prompt)
            st.write(response.content)
