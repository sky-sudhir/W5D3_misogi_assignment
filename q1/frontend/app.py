# streamlit_app.py

import streamlit as st
import requests
import os
import pandas as pd

API_URL = "http://localhost:8000"  # Change if deploying

st.set_page_config(page_title="Sports Analytics RAG", layout="wide")
st.title("⚽ Sports Analytics RAG System")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose Action", ["Ask a Question", "Upload Text", "Upload PDF", "Upload CSV"])

# -------------------------------
# 1. ASK A QUESTION
# -------------------------------
if page == "Ask a Question":
    st.subheader("📊 Ask your sports question")

    query = st.text_area("Enter your question", height=100, placeholder="e.g. Which team has the best defense?")
    if st.button("Ask"):
        if query.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                res = requests.post(f"{API_URL}/ask", json={"query": query})
                if res.status_code == 200:
                    st.success("Answer:")
                    st.markdown(res.json()["response"])
                else:
                    st.error("Something went wrong!")

# -------------------------------
# 2. UPLOAD TEXT FILE
# -------------------------------
elif page == "Upload Text":
    st.subheader("📝 Upload Text Document")

    text_content = st.text_area("Paste or type your document here", height=300)
    source = st.text_input("Source (e.g. match_report_2024.txt)")
    date = st.date_input("Date")

    if st.button("Ingest Text"):
        if text_content.strip() == "" or not source:
            st.warning("Please provide both content and source.")
        else:
            metadata = {"source": source, "date": str(date)}
            res = requests.post(f"{API_URL}/ingest", json={"content": text_content, "metadata": metadata})
            if res.status_code == 200:
                st.success(f"✅ {res.json()['chunks_added']} chunks added to vector DB.")
            else:
                st.error("Failed to ingest text.")

# -------------------------------
# 3. UPLOAD PDF FILE
# -------------------------------
elif page == "Upload PDF":
    st.subheader("📄 Upload PDF Match Report")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    source = st.text_input("Source name", placeholder="e.g. el_clasico_2024.pdf")
    date = st.date_input("Date")

    if st.button("Upload PDF"):
        if uploaded_file is None or not source:
            st.warning("Please upload a file and provide source name.")
        else:
            with st.spinner("Uploading and processing..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                data = {"source": source, "date": str(date)}
                res = requests.post(f"{API_URL}/upload-pdf/", files=files, data=data)
                if res.status_code == 200:
                    st.success(f"✅ {res.json()['chunks_added']} chunks added from PDF.")
                else:
                    st.error("Upload failed!")

# -------------------------------
# 4. UPLOAD CSV FILE
# -------------------------------
elif page == "Upload CSV":
    st.subheader("📈 Upload CSV Stats File")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    source = st.text_input("Source name", placeholder="e.g. team_stats_2024.csv")
    date = st.date_input("Date")

    if st.button("Upload CSV"):
        if uploaded_file is None or not source:
            st.warning("Please upload a file and provide source name.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                csv_text = df.to_csv(index=False)

                metadata = {"source": source, "date": str(date)}
                res = requests.post(f"{API_URL}/upload-pdf", json={"content": csv_text, "metadata": metadata})
                if res.status_code == 200:
                    st.success(f"✅ {res.json()['chunks_added']} chunks added from CSV.")
                else:
                    st.error("Upload failed!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
