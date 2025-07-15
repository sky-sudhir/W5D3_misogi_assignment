import streamlit as st
import requests
import os

FASTAPI_URL = "http://localhost:8000"  # change if hosted elsewhere

st.set_page_config(page_title="Smart Assessment Generator", layout="wide")
st.title("🧠 Advanced Assessment Generator")

# Upload Section
st.subheader("📤 Upload Educational Document")

uploaded_file = st.file_uploader("Upload any document (PDF, DOCX, TXT, PPTX...)", type=None)

if uploaded_file:
    with st.spinner("Uploading and processing..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(f"{FASTAPI_URL}/upload/", files=files)

        if response.status_code == 200:
            st.success(f"✅ Uploaded successfully. Chunks processed: {response.json().get('chunks')}")
        else:
            st.error("❌ Upload failed")

# Generation Section
st.subheader("📝 Generate Assessment")

topic = st.text_input("Topic", placeholder="e.g. Photosynthesis")
objectives = st.text_area("Learning Objectives (comma-separated)", placeholder="e.g. Light reaction, Calvin Cycle")
difficulty = st.selectbox("Difficulty", ["auto", "easy", "medium", "hard"])
user_id = st.text_input("User ID", value="sudhir123")

if st.button("Generate Assessment"):
    if not topic or not objectives or not user_id:
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Generating personalized assessment..."):
            payload = {
                "topic": topic,
                "objectives": [obj.strip() for obj in objectives.split(",")],
                "difficulty": difficulty,
                "user_id": user_id
            }

            res = requests.post(f"{FASTAPI_URL}/generate/", json=payload)
            if res.status_code == 200:
                data = res.json()
                st.success("✅ Assessment Generated!")
                st.markdown("---")
                st.markdown(data["assessment"])
            else:
                st.error("❌ Failed to generate assessment")
