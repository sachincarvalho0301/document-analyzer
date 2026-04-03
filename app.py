import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import time
import os

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Financial Analyzer (RAG)", layout="wide")  # only once!

st.title("📊 Document Analyzer (RAG)")
st.markdown("Upload a PDF and get AI-powered insights using Retrieval-Augmented Generation.")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    "This tool uses RAG (Retrieval-Augmented Generation) to analyze financial PDFs with higher accuracy."
)

# -------------------------------
# CHECK API KEY
# -------------------------------
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ Please set GROQ_API_KEY in environment")
    st.stop()

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    st.write(f"📄 Uploaded File: {uploaded_file.name}")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    st.success("✅ PDF loaded successfully!")

    # -------------------------------
    # DOCUMENT INFO
    # -------------------------------
    st.write(f"📄 Pages: {len(documents)}")

    # -------------------------------
    # CREATE RAG PIPELINE
    # -------------------------------
    with st.spinner("🔄 Creating embeddings... (one-time process)"):

        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()

        if not texts:
            st.error("❌ Could not extract text from this PDF. It may be a scanned/image-based PDF. Please try a text-based PDF.")
            st.stop()

        vector_db = FAISS.from_documents(texts, embeddings)

    st.success("✅ RAG system ready!")

    # -------------------------------
    # USER INPUT
    # -------------------------------
    question = st.text_input("💬 Ask something about the document (optional)")

    if st.button("🧪 Test API"):
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Hello"}]
            )
            st.success("✅ API Working")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ API Error: {e}")

    mode = st.selectbox(
        "🎯 Select Mode",
        ["Insights", "Summary", "Ask Question"]
    )

    # -------------------------------
    # GENERATE BUTTON
    # -------------------------------
    if st.button("🚀 Generate Output"):

        if mode == "Ask Question" and not question:
            st.warning("⚠️ Please enter a question")
        else:
            with st.spinner("Analyzing with RAG..."):
                try:
                    start = time.time()

                    client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # ← to this

                    # -------------------------------
                    # TASK LOGIC
                    # -------------------------------
                    if mode == "Insights":
                        task = "Return EXACTLY 5 financial insights."

                    elif mode == "Summary":
                        task = "Summarize the document clearly."

                    else:
                        task = f"""Answer the following question based ONLY on the document.

Question: {question}

Give a clear and direct answer."""

                    # -------------------------------
                    # RETRIEVAL (CORE RAG)
                    # -------------------------------
                    query = question if question else "financial insights"

                    docs = vector_db.similarity_search(query, k=5)

                    context = "\n".join([doc.page_content for doc in docs])

                    # -------------------------------
                    # PROMPT
                    # -------------------------------
                    prompt = f"""You are a financial analyst.

{task}

STRICT RULES:
- Do NOT include numbers
- Very short
- No repetition

Context:
{context}
"""

                    # -------------------------------
                    # MODEL CALL
                    # -------------------------------
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )

                    result = response.choices[0].message.content

                    end = time.time()

                    # -------------------------------
                    # OUTPUT
                    # -------------------------------
                    st.divider()

                    if mode == "Insights":
                        st.subheader("📊 Key Financial Insights")
                    elif mode == "Summary":
                        st.subheader("📝 Summary")
                    else:
                        st.subheader("💬 Answer")

                    # Clean bullet output
                    clean_lines = []
                    for line in result.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        if line.lower().startswith("most important"):
                            continue
                        clean_lines.append(line)

                    for line in clean_lines:
                        st.markdown(f"• {line}")

                    # -------------------------------
                    # SHOW RETRIEVED CONTEXT (IMPRESSIVE)
                    # -------------------------------
                    with st.expander("🔍 Retrieved Context"):
                        st.write(context)

                    # -------------------------------
                    # TIME
                    # -------------------------------
                    st.caption(f"⏱ Generated in {round(end - start, 2)} seconds")

                    # -------------------------------
                    # DOWNLOAD
                    # -------------------------------
                    st.download_button(
                        label="📥 Download Output",
                        data=result,
                        file_name="financial_output.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")

# -------------------------------
# RESET
# -------------------------------
if st.button("🔄 Reset"):
    st.rerun()