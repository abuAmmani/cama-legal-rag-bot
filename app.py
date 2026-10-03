import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load secret keys from your hidden .env file
load_dotenv()

# Page setting configurations
st.set_page_config(page_title="CAMA Legal AI", page_icon="⚖️", layout="centered")

st.title("⚖️ CAMA Legal Compliance AI Assistant")
st.write("Helping Nigerian small businesses navigate the Companies and Allied Matters Act (CAMA 2020).")

# 1. Connect to our localized vector database memory folder
DB_DIR = "cama_vector_db"

if not os.path.exists(DB_DIR):
    st.warning("⚠️ Database memory storage folder not found locally. Please run 'python index_db.py' to generate your searchable vector indexes.")
else:
    # Initialize the matching embedding model 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Connect directly to the existing local database
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    st.success("✅ Connected to the official CAMA 2020 Database Pipeline.")

    # 2. Main search prompt interface
    user_query = st.text_input("Ask a business compliance or regulatory question:")

    if user_query:
        with st.spinner("Searching the CAMA 2020 legal provisions..."):
            # Fetch the top 3 most contextually relevant legal paragraphs matching the query
            matching_clauses = db.similarity_search(user_query, k=3)
            
            st.markdown("### 🔍 Relevant CAMA 2020 Clauses Discovered:")
            
            for index, clause in enumerate(matching_clauses):
                st.info(f"**Source Page: {clause.metadata.get('page', 'Unknown')}**\n\n{clause.page_content}")
                
            st.caption("🤖 Next Step: These extracted context documents will be passed directly into the LLM chain to generate your structured response.")
