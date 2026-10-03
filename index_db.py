import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from process_doc import process_cama_document

def create_vector_database():
    # 1. Fetch your 1,597 text chunks
    chunks = process_cama_document()
    if not chunks:
        print("❌ Database creation aborted: Document chunks missing.")
        return

    persist_directory = "cama_vector_db"
    
    # 2. Use a highly efficient, free open-source embedding model that runs locally
    print("⏳ Step 3: Initializing free local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Generate vectors and save them locally
    print(f"⏳ Step 4: Storing 1,597 clauses inside ChromaDB at '{persist_directory}'...")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("✅ Local database created and saved successfully! Your RAG memory is locked in for free.")
    return vector_db

if __name__ == "__main__":
    create_vector_database()
