import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from process_doc import process_cama_document

# Load API keys from your secret .env file
load_dotenv()

def create_vector_database():
    # 1. Fetch the 1597 text chunks we successfully generated earlier
    chunks = process_cama_document()
    if not chunks:
        print("❌ Database creation aborted: Document chunks missing.")
        return

    # 2. Define where to save the database folder locally
    persist_directory = "cama_vector_db"
    
    # 3. Initialize the OpenAI embedding engine
    print("⏳ Step 3: Initializing OpenAI Embeddings model...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Generate embeddings and save them into ChromaDB
    print(f"⏳ Step 4: Storing 1,597 clauses inside ChromaDB at '{persist_directory}'...")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("✅ Database created and saved successfully! Your RAG memory is locked in.")
    return vector_db

if __name__ == "__main__":
    # Quick guard check for your key
    if os.environ.get("OPENAI_API_KEY") == "your_actual_openai_key_here" or not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Warning: Please replace the placeholder in your '.env' file with a valid OpenAI API Key before running.")
    else:
        create_vector_database()
