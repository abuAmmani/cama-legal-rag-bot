import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_cama_document():
    pdf_path = "cama_2020.pdf"
    
    # 1. Check if the user placed the file in the correct directory
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Please download CAMA 2020 and save it as '{pdf_path}'")
        return None

    print("⏳ Step 1: Loading official CAMA 2020 PDF text...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from the legal document.")

    # 2. Split the massive law book into smaller chunks so the LLM can read them
    print("⏳ Step 2: Splitting legal text into manageable clauses...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Character length of each text block
        chunk_overlap=150,     # Overlap to preserve contextual legal phrasing
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Successfully generated {len(chunks)} searchable text chunks.")
    return chunks

if __name__ == "__main__":
    process_cama_document()
