import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load keys from your local hidden configuration file
load_dotenv()

st.set_page_config(page_title="CAMA Legal AI", page_icon="⚖️", layout="centered")

st.title("⚖️ CAMA Legal Compliance AI Assistant")
st.write("Helping Nigerian small businesses navigate the Companies and Allied Matters Act (CAMA 2020).")

DB_DIR = "cama_vector_db"

if not os.path.exists(DB_DIR):
    st.warning("⚠️ Database storage folder not found locally. Please run 'python index_db.py' first.")
elif not os.environ.get("GROQ_API_KEY"):
    st.error("❌ Setup incomplete: Please add your free GROQ_API_KEY to your local hidden '.env' file.")
else:
    # 1. Connect to our free local vector storage
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 2. Setup our free blazing-fast cloud LLM text engine (Llama-3)
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2)

    # 3. Design professional system guardrails for a legal compliance expert assistant
    system_prompt = (
        "You are an expert legal assistant specializing in Nigerian corporate compliance under the CAMA 2020 Act.\n"
        "Use the following retrieved official legal clauses to answer the small business owner's question accurately.\n"
        "If you do not know the answer, say clearly that you cannot find it in the current text framework and advise consulting a lawyer.\n"
        "Structure your response cleanly using bullet points or paragraphs.\n\n"
        "Contextual Official Clauses:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Bind the retrieval data pipeline and the LLM generation chains together
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.success("✅ Free CAMA Database & Llama-3 AI Engine Fully Active.")

    # 5. Live UI input execution space
    user_query = st.text_input("Ask a business compliance or regulatory question:")

    if user_query:
        with st.spinner("Analyzing CAMA regulations and drafting response..."):
            response = rag_chain.invoke({"input": user_query})
            
            st.markdown("### 🤖 Legal Assistant Compliance Advice:")
            st.write(response["answer"])
            
            # Show original source material transparently beneath the response
            with st.expander("🔍 View Official CAMA Source Material Utilized"):
                for doc in response["context"]:
                    st.markdown(f"**Page Reference: {doc.metadata.get('page', 'N/A')}**")
                    st.info(doc.page_content)
