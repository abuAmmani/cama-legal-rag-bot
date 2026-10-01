import streamlit as st

# Configure the web page
st.set_page_config(page_title="CAMA Legal AI", page_icon="⚖️", layout="centered")

# Main Title and Subheader
st.title("⚖️ CAMA Legal Compliance AI Assistant")
st.write("Helping Nigerian small businesses navigate the Companies and Allied Matters Act (CAMA 2020).")

# Chat Interface Placeholder
st.info("The RAG intelligence is being connected. Type a test question below:")
user_input = st.text_input("Ask a CAMA compliance question:")

if user_input:
    st.write(f"🤖 **AI System Response Placeholder:** You asked: '{user_input}'. Soon, I will pull official clauses from the CAMA 2020 documents to answer this accurately!")
