import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# Streamlit UI
st.title("📊 Market Analyst")
st.sidebar.title("🔗 Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

button = st.sidebar.button("🔄 Process URLs")
file_path = "vectorstore-faiss-huggingface.pkl"

main_placeholder = st.empty()
llm=ChatGroq(model="moonshotai/kimi-k2-instruct")

# Process button logic
if button and urls:
    # Step 1: Load data from URLs
    loader = WebBaseLoader(urls)
    main_placeholder.text("🔄 Loading data from URLs...")
    data = loader.load()
    # Step 2: Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text("✂️ Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    # Step 3: OpenAI Embeddings
    if not docs:
        st.error("❌ Failed to extract content from the provided URLs. Please check the URLs or try different ones.")
    else:
        main_placeholder.text("🧠 Creating embeddings using HuggingFace...")
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Step 4: Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Step 5: Save vector store
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        main_placeholder.text("✅ Processing complete. You can now ask questions.")

# Query UI
query = main_placeholder.text_input("💬 Ask a question based on your provided articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)

            st.header("🧠 Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources")
                for source in sources.split("\n"):
                    st.write(source)
    else:
        st.error("Vector store not found. Please click 'Process URLs' first.")


#https://www.moneycontrol.com/automobile/tesla-model-y-buying-e-suv-in-delhi-gurugram-or-mumbai-this-is-how-much-you-will-have-to-pay-article-13449866.html
#https://www.moneycontrol.com/stocksmarketsindia/
#https://www.moneycontrol.com/news/business/ipo/gem-aromatics-sets-price-band-of-rs-309-325-a-share-for-ipo-13447334.html