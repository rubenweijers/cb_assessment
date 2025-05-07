import streamlit as st
from dotenv import load_dotenv
import os
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from PIL import Image

FILES_DIR = "files"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = "faiss_index"

prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
    Je bent een vriendelijke en behulpzame klantenservice chatbot...
    Context:
    {context}
    Gespreksgeschiedenis:
    {chat_history}
    Vraag:
    {question}
    """
)

st.set_page_config(page_title="Coolblue Support Chatbot", page_icon="favicon.png", layout="wide")
st.image(Image.open("favicon.png"), width=64)

def load_documents(data_dir):
    docs = []
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader,
                             show_progress=False, use_multithreading=True)
    docs.extend(loader.load())
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".json":
            data = json.load(open(path, encoding="utf-8"))
            records = data if isinstance(data, list) else [data]
            for i, rec in enumerate(records):
                content = "\n".join(f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in rec.items())
                docs.append(Document(page_content=content, metadata={"source": fname, "index": i}))
        if ext == ".csv":
            df = pd.read_csv(path)
            for i, rec in enumerate(df.to_dict(orient="records")):
                content = "\n".join(f"{k}: {v}" for k, v in rec.items())
                docs.append(Document(page_content=content, metadata={"source": fname, "index": i}))
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs) if docs else []

def get_vector_store(texts, embedder):
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embedder, allow_dangerous_deserialization=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    store = FAISS.from_documents(texts, embedder)
    store.save_local(VECTOR_STORE_PATH)
    return store

load_dotenv()
key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(openai_api_key=key, model_name="gpt-4o", temperature=0.3)
emb = OpenAIEmbeddings(openai_api_key=key)

if "vector_store" not in st.session_state:
    docs = load_documents(FILES_DIR)
    texts = chunk_documents(docs)
    st.session_state.vector_store = get_vector_store(texts, emb)

if "retriever" not in st.session_state:
    st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

if "chain" not in st.session_state:
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"}
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey, ik ben Sanne. Waar kan ik je mee helpen?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Stel een vraag"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages[:-1]
        ]
        result = st.session_state.chain.invoke({"question": user_input, "chat_history": history})
        answer = result["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
