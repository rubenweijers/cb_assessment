import streamlit as st
# from openai import OpenAI # Not directly used; ChatOpenAI from langchain is used
from dotenv import load_dotenv
import os
import json # For more flexible JSON loading
import pandas as pd # For CSV loading
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document # To manually create documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA # Replaced by ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain # Added for conversation history
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage # Added for formatting chat history
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

# --- Configuration ---
FILES_DIR = "files"  # Directory containing your knowledge base files
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = "faiss_index"

prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
    "Je bent een vriendelijke en behulpzame klantenservice chatbot van Coolblue genaamd Sanne. "
    "Je doel is om klanten te ondersteunen bij hun supportvragen over producten die ze hebben gekocht, "
    "zoals installatie, retourneren, reparatie of status van bestellingen. Geef duidelijke, beknopte en bruikbare antwoorden, "
    "gebaseerd op de context die je ontvangt. Antwoord in het Nederlands, en alleen in het Engels als de klant in het Engels wil praten. "
    "Wees vriendelijk, persoonlijk. Gebruik bold text en bullet point om duidelijk uit te leggen, gebruik emojis waar gepast. Je werkt voor het meest klantgerichte bedrijf in Nederland, en je hoogste prioriteit is klanttevredenheid, je doet alles voor een glimlach."
    "Als de benodigde informatie voor het verzoek van de gebruiker niet in de context staat, zeg dan eerlijk dat je het niet weet en verwijs naar de klantenservice. "
    "Geef nooit productadvies of commerciële aanbevelingen."
    
    Gebruik de onderstaande context om de vraag te beantwoorden. Als je het antwoord niet weet, zeg dat dan eerlijk.

    Als iets al is uitgelegd in de vorige beurt, herhaal het dan niet opnieuw. Geef een korte bevestiging of vraag of er iets onduidelijk is.

    Context:
    {context}

    Gespreksgeschiedenis:
    {chat_history}

    Vraag:
    {question}
    """
)

st.set_page_config(
    page_title="Coolblue Support Chatbot",
    page_icon="../favicon.png",  # Using an emoji as an example, ensure path is correct if using local file
    layout="wide",   # "centered" or "wide"
    initial_sidebar_state="expanded",  # or "collapsed"
)

# --- Helper Functions for RAG --- (Remain Unchanged)

def load_documents(data_dir):
    """Loads documents from various file types in the specified directory,
       with specific handling for known JSON and CSV formats."""
    documents = []
    loaded_files_info = [] # To keep track of what's loaded

    # 1. PDF Loader
    try:
        pdf_loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False,
            use_multithreading=True,
            silent_errors=True
        )
        pdf_documents = pdf_loader.load()
        if pdf_documents:
            documents.extend(pdf_documents)
            loaded_files_info.append(f"Successfully loaded {len(pdf_documents)} PDF document(s).")
        else:
            loaded_files_info.append("No PDF files found or loaded.")
    except Exception as e:
        loaded_files_info.append(f"Error loading PDFs: {e}")


    # 2. Specific JSON and CSV File Handling
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            try:
                if filename.lower() == "product_support_articles.json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for article in data:
                        content = f"Title: {article.get('title', 'N/A')}\n"
                        tags = article.get('tags')
                        if tags:
                            if isinstance(tags, list):
                                content += f"Tags: {', '.join(tags)}\n"
                            elif isinstance(tags, str):
                                content += f"Tags: {tags.replace('|', ', ')}\n"
                        content += f"Topic: {article.get('topicName', 'N/A')}\n"
                        if article.get('url'):
                            content += f"URL: {article.get('url')}\n"
                        metadata = {
                            "source": filename,
                            "id": article.get("id"),
                            "type": article.get("articleType"),
                            "language": article.get("lang"),
                            "lastReviewedAt": article.get("lastReviewedAt"),
                            "categoryId": article.get("categoryId")
                        }
                        documents.append(Document(page_content=content, metadata=metadata))
                    loaded_files_info.append(f"Processed Product Support Articles JSON: {filename}")

                elif filename.lower() == "faq.json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for item in data:
                        content = f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', '')}"
                        metadata = {"source": filename}
                        documents.append(Document(page_content=content, metadata=metadata))
                    loaded_files_info.append(f"Processed Q&A JSON: {filename}")

                elif filename.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(file_path)
                        for index, row in df.iterrows():
                            label = row.get('LabelCategoryItem', 'N/A')
                            item_desc = row.get('Item', 'N/A')
                            text_content = f"Topic: {item_desc}\n"
                            text_content += f"Category Details: {label}\n"
                            metadata = {
                                "source": filename,
                                "row": index,
                                "label_category_item": label,
                                "total_clicks": row.get('Total clicks'),
                                "contact_ratio": row.get('Contact ratio'),
                                "total_contacts": row.get('Total contacts')
                            }
                            documents.append(Document(page_content=text_content, metadata=metadata))
                        loaded_files_info.append(f"Processed CSV: {filename} with {len(df)} rows.")
                    except Exception as e:
                        loaded_files_info.append(f"Error processing CSV {filename}: {e}")
            except Exception as e:
                loaded_files_info.append(f"Error loading or parsing file {filename}: {e}")

    # Messages will be displayed during the initialization phase for visibility
    # This part will be called within the "vector_store_initialized" block later

    if not documents:
        # This warning will be shown if no documents are loaded at all.
        # Specific file loading messages are in loaded_files_info.
        pass # The main warning will be handled in the initialization block.
    return documents, loaded_files_info

def split_documents(documents):
    """Splits documents into smaller chunks."""
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    return texts

def create_or_load_vector_store(texts, embeddings_model):
    """Creates a FAISS vector store from text chunks or loads an existing one."""
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        st.info(f"Loading existing vector store from {VECTOR_STORE_PATH}...")
        try:
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
            st.success("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            st.error(f"Error loading vector store: {e}. Rebuilding...")
            if os.path.exists(VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(VECTOR_STORE_PATH) # Remove old potentially corrupt index
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True) # Ensure dir exists for new store

    st.info("Creating new vector store...")
    if not texts:
        st.error("No texts available to create vector store. Please check document loading and splitting.")
        return None
    try:
        vector_store = FAISS.from_documents(texts, embeddings_model)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        st.success("New vector store created and saved.")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


# --- Streamlit App ---
st.title("Coolblue Klantenservice")

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Initialize LLM and Embeddings
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0.7)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load and process documents on first run or if vector store is not found
if "vector_store_initialized" not in st.session_state:
    with st.spinner("Initializing knowledge base... This may take a moment on first run."):
        if not os.path.exists(FILES_DIR) or not os.listdir(FILES_DIR):
            st.error(f"The '{FILES_DIR}' directory is missing or empty. Please create it and add your support documents.")
            st.session_state.vector_store = None
        else:
            raw_documents, loaded_files_info_msgs = load_documents(FILES_DIR)
            for msg in loaded_files_info_msgs: # Display loading info within spinner
                st.info(msg)

            if raw_documents:
                text_chunks = split_documents(raw_documents)
                if text_chunks:
                    st.session_state.vector_store = create_or_load_vector_store(text_chunks, embeddings)
                else:
                    st.error("Failed to split documents into text chunks. Vector store cannot be created.")
                    st.session_state.vector_store = None
            else:
                st.warning("No documents were loaded. Cannot build vector store.") # More prominent warning
                st.session_state.vector_store = None
        st.session_state.vector_store_initialized = True # Mark as initialized

if "retriever" not in st.session_state and st.session_state.get("vector_store"):
    st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3

# --- MODIFIED: Initialize ConversationalRetrievalChain ---
if "conversational_qa_chain" not in st.session_state and st.session_state.get("retriever"):
    try:
        st.session_state.conversational_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
                }
            # The chain will condense the question and use chat_history.
            # Default prompts are usually good for this.
            # You can customize prompts via `combine_docs_chain_kwargs` and `question_generator_chain_kwargs` if needed.
        )
        st.success("Conversational RAG chain initialized.")
    except Exception as e:
        st.error(f"Failed to create Conversational QA chain: {e}")
        st.session_state.conversational_qa_chain = None


# Initialize chat history (stores dicts: {"role": "user/assistant", "content": "..."})
# Initialize chat history with intro message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey, ik ben Sanne, de support chatbot van Coolblue. Waar kan ik je mee helpen?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Stel een vraag"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.get("conversational_qa_chain"):
            response_content = "Sorry, the conversational RAG chain is not available. Please check the setup and document loading."
            st.error(response_content)
        else:
            with st.spinner("Thinking..."):
                try:
                    # --- MODIFIED: Prepare chat history for the chain ---
                    # ConversationalRetrievalChain expects history as a list of BaseMessages or (human_msg, ai_msg) tuples.
                    # We extract messages prior to the current user prompt.
                    
                    # `st.session_state.messages` contains the full history including the new prompt.
                    # We need history *before* this new prompt for the `chat_history` argument.
                    previous_messages = st.session_state.messages[:-1] # All but the last (current) user message

                    formatted_chat_history = []
                    for msg_dict in previous_messages:
                        if msg_dict["role"] == "user":
                            formatted_chat_history.append(HumanMessage(content=msg_dict["content"]))
                        elif msg_dict["role"] == "assistant":
                            formatted_chat_history.append(AIMessage(content=msg_dict["content"]))
                    
                    # Invoke the chain with the current question and the formatted chat history
                    result = st.session_state.conversational_qa_chain.invoke({
                        "question": prompt,
                        "chat_history": formatted_chat_history
                    })
                    
                    # The output of ConversationalRetrievalChain is a dict, typically with an 'answer' key.
                    response_content = result.get("answer", "Sorry, I couldn't find a specific answer to that in my current documents.")

                    # --- Optional: Display source documents ---
                    # st.sidebar.subheader("Retrieved Sources for this response:")
                    # if result.get("source_documents"):
                    #     for i, doc in enumerate(result["source_documents"]):
                    #         source_name = doc.metadata.get('source', 'N/A')
                    #         source_preview = doc.page_content[:200] # Show an excerpt
                    #         with st.sidebar.expander(f"Source {i+1}: {source_name}"):
                    #             st.markdown(f"**File:** {source_name}")
                    #             # You could add more metadata here if useful, e.g., row for CSVs
                    #             st.markdown("---")
                    #             st.markdown(source_preview + "...")
                    # else:
                    #     st.sidebar.markdown("No specific sources were retrieved for this response.")

                except Exception as e:
                    st.error(f"Error during RAG chain processing: {e}")
                    response_content = f"An error occurred while trying to answer your question. Details: {str(e)}" # Ensure error is a string
        st.markdown(response_content)
    # Append assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response_content})