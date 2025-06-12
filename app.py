import streamlit as st
import os
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
import atexit
import shutil

# Load environment variables
load_dotenv()

# Initialize models and components
llm = GoogleGenerativeAI(model="gemini-1.5-flash")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Folder path for temporary document storage
folder_path = "docs"
os.makedirs(folder_path, exist_ok=True)

#Auto delete alll files in docs folder on each run
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"colude not delete {file_path}:{e}")

# Cleanup function to delete .txt files
def delete_text_files(folder_path: str):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

atexit.register(delete_text_files, folder_path)

# Function to scrape and save web pages as .txt
def scrape_and_save(urls: List[str], save_folder: str):
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            filename = os.path.join(save_folder, f"page_{i+1}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")

# Function to load documents from folder
def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            continue
        documents.extend(loader.load())
    return documents

st.title("RAG Chatbot with LangChain & Gemini")

file_type = st.radio("Choose input type:", ["document", "url"])

if file_type == "document":
    uploaded_files = st.file_uploader("Upload documents (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(folder_path, file.name), "wb") as f:
                f.write(file.read())
        docs = load_documents(folder_path)
        st.success(f"Loaded {len(docs)} documents.")

elif file_type == "url":
    urls_input = st.text_area("Enter URLs (comma separated):")
    if st.button("Scrape URLs") and urls_input:
        urls = [url.strip() for url in urls_input.split(",") if url.strip()]
        scrape_and_save(urls, folder_path)
        docs = load_documents(folder_path)
        st.session_state.docs = docs  # ✅ persist docs
        st.success(f"Scraped and loaded {len(docs)} documents.")
        st.success(f"Scraped and loaded {len(docs)} documents.")
        st.write("Sample doc preview:")#debuS
        st.write(docs[0].page_content[:500] if docs else "No docs loaded.")
        
if os.path.exists('.chroma_db'):
    try:
        shutil.rmtree('.chroma_db')
        print("chromdb deleted")
    except Exception as e:
        print(f"Could not delete .chroma_db: {e}")

# Proceed if documents are loaded
if "docs" in st.session_state:
    docs = st.session_state.docs

    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        collection_name='my_collection',
        documents=splits,
        embedding=embedding_function,
        persist_directory='.chroma_db'
    )

    st.success("chromdb create for url")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    base_prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Answer:
    """)

    context_q_system_prompt = """
    Given a chat history and latest user question which might reference context in chat history,
    formulate a standalone question which can be understood without the chat history.
    Do not answer the question, just rephrase it if needed.
    """

    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask your question:")
    if st.button("Submit") and user_question:
        result = rag_chain.invoke({"input": user_question, "chat_history": st.session_state.chat_history})
        st.success("done")
        st.session_state.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=result['answer'])
        ])
        st.markdown(f"**You:** {user_question}")
        st.markdown(f"**Assistant:** {result['answer']}")
else:
    st.warning("No documents found. Did you scrape URLs or upload files?")
