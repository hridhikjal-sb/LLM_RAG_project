from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from re import template
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_core.messages import HumanMessage,AIMessage
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
import atexit


load_dotenv()
llm = GoogleGenerativeAI(model="gemini-1.5-flash")
# response = llm.invoke("Hello, how are you?")
# print(response)


text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)

def scrape_and_save(urls: List[str], save_folder: str):
    os.makedirs(save_folder, exist_ok=True)
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract readable text
            text = soup.get_text(separator='\n', strip=True)

            # Save to .txt file
            filename = os.path.join(save_folder, f"page_{i+1}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Error scraping {url}: {e}")

def delete_text_files(folder_path: str):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


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
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

# ====================
# USER INPUT SECTION
# ====================
file_type = input("What is your file type (document/url): ").strip().lower()

# Set destination folder
folder_path = r"C:\Users\VICTUS\OneDrive\Desktop\LLM_docs\docs"

if file_type == "document":
    docs = load_documents(folder_path)
    print(f"Loaded {len(docs)} documents")

elif file_type == "url":
    urls = input("Enter one or more URLs separated by commas:\n").split(",")
    urls = [url.strip() for url in urls if url.strip()]
    scrape_and_save(urls, folder_path)
    docs = load_documents(folder_path)
    print(f"Loaded {len(docs)} documents from web")

else:
    print("Invalid input. Skipping document load.")
    quit()

# ====================
# TEXT SPLITTING
# ====================

splits = text_splitter.split_documents(docs)
print(f"Split documents into {len(splits)} chunks")


#setting up vector store chromadb and embedding
#set ip the embedding model useing google generative AI embeddings
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
collection_name ='my_coolection'
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_function,
    persist_directory='.chroma_db'
)
print("vector store created to '.chroma_db'")

#covert the vector store into a retriever that can perform semantic search

retriever = vectorstore.as_retriever(search_kwarg={"k":2})

template="""

Answer the question based only on the following context :
{context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)


RAG_chain =(
    {"context":retriever,"question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)
# question="which college does the person studies?"
# response=RAG_chain.invoke(question)
# print(f"question: {question}")
# print(f"response: {response}")

#system instruction for rephrasing a question using prior chat history
context_q_system_prompt="""
Given a chat history and latest user question
which might refernce context in chat history,
formulate a standalone quesion which can be understood without the chat history.
Do not answer the question,
just formulate it if needed and otherwise return as it is.

"""

#define a prompt that includes the system instruction chat history and new user message

context_q_prompt = ChatPromptTemplate.from_messages([
    "system",context_q_system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")
])

# #checking
# context_chain=context_q_prompt | llm | StrOutputParser()
# print(context_chain.invoke({"input":"what is your number","chat_history":[]}))

history_aware_retriever = create_history_aware_retriever(
    llm,retriever,context_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpfull AI assistant. use following context to answer the user's question."),
     ("system","Context:{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
RAG_chain =create_retrieval_chain(history_aware_retriever,question_answer_chain)

chat_history =[]
question1 = input("Question:")
answer1 =RAG_chain.invoke({"input":question1,"chat_history":chat_history})['answer']

chat_history.extend([
    HumanMessage(content=question1),
    AIMessage(content=answer1)
])

print(question1)
print(answer1)
# Register the cleanup to run on program exit
atexit.register(delete_text_files, folder_path)

