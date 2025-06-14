# RAG Chatbot using LangChain & Gemini

This is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, Gemini (Google Generative AI), and Chroma as the vector database. The chatbot can answer questions based on uploaded documents (`.pdf`, `.docx`, `.txt`) or scraped web pages (via URL input).

---

## Features

- Upload and chat with `.pdf`, `.docx`, or `.txt` documents.
- Input URLs to scrape and extract content for QA.
- Maintains chat history for better context in multi-turn conversations.
- Uses Gemini (`gemini-1.5-flash`) for both question rephrasing and answering.
- Text splitting with chunking and overlap for context retention.
- Stores document embeddings in ChromaDB.
- Clean interface built with Streamlit.

---

## Tech Stack

- Streamlit
- LangChain
- Google Generative AI (Gemini)
- Chroma VectorDB
- BeautifulSoup (for web scraping)
- PyPDFLoader
- dotenv (for environment variable handling)

---

## Folder Structure

├── docs/ # Temporary folder to store uploaded/scraped files
├── .chroma_db/ # ChromaDB persistence directory
├── app.py # Main Streamlit![Screenshot 2025-06-14 185337](https://github.com/user-attachments/assets/920b4b95-5d9f-407a-9c41-c20c868af7c4)


https://github.com/user-attachments/assets/cac906cb-ba73-40a3-85da-142171d840c4


├── README.md # Project documentation
└── .gitignore # Specifies untracked files like .env to ignore
