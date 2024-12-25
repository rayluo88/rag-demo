import os
import gradio as gr
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

class RAGChat:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.vector_store = None
        
    def process_file(self, file_path: str) -> List:
        """Process a single file and return documents."""
        # Choose appropriate loader based on file extension
        if file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
            
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
        
    def update_vector_store(self, documents: List):
        """Update vector store with new documents."""
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add new documents to existing vector store
            self.vector_store.add_documents(documents)
            
    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.vector_store:
            return "Please upload some documents first."

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        response = qa_chain.invoke({"query": question})
        return response["result"]

# Initialize the RAG system
rag_chat = RAGChat()

def process_message(message: dict, history: list) -> str:
    """Process incoming messages and files."""
    # Handle file uploads
    if message.get("files"):
        processed_docs = []
        for file_path in message["files"]:
            try:
                # Check if the file has a valid extension
                if any(file_path.lower().endswith(ext) for ext in ['.txt', '.docx']):
                    processed_docs.extend(rag_chat.process_file(file_path))
                else:
                    return f"Unsupported file type: {file_path}. Please upload .txt or .docx files."
            except Exception as e:
                return f"Error processing file {file_path}: {str(e)}"
        
        if processed_docs:
            try:
                rag_chat.update_vector_store(processed_docs)
                return "Documents processed successfully! You can now ask questions about them."
            except Exception as e:
                return f"Error updating vector store: {str(e)}"
        else:
            return "No valid documents were uploaded. Please upload .txt or .docx files."
    
    # Handle text questions
    if message.get("text"):
        try:
            return rag_chat.query(message["text"])
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    return "Please upload a text file or ask a question."

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=process_message,
    title="RAG Chat Assistant",
    description="Upload documents (.txt or .docx) and ask questions about them! The assistant will use RAG to provide accurate answers based on the uploaded documents.",
    examples=[
        "What is RAG and how does it work?",
        "What are the main benefits of using RAG?",
        "What are some common use cases for RAG?"
    ],
    textbox=gr.MultimodalTextbox(file_types=[".txt", ".docx"]),
    multimodal=True
)

if __name__ == "__main__":
    demo.launch() 