import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class RAGExample:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # gpt-4 cost is $0.05 per call, gpt-3.5-turbo is much cheaper
        self.llm = ChatOpenAI(temperature=0)  # default model_name='gpt-3.5-turbo'
        print(f"Using model: {self.llm.model_name}")
        self.vector_store = None

    def load_documents(self, file_path: str) -> List:
        """Load and split documents."""
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List):
        """Create FAISS vector store from documents."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.vector_store:
            return "Please load documents first."

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        response = qa_chain.invoke({"query": question})
        return response["result"]

def main():
    # Initialize RAG system
    rag = RAGExample()
    
    # Load and process documents
    print("Loading documents...")
    documents = rag.load_documents("data/rag_info.txt")
    
    # Create vector store
    print("Creating vector store...")
    rag.create_vector_store(documents)
    
    # Example questions
    questions = [
        "What is RAG and how does it work?",
        "What are the main benefits of using RAG?",
        "What are some common use cases for RAG?"
    ]
    
    # Query the system
    print("\nAsking questions...\n")
    for question in questions:
        print(f"Q: {question}")
        answer = rag.query(question)
        print(f"A: {answer}\n")

if __name__ == "__main__":
    main() 