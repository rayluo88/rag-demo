import os
import gradio as gr
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    PDFMinerLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Constants
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB

# API Configuration
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
OPENAI_API_BASE = "https://api.openai.com/v1"  # OpenAI's default API endpoint

class RAGChat:
    def __init__(self):
        # Use OpenAI for embeddings
        self.embeddings = OpenAIEmbeddings(
            #model="text-embedding-ada-002"  # default model
        )
        
        # Whether to use Deepseek for chat completion (using OpenAI-compatible format), configured in .env
        use_deepseek = os.getenv("USE_DEEPSEEK", "false").lower() == "true"

        if use_deepseek:
            self.llm = ChatOpenAI(
                temperature=0,
                model_name=DEEPSEEK_MODEL,
                base_url=DEEPSEEK_API_BASE,
                api_key=os.getenv("DEEPSEEK_API_KEY")
            )
            print(f"Using DeepSeek chat model: {DEEPSEEK_MODEL} at {DEEPSEEK_API_BASE}")
        else:
            self.llm = ChatOpenAI(
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            print(f"Using OpenAI chat model at {OPENAI_API_BASE}")
        self.vector_store = None
        
        # Load initial knowledge base
        try:
            initial_docs = self.load_knowledge_base()
            if initial_docs:
                self.update_vector_store(initial_docs)
                print(f"Loaded {len(initial_docs)} documents from knowledge base")
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {str(e)}")
        
    def check_file_size(self, file_path: str) -> bool:
        """Check if file size is within limits."""
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return file_size_mb <= MAX_FILE_SIZE_MB
        
    def process_file(self, file_path: str, progress=gr.Progress()) -> List:
        """Process a single file and return documents."""
        if not self.check_file_size(file_path):
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
            
        # Choose appropriate loader based on file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            progress(0, desc=f"Loading {os.path.basename(file_path)}")
            
            if file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.docx':
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == '.pdf':
                progress(0.2, desc="Initializing PDF processor")
                # Try PyPDFLoader first, fallback to PDFMinerLoader if it fails
                try:
                    loader = PyPDFLoader(file_path)
                except Exception as e:
                    progress(0.3, desc="Falling back to alternative PDF processor")
                    loader = PDFMinerLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            progress(0.4, desc="Loading document content")
            documents = loader.load()
            
            progress(0.6, desc="Splitting document into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            
            progress(1.0, desc="Document processing complete")
            return split_docs
            
        except Exception as e:
            raise ValueError(f"Error processing {file_path}: {str(e)}")
        
    def update_vector_store(self, documents: List, progress=gr.Progress()):
        """Update vector store with new documents."""
        progress(0, desc="Initializing vector store")
        try:
            if self.vector_store is None:
                progress(0.3, desc="Creating new vector store")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                progress(0.3, desc="Updating existing vector store")
                self.vector_store.add_documents(documents)
            progress(1.0, desc="Vector store update complete")
        except Exception as e:
            raise ValueError(f"Error updating vector store: {str(e)}")
            
    def query(self, question: str) -> str:
        """Query the system with or without RAG."""
        try:
            if self.vector_store:
                # Use RAG when documents are available
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(),
                    return_source_documents=True
                )
                response = qa_chain.invoke({"query": question})
                answer = response["result"]
                
                # Add a note that this answer is based on uploaded documents
                return f"Based on the uploaded documents:\n\n{answer}"
            else:
                # Use direct LLM when no documents are available
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Answer the question based on your general knowledge."),
                    ("human", "{question}")
                ])
                chain = prompt | self.llm
                response = chain.invoke({"question": question})
                return response.content
                
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def load_knowledge_base(self, data_dir: str = "data", progress=gr.Progress()) -> List:
        """Load all documents from knowledge base directory."""
        all_docs = []
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} directory not found")
            return all_docs
        
        progress(0, desc="Loading knowledge base")
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        
        for idx, file in enumerate(files):
            file_path = os.path.join(data_dir, file)
            try:
                if any(file.lower().endswith(ext) for ext in ['.txt', '.docx', '.pdf']):
                    progress(idx/len(files), desc=f"Processing {file}")
                    docs = self.process_file(file_path)
                    all_docs.extend(docs)
            except Exception as e:
                print(f"Warning: Error processing {file}: {str(e)}")
            
        progress(1.0, desc="Knowledge base loaded")
        return all_docs

# Initialize the RAG system
rag_chat = RAGChat()

def process_message(message: dict, history: list) -> str:
    """Process incoming messages and files."""
    has_files = bool(message.get("files"))
    has_question = bool(message.get("text"))
    
    # Process files if present
    if has_files:
        processed_docs = []
        total_files = len(message["files"])
        
        for idx, file_path in enumerate(message["files"], 1):
            try:
                # Check if the file has a valid extension
                if any(file_path.lower().endswith(ext) for ext in ['.txt', '.docx', '.pdf']):
                    processed_docs.extend(rag_chat.process_file(file_path))
                else:
                    return {"role": "assistant", "content": f"Unsupported file type: {file_path}. Please upload .txt, .docx, or .pdf files."}
            except ValueError as e:
                if "file size exceeds" in str(e):
                    return {"role": "assistant", "content": f"Error: {str(e)}. Please upload smaller files."}
                return {"role": "assistant", "content": f"Error processing file {file_path}: {str(e)}"}
            except Exception as e:
                return {"role": "assistant", "content": f"Unexpected error processing file {file_path}: {str(e)}"}
        
        if processed_docs:
            try:
                rag_chat.update_vector_store(processed_docs)
                # If there's also a question, process it immediately
                if has_question:
                    try:
                        answer = rag_chat.query(message["text"])
                        return {"role": "assistant", "content": f"Documents processed successfully!\n\n{answer}"}
                    except Exception as e:
                        return {"role": "assistant", "content": f"Documents processed successfully, but there was an error answering your question: {str(e)}"}
                return {"role": "assistant", "content": "Documents processed successfully! You can now ask questions about them or continue with general questions."}
            except Exception as e:
                return {"role": "assistant", "content": f"Error updating vector store: {str(e)}"}
        else:
            return {"role": "assistant", "content": "No valid documents were uploaded. Please upload .txt, .docx, or .pdf files."}
    
    # Handle text questions when no files are present
    elif has_question:
        try:
            answer = rag_chat.query(message["text"])
            return {"role": "assistant", "content": answer}
        except Exception as e:
            return {"role": "assistant", "content": f"Error processing question: {str(e)}"}
    
    return {"role": "assistant", "content": "Please ask a question or upload documents to enhance my knowledge."}

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=process_message,
    title="RAG Chat Assistant",
    description=f"""Chat with me about anything! Upload documents (.txt, .docx, or .pdf) to enhance my knowledge about specific topics.
    \nNote: Maximum file size is {MAX_FILE_SIZE_MB}MB per file.""",
    examples=[
        "What is RAG and how does it work?",
        "What are the main benefits of using RAG?",
        "What are some common use cases for RAG?",
        "What is the capital of France?",
        "How does photosynthesis work?"
    ],
    textbox=gr.MultimodalTextbox(file_types=[".txt", ".docx", ".pdf"]),
    multimodal=True,
    type="messages"  # Use the newer messages format
)

if __name__ == "__main__":
    demo.launch(share=True) 