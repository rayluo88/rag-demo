# Simple RAG (Retrieval Augmented Generation) Example

This is a simple implementation of RAG using LangChain, FAISS, and OpenAI. The project demonstrates how to:
1. Load and process documents (supports .txt and .docx files)
2. Create embeddings and store them in a vector database
3. Retrieve relevant context for questions
4. Generate answers using the retrieved context

## Prerequisites

### Python Requirements
- Python 3.9+
- OpenAI API key
- (Optional but recommended) Virtual environment

### System Dependencies
For Word document support on Linux/WSL:
```bash
sudo apt-get update && sudo apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libreoffice
```

For Windows users:
- The required dependencies are included in the Python packages

## Setup

1. Clone the repository and navigate to the project directory.

2. (Recommended) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Copy the `.env.template` file to `.env`:
     ```bash
     cp .env.template .env
     ```
   - Edit `.env` and replace `your_api_key_here` with your actual OpenAI API key

5. Run the example:
   - For command-line interface:
     ```bash
     python rag_example.py
     ```
   - For web interface:
     ```bash
     python app.py
     ```
     Then open your browser to the URL shown in the terminal (usually http://localhost:7860)

## Project Structure

- `rag_example.py`: Command-line implementation of the RAG system
- `app.py`: Gradio web interface for the RAG system
- `data/`: Directory containing example documents
  - `rag_info.txt`: Sample document about RAG concepts
- `requirements.txt`: Project dependencies

## Features

- Uses LangChain's latest components (langchain-community, langchain-openai)
- FAISS vector store for efficient similarity search
- OpenAI's GPT-3.5-turbo for generation (configurable to use GPT-4)
- Document chunking with overlap for better context preservation
- Interactive web interface with file upload support
- Support for both text (.txt) and Word (.docx) documents
- Chat-based interface for natural interaction

## Web Interface

The project includes a Gradio-based web interface (`app.py`) that provides:
- File upload capability for text and Word documents
- Chat interface for asking questions
- Example questions to get started
- Real-time document processing and querying

## Document Processing

The system supports:
- Text files (.txt)
- Word documents (.docx)

Each document is:
1. Loaded using appropriate document loaders
2. Split into chunks with overlap for context preservation
3. Converted to embeddings and stored in FAISS
4. Made available for question answering

## Troubleshooting

### General Issues
1. Ensure you're using a fresh virtual environment
2. Try installing dependencies one by one if bulk installation fails
3. Check Python version compatibility (3.9+ recommended)

### Document Processing Issues
1. For Word documents:
   - Ensure all system dependencies are installed (Linux/WSL)
   - Check that python-docx and unstructured packages are installed
   - Verify the document isn't corrupted or password-protected
2. For text files:
   - Ensure proper file encoding (UTF-8 recommended)
   - Check file permissions

### API Issues
1. Verify your OpenAI API key is correctly set in .env
2. Check your API quota and limits
3. Ensure you have billing set up in your OpenAI account

## Note on Costs

This example uses OpenAI's API which has associated costs:
- Embedding API calls for document indexing
- GPT-3.5-turbo for answer generation (default)
- You can modify `model_name` in the code to use different models

## Contributing

Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your experience or suggestions 