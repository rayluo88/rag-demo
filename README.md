# Simple RAG (Retrieval Augmented Generation) Example

This is a simple implementation of RAG using LangChain, FAISS, and a hybrid approach with OpenAI (embeddings) and DeepSeek (chat completion). The project demonstrates how to:
1. Load and process documents (supports .txt, .docx, and .pdf files)
2. Create embeddings using OpenAI's embedding service
3. Store embeddings in a vector database
4. Generate answers using DeepSeek's chat model

## Prerequisites

### API Requirements
- OpenAI API key (for embeddings)
- DeepSeek API key (for chat completion)

### Python Requirements
- Python 3.9+
- Virtual environment (recommended)

### System Dependencies
For document processing on Linux/WSL:
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

4. Set up your API configuration:
   - Copy the `.env.template` file to `.env`:
     ```bash
     cp .env.template .env
     ```
   - Edit `.env` and configure both APIs:
     ```
     # DeepSeek API Configuration (using OpenAI-compatible format)
     DEEPSEEK_API_KEY=your_deepseek_api_key_here
     DEEPSEEK_API_BASE=https://api.deepseek.com
     DEEPSEEK_MODEL=deepseek-chat

     # OpenAI API Configuration (for embeddings)
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   Note: DeepSeek uses an OpenAI-compatible API format, but it's a separate service with its own API key.

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

- Uses LangChain's latest components
- FAISS vector store for efficient similarity search
- Hybrid API approach:
  - OpenAI's proven embeddings service for document vectorization
  - DeepSeek's powerful chat model for response generation
- Document chunking with overlap for better context preservation
- Interactive web interface with file upload support
- Support for multiple document formats:
  - Text files (.txt)
  - Word documents (.docx)
  - PDF documents (.pdf)
- Chat-based interface for natural interaction

## API Configuration

The system uses a hybrid approach combining two different services:

### OpenAI API (Embeddings)
- Used for converting documents into vector embeddings
- Requires OpenAI API key in `OPENAI_API_KEY`
- Uses the proven `text-embedding-ada-002` model
- Uses OpenAI's standard endpoint (https://api.openai.com/v1)

### DeepSeek API (Chat Completion)
- Used for generating responses to questions
- Requires DeepSeek API key in `DEEPSEEK_API_KEY`
- Uses DeepSeek's API endpoint (https://api.deepseek.com)
- Uses DeepSeek's chat model
- Uses OpenAI-compatible API format but is a separate service

## Web Interface

The project includes a Gradio-based web interface (`app.py`) that provides:
- File upload capability for multiple document formats
- Chat interface for asking questions
- Example questions to get started
- Real-time document processing and querying

## Document Processing Pipeline

1. Document Loading:
   - Supports multiple formats (.txt, .docx, .pdf)
   - Uses appropriate loaders for each format
   - Handles file size limits and validation

2. Text Processing:
   - Splits documents into manageable chunks
   - Maintains context with chunk overlap
   - Prepares text for embedding

3. Embedding Generation:
   - Uses OpenAI's embedding service
   - Converts text chunks to vector representations
   - Stores vectors in FAISS database

4. Question Answering:
   - Retrieves relevant context using FAISS
   - Processes questions using DeepSeek's chat model
   - Generates context-aware responses

## Troubleshooting

### API Configuration Issues
1. OpenAI API (Embeddings):
   - Verify `OPENAI_API_KEY` is set correctly
   - Ensure you have access to the embeddings API
   - Check OpenAI API quota and billing
   - Verify network access to api.openai.com

2. DeepSeek API (Chat):
   - Verify `DEEPSEEK_API_KEY` is set correctly
   - Ensure `DEEPSEEK_API_BASE` is set to https://api.deepseek.com
   - Check DeepSeek API quota and limits
   - Verify network access to api.deepseek.com

### Document Processing Issues
1. For PDF documents:
   - Ensure poppler-utils is installed (Linux/WSL)
   - Try alternative PDF loader if one fails
   - Check if PDF is text-based (not scanned)

2. For Word documents:
   - Ensure all system dependencies are installed
   - Check that python-docx and unstructured packages are installed
   - Verify document isn't corrupted or password-protected

3. For text files:
   - Ensure proper file encoding (UTF-8 recommended)
   - Check file permissions

### General Issues
1. Ensure you're using a fresh virtual environment
2. Try installing dependencies one by one if bulk installation fails
3. Check Python version compatibility (3.9+ recommended)
4. Verify all required system dependencies are installed

## Costs and API Usage

This implementation uses two separate API services:

1. OpenAI API Costs:
   - Charged for document embedding generation
   - Based on input token count
   - Uses text-embedding-ada-002 model pricing

2. DeepSeek API Costs:
   - Charged for chat completion responses
   - Based on input/output token count
   - Uses DeepSeek chat model pricing

Monitor both API accounts for usage and costs separately.

## Contributing

Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your experience or suggestions 