# Simple RAG (Retrieval Augmented Generation) Example

This is a simple implementation of RAG using LangChain, FAISS, and OpenAI. The project demonstrates how to:
1. Load and process documents
2. Create embeddings and store them in a vector database
3. Retrieve relevant context for questions
4. Generate answers using the retrieved context

## Prerequisites

- Python 3.9+
- OpenAI API key
- (Optional but recommended) Virtual environment

## Setup

1. (Recommended) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Copy the `.env.template` file to `.env`:
     ```bash
     cp .env.template .env
     ```
   - Edit `.env` and replace `your_api_key_here` with your actual OpenAI API key

4. Run the example:
```bash
python rag_example.py
```

## Project Structure

- `rag_example.py`: Main implementation of the RAG system
- `data/`: Directory containing example documents
  - `rag_info.txt`: Sample document about RAG concepts
- `requirements.txt`: Project dependencies

## Features

- Uses LangChain's latest components (langchain-community, langchain-openai)
- FAISS vector store for efficient similarity search
- OpenAI's GPT-3.5-turbo for generation (configurable to use GPT-4)
- Document chunking with overlap for better context preservation

## Troubleshooting

If you encounter dependency conflicts:
1. Ensure you're using a fresh virtual environment
2. Try installing dependencies one by one if bulk installation fails
3. Check Python version compatibility (3.9+ recommended)

## Note on Costs

This example uses OpenAI's API which has associated costs:
- Embedding API calls for document indexing
- GPT-3.5-turbo for answer generation (default)
- You can modify `model_name` in the code to use different models 