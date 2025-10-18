# =============================================================================
# SIMPLE FLASK API ENDPOINT FOR RAG APPLICATION
# Save this as: api_server.py
# Run with: python api_server.py
# =============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
# from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import tiktoken

# Load environment variables
# load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for testing

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Simple configuration"""
    DATA_PATH = "data/raw/"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o"
    TEMPERATURE = 0
    K = 5  # Number of documents to retrieve

config = Config()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def tiktoken_len(text):
    """Count tokens using tiktoken"""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

# =============================================================================
# LOAD AND SETUP RAG SYSTEM (On Startup)
# =============================================================================

print("🚀 Starting RAG API Server...")
print("=" * 60)

# Load documents
print("📄 Loading documents...")
loader = DirectoryLoader(
    config.DATA_PATH,
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

# Chunk documents
print("✂️ Chunking documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=tiktoken_len
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Create embeddings and vector store
print("🔢 Creating vector store...")
embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
vectorstore = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="legal_aid_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": config.K})
print("✅ Vector store ready")

# Initialize LLM
print("🤖 Initializing LLM...")
llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE)
print("✅ LLM ready")

# Create prompt template
RAG_PROMPT = """Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# Build RAG chain
rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

print("✅ RAG chain ready")
print("=" * 60)
print("🎉 Server is ready to accept requests!")
print("=" * 60)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Legal Aid Navigator RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API documentation (this page)",
            "/health": "Health check",
            "/ask": "Ask a question (POST)",
            "/info": "Get system information"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/ask",
            "body": {
                "question": "Your question here"
            },
            "example": "curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What are tenant rights?\"}'"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "RAG API is running",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "vector_store": "ready"
    })

@app.route('/info')
def info():
    """System information endpoint"""
    return jsonify({
        "configuration": {
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "retrieval_k": config.K
        },
        "statistics": {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "average_chunk_size": sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
        }
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Main RAG endpoint - ask a question"""
    try:
        # Get question from request
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' in request body",
                "usage": "POST /ask with JSON body: {\"question\": \"Your question\"}"
            }), 400
        
        question = data['question']
        
        if not question or not question.strip():
            return jsonify({
                "error": "Question cannot be empty"
            }), 400
        
        # Get answer from RAG chain
        answer = rag_chain.invoke({"question": question})
        
        # Return response
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/ask_with_sources', methods=['POST'])
def ask_with_sources():
    """RAG endpoint that returns sources"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' in request body"
            }), 400
        
        question = data['question']
        
        # Retrieve documents
        retrieved_docs = retriever.invoke(question)
        
        # Get answer
        answer = rag_chain.invoke({"question": question})
        
        # Format sources
        sources = []
        for i, doc in enumerate(retrieved_docs):
            sources.append({
                "source_id": i + 1,
                "content": doc.page_content[:200] + "...",  # First 200 chars
                "metadata": doc.metadata
            })
        
        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Starting Flask server on http://localhost:5000")
    print("=" * 60)
    print("\nTest the API:")
    print("  1. Health check: http://localhost:5000/health")
    print("  2. System info: http://localhost:5000/info")
    print("  3. Ask question: POST to http://localhost:5000/ask")
    print("\nExample curl command:")
    print("  curl -X POST http://localhost:5000/ask \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"question\": \"What are tenant rights in California?\"}'")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)


# =============================================================================
# ALTERNATIVE: SIMPLE TEST SCRIPT
# Save this as: test_api.py
# =============================================================================

"""
#!/usr/bin/env python
\"\"\"
Simple script to test the RAG API
Usage: python test_api.py
\"\"\"

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    print("\\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_info():
    print("\\n" + "="*60)
    print("Testing /info endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_ask(question):
    print("\\n" + "="*60)
    print(f"Testing /ask endpoint")
    print("="*60)
    print(f"Question: {question}")
    
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question},
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\\nStatus Code: {response.status_code}")
    result = response.json()
    
    if response.status_code == 200:
        print(f"\\nAnswer: {result['answer']}")
    else:
        print(f"\\nError: {result.get('error', 'Unknown error')}")

def test_ask_with_sources(question):
    print("\\n" + "="*60)
    print(f"Testing /ask_with_sources endpoint")
    print("="*60)
    print(f"Question: {question}")
    
    response = requests.post(
        f"{BASE_URL}/ask_with_sources",
        json={"question": question},
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\\nStatus Code: {response.status_code}")
    result = response.json()
    
    if response.status_code == 200:
        print(f"\\nAnswer: {result['answer']}")
        print(f"\\nSources ({result['num_sources']}):")
        for source in result['sources']:
            print(f"  [{source['source_id']}] {source['content'][:100]}...")
    else:
        print(f"\\nError: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    print("🧪 Testing RAG API")
    print("Make sure the API server is running!")
    
    # Test health
    test_health()
    
    # Test info
    test_info()
    
    # Test questions
    test_questions = [
        "What are tenant rights in California?",
        "Can my landlord raise rent by 12%?",
        "What should I do if my landlord won't make repairs?"
    ]
    
    for question in test_questions:
        test_ask(question)
    
    # Test with sources
    test_ask_with_sources("What are my rights regarding security deposits?")
    
    print("\\n" + "="*60)
    print("✅ All tests complete!")
    print("="*60)
"""


