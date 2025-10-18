# 🏛️ Legal Aid Navigator

An AI-powered RAG system that helps tenants understand their legal rights and navigate complex housing laws across different jurisdictions.

## 📋 Project Overview

**Problem:** Tenants facing housing issues struggle to understand their legal rights and navigate complex housing laws that vary by location.

**Solution:** Advanced RAG system with multi-agent reasoning for complex queries and advanced retrieval techniques.

**Tech Stack:**
- **LLM:** GPT-4o
- **Embeddings:** text-embedding-3-small  
- **Vector DB:** Qdrant (in-memory)
- **Orchestration:** LangChain
- **Evaluation:** RAGAS
- **Monitoring:** LangSmith

## 🏗️ Project Structure

The project is organized into task-wise modules under the `src/` directory:

```
src/
├── __init__.py
├── main.py                    # Main application orchestrator
├── data_ingestion.py          # Task 1: Data ingestion pipeline
├── chunking_strategy.py       # Task 2: Document chunking strategy
├── embedding_pipeline.py      # Task 3: Embedding pipeline with Qdrant
├── rag_core.py               # Task 4: Core RAG system
├── multi_agent_system.py     # Task 5: Multi-agent system architecture
├── advanced_retrieval.py     # Task 6: Advanced retrieval techniques
└── evaluation.py             # Task 7: Evaluation framework with RAGAS
```

## 🚀 Quick Start

### Prerequisites

1. Python 3.13+
2. Required API keys:
   - OpenAI API Key
   - Cohere API Key (optional)
   - Tavily API Key (optional)
   - LangSmith API Key (optional)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Running the Application

1. **Run the complete pipeline:**
   ```bash
   python src/main.py
   ```

2. **Run individual components:**
   ```python
   from src.data_ingestion import LegalDocumentIngester
   from src.chunking_strategy import LegalDocumentChunker
   from src.embedding_pipeline import LegalVectorStore
   from src.rag_core import LegalRAGSystem
   from src.multi_agent_system import LegalMultiAgentSystem
   from src.advanced_retrieval import AdvancedRetrievalSystem
   from src.evaluation import LegalRAGEvaluator
   ```

## 📚 Module Descriptions

### 1. Data Ingestion (`data_ingestion.py`)
- Creates sample legal documents for demonstration
- Handles loading of legal documents from various sources
- Supports municipal codes, state laws, federal laws, and legal aid guides

### 2. Chunking Strategy (`chunking_strategy.py`)
- Implements recursive text splitting optimized for legal documents
- Uses legal-specific separators (sections, subsections, etc.)
- Analyzes chunking quality metrics

### 3. Embedding Pipeline (`embedding_pipeline.py`)
- Manages Qdrant vector store for legal documents
- Handles similarity search and retrieval
- Provides retriever interface for LangChain chains

### 4. Core RAG System (`rag_core.py`)
- Implements the main RAG system with GPT-4o
- Uses specialized legal prompt templates
- Provides comprehensive answers with source citations

### 5. Multi-Agent System (`multi_agent_system.py`)
- **Research Agent**: Finds relevant laws and regulations
- **Empathy Agent**: Understands emotional context of tenant situations  
- **Action Plan Agent**: Creates concrete, step-by-step guidance
- **Validation Agent**: Ensures legal accuracy and proper citations
- **Supervisor Agent**: Coordinates all specialized agents

### 6. Advanced Retrieval (`advanced_retrieval.py`)
- **Hybrid Search**: Combines vector similarity + keyword matching
- **Query Expansion**: Generates related queries to improve recall
- **Multi-query Generation**: Breaks complex questions into sub-queries
- **Re-ranking**: Uses cross-encoder models to reorder results

### 7. Evaluation Framework (`evaluation.py`)
- Comprehensive evaluation using RAGAS metrics
- **Faithfulness**: How well the answer reflects retrieved context
- **Answer Relevance**: How directly the answer addresses the question  
- **Context Precision**: How relevant retrieved context is to the question
- **Context Recall**: How completely the system retrieves all relevant info

## 🧪 Testing

The system includes comprehensive testing capabilities:

1. **Sample Questions**: Tests with predefined legal questions
2. **Multi-Agent Testing**: Tests complex query processing
3. **Advanced Retrieval Testing**: Tests hybrid search and query expansion
4. **Evaluation**: Runs RAGAS metrics on golden dataset

## 📊 Evaluation Metrics

The system evaluates performance using:

- **Faithfulness**: Measures how well answers reflect the retrieved context
- **Answer Relevance**: Measures how directly answers address questions
- **Context Precision**: Measures relevance of retrieved context
- **Context Recall**: Measures completeness of retrieved information

## 🔧 Configuration

Key configuration options:

- **Chunk Size**: 1000 tokens (configurable in `chunking_strategy.py`)
- **Chunk Overlap**: 200 tokens (configurable in `chunking_strategy.py`)
- **Retrieval Count**: 5 documents (configurable in `rag_core.py`)
- **Model**: GPT-4o (configurable in `main.py`)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with LangChain and OpenAI
- Uses Qdrant for vector storage
- Evaluation powered by RAGAS
- Monitoring with LangSmith