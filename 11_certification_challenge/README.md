# 🏛️ Legal Aid Navigator

An advanced AI-powered RAG (Retrieval-Augmented Generation) system that helps tenants understand their legal rights and navigate complex housing laws across different jurisdictions. Built with state-of-the-art LLM technology and comprehensive evaluation frameworks.

## 📋 Project Overview

**Problem:** Tenants facing housing issues struggle to understand their legal rights and navigate complex housing laws that vary by location, jurisdiction, and specific circumstances.

**Solution:** A sophisticated RAG system with multi-agent reasoning capabilities, advanced retrieval techniques, and comprehensive evaluation metrics to provide accurate, contextual legal guidance.

**Tech Stack:**
- **LLM:** GPT-4o (OpenAI)
- **Embeddings:** text-embedding-3-small (OpenAI)
- **Vector DB:** Qdrant (in-memory)
- **Orchestration:** LangChain & LangGraph
- **Evaluation:** RAGAS (comprehensive metrics)
- **Monitoring:** LangSmith
- **Search:** Tavily Search API
- **Research:** ArXiv integration
- **Document Processing:** PyMuPDF, Unstructured

## 🏗️ Project Structure

```
📁 11_certification_challenge/
├── 📄 legal_aid_navigator.ipynb    # Main demo notebook with complete workflow
├── 📄 README.md                    # This comprehensive documentation
├── 📄 pyproject.toml               # Project dependencies and configuration
├── 📄 uv.lock                      # Lock file for reproducible builds
├── 📄 test_pdf_processing.py       # PDF processing test script
├── 📄 test_rag.html               # Interactive RAG testing interface
├── 📄 usecase_data_kg.json        # Knowledge graph data for use cases
├── 📄 rag_chain.joblib            # Saved RAG chain (joblib format)
├── 📄 rag_chain.pkl               # Saved RAG chain (pickle format)
│
├── 📁 src/                         # Source code modules
│   ├── 📄 __init__.py             # Package initialization
│   ├── 📄 data_ingestion.py       # Task 1: Data ingestion pipeline
│   ├── 📄 chunking_strategy.py    # Task 2: Document chunking strategy
│   └── 📄 dynamic_rag.py          # Dynamic RAG implementation
│
├── 📁 data/                        # Data directory
│   ├── 📁 raw/                     # Raw documents (PDFs + JSON)
│   │   ├── 📄 Fair Housing Act Summary.pdf
│   │   ├── 📄 NOLO California Quick Guide.pdf
│   │   ├── 📄 California Tenant Rights Guide.pdf
│   │   ├── 📄 HUD Tenant Rights and Responsibilities.pdf
│   │   ├── 📄 Texas Tenant Handbook.pdf
│   │   ├── 📄 san_francisco_rent_control.json
│   │   ├── 📄 california_civil_code.json
│   │   ├── 📄 fair_housing_act.json
│   │   └── 📄 austin_tenant_rights.json
│   ├── 📁 processed/               # Processed documents
│   ├── 📁 vector_store/            # Vector database storage
│   └── 📁 golden_set/              # Golden dataset for evaluation
│
├── 📁 tests/                       # Test suite
│   ├── 📄 __init__.py             # Test package initialization
│   ├── 📄 test_retrieval.py       # Retrieval testing
│   └── 📁 golden_dataset/          # Golden dataset for testing
│
├── 📁 config/                      # Configuration files
├── 📁 frontend/                    # Frontend components
└── 📁 evaluation_results/          # Evaluation and metrics results
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.13+** (as specified in pyproject.toml)
2. **Required API Keys:**
   - OpenAI API Key (required)
   - Cohere API Key (optional)
   - Tavily API Key (optional)
   - LangSmith API Key (optional)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd 11_certification_challenge
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export COHERE_API_KEY="your-cohere-key"  # Optional
   export TAVILY_API_KEY="your-tavily-key"  # Optional
   export LANGCHAIN_TRACING_V2="true"       # Optional
   export LANGCHAIN_PROJECT="Legal-Aid-Navigator"
   export LANGCHAIN_API_KEY="your-langsmith-key"  # Optional
   ```

### Running the Application

#### Option 1: Interactive Jupyter Notebook
```bash
jupyter notebook legal_aid_navigator.ipynb
```

#### Option 2: Test PDF Processing
```bash
python test_pdf_processing.py
```

#### Option 3: Interactive Web Interface
```bash
# Open test_rag.html in your browser for interactive testing
```

#### Option 4: Import Individual Components
```python
from src.data_ingestion import LegalDocumentIngester
from src.chunking_strategy import LegalDocumentChunker
from src.dynamic_rag import DynamicPaperFetcher

# Initialize components
ingester = LegalDocumentIngester()
chunker = LegalDocumentChunker(chunk_size=1000, chunk_overlap=200)
paper_fetcher = DynamicPaperFetcher(max_results=5)
```

## 📚 Module Descriptions

### 1. Data Ingestion (`src/data_ingestion.py`)
**Purpose:** Handles ingestion of legal documents from various sources

**Features:**
- Creates sample legal documents for demonstration
- Supports multiple document formats (PDF, JSON)
- Handles municipal codes, state laws, federal laws, and legal aid guides
- Automatic directory structure creation
- Document metadata extraction and management

**Key Methods:**
- `create_sample_legal_documents()`: Generates sample legal documents
- `load_documents()`: Loads documents from raw directory
- `process_pdf_documents()`: Processes PDF files with metadata extraction

### 2. Chunking Strategy (`src/chunking_strategy.py`)
**Purpose:** Implements intelligent document chunking optimized for legal documents

**Features:**
- Token-based chunking using tiktoken for accurate sizing
- Legal-specific separators (sections, subsections, legal citations)
- Configurable chunk size and overlap
- Chunk quality analysis and metrics
- Support for both JSON and LangChain Document formats

**Key Methods:**
- `chunk_documents()`: Converts documents into optimized chunks
- `analyze_chunking()`: Provides quality metrics and analysis
- `tiktoken_len()`: Accurate token counting for GPT-4o

### 3. Dynamic RAG (`src/dynamic_rag.py`)
**Purpose:** Implements dynamic paper fetching and research capabilities

**Features:**
- ArXiv paper search and retrieval
- Research paper metadata extraction
- Integration with LangChain Document format
- Configurable search parameters
- Support for specific paper ID fetching

**Key Methods:**
- `search_papers()`: Searches and fetches papers from ArXiv
- `fetch_paper_by_id()`: Fetches specific papers by ID

## 📊 Data Sources

The system processes multiple types of legal documents:

### PDF Documents (5 files)
- **Fair Housing Act Summary.pdf**: Federal housing discrimination protections
- **NOLO California Quick Guide.pdf**: California-specific tenant rights
- **California Tenant Rights Guide.pdf**: Comprehensive CA tenant information
- **HUD Tenant Rights and Responsibilities.pdf**: HUD official guidelines
- **Texas Tenant Handbook.pdf**: Texas-specific tenant rights

### JSON Documents (4 files)
- **san_francisco_rent_control.json**: SF rent control ordinance details
- **california_civil_code.json**: CA civil code tenant rights
- **fair_housing_act.json**: Federal fair housing protections
- **austin_tenant_rights.json**: Austin-specific tenant rights

## 🧪 Testing & Evaluation

### Test Scripts
- **`test_pdf_processing.py`**: Comprehensive PDF processing test
- **`test_retrieval.py`**: Retrieval system testing
- **`test_rag.html`**: Interactive web-based testing interface

### Evaluation Framework
The system uses RAGAS (Retrieval-Augmented Generation Assessment) for comprehensive evaluation:

- **Faithfulness**: How well answers reflect retrieved context
- **Answer Relevance**: How directly answers address questions
- **Context Precision**: Relevance of retrieved context
- **Context Recall**: Completeness of retrieved information

### Golden Dataset
Located in `tests/golden_dataset/` and `data/golden_set/` for evaluation benchmarks.

## 🔧 Configuration

### Key Configuration Options

**Chunking Parameters:**
- Chunk Size: 1000 tokens (configurable)
- Chunk Overlap: 200 tokens (configurable)
- Separators: Legal-specific (sections, subsections, citations)

**Retrieval Parameters:**
- Retrieval Count: 5 documents (configurable)
- Model: GPT-4o (configurable)
- Embeddings: text-embedding-3-small

**Search Parameters:**
- Max Results: 5 papers (ArXiv)
- Search Tools: Tavily, ArXiv

## 🌟 Key Features

### Advanced Document Processing
- **Multi-format Support**: PDF and JSON document processing
- **Metadata Extraction**: Automatic extraction of document metadata
- **Legal-specific Chunking**: Optimized for legal document structure

### Intelligent Retrieval
- **Vector Search**: Semantic similarity using OpenAI embeddings
- **Hybrid Search**: Combines vector and keyword matching
- **Query Expansion**: Generates related queries for better recall

### Research Integration
- **ArXiv Integration**: Access to latest legal research papers
- **Dynamic Fetching**: Real-time paper retrieval based on queries
- **Research Metadata**: Comprehensive paper information extraction

### Evaluation & Monitoring
- **RAGAS Metrics**: Comprehensive evaluation framework
- **LangSmith Integration**: Real-time monitoring and tracing
- **Golden Dataset**: Benchmark testing capabilities

## 📈 Performance Metrics

The system is designed to achieve high performance across key metrics:

- **Accuracy**: High-quality legal information retrieval
- **Relevance**: Contextually appropriate responses
- **Completeness**: Comprehensive coverage of legal topics
- **Speed**: Efficient processing and retrieval

## 🔒 Security & Privacy

- **API Key Management**: Secure environment variable handling
- **Data Privacy**: Local processing of sensitive legal documents
- **Access Control**: Configurable API access levels

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add comprehensive docstrings to all functions
- Include tests for new functionality
- Update documentation for any API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: Orchestration and chain management
- **OpenAI**: GPT-4o and embedding models
- **Qdrant**: Vector database and similarity search
- **RAGAS**: Comprehensive evaluation framework
- **LangSmith**: Monitoring and observability
- **PyMuPDF**: PDF document processing
- **ArXiv**: Research paper access

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the Jupyter notebook for detailed examples
- Review the test scripts for usage patterns

## 🔄 Version History

- **v0.1.0**: Initial release with core RAG functionality
- Current version includes comprehensive evaluation and multi-agent capabilities

---

**Built with ❤️ for the legal aid community**