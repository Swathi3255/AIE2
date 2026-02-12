# ✂️ Chunking Strategy Implementation
import numpy as np
import tiktoken
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def tiktoken_len(text):
    """Calculate token length using tiktoken for accurate chunking"""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

class LegalDocumentChunker:
    """
    Handles chunking of legal documents with token-based strategy
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter with token-based length function
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=tiktoken_len,
            separators=[
                "\n\nSection",  # Legal sections
                "\n\nSubsection", 
                "\n\n(a)",      # Legal subsections
                "\n\n(b)",
                "\n\n(c)",
                "\n\n(d)",
                "\n\n(e)",
                "\n\n(f)",
                "\n\n",         # Paragraph breaks
                "\n",           # Line breaks
                " ",            # Spaces
                ""              # Character level
            ]
        )
    
    def chunk_documents(self, documents: List[Any]) -> List[Document]:
        """
        Convert legal documents into LangChain Document objects with metadata
        Handles both JSON documents and LangChain Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            # Handle different document types
            if isinstance(doc, Document):
                # Already a LangChain Document (from PDF loader)
                base_doc = doc
            elif isinstance(doc, dict):
                # JSON document format - convert to Document
                base_doc = Document(
                    page_content=doc["content"],
                    metadata={
                        "title": doc["title"],
                        "jurisdiction": doc["jurisdiction"],
                        "source": doc["source"],
                        "last_updated": doc["last_updated"],
                        "document_type": "legal_document"
                    }
                )
            else:
                # Skip unknown document types
                print(f"⚠️ Skipping unknown document type: {type(doc)}")
                continue
            
            # Split the document using token-based chunking
            chunks = self.text_splitter.split_documents([base_doc])
            
            # Add chunk-specific metadata
            doc_title = base_doc.metadata.get("title", "unknown")
            for j, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc_title.replace(' ', '_').lower()}_{j}",
                    "chunk_index": j,
                    "total_chunks": len(chunks)
                })
            
            chunked_docs.extend(chunks)
        
        print(f"📄 Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def analyze_chunking(self, chunked_docs: List[Document]) -> Dict[str, Any]:
        """
        Analyze the chunking results for quality metrics using token-based analysis
        """
        chunk_token_lengths = [tiktoken_len(chunk.page_content) for chunk in chunked_docs]
        
        # Analyze document types
        doc_types = {}
        for chunk in chunked_docs:
            doc_type = chunk.metadata.get("document_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        analysis = {
            "total_chunks": len(chunked_docs),
            "avg_chunk_tokens": np.mean(chunk_token_lengths),
            "min_chunk_tokens": min(chunk_token_lengths),
            "max_chunk_tokens": max(chunk_token_lengths),
            "std_chunk_tokens": np.std(chunk_token_lengths),
            "document_types": doc_types
        }
        
        return analysis
