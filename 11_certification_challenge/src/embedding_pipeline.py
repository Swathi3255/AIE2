# 🗄️ Embedding Pipeline with Qdrant
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

class LegalVectorStore:
    """
    Manages the Qdrant vector store for legal documents
    """
    
    def __init__(self, collection_name: str = "legal_documents"):
        self.collection_name = collection_name
        self.vectorstore = None
        
        # Initialize Qdrant client in memory
        self.client = QdrantClient(":memory:")
        
    def create_vectorstore(self, documents: List[Document], embeddings_model):
        """
        Create and populate the Qdrant vector store
        """
        print("🔄 Creating Qdrant vector store...")
        
        # Create Qdrant vector store
        self.vectorstore = Qdrant.from_documents(
            documents=documents,
            embedding=embeddings_model,
            collection_name=self.collection_name,
            client=self.client
        )
        
        print(f"✅ Qdrant vector store created with {len(documents)} documents")
        return self.vectorstore
    
    def load_existing_vectorstore(self, embeddings_model):
        """
        Load existing vector store from memory
        """
        try:
            self.vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=embeddings_model
            )
            print("✅ Loaded existing Qdrant vector store")
            return self.vectorstore
        except Exception as e:
            print(f"❌ Could not load existing vector store: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search on the vector store
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """
        Get a retriever for use with LangChain chains
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        return retriever
