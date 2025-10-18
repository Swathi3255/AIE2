# 🔍 Advanced Retrieval Techniques Implementation
from typing import List, Dict, Any
from langchain.schema import Document

class AdvancedRetrievalSystem:
    """
    Advanced retrieval system with multiple techniques for improved accuracy
    """
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        
    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Combine vector similarity search with keyword matching
        """
        # Vector similarity search
        vector_results = self.vectorstore.similarity_search(query, k=k*2)
        
        # Simple keyword matching (in production, would use BM25)
        keyword_results = self._keyword_search(query, vector_results)
        
        # Combine and deduplicate results
        combined_results = self._combine_results(vector_results, keyword_results, k)
        
        return combined_results
    
    def _keyword_search(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Simple keyword matching implementation
        """
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                scored_docs.append((doc, overlap))
        
        # Sort by keyword overlap
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]
    
    def _combine_results(self, vector_results: List[Document], keyword_results: List[Document], k: int) -> List[Document]:
        """
        Combine vector and keyword results with deduplication
        """
        seen_content = set()
        combined = []
        
        # Add vector results first (higher priority)
        for doc in vector_results:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined.append(doc)
        
        # Add keyword results
        for doc in keyword_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined.append(doc)
        
        return combined[:k]
    
    def query_expansion(self, query: str) -> List[str]:
        """
        Generate related queries to improve recall
        """
        expansion_prompt = f"""
        Given this legal question about tenant rights, generate 3 related questions that might help find relevant information:
        
        Original question: {query}
        
        Generate related questions that cover:
        1. Different aspects of the same legal issue
        2. Related legal concepts
        3. Different jurisdictions or scenarios
        
        Return only the questions, one per line.
        """
        
        try:
            response = self.llm.invoke(expansion_prompt)
            expanded_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
            return expanded_queries[:3]  # Limit to 3 expansions
        except:
            # Fallback to simple keyword variations
            return self._simple_query_expansion(query)
    
    def _simple_query_expansion(self, query: str) -> List[str]:
        """
        Simple query expansion using keyword variations
        """
        expansions = [query]
        
        # Add variations
        if "rights" in query.lower():
            expansions.append(query.replace("rights", "protections"))
        if "landlord" in query.lower():
            expansions.append(query.replace("landlord", "property owner"))
        if "tenant" in query.lower():
            expansions.append(query.replace("tenant", "renter"))
        
        return expansions[:3]
    
    def multi_query_retrieval(self, query: str, k: int = 5) -> List[Document]:
        """
        Break complex queries into sub-queries and retrieve from each
        """
        # Generate sub-queries
        sub_queries = self.query_expansion(query)
        sub_queries.append(query)  # Include original query
        
        all_results = []
        
        # Retrieve for each sub-query
        for sub_query in sub_queries:
            results = self.hybrid_search(sub_query, k=k)
            all_results.extend(results)
        
        # Deduplicate and rank
        return self._deduplicate_and_rank(all_results, k)
    
    def _deduplicate_and_rank(self, results: List[Document], k: int) -> List[Document]:
        """
        Remove duplicates and rank results
        """
        seen_content = set()
        deduplicated = []
        
        for doc in results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(doc)
        
        return deduplicated[:k]
    
    def rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank results using cross-encoder approach (simplified)
        """
        # Simple re-ranking based on query-document similarity
        scored_docs = []
        
        for doc in documents:
            # Calculate simple relevance score
            score = self._calculate_relevance_score(query, doc.page_content)
            scored_docs.append((doc, score))
        
        # Sort by relevance score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
    
    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """
        Calculate relevance score between query and content
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union
