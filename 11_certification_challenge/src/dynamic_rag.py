# src/dynamic_paper_fetcher.py
import arxiv
from typing import List, Dict, Any
from langchain.schema import Document

class DynamicPaperFetcher:
    """
    Dynamically fetch Arxiv papers based on search queries
    """
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
    
    def search_papers(self, query: str, max_results: int = None) -> List[Document]:
        """
        Search and fetch papers from Arxiv
        """
        if max_results is None:
            max_results = self.max_results
            
        # Search Arxiv
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        documents = []
        for paper in search.results():
            # Create LangChain Document
            doc = Document(
                page_content=f"""
                Title: {paper.title}
                
                Abstract: {paper.summary}
                
                Authors: {', '.join([author.name for author in paper.authors])}
                
                Published: {paper.published}
                
                Categories: {', '.join(paper.categories)}
                """,
                metadata={
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "published": str(paper.published),
                    "categories": paper.categories,
                    "arxiv_id": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "source": "arxiv",
                    "document_type": "research_paper"
                }
            )
            documents.append(doc)
        
        print(f"📄 Fetched {len(documents)} papers for query: '{query}'")
        return documents
    
    def fetch_paper_by_id(self, arxiv_id: str) -> Document:
        """
        Fetch a specific paper by Arxiv ID
        """
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        doc = Document(
            page_content=f"""
            Title: {paper.title}
            
            Abstract: {paper.summary}
            
            Authors: {', '.join([author.name for author in paper.authors])}
            
            Published: {paper.published}
            
            Categories: {', '.join(paper.categories)}
            """,
            metadata={
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": str(paper.published),
                "categories": paper.categories,
                "arxiv_id": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "source": "arxiv",
                "document_type": "research_paper"
            }
        )
        
        print(f"📄 Fetched paper: {paper.title}")
        return doc