"""
Enhanced RAG implementation with thoughtful prompting strategies
"""

from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.vectordatabase import VectorDatabase
from enhanced_prompts import get_enhanced_prompt_strategy

class ThoughtfulRAG:
    def __init__(self, vector_db: VectorDatabase, strategy="chain_of_thought"):
        self.vector_db = vector_db
        self.chat_openai = ChatOpenAI()
        self.prompts = get_enhanced_prompt_strategy(strategy)
        
    def query(self, user_query: str, k: int = 5, include_similarity_scores: bool = True):
        """
        Enhanced RAG query with thoughtful prompting
        
        Args:
            user_query: The user's question
            k: Number of relevant documents to retrieve
            include_similarity_scores: Whether to include similarity scores in context
        """
        # Retrieve relevant contexts
        context_list = self.vector_db.search_by_text(user_query, k=k)
        
        # Prepare context
        context_prompt = ""
        similarity_scores_text = ""
        
        for i, (text, score) in enumerate(context_list):
            context_prompt += f"Source {i+1}: {text}\n\n"
            if include_similarity_scores:
                similarity_scores_text += f"Source {i+1} similarity: {score:.3f}\n"
        
        # Create messages with enhanced prompts
        messages = [
            self.prompts['system'].create_message(),
            self.prompts['user'].create_message(
                context=context_prompt.strip(),
                user_query=user_query,
                context_count=len(context_list),
                similarity_scores=similarity_scores_text.strip()
            )
        ]
        
        # Get response
        response = self.chat_openai.invoke(messages)
        return response.content

# Example usage with different strategies
def demonstrate_thoughtful_rag():
    """Demonstrate different thoughtful prompting strategies"""
    
    # Assuming you have a vector database set up
    # vector_db = VectorDatabase(...)
    
    strategies = ["chain_of_thought", "expert_role", "meta_cognitive"]
    
    for strategy in strategies:
        print(f"\n=== {strategy.upper()} STRATEGY ===")
        rag = ThoughtfulRAG(None, strategy=strategy)  # Replace None with actual vector_db
        
        # Example query
        query = "What are the key principles of effective leadership?"
        
        # This would normally work with a real vector database
        # response = rag.query(query)
        # print(f"Response: {response}")
        
        # For demonstration, show the prompt structure
        system_msg = rag.prompts['system'].create_message()
        user_msg = rag.prompts['user'].create_message(
            context="Sample context about leadership...",
            user_query=query,
            context_count=3,
            similarity_scores="Source 1 similarity: 0.85\nSource 2 similarity: 0.78"
        )
        
        print("System Prompt Preview:")
        print(system_msg['content'][:200] + "...")
        print("\nUser Prompt Preview:")
        print(user_msg['content'][:200] + "...")

if __name__ == "__main__":
    demonstrate_thoughtful_rag()
