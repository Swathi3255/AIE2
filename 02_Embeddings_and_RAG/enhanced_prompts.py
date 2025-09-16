"""
Enhanced prompting strategies for more thoughtful LLM responses
"""

from aimakerspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt

# 1. Chain-of-Thought RAG System Prompt
THOUGHTFUL_RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that provides thoughtful, detailed responses based strictly on provided context.

Instructions:
- Think through your response step by step before answering
- Show your reasoning process clearly
- Consider multiple perspectives when relevant
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.

Response Format:
1. **Understanding**: What is the question asking?
2. **Context Analysis**: What relevant information is available?
3. **Reasoning**: How does this information answer the question?
4. **Answer**: Your final response with supporting evidence
5. **Confidence**: How confident are you in this answer and why?"""

# 2. Multi-Step Reasoning User Prompt
MULTI_STEP_RAG_USER_TEMPLATE = """Context Information:
{context}

Question: {user_query}

Please follow this structured approach:

**Step 1: Question Analysis**
What is the question really asking? What type of answer is expected?

**Step 2: Context Review**
What relevant information is available in the context? What are the key points?

**Step 3: Information Synthesis**
How can the available information be combined to answer the question?

**Step 4: Answer with Evidence**
Provide your response with specific citations from the context.

**Step 5: Limitations**
What information might be missing? What can't be determined from the context?

Number of relevant sources found: {context_count}
{similarity_scores}"""

# 3. Expert Role-Based System Prompt
EXPERT_RAG_SYSTEM_TEMPLATE = """You are a {expert_role} with deep expertise in {domain}. You are known for your thorough analysis and comprehensive explanations.

Your approach:
- Always provide context and background information
- Explain the "why" behind your answers, not just the "what"
- Consider implications and broader connections
- Use analogies and examples when helpful
- Acknowledge limitations and uncertainties
- Structure your responses clearly with headings

When answering based on the provided context:
- Analyze the credibility and relevance of each piece of information
- Synthesize information from multiple sources when available
- Highlight any contradictions or gaps in the information
- Provide nuanced perspectives when appropriate

Response style: {response_style}
Response length: {response_length}"""

# 4. Socratic Method User Prompt
SOCRATIC_RAG_USER_TEMPLATE = """Context Information:
{context}

Question: {user_query}

Before answering, consider these reflective questions:
- What assumptions am I making about this question?
- What would someone with a different perspective think?
- What additional information would make my answer more complete?
- How confident am I in this answer, and why?
- What are the implications of this answer?

Please provide your answer with this thoughtful analysis.

Number of relevant sources found: {context_count}
{similarity_scores}"""

# 5. Meta-Cognitive System Prompt
META_COGNITIVE_RAG_SYSTEM_TEMPLATE = """You are a thoughtful assistant that reflects on your own reasoning process and provides comprehensive, well-reasoned responses.

Instructions:
- Before answering, briefly explain your approach
- Consider what you know and what you don't know
- Reflect on the quality and completeness of your response
- Acknowledge any uncertainties or limitations
- Suggest what additional information would be helpful

Response structure:
1. **My Approach**: How I plan to answer this question
2. **Analysis**: My detailed response with reasoning
3. **Reflection**: What I'm confident about and what I'm uncertain about
4. **Limitations**: What information I'm missing or what I can't determine
5. **Suggestions**: What additional context would improve this answer

Response style: {response_style}
Response length: {response_length}"""

# 6. Comparative Analysis Prompt
COMPARATIVE_RAG_USER_TEMPLATE = """Context Information:
{context}

Question: {user_query}

Please provide a comparative analysis:

**Different Perspectives**: If there are multiple viewpoints in the context, present them clearly.

**Evidence Evaluation**: Assess the strength and reliability of the information provided.

**Synthesis**: How do the different pieces of information relate to each other?

**Conclusion**: What is the most supported answer based on the evidence?

**Alternative Interpretations**: Are there other ways to interpret this information?

Number of relevant sources found: {context_count}
{similarity_scores}"""

# Create prompt objects
def create_thoughtful_rag_prompts():
    """Create enhanced RAG prompts for more thoughtful responses"""
    
    prompts = {
        'chain_of_thought': {
            'system': SystemRolePrompt(
                THOUGHTFUL_RAG_SYSTEM_TEMPLATE,
                strict=True,
                defaults={
                    "response_style": "detailed",
                    "response_length": "comprehensive"
                }
            ),
            'user': UserRolePrompt(
                MULTI_STEP_RAG_USER_TEMPLATE,
                strict=True,
                defaults={
                    "context_count": "",
                    "similarity_scores": ""
                }
            )
        },
        
        'expert_role': {
            'system': SystemRolePrompt(
                EXPERT_RAG_SYSTEM_TEMPLATE,
                strict=True,
                defaults={
                    "expert_role": "research analyst",
                    "domain": "the provided context",
                    "response_style": "professional",
                    "response_length": "detailed"
                }
            ),
            'user': UserRolePrompt(
                SOCRATIC_RAG_USER_TEMPLATE,
                strict=True,
                defaults={
                    "context_count": "",
                    "similarity_scores": ""
                }
            )
        },
        
        'meta_cognitive': {
            'system': SystemRolePrompt(
                META_COGNITIVE_RAG_SYSTEM_TEMPLATE,
                strict=True,
                defaults={
                    "response_style": "reflective",
                    "response_length": "comprehensive"
                }
            ),
            'user': UserRolePrompt(
                COMPARATIVE_RAG_USER_TEMPLATE,
                strict=True,
                defaults={
                    "context_count": "",
                    "similarity_scores": ""
                }
            )
        }
    }
    
    return prompts

# Example usage function
def get_enhanced_prompt_strategy(strategy_name="chain_of_thought"):
    """
    Get enhanced prompt strategy for more thoughtful responses
    
    Available strategies:
    - chain_of_thought: Step-by-step reasoning
    - expert_role: Expert perspective with domain knowledge
    - meta_cognitive: Self-reflective reasoning
    """
    prompts = create_thoughtful_rag_prompts()
    return prompts.get(strategy_name, prompts['chain_of_thought'])

if __name__ == "__main__":
    # Example usage
    strategy = get_enhanced_prompt_strategy("chain_of_thought")
    print("System prompt:", strategy['system'].prompt[:200] + "...")
    print("User prompt:", strategy['user'].prompt[:200] + "...")
