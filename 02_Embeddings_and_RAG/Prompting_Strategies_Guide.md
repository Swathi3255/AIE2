## Prompting Strategies for Thoughtful, Detailed LLM Responses

This guide summarizes practical prompting patterns you can reuse across tasks and in RAG systems.

---

### 1) Chain-of-Thought (Step-by-step Reasoning)
- **Use when**: You need analytical, well-justified answers.
- **Key idea**: Ask the model to think step-by-step and show its reasoning before concluding.

```text
System (thoughtful CoT):
You are a knowledgeable assistant that provides thoughtful, detailed responses based strictly on provided context.
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
1) Understanding  2) Context Analysis  3) Reasoning  4) Answer  5) Confidence
```

---

### 2) Multi-Step Reasoning (Structured Decomposition)
- **Use when**: The question is complex or multi-part.
- **Key idea**: Force an ordered, labeled thought process and evidence-based synthesis.

```text
User (multi-step):
Context Information:
{context}

Question: {user_query}

Please follow this structured approach:
1) Question Analysis  2) Context Review  3) Information Synthesis
4) Answer with Evidence (cite context)  5) Limitations

Number of relevant sources found: {context_count}
{similarity_scores}
```

---

### 3) Role-Based Expert Prompting
- **Use when**: You want depth and domain nuance.
- **Key idea**: Assign an expert role and specify style, structure, and expectations.

```text
System (expert role):
You are a {expert_role} with deep expertise in {domain}.
Approach:
- Provide context and background
- Explain the "why", not just the "what"
- Consider implications and connections
- Use examples when helpful
- Acknowledge limitations and uncertainties
- Structure responses with clear headings

Response style: {response_style}
Response length: {response_length}
```

---

### 4) Socratic Method (Reflective Questions)
- **Use when**: You want deeper analysis or challenge assumptions.
- **Key idea**: Have the model interrogate its own assumptions before answering.

```text
User (socratic):
Context Information:
{context}

Question: {user_query}

Before answering, reflect on:
- Assumptions being made
- Alternative perspectives
- Missing information
- Confidence level and why

Please provide your answer with this thoughtful analysis.
```

---

### 5) Meta-Cognitive Prompting (Self-Reflection)
- **Use when**: You want the model to audit its own reasoning.
- **Key idea**: Require “Approach → Analysis → Reflection → Limitations → Suggestions”.

```text
System (meta-cognitive):
Instructions:
1) My Approach  2) Analysis (with reasoning)  3) Reflection (confidence/uncertainty)
4) Limitations (what cannot be determined)  5) Suggestions (what extra info helps)

Response style: {response_style}
Response length: {response_length}
```

---

### 6) Comparative Analysis (Weigh Evidence)
- **Use when**: Sources conflict or you want pros/cons.
- **Key idea**: Present perspectives, evaluate evidence, synthesize, conclude, note alternatives.

```text
User (comparative):
Context Information:
{context}

Question: {user_query}

Provide a comparative analysis:
- Different Perspectives
- Evidence Evaluation
- Synthesis
- Conclusion (best-supported answer)
- Alternative Interpretations

Number of relevant sources found: {context_count}
{similarity_scores}
```

---

### 7) Few-Shot (Exemplars)
- **Use when**: You can show “great answers” and want the model to imitate them.
- **Key idea**: Include 1–3 concise, high-quality examples formatted exactly as desired.

```text
System (few-shot preface):
Here are examples of thoughtful responses (style, depth, structure). Match them in future answers.
[Example 1: ...]
[Example 2: ...]
```

---

### 8) Constraint-Based (Quality Gates)
- **Use when**: You need consistent quality/format.
- **Key idea**: Enforce depth, evidence, perspectives, limitations, and structure.

```text
System (constraints):
1) Depth: at least 3 levels of detail
2) Evidence: cite specific context
3) Perspectives: cover ≥2 viewpoints
4) Limitations: state what’s unknown
5) Structure: clear headings and bullets
```

---

### 9) Progressive Disclosure (Layered Detail)
- **Use when**: Mixed audiences need both quick and deep views.
- **Key idea**: Ask for Level 1 (quick) → Level 2 (detail) → Level 3 (deep analysis) → Level 4 (extras).

```text
User (progressive):
Level 1: Quick Answer (1–2 sentences)
Level 2: Detailed Explanation
Level 3: Deep Analysis (perspectives, evidence, implications, limits)
Level 4: Additional Context / Related questions
```

---

### 10) Parameter Tuning (Sampling)
- **Use when**: You want more creativity vs. determinism.
- **Guidance**:
  - Analytical: temperature ≈ 0.1–0.4, top_p ≈ 0.8
  - Creative: temperature ≈ 0.6–0.9, top_p ≈ 0.9

```python
# Analytical
chat = ChatOpenAI(temperature=0.3, top_p=0.8)
# Creative
chat = ChatOpenAI(temperature=0.7, top_p=0.9)
```

---

### How To Use With Your RAG Notebook
- Import the enhanced strategies:
```python
from enhanced_prompts import get_enhanced_prompt_strategy
strategy = get_enhanced_prompt_strategy("chain_of_thought")  # or "expert_role", "meta_cognitive"
rag_system_prompt = strategy['system']
rag_user_prompt = strategy['user']
```
- Swap strategy names to compare behaviors quickly.

---

### Practical Tips
- **Be explicit** about structure, length, and style.
- **Show reasoning** steps where appropriate.
- **Cite evidence** from context in RAG.
- **Acknowledge limits** and uncertainties.
- **Iterate**: test variations and measure outcomes.

---

Prepared for: AIE2 / 02_Embeddings_and_RAG
