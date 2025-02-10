# Evaluating Prompt Strategies with an LLM Judge
## Technologies Used: LLMs, Prompt Designing, OpenAI API

- Designed and implemented seven prompt strategies (Zero Shot, Few Shot, Chain of Thought, etc.) to systematically test LLM-generated responses on 150 queries from the TREC ’24 RAG Track dataset (MS Marco V2.1).
- Developed an LLM Judge, an automated evaluation framework that scored over 1,050 responses based on Relevance, Correctness, Coherence, Conciseness, and Consistency, using the GPT-4o-mini API.
- Engineered a Python-based pipeline to automate response generation, scoring, and visualisation, revealing thatstructured prompting techniques like Chain of Thought achieved the highest average normalised score (9.36/10), improving LLM performance in complex reasoning tasks.
  
