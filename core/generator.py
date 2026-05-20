"""Answer generation module."""

from __future__ import annotations

from langchain_ollama import ChatOllama


PROMPT_TEMPLATE = """
Use ONLY the following context to answer the question.
If the answer is not found in the context, say "I don't know."
Do not add any information not present in the context.
Keep the answer concise.
Always say "Merci pour votre question!" at the end of the answer.
Answer in the same language as the question.

{memory_context}
Context:
{context}

Question: {input}
Answer:
""".strip()


class AnswerGenerator:
    """LLM generator wrapper."""

    def __init__(self, model_name: str = "gemma3:4b", temperature: float = 0.0, num_predict: int = 256):
        self.llm = ChatOllama(model=model_name, temperature=temperature, seed=42, validate_model_on_init=True, num_predict=num_predict)

    def render_prompt(self, question: str, context_docs: list, memory_context: str = "") -> str:
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
        return PROMPT_TEMPLATE.format(memory_context=memory_context or "", context=context, input=question)

    def answer(self, question: str, context_docs: list, memory_context: str = "") -> str:
        prompt = self.render_prompt(question, context_docs, memory_context=memory_context)
        return str(self.llm.invoke(prompt).content)
