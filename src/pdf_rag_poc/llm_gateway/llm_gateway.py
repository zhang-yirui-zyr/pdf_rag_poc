from abc import ABC, abstractmethod
from litellm import completion

class LLMHandler(ABC):
    @abstractmethod
    def generate(self, question: str, context: list[str], prompt: str) -> str:
        pass

class OllamaLLMHandler(LLMHandler):
    def __init__(self, model: str = "ollama_chat/llama3.2:latest", api_base: str = "http://localhost:11434"):
        self.model = model
        self.api_base = api_base
    def generate(self, question: str, context: list[str], prompt: str) -> str:
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt.format(context=context, question=question)}],
            api_base=self.api_base
        )
        return response.choices[0].message.content # type: ignore