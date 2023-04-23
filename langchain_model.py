from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

from typing import Any, List, Optional


class LangChainModel(LLM):
    model: Any  #: :meta private:

    @property
    def _llm_type(self) -> str:
        return "my_langchain_model"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate(
            prompt,
            stop,
        )

        text = response[0]["generated_text"][len(prompt) :]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text
