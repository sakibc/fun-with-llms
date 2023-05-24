from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

from typing import Any, List, Optional


class LangChainWrapper(LLM):
    model: Any  #: :meta private:

    @property
    def _llm_type(self) -> str:
        return self.model.model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate(
            prompt,
            stop,
        )

        text = response[len(prompt) :]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text.strip()
