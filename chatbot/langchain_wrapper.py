from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

from typing import Any, List, Optional


class LangChainWrapper(LLM):
    model: Any  #: :meta private:
    default_ai_label = "AI: "
    default_human_label = "Human: "

    @property
    def _llm_type(self) -> str:
        return self.model.model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.model.model_config.get("ai_label") is not None:
            prompt = prompt.replace(
                self.default_ai_label,
                self.model.model_config.get("ai_label"),
            )

        if self.model.model_config.get("human_label") is not None:
            prompt = prompt.replace(
                self.default_human_label,
                self.model.model_config.get("human_label"),
            )

            if stop is not None:
                stop = [
                    s.replace(
                        self.default_human_label,
                        self.model.model_config.get("human_label"),
                    )
                    for s in stop
                ]

        if "stop" in self.model.model_config:
            stop.append(self.model.model_config.get("stop"))

        response = self.model.generate(
            prompt,
            stop,
        )

        text = response[len(prompt) :]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text.strip()
