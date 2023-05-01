from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
import requests

from typing import Any, List, Optional


class HostedLangChainModel(LLM):
    url: str
    token: str
    model_name: str

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
        }

        data = {
            "prompt": prompt,
            "stop": stop,
        }

        response = requests.post(
            f"{self.url}/generate-text",
            headers=headers,
            json=data,
        )

        if response.status_code == 200:
            generated_text = response.json()["generated_text"]

            text = generated_text[len(prompt) :]

            if stop is not None:
                text = enforce_stop_tokens(text, stop)

            return text
        else:
            print(f"Error: {response.status_code}, {response.text}")
