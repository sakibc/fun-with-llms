import requests


class HostedModel:
    url: str
    token: str

    model_config: dict[str, any]

    def __init__(
        self, model_name: str, model_config: dict[str, any], url: str, token: str
    ):
        self.model_config = model_config
        self.url = url
        self.token = token

        self.model_name = model_name

        if "pipeline" in model_config:
            self.model_type = "pipeline"
        elif "seq2seq" in model_config:
            self.model_type = "seq2seq"
        else:
            self.model_type = "causal"

        self.headers = {
            "Authorization": f"Bearer {self.token}",
        }

        health_check = requests.get(f"{self.url}/health", headers=self.headers)
        if health_check.status_code != 200:
            raise Exception(f"Error: {health_check.status_code}, {health_check.text}")

        if health_check.json()["model_name"] != self.model_name:
            raise Exception(
                f"Error: Model name mismatch, {health_check.json()['model_name']} != {self.model_name}"
            )

    def generate(self, input: str, stop):
        data = {
            "prompt": input,
            "stop": stop,
        }

        response = requests.post(
            f"{self.url}/generate-text",
            headers=self.headers,
            json=data,
        )

        if response.status_code == 200:
            return response.json()["generated_text"]

        else:
            print(f"Error: {response.status_code}, {response.text}")
