from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModelForCausalLM

import torch


class ConversationalStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, user_label):
        StoppingCriteria.__init__(self)
        self.user_label = user_label
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(
        self,
        input_ids: torch.LongTensor,
        score: torch.FloatTensor,
    ) -> bool:
        output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        output = output[self.prompt_len :]
        if self.user_label in output:
            return True


class Model:
    model = None
    tokenizer = None
    lora_applied = False

    def __init__(self, model_config: dict[str, any]):
        self.model_path = model_config["path"]
        self.lora_path = model_config.get("lora")

        self.model_name = self.model_path

        if self.lora_path:
            self.model_name += f" with {self.lora_path}"

        print(f"Using {self.model_name}")

        if not Model.tokenizer:
            self.load_tokenizer()

        if not Model.model:
            self.load_model()

        if self.lora_path and not Model.lora_applied:
            self.apply_lora()

        self.stopping_string = model_config.get("stopping_string")

    def load_tokenizer(self):
        print("Loading tokenizer...")

        if "llama" in self.model_path:
            Model.tokenizer = LlamaTokenizer.from_pretrained(
                self.model_path,
            )
        else:
            Model.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
            )

        print("Tokenizer loaded.")

    def load_model(self):
        print("Loading model...")

        Model.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
        )

        print("Model loaded.")

    def apply_lora(self):
        print("Applying LoRA...")

        Model.model = PeftModelForCausalLM.from_pretrained(
            Model.model,
            self.lora_path,
        )

        self.lora_applied = True

        print("LoRA applied.")

    def generate_response(self, input: str):
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to("cuda")

        stopping_criteria_list = None

        if self.stopping_string:
            stopping_criteria_list = StoppingCriteriaList(
                [
                    ConversationalStoppingCriteria(
                        self.tokenizer, len(input), self.stopping_string
                    )
                ]
            )

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria_list,
        )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output[len(input) :]

        if self.stopping_string:
            response = response.split(self.stopping_string)[0]

        return response.strip()
