from __future__ import annotations
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
    def __init__(self, tokenizer, prompt_len, stop):
        StoppingCriteria.__init__(self)
        self.stop = stop
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(
        self,
        input_ids,
        score,
    ) -> bool:
        output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        output = output[self.prompt_len :]

        for stop in self.stop:
            if stop in output:
                return True


class Model:
    model = None
    tokenizer = None
    lora_applied = False

    model_config: dict[str, any]

    def __init__(self, model_config: dict[str, any]):
        self.model_path = model_config["path"]
        self.lora_path = model_config.get("lora")

        self.model_config = model_config

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

    def generate(self, input: str, stop):
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to("cuda")

        stopping_criteria_list = None

        if stop is not None:
            stopping_criteria_list = StoppingCriteriaList(
                [ConversationalStoppingCriteria(self.tokenizer, len(input), stop)]
            )

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria_list,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
