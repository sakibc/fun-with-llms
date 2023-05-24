from __future__ import annotations
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
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
    pipeline = None
    tokenizer = None
    lora_applied = False

    model_config: dict[str, any]

    def __init__(self, model_name: str, model_config: dict[str, any]):
        # if model_config contains a pipeline, use that instead
        if "pipeline" in model_config:
            self.pipeline_name = model_config["pipeline"]

            if not Model.pipeline:
                self.load_pipeline()
        else:
            self.model_path = model_config["path"]
            self.lora_path = model_config.get("lora")

            self.model_config = model_config

            self.model_name = model_name

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

    def load_pipeline(self):
        print("Loading pipeline...")

        Model.pipeline = pipeline(
            model=self.pipeline_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            return_full_text=True,
        )

        print("Pipeline loaded.")

    def generate(self, input: str, stop):
        if self.pipeline:
            return self.pipeline(input)[0]["generated_text"]
        else:
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
