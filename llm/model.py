from __future__ import annotations
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from peft import PeftModelForCausalLM, PeftModelForSeq2SeqLM

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

    def __init__(self, model_name: str, model_config: dict[str, any], size: str):
        self.model_name = model_name

        print(f"Using {self.model_name}")

        if "pipeline" in model_config:
            self.model_type = "pipeline"
            model_path = model_config["pipeline"]
        elif "seq2seq" in model_config:
            self.model_type = "seq2seq"
            model_path = model_config["seq2seq"]
        else:
            self.model_type = "causal"
            model_path = model_config["causal"]

        if "sizes" in model_config:
            if size not in model_config["sizes"]:
                raise ValueError(f"Invalid size: {size}")

            model_path = model_path.format(size=size)

        if self.model_type == "pipeline":
            if not Model.pipeline:
                self.load_pipeline(model_path)
        else:
            if not Model.tokenizer:
                self.load_tokenizer(model_path)

            if not Model.model:
                self.load_model(model_path)

            if "lora" in model_config and not Model.lora_applied:
                self.apply_lora(model_config["lora"])

    def load_tokenizer(self, model_path: str):
        print("Loading tokenizer...")

        if "llama" in model_path:
            Tokenizer = LlamaTokenizer
        else:
            Tokenizer = AutoTokenizer

        Model.tokenizer = Tokenizer.from_pretrained(model_path)

        print("Tokenizer loaded.")

    def load_model(self, model_path: str):
        print("Loading model...")

        if self.model_type == "seq2seq":
            AutoModel = AutoModelForSeq2SeqLM
        else:
            AutoModel = AutoModelForCausalLM

        Model.model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
        )

        print("Model loaded.")

    def apply_lora(self, lora_path: str):
        print("Applying LoRA...")

        if self.model_type == "seq2seq":
            PeftModel = PeftModelForSeq2SeqLM
        else:
            PeftModel = PeftModelForCausalLM

        Model.model = PeftModel.from_pretrained(
            Model.model,
            lora_path,
        )

        self.lora_applied = True

        print("LoRA applied.")

    def load_pipeline(self, pipeline_name: str):
        print("Loading pipeline...")

        Model.pipeline = pipeline(
            model=pipeline_name,
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

            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            if self.model_type == "seq2seq":
                return input + decoded
            else:
                return decoded
