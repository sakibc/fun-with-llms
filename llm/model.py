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
    def __init__(self, stop_ids):
        StoppingCriteria.__init__(self)
        self.stop_ids = stop_ids

    def __call__(
        self,
        input_ids,
        scores,
    ) -> bool:
        input_ids = input_ids[0]
        len_input = input_ids.shape[0]

        for stop in self.stop_ids:
            len_stop = stop.shape[0]

            if len_input < len_stop:
                continue

            if torch.equal(input_ids[-len_stop:], stop):
                return True


class Model:
    model = None
    pipeline = None
    tokenizer = None
    lora_applied = False

    model_config: dict[str, any]

    def __init__(
        self, model_name: str, model_config: dict[str, any], size: str, backend: str
    ):
        self.model_name = model_name
        self.backend = backend

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

        kwargs = {}

        if self.backend == "mps":
            kwargs["torch_dtype"] = torch.float16
        if self.backend == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["load_in_8bit"] = True

        Model.model = AutoModel.from_pretrained(
            model_path,
            **kwargs,
        )

        if self.backend == "mps":
            Model.model = Model.model.to("mps")

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
            input_ids = self.tokenizer.encode(input, return_tensors="pt")

            if self.backend == "mps" or self.backend == "cuda":
                input_ids = input_ids.to(self.backend)

            stopping_criteria_list = None

            if stop is not None:
                if self.backend == "mps" or self.backend == "cuda":
                    stop_ids = [
                        self.tokenizer.encode(s, return_tensors="pt")[0].to(
                            self.backend
                        )
                        for s in stop
                    ]
                else:
                    stop_ids = [
                        self.tokenizer.encode(s, return_tensors="pt")[0] for s in stop
                    ]

                stopping_criteria_list = StoppingCriteriaList(
                    [ConversationalStoppingCriteria(stop_ids)]
                )

            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                stopping_criteria=stopping_criteria_list,
            )

            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            return decoded
