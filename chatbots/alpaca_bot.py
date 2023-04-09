from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModelForCausalLM

import torch

default_chatbot_name = "Chatbot"
default_instruction = "Below is a dialogue of instructions and responses that each describe a task, paired with an input that provides further context. Write a response that appropriately completes the latest request.\n\n### Input:\nYou are a chatbot created by Facebook and tuned by Stanford. The current time is 5:44 PM. You know that the user is located in a condo apartment in Downtown Toronto, but you do not know their precise location. You are running on a desktop computer."
default_user_label = "### Instruction:"
default_chatbot_label = "### Response:"


class Conversation:
    def __init__(
        self,
        instruction=default_instruction,
        user_label=default_user_label,
        chatbot_label=default_chatbot_label,
    ) -> None:
        self.instruction = instruction
        self.dialogue = ""
        self.chatbot_name = default_chatbot_name
        self.user_label = user_label
        self.chatbot_label = chatbot_label
        self.last_response = ""


class AlpacaBot:
    model = None
    tokenizer = None

    def __init__(self) -> None:
        llama_path = "/extended-storage/llama-hf/7B"
        self.model_name = f"{llama_path} with tloen/alpaca-lora-7b"
        print(f"Using {self.model_name}")
        self.chatbot_name = default_chatbot_name

        self.load_tokenizer(llama_path)
        self.load_model(llama_path)

    def load_tokenizer(self, model_name):
        if AlpacaBot.tokenizer is None:
            print("Loading tokenizer...")

            AlpacaBot.tokenizer = LlamaTokenizer.from_pretrained(model_name)

            print("Tokenizer loaded.")

    def load_model(self, model_name):
        if AlpacaBot.model is None:
            print("Loading model...")

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

            AlpacaBot.model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )

            print("Applying LoRA...")

            AlpacaBot.model = PeftModelForCausalLM.from_pretrained(
                AlpacaBot.model,
                "tloen/alpaca-lora-7b",
            )

            print("Model loaded.")

    def generate_response(self, conversation: Conversation):
        conversation.dialogue += f"\n{conversation.chatbot_label}\n"

        input = f"{conversation.instruction}\n\n{conversation.dialogue}"

        inputs = self.tokenizer.encode(input, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        conversation.last_response = (
            output[len(input) :].split(conversation.user_label)[0].strip()
        )
        conversation.dialogue += conversation.last_response

    def submit_message(self, conversation, user_message):
        if conversation.dialogue != "":
            conversation.dialogue += "\n"

        conversation.dialogue += f"{conversation.user_label}\n{user_message.strip()}"

    def get_response(self, conversation):
        return conversation.last_response

    def raw_dialogue(self, conversation):
        return conversation.dialogue

    def get_instruction(self, conversation=None):
        if conversation == None:
            return default_instruction

        return conversation.instruction

    def set_instruction(self, conversation: Conversation, instruction: str):
        conversation.instruction = instruction

    def get_user_label(self, conversation=None):
        if conversation == None:
            return default_user_label

        return conversation.user_label

    def set_user_label(self, conversation: Conversation, user_label: str):
        conversation.user_label = user_label

    def get_chatbot_label(self, conversation=None):
        if conversation == None:
            return default_chatbot_label

        return conversation.chatbot_label

    def set_chatbot_label(self, conversation: Conversation, chatbot_label: str):
        conversation.chatbot_label = chatbot_label

    def new_session_state(self):
        return Conversation()

    def new_session_state_non_default(self, instruction, user_label, chatbot_label):
        return Conversation(instruction, user_label, chatbot_label)
