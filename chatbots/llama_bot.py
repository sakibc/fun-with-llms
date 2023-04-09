from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch

default_chatbot_name = "Chatbot"
default_instruction = f"This is a dialogue between User and {default_chatbot_name}. {default_chatbot_name} is helpful, friendly, and eager to please. An example dialogue looks like this:\nUser: Hello, how are you?\n{default_chatbot_name}: Fine, thank you. How may I be of assistance?\nAs you can see, {default_chatbot_name} provides long, meaningful answers to all of User's questions.\n"
default_user_label = "User:"
default_chatbot_label = "Chatbot:"


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


class ConversationalStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, user_label=default_user_label):
        StoppingCriteria.__init__(self)
        self.user_label = user_label
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(
        self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs
    ) -> bool:
        output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        output = output[self.prompt_len :]
        if self.user_label in output:
            return True


class LlamaBot:
    model = None
    tokenizer = None

    def __init__(self) -> None:
        self.model_name = "/extended-storage/llama-hf/7B"
        print(f"Using {self.model_name}")
        self.chatbot_name = default_chatbot_name

        self.load_tokenizer(self.model_name)
        self.load_model(self.model_name)

    def load_tokenizer(self, model_name):
        if LlamaBot.tokenizer is None:
            print("Loading tokenizer...")

            LlamaBot.tokenizer = LlamaTokenizer.from_pretrained(model_name)

            print("Tokenizer loaded.")

    def load_model(self, model_name):
        if LlamaBot.model is None:
            print("Loading model...")

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

            LlamaBot.model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )

            print("Model loaded.")

    def generate_response(self, conversation: Conversation):
        conversation.dialogue += f"\n{conversation.chatbot_label} "

        input = f"{conversation.instruction}\n{conversation.dialogue}"

        inputs = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            inputs,
            max_new_tokens=64,
            stopping_criteria=StoppingCriteriaList(
                [
                    ConversationalStoppingCriteria(
                        self.tokenizer, len(input), conversation.user_label
                    )
                ]
            ),
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        conversation.last_response = (
            output[len(input) :].split(conversation.user_label)[0].strip()
        )
        conversation.dialogue += conversation.last_response

    def submit_message(self, conversation, user_message):
        if conversation.dialogue != "":
            conversation.dialogue += "\n"

        conversation.dialogue += f"{conversation.user_label} {user_message.strip()}"

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
