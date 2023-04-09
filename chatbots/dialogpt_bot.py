from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Conversation:
    def __init__(self) -> None:
        last_generation_length = 0
        self.chat_history = None


class DialoGPTBot:
    model = None
    tokenizer = None

    def __init__(self, model_name) -> None:
        model_name = "microsoft/DialoGPT-large"
        print(f"Using {model_name}")

        self.load_tokenizer(model_name)
        self.load_model(model_name)

    def load_tokenizer(self, model_name):
        if DialoGPTBot.tokenizer is None:
            print("Loading tokenizer...")

            DialoGPTBot.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print("Tokenizer loaded.")

    def load_model(self, model_name):
        if DialoGPTBot.model is None:
            print("Loading model...")

            DialoGPTBot.model = AutoModelForCausalLM.from_pretrained(model_name).to(
                "cuda"
            )

            print("Model loaded.")

    def generate_response(self, conversation):
        old_history_length = conversation.chat_history.shape[-1]

        conversation.chat_history = self.model.generate(
            conversation.chat_history,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        conversation.new_output_length = (
            conversation.chat_history.shape[-1] - old_history_length
        )

    def submit_message(self, conversation, user_message):
        inputs = self.tokenizer.encode(
            user_message + self.tokenizer.eos_token,
            return_tensors="pt",
        ).to("cuda")

        if conversation.chat_history != None:
            conversation.chat_history = torch.cat(
                [conversation.chat_history, inputs], dim=-1
            )
        else:
            conversation.chat_history = inputs

    def get_response(self, conversation):
        return self.tokenizer.decode(
            conversation.chat_history[
                :,
                conversation.chat_history.shape[-1] - conversation.new_output_length :,
            ][0],
            skip_special_tokens=True,
        )

    def new_session_state(self):
        return Conversation()
