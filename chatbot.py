from __future__ import annotations
from model import Model


class Conversation:
    def __init__(self, model_config: dict[str, any]) -> None:
        self.instruction = model_config["instruction"]
        self.user_label = model_config["user_label"]
        self.chatbot_name = model_config["chatbot_name"]
        self.chatbot_label = model_config["chatbot_label"]
        self.dialogue = ""
        self.last_response = ""


class Chatbot:
    def __init__(self, model_config: dict[str, any]) -> None:
        self.model_config = model_config

        self.model = Model(self.model_config)
        self.model_name = self.model.model_name

    def new_session_state(self, custom_state: dict[str, str] = None):
        if custom_state:
            return Conversation(custom_state)

        return Conversation(self.model_config)

    def submit_message(self, conversation: Conversation, user_message: str):
        if conversation.dialogue != "":
            conversation.dialogue += "\n\n"

        conversation.dialogue += f"{conversation.user_label}{user_message.strip()}"

    def generate_response(self, conversation: Conversation):
        conversation.dialogue += f"\n\n{conversation.chatbot_label}"

        input = f"{conversation.instruction}\n\n{conversation.dialogue}"

        conversation.last_response = self.model.generate_response(input)

        conversation.dialogue += conversation.last_response

    def get_last_response(self, conversation: Conversation):
        return conversation.last_response
