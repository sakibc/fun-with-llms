from __future__ import annotations
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class Conversation:
    def __init__(self, model_config: dict[str, any]) -> None:
        self.system_message = SystemMessage(content=model_config["instruction"])
        self.user_label = model_config["user_label"]
        self.chatbot_label = model_config["chatbot_label"]
        self.chatbot_name = model_config["chatbot_name"]
        self.dialogue = []
        self.last_response = None


class LangChainChatbot:
    def __init__(self, model_config: dict[str, any]) -> None:
        self.model_config = model_config

        self.chat = ChatOpenAI(temperature=0)
        self.model_name = "OpenAI"

    def new_session_state(self, custom_state: dict[str, str] = None):
        if custom_state:
            return Conversation(custom_state)

        return Conversation(self.model_config)

    def submit_message(self, conversation: Conversation, user_message: str):
        conversation.dialogue.append(HumanMessage(content=user_message.strip()))

    def generate_response(self, conversation: Conversation):
        conversation.last_response = self.chat(
            [conversation.system_message] + conversation.dialogue
        )

        conversation.dialogue.append(conversation.last_response)

    def get_last_response(self, conversation: Conversation):
        return conversation.last_response.content
