from chatbot_cmd import ChatbotCmd
from chatbot_server import ChatbotServer
from dotenv import load_dotenv

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python chat.py <bot_name> <server | cmd>")
        return

    load_dotenv()

    bot_name = sys.argv[1]
    app_type = sys.argv[2]

    if bot_name == "llama":
        model_config = {
            "path": "decapoda-research/llama-7b-hf",
            "chatbot_name": "Chatbot",
            "instruction": "This is a dialogue between User and Chatbot. Chatbot is helpful, friendly, and eager to please. An example dialogue looks like this:\n\nUser: Hello, how are you?\n\nChatbot: Fine, thank you. How may I be of assistance?\n\nAs you can see, Chatbot provides long, meaningful answers to all of User's questions.",
            "user_label": "User: ",
            "chatbot_label": "Chatbot: ",
            "stopping_string": "User: ",
        }

    elif bot_name == "alpaca":
        model_config = {
            "path": "decapoda-research/llama-7b-hf",
            "lora": "tloen/alpaca-lora-7b",
            "chatbot_name": "Chatbot",
            "instruction": "Below is a dialogue of instructions given by the user and responses given by you in the past, paired with an input that provides further context. Write a response that appropriately completes the latest user request.\n\n### Input:\nYou are a chatbot running on a desktop computer.",
            "user_label": "### Instruction:\n",
            "chatbot_label": "### Response:\n",
            "stopping_string": "### Instruction:\n",
        }

    elif bot_name == "bloomz":
        model_config = {
            "path": "bigscience/bloomz-7b1",
            "chatbot_name": "Chatbot",
            "instruction": "This is a dialogue between User and Chatbot. Chatbot is helpful, friendly, and eager to please. An example dialogue looks like this:\n\nUser: Hello, how are you?\n\nChatbot: Fine, thank you. How may I be of assistance?\n\nAs you can see, Chatbot provides long, meaningful answers to all of User's questions.",
            "user_label": "User: ",
            "chatbot_label": "Chatbot: ",
            "stopping_string": "User: ",
        }

    elif bot_name == "openai":
        model_config = {
            "path": "openai",
            "chatbot_name": "Chatbot",
            "instruction": "This is a dialogue between User and Chatbot. Chatbot is helpful, friendly, and eager to please. An example dialogue looks like this:\n\nUser: Hello, how are you?\n\nChatbot: Fine, thank you. How may I be of assistance?\n\nAs you can see, Chatbot provides long, meaningful answers to all of User's questions.",
            "user_label": "User: ",
            "chatbot_label": "Chatbot: ",
            "stopping_string": "User: ",
        }

    else:
        raise

    if bot_name == "openai":
        from langchain_chatbot import LangChainChatbot

        bot = LangChainChatbot(model_config)
    else:
        from chatbot import Chatbot

        bot = Chatbot(model_config)

    if app_type == "cmd":
        server = ChatbotCmd(bot)
    elif app_type == "server":
        server = ChatbotServer(bot)

    server.run()


if __name__ == "__main__":
    main()
