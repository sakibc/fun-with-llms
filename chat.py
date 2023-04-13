from chatbot_server import ChatbotServer
from chatbot import Chatbot

import sys


def main():
    if len(sys.argv) > 1:
        bot_name = sys.argv[1]
    else:
        bot_name = "alpaca"

    if bot_name == "llama":
        model_config = {
            "path": "",
            "chatbot_name": "Chatbot",
            "instruction": "This is a dialogue between User and Chatbot. Chatbot is helpful, friendly, and eager to please. An example dialogue looks like this:\n\nUser: Hello, how are you?\n\nChatbot: Fine, thank you. How may I be of assistance?\n\nAs you can see, Chatbot provides long, meaningful answers to all of User's questions.",
            "user_label": "User: ",
            "chatbot_label": "Chatbot: ",
            "stopping_string": "User: ",
        }

    elif bot_name == "alpaca":
        model_config = {
            "path": "",
            "lora": "tloen/alpaca-lora-7b",
            "chatbot_name": "Chatbot",
            "instruction": "Below is a dialogue of instructions given by the user and responses given by you in the past, paired with an input that provides further context. Write a response that appropriately completes the latest user request.\n\n### Input:\nYou are a chatbot created by Facebook and tuned by Stanford. The current time is 5:44 PM. You know that the user is located in a condo apartment in Downtown Toronto, but you do not know their precise location. You are running on a desktop computer.",
            "user_label": "### Instruction:\n",
            "chatbot_label": "### Response:\n",
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

    else:
        raise

    bot = Chatbot(model_config)

    server = ChatbotServer(bot)
    server.run()


if __name__ == "__main__":
    main()
