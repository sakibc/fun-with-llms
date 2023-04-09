from chatbot_server import ChatbotServer
from chatbots.dialogpt_bot import DialoGPTBot
from chatbots.dialogpt_pipeline_bot import DialoGPTPipelineBot
from chatbots.godel_pipeline_bot import GodelPipelineBot
from chatbots.godel_bot import GodelBot
from chatbots.bloomz_bot import BloomzBot
from chatbots.llama_bot import LlamaBot
from chatbots.alpaca_bot import AlpacaBot

import sys


def main():
    if len(sys.argv) > 1:
        bot_name = sys.argv[1]
    else:
        bot_name = "dialogpt"

    if bot_name == "dialogpt-pipeline":
        bot = DialoGPTPipelineBot()
    elif bot_name == "dialogpt":
        bot = DialoGPTBot()
    elif bot_name == "godel-pipeline":
        bot = GodelPipelineBot()
    elif bot_name == "godel":
        bot = GodelBot()
    elif bot_name == "bloomz":
        bot = BloomzBot()
    elif bot_name == "llama":
        bot = LlamaBot()
    elif bot_name == "alpaca":
        bot = AlpacaBot()
    else:
        raise

    server = ChatbotServer(bot)
    server.run()


if __name__ == "__main__":
    main()
