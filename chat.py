from chatbot_cmd import ChatbotCmd
from chatbot_server import ChatbotGradio
from model import Model
from langchain_chatbot import LangChainChatbot
from langchain.llms import OpenAI
from langchain_model import LangChainModel
from dotenv import load_dotenv

import sys

basic_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""

basic_stop = ["\nHuman:", "Human:"]

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### History:
{history}

### Instruction:
{input}

### Response:
"""

alpaca_stop = ["\n### Instruction:", "### Instruction:"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python chat.py <bot_name> <server | cmd>")
        return

    load_dotenv()

    bot_name = sys.argv[1]
    app_type = sys.argv[2]

    template = basic_template
    stop = basic_stop

    if bot_name == "llama":
        model_config = {
            "path": "decapoda-research/llama-7b-hf",
        }

        model = Model(model_config)
        llm = LangChainModel(model=model)

    if bot_name == "alpaca":
        model_config = {
            "path": "decapoda-research/llama-7b-hf",
            "lora": "tloen/alpaca-lora-7b",
        }

        model = Model(model_config)
        llm = LangChainModel(model=model)
        template = alpaca_template
        stop = alpaca_stop

    elif bot_name == "bloomz":
        model_config = {
            "path": "bigscience/bloomz-7b1",
        }

        model = Model(model_config)
        llm = LangChainModel(model=model)

    elif bot_name == "openai":
        llm = OpenAI(temperature=0)

    bot = LangChainChatbot(llm=llm, template=template, stop=stop)

    if app_type == "cmd":
        server = ChatbotCmd(bot)
    elif app_type == "gradio":
        server = ChatbotGradio(bot)

    server.run()


if __name__ == "__main__":
    main()
