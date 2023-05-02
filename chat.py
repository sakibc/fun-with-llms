from ui.chatbot_cmd import ChatbotCmd
from ui.chatbot_gradio import ChatbotGradio
from chatbot.langchain_chatbot import LangChainChatbot
from langchain.llms import OpenAI
from chatbot.langchain_model import LangChainModel
from dotenv import load_dotenv
from chatbot.hosted_langchain_model import HostedLangChainModel
from ui.chatbot_server import ChatbotServer
import os

import argparse
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
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_name",
        help="Name of the model to use",
        choices=["llama", "alpaca", "bloomz", "openai", "hosted_alpaca"],
    )
    parser.add_argument("ui_type", help="Type of UI to use", choices=["cmd", "gradio", "server"])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    model_name = args.model_name
    ui_type = args.ui_type
    verbose = args.verbose

    template = basic_template
    stop = basic_stop

    model_type = None

    if model_name == "llama":
        model_config = { "path": "decapoda-research/llama-7b-hf" }
        model_type = "local"

    if model_name == "alpaca":
        model_config = {
            "path": "decapoda-research/llama-7b-hf",
            "lora": "tloen/alpaca-lora-7b",
        }

        model_type = "local"

        template = alpaca_template
        stop = alpaca_stop

    elif model_name == "bloomz":
        model_config = { "path": "bigscience/bloomz-7b1" }
        model_type = "local"

    elif model_name == "openai":
        model_type = "openai"

    elif model_name == "hosted_alpaca":
        model_type = "hosted"

        template = alpaca_template
        stop = alpaca_stop

    if model_type == "local":
        from llm.model import Model

        model = Model(model_config)
        llm = LangChainModel(model=model)
    elif model_type == "openai":
        llm = OpenAI(temperature=0.2)
    elif model_type == "hosted":
        url = os.getenv("URL")
        token = os.getenv("HOSTED_MODEL_TOKEN")

        llm = HostedLangChainModel(model_name=model_name, url=url, token=token)

    bot = LangChainChatbot(llm=llm, template=template, stop=stop, verbose=verbose)

    if ui_type == "cmd":
        server = ChatbotCmd(bot)
    elif ui_type == "gradio":
        server = ChatbotGradio(bot)
    elif ui_type == "server":
        server = ChatbotServer(bot)

    server.run()


if __name__ == "__main__":
    main()
