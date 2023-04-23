from chatbot_cmd import ChatbotCmd
from chatbot_gradio import ChatbotGradio
from model import Model
from langchain_chatbot import LangChainChatbot
from langchain.llms import OpenAI
from langchain_model import LangChainModel
from dotenv import load_dotenv

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
        choices=["llama", "alpaca", "bloomz", "openai"],
    )
    parser.add_argument("ui_type", help="Type of UI to use", choices=["cmd", "gradio"])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    model_name = args.model_name
    ui_type = args.ui_type
    verbose = args.verbose

    template = basic_template
    stop = basic_stop

    if model_name == "llama":
        model_config = {
            "path": "decapoda-research/llama-7b-hf",
        }

        model = Model(model_config)
        llm = LangChainModel(model=model)

    if model_name == "alpaca":
        model_config = {
            "path": "decapoda-research/llama-7b-hf",
            "lora": "tloen/alpaca-lora-7b",
        }

        model = Model(model_config)
        llm = LangChainModel(model=model)
        template = alpaca_template
        stop = alpaca_stop

    elif model_name == "bloomz":
        model_config = {
            "path": "bigscience/bloomz-7b1",
        }

        model = Model(model_config)
        llm = LangChainModel(model=model)

    elif model_name == "openai":
        llm = OpenAI(temperature=0.2)

    bot = LangChainChatbot(llm=llm, template=template, stop=stop, verbose=verbose)

    if ui_type == "cmd":
        server = ChatbotCmd(bot)
    elif ui_type == "gradio":
        server = ChatbotGradio(bot)

    server.run()


if __name__ == "__main__":
    main()
