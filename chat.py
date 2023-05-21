from ui.chatbot_cmd import ChatbotCmd
from ui.chatbot_gradio import ChatbotGradio
from chatbot.langchain_chatbot import LangChainChatbot
from langchain.llms import OpenAI
from chatbot.langchain_model import LangChainModel
from dotenv import load_dotenv
from ui.chatbot_server import ChatbotServer
import os
import json

import argparse


def main():
    load_dotenv()

    # Get list of models from the folder names that are subfolders of the models folder
    models = [
        name
        for name in os.listdir("models")
        if os.path.isdir(os.path.join("models", name))
    ] + ["openai"]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_name",
        help="Name of the model to use",
        choices=models,
    )
    parser.add_argument(
        "--hosted", action="store_true", help="Use the model hosted in the cloud"
    )
    parser.add_argument(
        "ui_type", help="Type of UI to use", choices=["cmd", "gradio", "server"]
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    model_name = args.model_name
    ui_type = args.ui_type
    verbose = args.verbose

    model_type = None

    if model_name == "openai":
        model_type = "openai"
    elif args.hosted:
        model_type = "hosted"
    else:
        model_type = "local"

    # read model.json from the model's folder into model_config
    with open(os.path.join("models", model_name, "model.json")) as f:
        model_config = json.load(f)

    if model_type == "local":
        from llm.model import Model

        model = Model(model_config)
        llm = LangChainModel(model=model)
    elif model_type == "hosted":
        from llm.hosted_model import HostedModel

        url = os.getenv("URL")
        token = os.getenv("HOSTED_MODEL_TOKEN")

        model = HostedModel(model_config, url=url, token=token)
        llm = LangChainModel(model=model)
    elif model_type == "openai":
        llm = OpenAI(temperature=0.2)

    bot = LangChainChatbot(llm=llm, verbose=verbose)

    if ui_type == "cmd":
        server = ChatbotCmd(bot)
    elif ui_type == "gradio":
        server = ChatbotGradio(bot)
    elif ui_type == "server":
        server = ChatbotServer(bot)

    server.run()


if __name__ == "__main__":
    main()
