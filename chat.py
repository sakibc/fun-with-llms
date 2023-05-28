from ui.chatbot_cmd import ChatbotCmd
from ui.chatbot_gradio import ChatbotGradio
from chatbot.langchain_chatbot import LangChainChatbot
from langchain.llms import OpenAI
from llm.langchain_wrapper import LangChainWrapper
from dotenv import load_dotenv
from ui.chatbot_server import ChatbotServer
import os
import json
from knowledge.vectorstores import load_vectorstores
import langchain

import argparse


def main():
    load_dotenv()

    models = [name.split(".")[0] for name in os.listdir("models")] + ["openai"]

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

    langchain.verbose = verbose

    templates = {}

    for template_name in os.listdir("templates"):
        with open(os.path.join("templates", template_name)) as f:
            template_name = os.path.splitext(template_name)[0]
            templates[template_name] = f.read()

    memory_vectorstore, vectorstores = load_vectorstores()

    if model_name == "openai":
        llm = OpenAI(temperature=0.2)

    else:
        if args.hosted:
            model_type = "hosted"
        else:
            model_type = "local"

        with open(os.path.join("models", f"{model_name}.json")) as f:
            model_config = json.load(f)

        if model_type == "local":
            from llm.model import Model

            model = Model(model_name, model_config)
        elif model_type == "hosted":
            from llm.hosted_model import HostedModel

            url = os.getenv("URL")
            token = os.getenv("HOSTED_MODEL_TOKEN")

            model = HostedModel(model_name, model_config, url=url, token=token)

        llm = LangChainWrapper(model=model)

    bot = LangChainChatbot(
        llm=llm,
        templates=templates,
        memory_vectorstore=memory_vectorstore,
        vectorstores=vectorstores,
    )

    if ui_type == "cmd":
        server = ChatbotCmd(bot)
    elif ui_type == "gradio":
        server = ChatbotGradio(bot)
    elif ui_type == "server":
        server = ChatbotServer(bot)

    server.run()


if __name__ == "__main__":
    main()
