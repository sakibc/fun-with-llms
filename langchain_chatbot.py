from __future__ import annotations
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts.prompt import PromptTemplate


class LangChainChatbot:
    def __init__(self, llm, template, stop) -> None:
        llm = llm

        self.memory = ConversationBufferMemory()

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        self.stop = stop

        self.chain = LLMChain(llm=llm, memory=self.memory, prompt=prompt)

        self.model_name = "openai"

    def generate_response(self, user_message: str):
        return self.chain.predict(
            input=user_message.strip(),
            stop=self.stop,
        ).strip()
