from __future__ import annotations
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts.prompt import PromptTemplate


class LangChainChatbot:
    def __init__(self, llm, template, verbose=False) -> None:
        self.llm = llm

        self.memory = ConversationBufferMemory()

        self.prompt = PromptTemplate(
            input_variables=["history", "input"], template=template
        )

        self.stop = ["Human: "]

        self.llm_chain = LLMChain(
            llm=self.llm, prompt=self.prompt, verbose=verbose, memory=self.memory
        )

        # load safety template
        with open("./templates/safety_check.txt", "r") as f:
            safety_template = f.read()

        safety_prompt = PromptTemplate(
            input_variables=["response"], template=safety_template
        )

        self.safety_chain = LLMChain(llm=llm, prompt=safety_prompt, verbose=verbose)

        with open("./templates/rewrite.txt", "r") as f:
            safety_rewrite_template = f.read()

        safety_rewrite_prompt = PromptTemplate(
            input_variables=["input", "response"], template=safety_rewrite_template
        )

        self.safety_rewrite_chain = LLMChain(
            llm=llm, prompt=safety_rewrite_prompt, verbose=verbose
        )

        self.model_name = self.llm._llm_type

    def generate_response(self, user_message: str):
        response = self.llm_chain.predict(
            input=user_message.strip(),
            stop=self.stop,
        )

        is_safe = self.safety_chain.predict(response=response)

        if "True" not in is_safe:
            response = self.safety_rewrite_chain.predict(
                input=user_message.strip(), response=response
            )
            self.memory.chat_memory.messages[-1].content = response

        return response.strip()
