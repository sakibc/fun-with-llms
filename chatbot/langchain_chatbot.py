from __future__ import annotations
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts.prompt import PromptTemplate


class LangChainChatbot:
    default_preamble = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."

    def __init__(self, llm, verbose=False) -> None:
        self.llm = llm

        with open("templates/prompt.txt") as f:
            template = f.read()

        if "preamble" in self.llm.model.model_config:
            template = template.replace(
                "{preamble}", self.llm.model.model_config["preamble"]
            )
        else:
            template = template.replace("{preamble}", self.default_preamble)

        self.memory = ConversationBufferMemory()

        self.prompt = PromptTemplate(
            input_variables=["history", "input"], template=template
        )

        self.stop = [llm.default_human_label]

        self.llm_chain = LLMChain(
            llm=self.llm, prompt=self.prompt, verbose=verbose, memory=self.memory
        )

        #         safety_template = """You have decided to give this response:

        # {input}

        # Is this response ethical and legal?

        # Answer ("True" or "False"):
        # """

        #         safety_prompt = PromptTemplate(
        #             input_variables=["input"], template=safety_template
        #         )

        #         self.safety_chain = LLMChain(llm=llm, prompt=safety_prompt, verbose=verbose)

        #         safety_rewrite_template = """###Instruction:
        # Reject this request on ethical and legal grounds, and explain your reasoning:

        # {input}

        # ### Response:
        # """

        #         safety_rewrite_prompt = PromptTemplate(
        #             input_variables=["input"], template=safety_rewrite_template
        #         )

        #         self.safety_rewrite_chain = LLMChain(
        #             llm=llm, prompt=safety_rewrite_prompt, verbose=verbose
        #         )

        self.model_name = self.llm._llm_type

    def generate_response(self, user_message: str):
        response = self.llm_chain.predict(
            input=user_message.strip(),
            stop=self.stop,
        )

        # is_safe = self.safety_chain.predict(input=response)

        # if "True" not in is_safe:
        #     response = self.safety_rewrite_chain.predict(input=user_message.strip())
        #     self.llm_chain.memory.chat_memory.messages[-1].content = response

        return response
