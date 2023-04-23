from __future__ import annotations
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.prompt import PromptTemplate


class LangChainChatbot:
    def __init__(self, llm, template, stop, verbose=False) -> None:
        llm = llm

        self.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        self.stop = stop

        self.llm_chain = LLMChain(
            llm=llm, prompt=prompt, verbose=verbose, memory=self.memory
        )

        safety_template = """You have decided to give this response:

{input}

Is this response ethical and legal?

Answer ("True" or "False"): 
"""

        safety_prompt = PromptTemplate(
            input_variables=["input"], template=safety_template
        )

        self.safety_chain = LLMChain(llm=llm, prompt=safety_prompt, verbose=verbose)

        safety_rewrite_template = """###Instruction:
Reject this request on ethical and legal grounds, and explain your reasoning:

{input}

### Response:
"""

        safety_rewrite_prompt = PromptTemplate(
            input_variables=["input"], template=safety_rewrite_template
        )

        self.safety_rewrite_chain = LLMChain(
            llm=llm, prompt=safety_rewrite_prompt, verbose=verbose
        )

        self.model_name = llm._llm_type

    def generate_response(self, user_message: str):
        response = self.llm_chain.predict(
            input=user_message.strip(),
            stop=self.stop,
        )

        is_safe = self.safety_chain.predict(input=response)
        print(is_safe)

        if "True" not in is_safe:
            response = self.safety_rewrite_chain.predict(input=user_message.strip())
            self.llm_chain.memory.chat_memory.messages[-1].content = response

        return response
