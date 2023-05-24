from __future__ import annotations
from langchain.memory import (
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory,
    CombinedMemory,
)
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo,
)
from typing import List


class LangChainChatbot:
    def __init__(
        self,
        llm,
        templates,
        memory_vectorstore: Chroma,
        vectorstores: List[VectorStoreInfo],
        verbose=False,
    ) -> None:
        self.llm = llm

        self.chat_history = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=500,
            memory_key="chat_history",
            input_key="input",
        )

        retriever = memory_vectorstore.as_retriever(search_kwargs=dict(k=5))
        self.vectorstore_memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="memories",
            input_key="input",
        )

        self.memory = CombinedMemory(
            memories=[self.chat_history, self.vectorstore_memory]
        )

        router_toolkit = VectorStoreRouterToolkit(
            vectorstores=vectorstores,
            llm=self.llm,
        )

        tools = router_toolkit.get_tools()

        self.agent_executor = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=verbose,
            memory=self.memory,
            agent_kwargs={
                "prefix": templates["agent_prefix"],
                "suffix": templates["agent_suffix"],
                "input_variables": [
                    "input",
                    "memories",
                    "chat_history",
                    "agent_scratchpad",
                ],
            },
        )

        # self.agent_executor = create_vectorstore_router_agent(
        #     llm=llm,
        #     toolkit=router_toolkit,
        #     verbose=verbose,
        # )

        # tools = load_tools(["llm-math"], llm=llm)
        # agent = initialize_agent(
        #     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose
        # )

        # self.prompt = PromptTemplate(
        #     input_variables=["history", "input"], template=templates["chat"]
        # )

        # self.stop = ["Human: "]

        # self.llm_chain = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     retriever=vectorstore.as_retriever(),
        #     memory=self.memory,
        # )

        # safety_precheck_prompt = PromptTemplate(
        #     input_variables=["input"], template=templates["safety_precheck"]
        # )

        # self.safety_precheck_chain = LLMChain(
        #     llm=llm, prompt=safety_precheck_prompt, verbose=verbose
        # )

        # self.safety_reject_prompt = PromptTemplate(
        #     input_variables=["input"], template=templates["safety_reject"]
        # )

        # self.safety_reject_chain = LLMChain(
        #     llm=llm, prompt=self.safety_reject_prompt, verbose=verbose
        # )

        # principles = ConstitutionalChain.get_principles(["uo-ethics-1", "offensive"])

        # self.constitutional_chain = ConstitutionalChain.from_llm(
        #     chain=self.llm_chain,
        #     constitutional_principles=principles,
        #     llm=llm,
        #     verbose=verbose,
        #     memory=self.memory,
        # )

        self.model_name = self.llm._llm_type

    def generate_response(self, user_message: str):
        # possibly_unsafe = self.safety_precheck_chain.predict(input=user_message.strip())
        # print(possibly_unsafe)
        # if "False" not in possibly_unsafe:
        #     return self.safety_reject_chain.predict(input=user_message.strip()).strip()
        # else:
        return self.agent_executor.run(
            input=user_message.strip(),
        ).strip()

        # return self.constitutional_chain.run(
        #     input=user_message.strip(),
        # ).strip()
