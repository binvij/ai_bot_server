from fastapi import FastAPI
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from engine.helper.prompt_helper import PromptHelper

import os


class AIChat:
    def __init__(self, app: FastAPI):
        self.app = app
        self.prompt_helper = PromptHelper()
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL_NAME"),
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def get_llm_rag_chain(self):
        # get the rag vector store retreiver
        retriever = self.app.state.doc_storage.as_retriever()
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt_helper.custom_prompt()
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
