from typing import Union
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import find_dotenv, load_dotenv
from engine.store.data_loader import DataLoader
from engine.ai_chat import AIChat
from model.user_query import UserQuery
import os
import logging


logger = logging.getLogger(__name__)


# lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("lifespan init!")
    logger.info("lifespan() run")
    # run at startup
    # load the pdf at startup and store the docstorage to be used later
    env_file = find_dotenv(".keys")
    load_dotenv(env_file)
    open_api_key = os.getenv("OPENAI_API_KEY")
    doc_name = os.getenv("PDF_DOC_NAME")
    data_loader = DataLoader(api_key=open_api_key, pdf_name=doc_name)
    app.state.doc_storage = data_loader.get_vector_store()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/bot/v1/ask")
async def ask_bot(user_query: UserQuery):
    ai_chat = AIChat(app=app)
    rag_chain = ai_chat.get_llm_rag_chain()
    resp = rag_chain.invoke(user_query.query)
    return {"response": resp}
