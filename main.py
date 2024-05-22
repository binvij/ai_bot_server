from typing import Union
from fastapi import FastAPI
from utils import rag_helper
from contextlib import asynccontextmanager
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import os

env_file = find_dotenv(".keys")
load_dotenv(env_file)

# lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    # run at startup
    # load the pdf at startup and store the docstorage to be used later
    app.state.doc_storage = rag_helper.load_pdf_doc()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/bot/{query}")
def ask_bot(query: str):
    print(f"user query={query}")
    qa = RetrievalQA.from_chain_type(
        OpenAI(
            model_name="gpt-3.5-turbo-instruct",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        ),
        chain_type="stuff",
        retriever=app.state.doc_storage.as_retriever(),
    )
    resp = qa.invoke(query)
    return {"response": resp}


@app.get("/")
def read_root():
    return {"Hello": "World"}
