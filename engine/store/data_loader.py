from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class DataLoader:

    def __init__(self, api_key: str, pdf_name: str):
        self.api_key = api_key
        self.pdf_name = pdf_name

    def get_vector_store(self) -> Chroma:
        loader = PyPDFLoader(self.pdf_name)
        data = loader.load()
        chunk_size = 200
        chunk_overlap = 50
        # Split the quote using RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = splitter.split_documents(data)
        # Define an OpenAI embeddings model
        embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)

        # Create the Chroma vector DB using the OpenAI embedding function; persist the database
        vectordb = Chroma(
            persist_directory="embedding/chrome", embedding_function=embedding_model
        )
        docstorage = Chroma.from_documents(docs, embedding_model)
        return docstorage
