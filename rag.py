from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
import os
import pandas as pd
import numpy as np
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def split_data(url):
    """
    data file location url: str
    content_column: str

    splitted_text: Document(page_content= str, metadata={})
    ids: str(uuid4)
    """
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()
    return pages

def split_data_csv(url, content_column):
    """
    data file location url: str
    content_column: str

    splitted_text: Document(page_content= str, metadata={})
    ids: str(uuid4)
    """
    df = pd.read_csv(url)
    articles = DataFrameLoader(df, page_content_column=content_column)
    documents = articles.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                              chunk_overlap = 20)
    splitted_text = splitter.split_documents(documents)
    ids = [str(uuid4()) for _ in splitted_text]
    return splitted_text, ids

def vectorstore(splitted_texts):
    """
    splitted_text: List[Documents]
    ids: str(uuid4)
    """
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    chromadb = Chroma.from_documents(splitted_texts,
                                 embeddings_model,
                                #  ids=ids,
                                 persist_directory = './chromadb')
    return chromadb

def genchain():
    """
    generates the retrieval chain
    :type chromadb: Chroma
    :rtype chain: ConversationalRetrievalChain
    """
    embedding_function = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    chromadb = Chroma(persist_directory="./chromadb", embedding_function=embedding_function)
    retriever = chromadb.as_retriever()
    llm = ChatOpenAI(temperature=0.3,
                 model_name = 'gpt-3.5-turbo',
                api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        memory=memory
    )
    return chain

def retreival_chain(chain, query):
    """
    chain: RetreivalChain
    query: str
    """
    response = chain({'question':query})
    return response['answer']


if  __name__ == '__main__':
    # data prep
    # url = r'C:\Users\hrush\OneDrive - Student Ambassadors\Desktop\discord bot\biconomyPay.pdf'
    # split_doc= split_data(url)
    # chromadb = vectorstore(split_doc)
    
    chain = genchain()

    # retreival
    query = 'What are paymaster methods and provide the code write smart contract?'
    response = retreival_chain(chain,query)
    print(response)