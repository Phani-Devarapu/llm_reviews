import os


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain



directory = './data'
os.environ['OPENAI_API_KEY'] = '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'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)


def split_docs(documents,chunk_size=1000,chunk_overlap=200):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

ARTICLE_1 ="The hotel has a great location, the staff is friendly and helpful, the rooms are clean and comfortable, and the breakfast is delicious. However, the rooms are small and some of the facilities are outdated."
ARTICLE_2="The hotel is not good for family"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = ".\\chroma_db"


       
db = Chroma.from_documents(documents=docs, embedding=embeddings,persist_directory=persist_directory)


# db = Chroma.from_texts([ARTICLE_1,ARTICLE_2], embeddings,metadatas=None,ids=["Myntra Hotel","ZOO Hotel"])
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name,temperature=0.6)


chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

# query = "Which hotel is not recommended for family"
# matching_docs = db.similarity_search(query)
# # print("I am here")
# # print(matching_docs)
# answer =  chain.run(input_documents=matching_docs, question=query)
# print(answer)


#######################################################
#WEBAPP
st.write(" # Get the best hotel")

prompt = st.text_input("Your Preference")

if prompt:
    matching_docs = db.similarity_search(prompt)
    answer =  chain.run(input_documents=matching_docs, question=prompt)
    print("the respons eis " + answer)
    st.write(answer)






#python -m streamlit run chroma_with_llm.py











# client = chromadb.Client()
# collection = client.create_collection(name="hotel_review")


# collection.add(
#     documents=[ARTICLE_1, ARTICLE_2],
#     metadatas=[{"tag": "1"}, {"tag": "2"}],
#     ids=["HOtel_Myntra", "Hotel_Amazon"]
# )

# results = collection.query(
#     query_texts=[query],
#     n_results=1
# )

# answer =  chain.run(input_documents=results, question=query)
# print(answer)














# from langchain.agents.agent_toolkits import(
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     VectorStoreInfo
# )



# os.environ['OPENAI_API_KEY'] = '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'

# persist_directory = 'db'

# embedding = OpenAIEmbeddings()


# client = chromadb.Client()

# ARTICLE_1 ="The hotel has a great location, the staff is friendly and helpful, the rooms are clean and comfortable, and the breakfast is delicious. However, the rooms are small and some of the facilities are outdated."
# ARTICLE_2="The hotel is not good for family"

# embedding = OpenAIEmbeddings()

# string_list = [ARTICLE_2,ARTICLE_1]
# store = Chroma.from_texts(
#     client,
#     texts=string_list,
#     embedding=embedding,
#     metadatas=[{"tag": "1"}, {"tag": "2"}],
#     ids=["id1", "id2"]
#     )

# vectorStore_Info = VectorStoreInfo(
#     name="annual_report",
#     description="annual_report_from_comp",
#     vectorstore=store
# )


# toolkit = VectorStoreToolkit(vectorstore_info=vectorStore_Info)

# llm = OpenAI(temperature=0.3)

# agent_exec = create_vectorstore_agent(
#     llm =llm,
#     toolkit=toolkit,
#     verbose=True
# )



# print(agent_exec.run("get me hotel which is good for famiily"))
