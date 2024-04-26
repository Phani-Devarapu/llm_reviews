from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os
import json
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.callbacks import get_openai_callback
import requests
import streamlit as st


## CONFIG
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]


APEC ="""openapi: 3.0.1
info:
  title: OpenAPI definition
  version: v0
servers:
- url: http://localhost:8080
  description: Generated server url
paths:
  /weather:
    get:
      tags:
      - home-controller
      summary: Get the temp of a city by providing cityname
      operationId: getdata
      parameters:
      - name: cityName
        in: query
        required: true
        schema:
          type: string
      responses:
        "200":
          description: OK
          content:
            '*/*':
              schema:
                type: string
  /hotels:
    get:
      tags:
      - home-controller
      summary: Get the city details provided the cityName and countryName
      operationId: getHotels
      parameters:
      - name: cityName
        in: query
        required: true
        schema:
          type: string
      - name: country
        in: query
        required: true
        schema:
          type: string
      - name: checkInDate
        in: query
        required: true
        schema:
          type: string
      - name: checkOutDate
        in: query
        required: true
        schema:
          type: string
      - name: noOfAdults
        in: query
        required: true
        schema:
          type: integer
          format: int32
      responses:
        "200":
          description: OK
          content:
            '*/*':
              schema:
                type: string
  /city:
    get:
      tags:
      - home-controller
      summary: Get the city details provided the cityName and countryName
      operationId: getcityDetails
      parameters:
      - name: cityName
        in: query
        required: true
        schema:
          type: string
      - name: country
        in: query
        required: true
        schema:
          type: string
      responses:
        "200":
          description: OK
          content:
            '*/*':
              schema:
                type: string
components: {}
"""

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_text(APEC)
vector = FAISS.from_texts([APEC], embeddings)

option = st.selectbox(
            'Select your GPT model',
            ('gpt-3.5-turbo-1106', 'gpt-3.5-turbo', 'gpt-4-turbo-preview'))

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=option,temperature=0.5)

final_prompt = PromptTemplate.from_template(  """API URL: {api_ourl}

Here is the response from the API:

{api_response}

Summarize this response to answer the original question.
                                  
Summary:""")

answer_chain = LLMChain(llm=llm, prompt=final_prompt)
retriever = vector.as_retriever()

URL =""
main_prompt ="";

## web app
prompt = ChatPromptTemplate.from_template("""Answer the following question based on the Open API spec provided:
<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
col1, col2 = st.columns([3, 1])

with get_openai_callback() as cb:
  with col1:
        st.header("Get your URL from SPEC file")
        input = st.text_input("Your Preference")
        if input:
                 response = retrieval_chain.invoke({"input": input + 'GIVE ME ONLY URL',
                                      "context": "Answer based on OPEN API 3 SPEC"})
                 URL = response["answer"].strip()
                 st.write(URL)
                 print(URL)
                #  r = requests.get(URL, headers={"Content-Type":"json"})
                #  content_str = r.content.decode("utf-8")
                #  api_response = json.loads(content_str)
                #  result =answer_chain.run(api_ourl=URL,api_response =api_response)
                #  print(result)
                #  st.write(result)

  with col2:
        st.header("Token Usage")
        st.write(f"Total Tokens: {cb.total_tokens}")
        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
        st.write(f"Completion Tokens: {cb.completion_tokens}")
        st.write(f"Total Cost (USD): ${cb.total_cost}")
   

