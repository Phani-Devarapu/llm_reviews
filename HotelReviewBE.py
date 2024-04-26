
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.callbacks import get_openai_callback
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

## CONFIG

embeddings = OpenAIEmbeddings()
persist_directory = ".\\chroma_db"

directory = './data/final_reviews'
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]


##--------------USEFUL CODE STARTS -----------

# vertexai.init(project="stockwatcher-389315", location="us-central1")

# parameters = {
#         "temperature": 0.6,
#         "max_output_tokens": 2048,
#         "top_p": 0.6,
#         "top_k": 15
#     }

# df = pd.read_csv('E:\\Downloads\\archive\\output_opencsv_barcelona.csv')
# def getSummaryFromVertex(reviewsAppended):
#     model = TextGenerationModel.from_pretrained("text-bison@002")
#     response = model.predict(
#         """Please provide consolidated sumamry in a paragraph by analysing the following hotel reviews from different users who provided review after staying in that hotel: """ + reviewsAppended,
#         **parameters
#     )
#     print(f"Response from Model: {response.text}")
#     return response.text;

# hotel_name_arr=[]
# hotel_consolidated=[]
# hotel_address =[]

# # #Loop through each row in the DataFrame
# for index, row in df.iterrows():
#     hotel_name = row[0];
#     hotel_addr = row[2];
#     print(hotel_name)
#     result_eachHotel = getSummaryFromVertex(row[1]);
#     hotel_name_arr.append(hotel_name)
#     hotel_consolidated.append(result_eachHotel)
#     hotel_address.append(hotel_addr)


# base_path ="E:\\Downloads\\archive\\";

# def writetoTextFile():
#    for i in range(len(hotel_name_arr)):
     
#      file_path = base_path+ hotel_name_arr[i].replace(' ', '-')+".txt"
#      with open(file_path, 'w') as file:
#         file.write(f"Hotel name: {hotel_name_arr[i]} ")
#         file.write("\n")
#         file.write("\n")
#         file.write(f"Hotel name: {hotel_consolidated[i]} ")
#         file.write("\n")
#         file.write("\n")
#         file.write(f"Hotel Address: {hotel_address[i]} ")
#         file.write("\n")  


# writetoTextFile()

# print("writing done")


##--------------USEFUL CODE ENDS -----------


# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_text(documents)

##--------------USEFUL CODE STARTS -----------

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

document_name_map= {}

documents = load_docs(directory)
for i in range(len(documents)):
   doc = documents[i]
   source = doc.metadata['source']
   document_name_map[source] = doc


hotel_ids = list(document_name_map.keys()) 
final_docs = list(document_name_map.values())

##--------------USEFUL CODE END -----------

# def split_docs(documents,chunk_size=1000,chunk_overlap=400):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   docs = text_splitter.split_documents(documents)
#   return docs

# docs = split_docs(documents)
# print(len(docs))


# client = chromadb.Client()

# collection = client.create_collection(name="hotel_review")


# collection.add(
#     documents=[ARTICLE_1, ARTICLE_2],
#     metadatas=[{"tag": "1"}, {"tag": "2"}],
#     ids=["HOtel_Myntra", "Hotel_Amazon"]
# )


##--------------USEFUL CODE STARTS -----------

db = Chroma.from_documents(documents=final_docs, embedding=embeddings,ids = hotel_ids,persist_directory=persist_directory)
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name,temperature=0.6)
# query = "How many hotels are there"
# matching_docs = db.similarity_search(query,k=100)
chain = load_qa_chain(llm,verbose=True)


##--------------USEFUL CODE END -----------

# # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# # retriever = ...
# # combine_docs_chain = create_stuff_documents_chain(
# #     llm, retrieval_qa_chat_prompt
# # )
# # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)



## WEB APP CODE
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name,temperature=0.6)
chain = load_qa_chain(llm,verbose=True)

st.header("Welcome to Hotel Search based on reviews")
option = st.selectbox(
            'Select your GPT model',
            ('gpt-3.5-turbo-1106', 'gpt-3.5-turbo', 'gpt-4-turbo-preview'))

llm = ChatOpenAI(model_name=option,temperature=0.6)
chain = load_qa_chain(llm,verbose=True)
col1, col2 = st.columns([3, 1])
with get_openai_callback() as cb:

    with col1:
        st.header("Serch here")
        prompt = st.text_input("Your Preference")
        if prompt:
                matching_docs = db.similarity_search(prompt)
                print(matching_docs)
                answer =  chain.run(input_documents=matching_docs, question=prompt)
                print("the respons is " + answer)
                st.write(answer)

    with col2:
        st.header("Usage Stats")
        st.write(f"Total Tokens: {cb.total_tokens}")
        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
        st.write(f"Completion Tokens: {cb.completion_tokens}")
        st.write(f"Total Cost (USD): ${cb.total_cost}")



## python -m streamlit run vertex.py

