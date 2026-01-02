#create the indexing in it

# to save the vectors
# import pickle
#for doing chunks .first we have to import api key through dotenv
import os 
from dotenv import load_dotenv
load_dotenv()

# now we simply extract the pdf file 
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('C:/Users/Ch Computers/Desktop/Digifloat/langchain practice/sciencerag.pdf')

# create a document variable to load the loader 
document =loader.load()
# print(document)
#document

#  now we basically used the recursivecharactertextsplitter to make its chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(
     chunk_size=500,
     chunk_overlap=100,  
     )

chunks=text_splitter.split_documents(document)
 # print (chunks)
# for chunk in chunks:
#     print (chunk)
#     print ("=="*50)

# len(chunks) # len=10

# now create the embedding of the created chunks 
from langchain_openai import AzureOpenAIEmbeddings

embeddings=AzureOpenAIEmbeddings(
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  deployment=os.getenv("AZURE_EMBEDDING_MODEL_DEPLOYMENT"),
)

# vector=[embedding.embed_query(chunk.page_content)for chunk in chunks]
# print(vector[:1])

# We used Faiss vector db 
from langchain_community.vectorstores import FAISS

faiss_db =FAISS.from_documents(chunks,embeddings)
faiss_db.save_local("faiss_vector_db")
# print("yes")


# with open("vector_store.pkl","wb") as f:
#     pickle.dump(vector_store,f)

# print ("v s succes")

# print(dir(retriever))

# ==================================================================================


