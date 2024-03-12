import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

def get_docs_from_pdf(path):
    loader = PyPDFLoader(path, extract_images=True)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    splitDocs = splitter.split_documents(pages)
    print("Text spiltted.")
    return splitDocs

def create_embeddings(splittedDocs):
    embeddings = OpenAIEmbeddings()
    index_name="wilson-embeddings"
    PineconeVectorStore.from_documents(documents=splittedDocs, embedding=embeddings, index_name=index_name)
    print("Embeddings created")

pdf_directory = os.path.join(os.getcwd(), "pdfs")

# List all files in the directory
files = os.listdir(pdf_directory)

for file in files:
    file_path = os.path.join(pdf_directory, file)
    docs = get_docs_from_pdf(file_path)
    vectorStore = create_embeddings(docs)
