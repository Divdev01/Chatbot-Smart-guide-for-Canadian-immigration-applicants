import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os
import openai
# from src.data_ingestion import JSONLoader
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers, OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# llm = CTransformers(model='D:\Divya\Job\Volunteering\Omdena_Projects\Toronto_Chapter\llama-2-7b-chat.ggmlv3.q8_0.bin')
# llm = CTransformers(
#         model = 'D:\Divya\Job\Volunteering\Omdena_Projects\Toronto_Chapter\llama-2-7b-chat.ggmlv3.q8_0.bin',
#         model_type="llama",
#         max_new_tokens = 2000,
#         temperature = 0.1
#     )

_ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key = os.environ['OPENAI_API_KEY']
pinecone_api_key = os.environ['PINECONE_API_KEY']

# Initialize pinecone vector database
pinecone.init(api_key=pinecone_api_key, 
              environment="gcp-starter")


hugging_face_api_key = os.environ['HUGGING_FACE_API_KEY']
model_id = 'tiiuae/falcon-7b-instruct'

llm = OpenAI(temperature=0, openai_api_key=openai.api_key)


# Open source model falcon 7billion parameter instruct model from hugging face inference API
falcon_llm = HuggingFaceHub(huggingfacehub_api_token=hugging_face_api_key,
                            repo_id=model_id,
                            model_kwargs={"temperature":0.1,"max_new_tokens":2000})


# file_path='src/dataset_final_with_questions.json'
# loader = JSONLoader(file_path=file_path)
# data = loader.load()

# # Split the documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 700,
#     chunk_overlap = 150
# )
# splits = text_splitter.split_documents(data)

# embeddings using sentence transformer
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
index_name = "chatbot"

# # load embeddings from existing index in vector database
vector_store = Pinecone.from_existing_index(index_name=index_name, embedding = embeddings)
def create_prompt_template():
    template = """
            You are a virtual assistant, who can help with questions on Canadian immigration\
            Use the following pieces of context to answer the question at the end.\
            if the question is not related to canadian immigration, say that you are a chatbot \
                designed to answer question on canadian immigration\
            If you don't know the answer  just say that you don't know, don't try to make up an answer.\
            Keep the answer as concise as possible. \
            Add bullet points in the reply while explaining a process or steps.\             
            
            {context}
            Question: {question}
            Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    return QA_CHAIN_PROMPT

def retrieval_qa_chain(llm, vector_store):
    PROMPT = create_prompt_template()
    
    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vector_store.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
    return qa_chain

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
   
    llm_chain = retrieval_qa_chain(llm, vector_store)
      
    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    
    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True, 
                                                                                     answer_prefix_tokens=["FINAL", "ANSWER"])])
    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["result"]).send()