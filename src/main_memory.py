import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os
import openai
from src.data_ingestion import JSONLoader
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
pinecone_api_key = os.environ['PINECONE_API_KEY']

# os.environ['API_KEY'] = os.environ['HUGGING_FACE_API_KEY']
hugging_face_api_key = os.environ['HUGGING_FACE_API_KEY']
model_id = 'tiiuae/falcon-7b-instruct'

# Open source model falcon 7billion parameter instruct model from hugging face inference API
falcon_llm = HuggingFaceHub(huggingfacehub_api_token=hugging_face_api_key,
                            repo_id=model_id,
                            model_kwargs={"temperature":0.1,"max_new_tokens":1000})
llm = OpenAI(temperature=0, openai_api_key=openai.api_key)

# file_path='src/dataset_final_with_questions.json'
# loader = JSONLoader(file_path=file_path)
# data = loader.load()

# # Split the documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 700,
#     chunk_overlap = 150
# )
# splits = text_splitter.split_documents(data)


# Initialize pinecone vector database
pinecone.init(api_key=pinecone_api_key, 
              environment="gcp-starter")


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
index_name = "chatbot"

# load embeddings from existing index in vector database
vector_store = Pinecone.from_existing_index(index_name=index_name, embedding = embeddings)

def create_condense_ques_prompt():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
                You can assume the question is about canadian immigration.

                Chat History:
                {chat_history} 
 
                Follow Up Input: {question}
                Standalone question:"""
                    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONDENSE_QUESTION_PROMPT

def create_qa_chain_prompt():
    template = """
        You are a virtual assistant, who can help with questions on Canadian immigration\
        You are given the following extracted parts of a long document and a question.\
        if the question is not related to canadian immigration, say that you are a chatbot \
            designed to answer question on canadian immigration\
        If you don't know the answer  just say that you don't know, don't try to make up an answer.\
        Keep the answer as concise as possible. \
        Add bullet points in the reply while explaining a process or steps.\             
         
        
        Question: {question}
        {context}
        Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    return QA_CHAIN_PROMPT

def conv_retrieval_qa_chain(llm, vector_store):
    memory = ConversationBufferWindowMemory( k=2, 
                                        memory_key='chat_history',
                                        return_messages=True,
                                        verbose = True)
    CONDENSE_QUESTION_PROMPT = create_condense_ques_prompt()
    QA_CHAIN_PROMPT = create_qa_chain_prompt()    
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        memory=memory
    )
    return qa


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    
   
    qa = conv_retrieval_qa_chain(llm, vector_store)
    
   
      
    # Store the chain in the user session
    cl.user_session.set("llm_chain", qa)


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
    await cl.Message(content=res["answer"]).send()