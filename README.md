# Chatbot-Smart-guide-for-Canadian-immigration-applicants
This project is done as a part Omdena local chapter challenge.\

https://omdena.com/chapter-challenges/smartguide-empowering-canadas-immigration-applicants-with-accurate-ai-chatbot-assistance/


# Chatbot using LLMS (OpenAI, Falcon)
- This method uses Retrieval Augmented Generation (RAG) by which retrieved information from the dataset is used as the context for llm and result is generated.
- Pinecone is used to store vector embeddings(pinecone allows to create one free index). Before running the script. Pinecone index should be created with a name 'chatbot' with dimension 384 (since we are using sentence transformer embeddings here) and metrics as 'cosine'.
Pinecone index can be create by signing up here for free https://www.pinecone.io/

## Required libraries
chainlit, langchain, pinecone

## main.py -  chatbot doesn't remember previous conversation

To run main.py, use the following command

chainlit run main.py -w

## main_memory.py -  the chatbot remember previous 2 conversation (limited to 2 because of the token limit. But can add more)

To run main_memory.py, use the following command

chainlit run main_memory.py -w


