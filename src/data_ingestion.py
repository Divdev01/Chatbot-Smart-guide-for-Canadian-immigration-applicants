"""Loader that loads data from JSON."""
from pprint import pprint
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone

class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        
    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs=[]
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)

            # Iterate through 'pages'
        for c in data:
            # print(c)
            parenturl = c['source']
            pagetitle = c['title']
            # indexeddate = page['indexeddate']
            
            snippets = c['subheadings']
            metadata={"title":pagetitle, "source": parenturl}
            
            print(pagetitle)
            # # Process snippets for each page
            for snippet in snippets:
                content = snippet['content']
                subheading = snippet['subheading']
                # text = snippet['text']
                docs.append(Document(page_content=content, metadata=metadata))
    # break

        return docs
    

if __name__ == "__main__":

    file_path='src/dataset_final_with_questions.json'
    loader = JSONLoader(file_path=file_path)
    data = loader.load()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 150
    )
    splits = text_splitter.split_documents(data)

    # embeddings using sentence transformer
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index_name = "chatbot"

    # run below line only if we want to add embedding to the vector database(in this case pinecone)
    vector_store = Pinecone.from_texts([split.page_content for split in splits], embeddings, index_name=index_name)
    print("Data ingestion completed!!")


