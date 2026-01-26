#load file json data
import json
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
json_path = "./Vinmec_output.json"

def load_data_doc(path: str) -> List[Document]:
    """
    File có dạng:
    [
      {
        "article": { "title": ..., "category": ..., "url": ..., "tags": (..) },
        "chunks": [
          {"chunk_id": 0, "text": "...", "subcategories": [...]},
          ...
        ]
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []
    for item in data:
        article = item.get("article", {})
        chunks = item.get("chunks", [])

        base_meta = {
            "title": article.get("title",""),
            "category": article.get("category", ""),
            "url": article.get("url",""),
            "tags": article.get("tags",""),
        }    

        for ch in chunks:
            text = ch.get("text","")
            if not text.strip():
                continue
            
            meta = base_meta.copy()
            meta["chunk_id"] = ch.get("chunk_id")
            meta["subcategories"] = ch.get("subcategories,()")
            docs.append(Document(page_content=text, metadata=meta))

    print(f"Load {len(docs)} Vinmec chunks Documents")
    return docs

docs = load_data_doc(json_path)

for doc in docs:
    if "tags" in doc.metadata and isinstance(doc.metadata["tags"], list):
        doc.metadata["tags"] = ", ".join(doc.metadata["tags"])


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100,
    add_start_index = True
)

all_splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    dimension=768
)

persist_dir = "/mnt/d/agent_chatbot/chroma_db"
collection_name = "mental_health"

if os.path.exists(persist_dir):
    print("Load chroma db")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
else:
    print("Tao chroma db")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    
    vector_store.add_documents(all_splits)
    print(f"Successfully added {len(all_splits)} documents to Chroma.")
