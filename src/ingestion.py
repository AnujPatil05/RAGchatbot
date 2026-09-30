import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from src.config import PINECONE_API_KEY, PDF_URL

INDEX_NAME = "agentic-ai-384"

def download_pdf(url: str, output_path: str = "data/Ebook-Agentic-AI.pdf") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        print(f"Downloading eBook from {url}...")
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(res.content)
        print("Download complete.")
    return output_path

def ingest_to_pinecone():
    pdf_path = download_pdf(PDF_URL)
    
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = text_splitter.split_documents(raw_docs)
    print(f"Generated {len(docs)} chunks from {len(raw_docs)} pages.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {INDEX_NAME} (dimension=384)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(INDEX_NAME)
    print("Loading lightweight embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    batch_size = 50
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [
            {
                "page": doc.metadata.get("page", 0),
                "source": "Ebook-Agentic-AI.pdf",
                "text": doc.page_content
            }
            for doc in batch
        ]
        vectors = embeddings.embed_documents(texts)
        records = [
            (f"doc_{i + idx}", vec, meta)
            for idx, (vec, meta) in enumerate(zip(vectors, metadatas))
        ]
        index.upsert(vectors=records)
        print(f"Upserted chunks {i} to {min(i + batch_size, len(docs))}")

    print("Knowledge base indexing completed successfully.")

if __name__ == "__main__":
    ingest_to_pinecone()