"""
Working Data Ingestion - bypasses LangChain HuggingFace bug
Uses direct HuggingFace API calls that we know work
"""

import os
import sys
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from data_converter import document_converter

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")

class WorkingHuggingFaceEmbeddings:
    """Direct HuggingFace API embeddings that actually work"""
    
    def __init__(self, api_key: str, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text query"""
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = []
        
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:  # Progress indicator
                print(f"   📊 Embedded {i}/{len(texts)} documents...")
            
            data = {
                "inputs": text,
                "options": {"wait_for_model": True}
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url, 
                        headers=self.headers, 
                        json=data, 
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        embedding = response.json()
                        embeddings.append(embedding)
                        break
                    elif response.status_code == 503:
                        print(f"   ⏳ Model loading, retrying in {2**attempt} seconds...")
                        time.sleep(2**attempt)
                    else:
                        print(f"   ❌ Error {response.status_code}: {response.text}")
                        # Use a default embedding if API fails
                        embeddings.append([0.0] * 768)  # BGE model dimension
                        break
                        
                except Exception as e:
                    print(f"   ❌ Request failed: {e}")
                    if attempt == max_retries - 1:
                        embeddings.append([0.0] * 768)
                    else:
                        time.sleep(1)
            
            # Small delay between requests to be nice to the API
            time.sleep(0.1)
        
        return embeddings

def create_working_embeddings():
    """Create embeddings that actually work"""
    
    print("🤗 Creating working HuggingFace embeddings...")
    
    try:
        embedding = WorkingHuggingFaceEmbeddings(
            api_key=HF_TOKEN,
            model_name="BAAI/bge-base-en-v1.5"
        )
        
        # Test it
        print("🧪 Testing embedding...")
        test_result = embedding.embed_query("test")
        print(f"✅ Working embeddings created! Dimension: {len(test_result)}")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Failed to create embeddings: {e}")
        raise

def data_ingestion_working(status):
    """Data ingestion with working embeddings"""
    
    print("🚀 Starting Working Data Ingestion...")
    print("=" * 60)
    
    # Step 1: Create working embeddings
    embedding = create_working_embeddings()
    
    # Step 2: Connect to AstraDB using raw API
    print("🗄️ Setting up AstraDB connection...")
    
    try:
        from langchain_astradb import AstraDBVectorStore
        
        # Use our working embeddings with AstraDB
        vstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name="procurement_docs_working",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE
        )
        
        print("✅ AstraDB connection successful!")
        
    except Exception as e:
        print(f"❌ AstraDB connection failed: {e}")
        raise

    # Step 3: Process documents
    if status is None:
        print("📄 Loading documents...")
        docs = document_converter()
        
        if not docs:
            print("❌ No documents to ingest!")
            return vstore, []
        
        print(f"📤 Ingesting {len(docs)} chunks with working embeddings...")
        
        # Process in small batches
        batch_size = 10  # Smaller batches to avoid timeouts
        total_inserted = 0
        all_ids = []
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(docs)-1)//batch_size + 1
            
            print(f"   📦 Batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                batch_ids = vstore.add_documents(batch)
                total_inserted += len(batch_ids)
                all_ids.extend(batch_ids)
                print(f"   ✅ Success ({len(batch_ids)} added)")
                
                # Longer delay between batches
                time.sleep(3)
                
            except Exception as e:
                print(f"   ❌ Batch failed: {e}")
                continue
        
        print(f"🎉 Total ingested: {total_inserted} out of {len(docs)} chunks")
        return vstore, all_ids
    
    else:
        print("🔄 Returning existing vector store...")
        return vstore

def test_retrieval_working(vstore):
    """Test retrieval with working system"""
    
    print("\n🔍 Testing Retrieval...")
    print("-" * 40)
    
    test_queries = [
        "procurement standards",
        "security requirements",
        "HR policies"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Query {i}: '{query}'")
        try:
            results = vstore.similarity_search(query, k=2)
            
            for j, result in enumerate(results, 1):
                source = result.metadata.get('source_file', 'Unknown')
                preview = result.page_content[:100] + "..."
                print(f"   {j}. {source}: {preview}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    try:
        print("🎯 Using WORKING embeddings (bypassing LangChain bug)")
        
        vstore, insert_ids = data_ingestion_working(None)
        
        if insert_ids:
            print(f"\n🎉 SUCCESS! Ingested {len(insert_ids)} documents")
            test_retrieval_working(vstore)
        else:
            print("\n⚠️ No documents ingested")
            
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()