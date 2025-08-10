"""
Interactive RAG Pipeline with Cohere Re-ranking
Enhanced retrieval with post-processing re-ranking for better relevance
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
import cohere

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document

load_dotenv()

# Environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Use the powerful model for better responses
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# Initialize Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Session management
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class SimpleWorkingEmbeddings:
    """Minimal embeddings class that works with AstraDB"""
    
    def __init__(self, api_key: str, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_query(self, text: str):
        """Embed a single query for similarity search"""
        data = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=data, 
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return [0.0] * 768
                
        except Exception:
            return [0.0] * 768
    
    def embed_documents(self, texts):
        """Compatibility method"""
        return [self.embed_query(text) for text in texts]

def rerank_documents(query: str, documents: list, top_k: int = 5) -> list:
    """
    Re-rank documents using Cohere's re-ranking API
    
    Args:
        query: The user's search query
        documents: List of Document objects from vector search
        top_k: Number of top documents to return after re-ranking
    
    Returns:
        List of re-ranked Document objects
    """
    
    if not documents:
        return documents
    
    try:
        print(f"🔄 Re-ranking {len(documents)} documents with Cohere...")
        
        # Prepare documents for Cohere API
        docs_for_rerank = []
        for doc in documents:
            # Use page_content for re-ranking
            text = doc.page_content[:1000]  # Truncate to avoid API limits
            docs_for_rerank.append(text)
        
        # Call Cohere re-rank API
        rerank_response = cohere_client.rerank(
            model="rerank-english-v3.0",  # Cohere's latest re-ranking model
            query=query,
            documents=docs_for_rerank,
            max_chunks_per_doc=1,  # Use max_chunks_per_doc instead of top_k
            return_documents=True
        )
        
        # Reorder original documents based on Cohere's ranking
        reranked_docs = []
        for i, result in enumerate(rerank_response.results[:top_k]):  # Limit to top_k results
            original_index = result.index
            original_doc = documents[original_index]
            
            # Add relevance score to metadata
            if hasattr(original_doc, 'metadata'):
                original_doc.metadata['cohere_relevance_score'] = result.relevance_score
            else:
                original_doc.metadata = {'cohere_relevance_score': result.relevance_score}
            
            reranked_docs.append(original_doc)
        
        print(f"✅ Re-ranking complete! Top document relevance: {reranked_docs[0].metadata.get('cohere_relevance_score', 'N/A'):.3f}")
        return reranked_docs
        
    except Exception as e:
        print(f"⚠️ Re-ranking failed: {e}")
        print("📋 Falling back to original vector search results...")
        return documents[:top_k]  # Return original results if re-ranking fails

class RerankingRetriever:
    """Custom retriever that adds Cohere re-ranking to vector search"""
    
    def __init__(self, vector_store, search_kwargs=None):
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {"k": 10}
    
    def get_relevant_documents(self, query: str):
        """Retrieve and re-rank documents"""
        initial_docs = self.vector_store.similarity_search(query, **self.search_kwargs)
        reranked_docs = rerank_documents(query, initial_docs, top_k=5)
        return reranked_docs
    
    def invoke(self, input_dict):
        """LangChain compatibility method"""
        if isinstance(input_dict, dict) and 'input' in input_dict:
            query = input_dict['input']
        else:
            query = str(input_dict)
        return self.get_relevant_documents(query)
    
    def _get_relevant_documents(self, query: str):
        """Private method for LangChain compatibility"""
        return self.get_relevant_documents(query)
    
    def __call__(self, inputs):
        """Make the retriever callable"""
        return self.get_relevant_documents(inputs)

def setup_rag_system_with_reranking():
    """Setup the complete RAG system with Cohere re-ranking"""
    
    print("🚀 Initializing Enhanced Procurement Assistant...")
    print("📊 Connecting to document database...")
    
    try:
        # Create embeddings
        embedding = SimpleWorkingEmbeddings(
            api_key=HF_TOKEN,
            model_name="BAAI/bge-base-en-v1.5"
        )
        
        # Create vector store
        from langchain_astradb import AstraDBVectorStore
        
        vstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name="procurement_docs_working",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
            setup_mode="off"
        )
        
        print("✅ Connected to document database!")
        print("🔗 Setting up AI pipeline with Cohere re-ranking...")
        
        # Query reformulation prompt
        retriever_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Focus on procurement, HR policies, security requirements, or organizational standards. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Create a simpler retriever approach to avoid compatibility issues
        from langchain_core.runnables import RunnableLambda
        
        def enhanced_retrieve(input_data):
            """Enhanced retrieval with re-ranking"""
            if isinstance(input_data, dict):
                query = input_data.get('input', str(input_data))
            else:
                query = str(input_data)
            
            # Get initial results
            initial_docs = vstore.similarity_search(query, k=10)
            
            # Re-rank them
            reranked_docs = rerank_documents(query, initial_docs, top_k=5)
            
            return reranked_docs
        
        # Create the enhanced retriever as a runnable
        enhanced_retriever = RunnableLambda(enhanced_retrieve)
        
        # Use the enhanced retriever in the chain
        history_aware_retriever = create_history_aware_retriever(
            model, enhanced_retriever, contextualize_q_prompt
        )

        # Answer generation prompt
        PROCUREMENT_BOT_TEMPLATE = """
You are an expert procurement and organizational policy assistant with access to comprehensive documentation including:

📋 **Available Documents:**
- Abu Dhabi Procurement Standards
- HR Bylaws and Policies  
- Information Security Guidelines
- Procurement Manuals (Ariba Aligned & Business Process)
- Document Q&A Examples

🎯 **Your Role:**
- Provide accurate, detailed answers based strictly on the provided context
- Focus on procurement standards, HR policies, security requirements, and organizational procedures
- Cite specific document sources when providing information
- If information isn't in the context, clearly state this limitation
- Use professional, clear language appropriate for business/government contexts

📝 **Response Guidelines:**
- Structure answers clearly with headers/bullets when helpful
- Include relevant policy numbers, sections, or document references
- Highlight important requirements, deadlines, or compliance issues
- Provide actionable guidance when appropriate
- Note: The context has been enhanced with AI re-ranking for maximum relevance

🔍 **Context Documents (Re-ranked by relevance):**
{context}

❓ **User Question:** {input}

📋 **Your Expert Response:**
"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", PROCUREMENT_BOT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        print("✅ Enhanced AI assistant ready with re-ranking!")
        return conversational_rag_chain
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise

def main():
    """Main interactive chat function"""
    
    print("🏛️ Abu Dhabi Procurement & Policy Assistant (Enhanced)")
    print("=" * 65)
    print("📚 I have access to your complete document library:")
    print("   • Abu Dhabi Procurement Standards")
    print("   • HR Bylaws and Policies")
    print("   • Information Security Guidelines") 
    print("   • Procurement Manuals")
    print("   • Document Q&A Examples")
    print()
    print("🧠 Enhanced with Cohere AI re-ranking for better relevance!")
    print("💬 Ask me anything about procurement, HR policies, or security requirements!")
    print("📝 Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 65)
    print()
    
    try:
        # Setup the enhanced RAG system
        rag_chain = setup_rag_system_with_reranking()
        session_id = "main_session"
        
        print("🤖 Assistant: Hello! I'm your enhanced procurement assistant with AI-powered")
        print("    document re-ranking for more accurate responses. I'm ready to help with")
        print("    questions about Abu Dhabi's procurement standards and policies.")
        print("    What would you like to know?\n")
        
        while True:
            user_input = input("🤔 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n🤖 Assistant: Thank you for using the Enhanced Procurement Assistant!")
                print("    Have a great day! 👋")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            try:
                # Get response from enhanced RAG system
                print("\n🔍 Searching documents...")
                response = rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                
                print(f"\n🤖 Assistant: {response['answer']}\n")
                
            except Exception as e:
                print(f"\n❌ Sorry, I encountered an error: {e}")
                print("   Please try rephrasing your question.\n")
    
    except Exception as e:
        print(f"\n❌ Failed to start the assistant: {e}")
        print("   Please check your configuration and try again.")

if __name__ == "__main__":
    main()