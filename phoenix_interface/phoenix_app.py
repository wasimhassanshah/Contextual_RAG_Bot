"""
Simple Phoenix Interface for RAG System
Working Phoenix setup with OpenInference tracing
"""

import os
import sys
from pathlib import Path
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class SimplePhoenixRAG:
    """Simple Phoenix RAG interface that works"""
    
    def __init__(self):
        self.rag_chain = None
        self.session = None
        self.query_count = 0
        
    def setup_phoenix(self, port=6006):
        """Setup Phoenix with OpenInference tracing"""
        
        print("🔥 Setting up Phoenix Interface...")
        print("=" * 50)
        
        # Set Phoenix port via environment variable (new method)
        os.environ["PHOENIX_PORT"] = str(port)
        
        # Launch Phoenix
        self.session = px.launch_app()
        
        # Setup OpenInference instrumentation
        tracer_provider = register(
            project_name="Contextual_RAG_Procurement",
            endpoint=f"http://localhost:{port}/v1/traces"
        )
        
        # Enable LangChain instrumentation
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        
        print(f"✅ Phoenix launched successfully!")
        print(f"🌐 Phoenix Dashboard: http://localhost:{port}")
        print(f"📊 Tracing enabled for all LangChain operations")
        
        return self.session
    
    def load_rag_system(self):
        """Load RAG system with automatic tracing"""
        
        print("🤖 Loading RAG system...")
        
        try:
            from Context_RAG.retrieval_generation import setup_rag_system_with_reranking
            
            # RAG chain will be automatically traced by OpenInference
            self.rag_chain = setup_rag_system_with_reranking()
            
            print("✅ RAG system loaded with automatic tracing")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load RAG system: {e}")
            return False
    
    def query_with_phoenix(self, user_input: str, session_id: str = None):
        """Execute query with Phoenix tracing (automatic via OpenInference)"""
        
        if not session_id:
            session_id = f"phoenix_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.query_count += 1
        
        print(f"🔍 Query {self.query_count}: {user_input[:50]}...")
        
        try:
            # Execute RAG query - automatically traced by OpenInference
            response = self.rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            answer = response["answer"]
            context_docs = response.get("context", [])
            sources = list(set([doc.metadata.get('source_file', 'Unknown') for doc in context_docs]))
            
            print(f"✅ Query {self.query_count} completed")
            
            return {
                "answer": answer,
                "sources": sources,
                "context_count": len(context_docs),
                "query_id": self.query_count
            }
            
        except Exception as e:
            print(f"❌ Query {self.query_count} failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "context_count": 0,
                "query_id": self.query_count
            }
    
    def interactive_chat(self):
        """Interactive chat with automatic Phoenix tracing"""
        
        print("\n💬 Interactive Phoenix Chat")
        print("=" * 50)
        print("🌐 Monitor queries in real-time: http://localhost:6006")
        print("📊 All queries are automatically traced!")
        print("💡 Type 'quit' to exit, 'stats' for session stats")
        print("-" * 50)
        
        session_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        while True:
            try:
                user_input = input("\n🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"\n🤖 Assistant: Thanks for using Phoenix RAG!")
                    print(f"📊 Session stats: {self.query_count} queries processed")
                    print(f"🌐 View detailed traces: http://localhost:6006")
                    break
                
                if user_input.lower() == 'stats':
                    print(f"\n📊 Session Statistics:")
                    print(f"   Total Queries: {self.query_count}")
                    print(f"   Session ID: {session_id}")
                    print(f"   Phoenix Dashboard: http://localhost:6006")
                    continue
                
                if not user_input:
                    continue
                
                # Process query with automatic Phoenix tracing
                print("🔍 Processing with Phoenix tracing...")
                result = self.query_with_phoenix(user_input, session_id)
                
                print(f"\n🤖 Assistant: {result['answer']}")
                if result['sources']:
                    print(f"📚 Sources: {', '.join(result['sources'])}")
                
                print(f"🔍 View trace #{result['query_id']}: http://localhost:6006")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def launch(self):
        """Launch complete Phoenix RAG system"""
        
        print("🚀 Launching Simple Phoenix RAG Interface")
        print("=" * 60)
        
        try:
            # Setup Phoenix
            self.setup_phoenix()
            
            # Load RAG system
            if not self.load_rag_system():
                return False
            
            print("\n🎉 Phoenix RAG System Ready!")
            print("=" * 50)
            print("🌐 Phoenix Dashboard: http://localhost:6006")
            print("📊 All queries will be automatically traced")
            print("💬 Starting interactive chat...")
            
            # Start interactive chat directly
            self.interactive_chat()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to launch Phoenix RAG: {e}")
            return False

def main():
    """Main function"""
    
    phoenix_rag = SimplePhoenixRAG()
    
    try:
        success = phoenix_rag.launch()
        
        if not success:
            print("❌ Phoenix RAG failed to launch")
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Phoenix RAG...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()