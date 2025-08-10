"""
Custom ChatGPT-like WebUI for RAG System with Phoenix Integration
Professional web interface using Streamlit (Python 3.10 compatible)
"""

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime
import uuid
import os

# Phoenix integration imports
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "Context_RAG"))

# Page configuration
st.set_page_config(
    page_title="🏛️ Abu Dhabi Procurement Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3);
        text-align: right;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #28a745, #1e7e34);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    
    .header-gradient {
        background: linear-gradient(135deg, #007bff, #28a745);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-loading {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.4);
    }
    
    /* Phoenix status indicator */
    .phoenix-status {
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .phoenix-online {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .phoenix-offline {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_phoenix_tracing():
    """Setup Phoenix with OpenInference tracing for Streamlit"""
    try:
        # Set Phoenix port
        phoenix_port = 6006
        os.environ["PHOENIX_PORT"] = str(phoenix_port)
        
        # Check if Phoenix is already running, if not launch it
        if 'phoenix_session' not in st.session_state:
            session = px.launch_app()
            st.session_state.phoenix_session = session
        
        # Setup OpenInference instrumentation
        tracer_provider = register(
            project_name="Streamlit_RAG_Procurement",
            endpoint=f"http://localhost:{phoenix_port}/v1/traces"
        )
        
        # Enable LangChain instrumentation
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        
        return True, f"http://localhost:{phoenix_port}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with Phoenix tracing (cached)"""
    try:
        # Setup Phoenix tracing first
        phoenix_success, phoenix_info = setup_phoenix_tracing()
        
        # Load RAG system
        from retrieval_generation import setup_rag_system_with_reranking
        rag_chain = setup_rag_system_with_reranking()
        
        return rag_chain, "success", phoenix_success, phoenix_info
    except Exception as e:
        return None, f"Error: {str(e)}", False, "Phoenix setup failed"

def display_message(message, role):
    """Display a chat message with appropriate styling"""
    if role == "user":
        st.markdown(f'<div class="user-message">🤔 You: {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 Assistant: {message}</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="header-gradient">🏛️ Abu Dhabi Procurement Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### 🔥 Powered by Contextual RAG with Cohere Re-ranking + Phoenix Tracing")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        # Initialize RAG system with Phoenix
        if 'rag_chain' not in st.session_state:
            with st.spinner("🤖 Initializing RAG system with Phoenix tracing..."):
                rag_chain, status, phoenix_status, phoenix_info = initialize_rag_system()
                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    st.session_state.rag_status = "success"
                    st.session_state.phoenix_status = phoenix_status
                    st.session_state.phoenix_url = phoenix_info if phoenix_status else None
                    st.success("✅ RAG System Online")
                    if phoenix_status:
                        st.success("🔥 Phoenix Tracing Online")
                    else:
                        st.warning("⚠️ Phoenix Tracing Offline")
                else:
                    st.session_state.rag_status = "error"
                    st.session_state.phoenix_status = False
                    st.error(f"❌ RAG System Failed: {status}")
        else:
            if st.session_state.rag_status == "success":
                st.success("✅ RAG System Online")
            else:
                st.error("❌ RAG System Offline")
            
            # Phoenix status
            if st.session_state.get('phoenix_status', False):
                st.markdown('<div class="phoenix-status phoenix-online">🔥 Phoenix Tracing: Online</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="phoenix-status phoenix-offline">🔥 Phoenix Tracing: Offline</div>', unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown("## 📈 Performance Metrics")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.session_state.total_response_time = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", st.session_state.query_count)
        with col2:
            avg_time = st.session_state.total_response_time / max(st.session_state.query_count, 1)
            st.metric("Avg Response", f"{avg_time:.1f}s")
        
        # System info
        st.markdown("## 🔧 System Features")
        st.markdown("""
        **🚀 Capabilities:**
        - Contextual RAG with Memory
        - Cohere AI Re-ranking
        - Multi-document Retrieval
        - Real-time Source Citations
        - 🔥 Phoenix Query Tracing
        
        **📚 Document Library:**
        - Abu Dhabi Procurement Standards
        - HR Bylaws and Policies
        - Information Security Guidelines
        - Procurement Manuals
        - Q&A Examples
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.session_state.total_response_time = 0
            st.rerun()
        
        # Phoenix integration
        st.markdown("## 🔥 Phoenix Monitoring")
        
        if st.session_state.get('phoenix_status', False):
            phoenix_url = st.session_state.get('phoenix_url', 'http://localhost:6006')
            st.markdown(f"**🌐 Dashboard:** [{phoenix_url}]({phoenix_url})")
            st.success("✅ Streamlit queries are being traced!")
            
            if st.button("📊 Open Phoenix Dashboard"):
                st.markdown(f"""
                **Phoenix Dashboard is running at:** [{phoenix_url}]({phoenix_url})
                
                🔍 **What you'll see:**
                - All Streamlit chat queries
                - RAG retrieval traces
                - Response generation timing
                - Document retrieval details
                """)
                st.balloons()
        else:
            st.error("❌ Phoenix tracing not available")
            st.info("💡 Restart the app to retry Phoenix setup")
    
    # Main chat interface
    st.markdown("## 💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        # Welcome message
        if not st.session_state.chat_history:
            phoenix_status_text = "with Phoenix tracing enabled" if st.session_state.get('phoenix_status', False) else "without Phoenix tracing"
            st.markdown(f"""
            <div class="assistant-message">
                🤖 Assistant: Hello! I'm your Abu Dhabi Procurement Assistant {phoenix_status_text}. 
                I have access to comprehensive procurement standards, HR policies, and security guidelines. 
                Ask me anything about:
                <br><br>
                • Procurement standards and procedures<br>
                • HR policies and regulations<br>
                • Information security requirements<br>
                • Supplier management guidelines<br>
                • Compliance and regulatory matters
                <br><br>
                🔥 All your queries will be traced in Phoenix for monitoring and analysis!
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_message(message["content"], message["role"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Document Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
            
            # Show response time and trace info
            if message["role"] == "assistant":
                col1, col2 = st.columns(2)
                with col1:
                    if message.get("response_time"):
                        st.caption(f"⏱️ Response time: {message['response_time']:.1f}s")
                with col2:
                    if st.session_state.get('phoenix_status', False):
                        st.caption("🔥 Traced in Phoenix")
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask about procurement, HR policies, or security requirements...",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Send 🚀")
        
        if submit_button and user_input and st.session_state.rag_status == "success":
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Generate response with Phoenix tracing
            trace_info = "🔥 with Phoenix tracing" if st.session_state.get('phoenix_status', False) else "without tracing"
            with st.spinner(f"🔍 Searching documents and generating response {trace_info}..."):
                start_time = time.time()
                
                try:
                    # Query RAG system (automatically traced by Phoenix if enabled)
                    session_id = f"streamlit_{uuid.uuid4().hex[:8]}"
                    response = st.session_state.rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    answer = response["answer"]
                    context_docs = response.get("context", [])
                    sources = list(set([doc.metadata.get('source_file', 'Unknown') for doc in context_docs]))
                    
                    response_time = time.time() - start_time
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "response_time": response_time,
                        "timestamp": datetime.now(),
                        "traced": st.session_state.get('phoenix_status', False)
                    })
                    
                    # Update metrics
                    st.session_state.query_count += 1
                    st.session_state.total_response_time += response_time
                    
                    # Success message
                    success_msg = f"✅ Response generated in {response_time:.1f}s"
                    if st.session_state.get('phoenix_status', False):
                        success_msg += " and traced in Phoenix!"
                    st.success(success_msg)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"I apologize, but I encountered an error: {str(e)}",
                        "sources": [],
                        "timestamp": datetime.now()
                    })
            
            # Rerun to show new messages
            st.rerun()
        
        elif submit_button and user_input and st.session_state.rag_status != "success":
            st.error("❌ RAG system is not available. Please check the system status.")
    
    # Footer
    st.markdown("---")
    phoenix_footer = " | 🔥 Phoenix Traced" if st.session_state.get('phoenix_status', False) else ""
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        🏛️ Abu Dhabi Procurement Assistant | 
        🧠 Powered by Contextual RAG | 
        🔥 Enhanced with Cohere Re-ranking{phoenix_footer}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()