# 🏛️ Contextual RAG for Abu Dhabi Procurement Standards

> Enterprise-grade Retrieval-Augmented Generation system with conversational memory, Cohere re-ranking, and Phoenix monitoring for Abu Dhabi government procurement standards, HR policies, and security guidelines.

# Demo_Video of the APP

**Please watch Contextual_RAG_Chatbot\Demo_Video of the APP**

## 🚀 Key Features

- **Contextual RAG**: Maintains conversation history and context across multiple document types
- **AI-Enhanced Ranking**: Cohere's rerank-english-v3.0 model achieving 99.8% relevance scores
- **Multi-Interface**: Streamlit WebUI, Phoenix monitoring, and interactive chat
- **Enterprise Monitoring**: Real-time tracing with Phoenix dashboard analytics
- **Scientific Evaluation**: RAGAs metrics for performance validation

---

## 📁 Project Structure

```
Context_RAG/
├── Context_RAG/              # Core RAG system modules
├── webui_interface/           # Streamlit web interface
├── phoenix_interface/         # Phoenix monitoring & chat
├── evaluation/               # RAGAs evaluation system
├── data/                     # Document storage
└── .github/                  # CI/CD workflows
```

---

## Command to Run APP

**streamlit run webui_interface\custom_webui.py**
or

**python -m streamlit run webui_interface\custom_webui.py**

## 🧠 Core Modules

### **Context_RAG/** - Core RAG Pipeline
| Module | Description | Command |
|--------|-------------|---------|
| `data_converter.py` | **PDF/DOCX Document Processing**: Extracts and chunks documents using PyMuPDF and python-docx for optimal vector storage. Handles multiple file formats with metadata preservation. | `python Context_RAG/data_converter.py` |
| `data_ingestion.py` | **Vector Database Ingestion**: Embeds document chunks using HuggingFace BGE models and stores in AstraDB with batch processing and error handling. | `python Context_RAG/data_ingestion.py` |
| `retrieval_generation.py` | **Enhanced RAG Pipeline**: Implements contextual retrieval with conversation memory, Cohere re-ranking, and DeepSeek LLM for accurate responses. | `python Context_RAG/retrieval_generation.py` |

### **webui_interface/** - Web Interface
| Module | Description | Command |
|--------|-------------|---------|
| `custom_webui.py` | **Streamlit ChatGPT-style Interface**: Professional web UI with Phoenix integration, real-time metrics, source citations, and conversation history. | `streamlit run webui_interface/custom_webui.py` |

### **phoenix_interface/** - Monitoring & Chat
| Module | Description | Command |
|--------|-------------|---------|
| `phoenix_app.py` | **Interactive Phoenix Chat**: Terminal-based chat interface with automatic OpenInference tracing, session management, and real-time monitoring. | `python phoenix_interface/phoenix_app.py` |
| `prompt_playground.py` | **Advanced Prompt Testing**: Compares 5 different prompt strategies with performance analytics, readability scoring, and automatic Phoenix tracing. | `python phoenix_interface/prompt_playground.py` |

### **evaluation/** - Performance Analysis
| Module | Description | Command |
|--------|-------------|---------|
| `ragas_evaluation.py` | **Scientific RAG Evaluation**: Uses RAGAs metrics (faithfulness, relevancy, precision) with Groq LLM and HuggingFace embeddings for comprehensive performance analysis. | `python evaluation/ragas_evaluation.py` |

---

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd Context_RAG

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. **Data Preparation**
```bash
# Add PDF/DOCX files to data/ folder
# Convert and ingest documents
python Context_RAG/data_converter.py
python Context_RAG/data_ingestion.py
```

### 3. **Launch Interfaces**

#### **Streamlit Web Interface** (Recommended)
```bash
streamlit run webui_interface/custom_webui.py
# Access: http://localhost:8501
# Phoenix Dashboard: http://localhost:6006
```

#### **Phoenix Interactive Chat**
```bash
python phoenix_interface/phoenix_app.py
# Terminal-based chat with monitoring
```

#### **Prompt Testing Playground**
```bash
python phoenix_interface/prompt_playground.py
# Test and compare different prompt strategies
```

### 4. **Evaluation & Testing**
```bash
# Run scientific evaluation
python evaluation/ragas_evaluation.py

# Test core RAG system
python Context_RAG/retrieval_generation.py
```

---

## 🔧 Configuration

### **Required API Keys** (.env)
```bash
GROQ_API_KEY="your_groq_api_key"
ASTRA_DB_API_ENDPOINT="your_astra_endpoint"
ASTRA_DB_APPLICATION_TOKEN="your_astra_token"
ASTRA_DB_KEYSPACE="your_keyspace"
HF_TOKEN="your_huggingface_token"
COHERE_API_KEY="your_cohere_api_key"
```

### **Model Configuration**
- **LLM**: llama-3.1-8b-instant (via Groq)
- **Embeddings**: BAAI/bge-base-en-v1.5 (HuggingFace)
- **Re-ranking**: rerank-english-v3.0 (Cohere)
- **Vector DB**: AstraDB (DataStax Cassandra)

---

## 📊 Performance Metrics

### **Production Results**
- **Relevance Scores**: 99.8% average with Cohere re-ranking
- **Response Time**: 5-8 seconds average
- **Document Coverage**: 2,011 contextualized chunks
- **Evaluation Metrics**: RAGAs validated performance

### **System Capabilities**
- **Context Awareness**: Maintains conversation history
- **Multi-Document**: Queries across procurement, HR, security docs
- **Real-time Monitoring**: Phoenix dashboard analytics
- **Source Attribution**: Automatic document citations

---

## 🐳 Docker Deployment

```bash
# Build container
docker build -t contextual-rag .

# Run with web interface
docker run -p 8501:8501 -p 6006:6006 contextual-rag

# Access interfaces
# Streamlit: http://localhost:8501
# Phoenix: http://localhost:6006
```

---

## ☁️ AWS Cloud Deployment

Complete AWS deployment guide available in `aws_ec2_deployment_steps.md`

### **Key Components**
- **EC2**: Application hosting
- **ECR**: Container registry
- **S3**: Artifact storage
- **GitHub Actions**: CI/CD pipeline

```bash
# Follow deployment guide
cat aws_ec2_deployment_steps.md
```

---

## 🧪 Testing & Evaluation

### **RAGAs Evaluation Metrics**
- **Faithfulness**: Answer grounding in context
- **Answer Relevancy**: Response quality assessment
- **Context Precision**: Retrieval accuracy
- **Context Recall**: Information completeness

### **Phoenix Monitoring**
- **Real-time Tracing**: All queries automatically traced
- **Performance Analytics**: Latency and token usage
- **Cost Tracking**: Token consumption monitoring

---

## 🏗️ Architecture

```
User Query → Context-Aware Reformulation → Vector Search → 
Cohere Re-ranking → Context Integration → LLM Generation → 
Contextual Response + Source Citations
```

**Enhanced Features:**
- Session-based conversation memory
- Multi-document context fusion
- AI-powered relevance ranking
- Real-time performance monitoring

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📞 Support

For issues and questions:
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check module docstrings and comments
- **Phoenix Dashboard**: Monitor system performance at `localhost:6006`

---

**🎯 Production-Ready Enterprise RAG System with Advanced Monitoring & Evaluation**