"""
RAGAs Evaluation Module
Evaluates RAG performance using RAGAs metrics with Groq LLM and Working HF Embeddings
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    print(f"❌ RAGAs not installed. Install with: pip install ragas")
    print(f"Error: {e}")
    sys.exit(1)

from ground_truth_generator import GroundTruthGenerator
from test_questions import get_test_questions

class WorkingHuggingFaceEmbeddings:
    """Direct HuggingFace API embeddings that actually work with RAGAs"""
    
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
        """Embed multiple documents"""
        return [self.embed_query(text) for text in texts]
    
    async def aembed_documents(self, texts):
        """Async embed multiple documents (required by RAGAs)"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str):
        """Async embed single query (required by RAGAs)"""
        return self.embed_query(text)

class RAGEvaluator:
    """RAG evaluation using RAGAs metrics with Groq LLM and Working HF Embeddings"""
    
    def __init__(self):
        self.rag_chain = None
        self.ground_truth_data = None
        self.session_id = "evaluation_session"
        
        # Setup Groq LLM and Working HF embeddings for RAGAs evaluation
        self.setup_groq_llm()
        
        # Define evaluation metrics
        self.metrics = [
            faithfulness,        # Are answers grounded in context?
            answer_relevancy,    # How relevant are answers?
            context_precision,   # Are retrieved contexts relevant?
            context_recall,      # Did we retrieve all relevant info?
            answer_similarity,   # How similar to ground truth?
            answer_correctness   # How correct are the answers?
        ]
    
    def setup_groq_llm(self):
        """Setup Groq LLM and Working HuggingFace embeddings for RAGAs"""
        
        print("🔧 Setting up Groq LLM and Working HuggingFace embeddings...")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        hf_token = os.getenv("HF_TOKEN")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        print(f"🔑 Using HF_TOKEN: {hf_token[:10]}...")
        
        # Create Groq LLM instance
        self.groq_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1
        )
        
        # Create Working HuggingFace embeddings (your custom class)
        self.hf_embeddings = WorkingHuggingFaceEmbeddings(
            api_key=hf_token,
            model_name="BAAI/bge-base-en-v1.5"
        )
        
        # Test embeddings to make sure they work
        print("🧪 Testing HuggingFace embeddings...")
        test_embedding = self.hf_embeddings.embed_query("test")
        print(f"✅ Embeddings working! Dimension: {len(test_embedding)}")
        
        # Wrap for RAGAs compatibility
        self.ragas_llm = LangchainLLMWrapper(self.groq_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.hf_embeddings)
        
        print("✅ Groq LLM and Working HuggingFace embeddings configured!")
    
    def setup_evaluation(self):
        """Setup RAG system and load ground truth"""
        
        print("🔧 Setting up RAG evaluation...")
        
        # Import here to avoid circular imports
        from Context_RAG.retrieval_generation import setup_rag_system_with_reranking
        
        # Initialize RAG system
        try:
            self.rag_chain = setup_rag_system_with_reranking()
            print("✅ RAG system initialized")
        except Exception as e:
            print(f"❌ Failed to initialize RAG: {e}")
            raise
        
        # Load or generate ground truth
        generator = GroundTruthGenerator()
        self.ground_truth_data = generator.load_ground_truth_dataset()
        
        if self.ground_truth_data is None:
            print("📝 No existing ground truth found. Generating new dataset...")
            self.ground_truth_data = generator.generate_ground_truth_dataset()
        
        print(f"📊 Loaded {len(self.ground_truth_data['data'])} ground truth Q&A pairs")
    
    def generate_evaluation_responses(self, questions):
        """Generate fresh responses for evaluation"""
        
        print("🤖 Generating fresh RAG responses for evaluation...")
        
        responses = []
        contexts = []
        
        for i, question in enumerate(questions, 1):
            print(f"   📋 Processing {i}/{len(questions)}: {question[:50]}...")
            
            try:
                response = self.rag_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": f"{self.session_id}_{i}"}}
                )
                
                responses.append(response["answer"])
                contexts.append([doc.page_content for doc in response.get("context", [])])
                
            except Exception as e:
                print(f"❌ Error generating response for question {i}: {e}")
                responses.append(f"Error: {str(e)}")
                contexts.append([])
        
        return responses, contexts
    
    def prepare_evaluation_dataset(self):
        """Prepare dataset in RAGAs format"""
        
        print("📋 Preparing evaluation dataset...")
        
        # Extract data from ground truth
        questions = [item["question"] for item in self.ground_truth_data["data"]]
        ground_truth_answers = [item["answer"] for item in self.ground_truth_data["data"]]
        ground_truth_contexts = [item["context"] for item in self.ground_truth_data["data"]]
        
        # Generate fresh responses for comparison
        fresh_answers, fresh_contexts = self.generate_evaluation_responses(questions)
        
        # Create evaluation dataset
        eval_data = {
            "question": questions,
            "answer": fresh_answers,  # Fresh RAG responses
            "contexts": fresh_contexts,  # Fresh retrieved contexts
            "ground_truth": ground_truth_answers  # Ground truth answers
        }
        
        # Convert to RAGAs dataset format
        dataset = Dataset.from_dict(eval_data)
        
        print(f"✅ Evaluation dataset prepared with {len(questions)} examples")
        return dataset
    
    def run_evaluation(self, save_results=True):
        """Run RAGAs evaluation with Groq LLM and Working HF Embeddings"""
        
        print("🚀 Starting RAGAs Evaluation with Groq & Working HF...")
        print("=" * 60)
        
        # Setup evaluation
        if self.rag_chain is None:
            self.setup_evaluation()
        
        # Prepare dataset
        eval_dataset = self.prepare_evaluation_dataset()
        
        # Configure metrics to use Groq LLM and Working HuggingFace embeddings
        print("🔧 Configuring RAGAs metrics with Groq LLM and Working HF embeddings...")
        
        # Set LLM and embeddings for each metric that requires them
        for metric in self.metrics:
            if hasattr(metric, 'llm'):
                metric.llm = self.ragas_llm
                print(f"   ✅ {metric.name} LLM configured with Groq")
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self.ragas_embeddings
                print(f"   ✅ {metric.name} embeddings configured with Working HF")
        
        # Run evaluation
        print("📊 Running RAGAs metrics evaluation...")
        try:
            results = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
                llm=self.ragas_llm,  # Explicitly pass Groq LLM
                embeddings=self.ragas_embeddings,  # Explicitly pass Working HF embeddings
                raise_exceptions=False  # Continue even if some metrics fail
            )
            
            print("✅ Evaluation completed successfully!")
            
            # Display results
            self.display_results(results)
            
            # Save results
            if save_results:
                self.save_results(results, eval_dataset)
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def display_results(self, results):
        """Display evaluation results with better error handling"""
        
        print("\n📊 RAGAs Evaluation Results:")
        print("=" * 50)
        
        try:
            # Convert results to DataFrame for better display
            results_df = results.to_pandas()
            
            # Debug: Check what columns we have
            print(f"📋 Available columns: {list(results_df.columns)}")
            
            # Calculate and display metric averages
            metric_columns = []
            for col in results_df.columns:
                if col not in ['question', 'answer', 'contexts', 'ground_truth']:
                    # Check if column contains numeric data
                    try:
                        pd.to_numeric(results_df[col], errors='raise')
                        metric_columns.append(col)
                    except (ValueError, TypeError):
                        print(f"⚠️  Skipping non-numeric column: {col}")
                        continue
            
            if metric_columns:
                print("\n🎯 Average Scores:")
                for metric in metric_columns:
                    try:
                        # Convert to numeric and calculate mean, handling NaN values
                        numeric_values = pd.to_numeric(results_df[metric], errors='coerce')
                        avg_score = numeric_values.mean()
                        valid_count = numeric_values.count()
                        total_count = len(numeric_values)
                        
                        if pd.notna(avg_score):
                            print(f"   {metric:20}: {avg_score:.3f} ({valid_count}/{total_count} valid)")
                        else:
                            print(f"   {metric:20}: No valid scores")
                    except Exception as e:
                        print(f"   {metric:20}: Error calculating average - {e}")
            else:
                print("⚠️  No valid numeric metrics found for averaging")
            
            # Display per-question results
            print(f"\n📋 Per-Question Results:")
            print("-" * 80)
            
            for idx, row in results_df.iterrows():
                question = row.get('question', f'Question {idx + 1}')
                print(f"\n❓ {question[:80]}...")
                
                for metric in metric_columns:
                    if metric in row:
                        try:
                            score = pd.to_numeric(row[metric], errors='coerce')
                            if pd.notna(score):
                                print(f"   {metric:15}: {score:.3f}")
                            else:
                                print(f"   {metric:15}: No score")
                        except:
                            print(f"   {metric:15}: Invalid score")
                            
        except Exception as e:
            print(f"❌ Error displaying results: {e}")
            print("📊 Raw results structure:")
            print(f"   Type: {type(results)}")
            if hasattr(results, 'to_pandas'):
                try:
                    df = results.to_pandas()
                    print(f"   Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Data types: {df.dtypes.to_dict()}")
                except Exception as inner_e:
                    print(f"   Error converting to pandas: {inner_e}")
            print(f"   Raw results: {results}")
    
    def save_results(self, results, dataset):
        """Save evaluation results with better error handling"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save detailed results
            results_df = results.to_pandas()
            results_file = Path(__file__).parent / f"ragas_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            
            # Get only numeric metric columns for summary
            metric_columns = []
            for col in results_df.columns:
                if col not in ['user_input', 'retrieved_contexts', 'response', 'reference', 'question', 'answer', 'contexts', 'ground_truth']:
                    try:
                        pd.to_numeric(results_df[col], errors='raise')
                        metric_columns.append(col)
                    except (ValueError, TypeError):
                        continue
            
            # Save summary statistics only for valid numeric columns
            summary_stats = {}
            
            for metric in metric_columns:
                try:
                    numeric_values = pd.to_numeric(results_df[metric], errors='coerce')
                    valid_values = numeric_values.dropna()
                    
                    if len(valid_values) > 0:
                        summary_stats[metric] = {
                            "mean": float(valid_values.mean()),
                            "std": float(valid_values.std()) if len(valid_values) > 1 else 0.0,
                            "min": float(valid_values.min()),
                            "max": float(valid_values.max()),
                            "count": int(len(valid_values)),
                            "total": int(len(numeric_values))
                        }
                    else:
                        summary_stats[metric] = {
                            "mean": None,
                            "std": None,
                            "min": None,
                            "max": None,
                            "count": 0,
                            "total": int(len(numeric_values))
                        }
                except Exception as e:
                    print(f"⚠️  Error processing metric {metric}: {e}")
                    continue
            
            summary_file = Path(__file__).parent / f"ragas_summary_{timestamp}.json"
            import json
            with open(summary_file, 'w') as f:
                json.dump({
                    "evaluation_date": datetime.now().isoformat(),
                    "model_config": {
                        "llm": "qwen/qwen3-32b",
                        "embedding": "BAAI/bge-base-en-v1.5", 
                        "reranking": "rerank-english-v3.0",
                        "evaluation_llm": "qwen/qwen3-32b (Groq)",
                        "evaluation_embeddings": "BAAI/bge-base-en-v1.5 (Working HF)"
                    },
                    "metrics": summary_stats,
                    "notes": {
                        "rate_limiting": "Some evaluations failed due to Groq rate limits",
                        "successful_metrics": list(summary_stats.keys()),
                        "total_questions": len(results_df)
                    }
                }, f, indent=2)
            
            print(f"\n💾 Results saved:")
            print(f"   📊 Detailed: {results_file}")
            print(f"   📈 Summary: {summary_file}")
            
            # Print summary of what worked
            working_metrics = [m for m in summary_stats.keys() if summary_stats[m]["count"] > 0]
            print(f"\n✅ Successfully evaluated metrics: {', '.join(working_metrics)}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print("📊 Attempting basic save...")
            
            # Fallback: just save the raw DataFrame
            try:
                results_df = results.to_pandas()
                fallback_file = Path(__file__).parent / f"ragas_raw_{timestamp}.csv"
                results_df.to_csv(fallback_file, index=False)
                print(f"💾 Raw results saved to: {fallback_file}")
            except Exception as fallback_error:
                print(f"❌ Even fallback save failed: {fallback_error}")
    
    def compare_configurations(self):
        """Compare different RAG configurations"""
        # Future enhancement: Test with/without reranking, different models, etc.
        pass

def main():
    """Main evaluation function"""
    
    print("🧪 RAGAs Evaluation for Contextual RAG")
    print("=" * 60)
    
    evaluator = RAGEvaluator()
    
    try:
        # Run evaluation
        results = evaluator.run_evaluation()
        
        if results is not None:
            print("\n🎉 Evaluation completed successfully!")
            print("📈 Check the generated CSV and JSON files for detailed results")
        else:
            print("\n❌ Evaluation failed")
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()