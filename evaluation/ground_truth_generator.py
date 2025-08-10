"""
Ground Truth Generator for RAG Evaluation
Generates ground truth answers using the current RAG system
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from test_questions import get_test_questions
from Context_RAG.retrieval_generation import setup_rag_system_with_reranking

class GroundTruthGenerator:
    """Generates ground truth answers using RAG system"""
    
    def __init__(self):
        self.rag_chain = None
        self.session_id = "ground_truth_generation"
    
    def initialize_rag(self):
        """Initialize the RAG system"""
        print("🤖 Initializing RAG system for ground truth generation...")
        try:
            self.rag_chain = setup_rag_system_with_reranking()
            print("✅ RAG system initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    def generate_answer(self, question):
        """Generate answer for a single question"""
        try:
            print(f"🔍 Generating answer for: {question[:60]}...")
            
            response = self.rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": self.session_id}}
            )
            
            return {
                "question": question,
                "answer": response["answer"],
                "context": [doc.page_content for doc in response.get("context", [])],
                "source_documents": [doc.metadata.get("source_file", "Unknown") 
                                   for doc in response.get("context", [])]
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "context": [],
                "source_documents": []
            }
    
    def generate_ground_truth_dataset(self, questions=None, save_file=True):
        """Generate ground truth for all test questions"""
        
        if questions is None:
            questions = get_test_questions()
        
        print(f"📝 Generating ground truth for {len(questions)} questions...")
        
        if self.rag_chain is None:
            self.initialize_rag()
        
        ground_truth_data = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n📋 Processing question {i}/{len(questions)}")
            
            result = self.generate_answer(question)
            ground_truth_data.append(result)
            
            print(f"✅ Answer generated ({len(result['answer'])} characters)")
        
        # Create dataset metadata
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_questions": len(questions),
                "rag_model": "llama-3.1-8b-instant",
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "reranking_model": "rerank-english-v3.0",
                "description": "Ground truth generated using enhanced RAG with Cohere re-ranking"
            },
            "data": ground_truth_data
        }
        
        if save_file:
            # Save to JSON file
            output_file = Path(__file__).parent / "ground_truth_dataset.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Ground truth dataset saved to: {output_file}")
        
        return dataset
    
    def load_ground_truth_dataset(self, file_path=None):
        """Load existing ground truth dataset"""
        
        if file_path is None:
            file_path = Path(__file__).parent / "ground_truth_dataset.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            print(f"📂 Loaded ground truth dataset with {len(dataset['data'])} questions")
            return dataset
            
        except FileNotFoundError:
            print(f"❌ Ground truth file not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return None
    
    def display_sample_qa(self, dataset, num_samples=2):
        """Display sample Q&A pairs from dataset"""
        
        print(f"\n📊 Sample Q&A Pairs (showing {num_samples}):")
        print("=" * 80)
        
        for i, item in enumerate(dataset["data"][:num_samples], 1):
            print(f"\n🔍 Question {i}:")
            print(f"Q: {item['question']}")
            print(f"\n🤖 Answer:")
            print(f"A: {item['answer'][:300]}...")
            print(f"\n📚 Sources: {', '.join(set(item['source_documents']))}")
            print("-" * 80)

def main():
    """Main function to generate ground truth"""
    
    print("🏗️ Ground Truth Generator for RAG Evaluation")
    print("=" * 60)
    
    generator = GroundTruthGenerator()
    
    try:
        # Generate ground truth dataset
        dataset = generator.generate_ground_truth_dataset()
        
        # Display sample results
        generator.display_sample_qa(dataset)
        
        print(f"\n🎉 Ground truth generation complete!")
        print(f"📊 Generated {len(dataset['data'])} Q&A pairs")
        print(f"💾 Saved to: ground_truth_dataset.json")
        
    except Exception as e:
        print(f"❌ Ground truth generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()