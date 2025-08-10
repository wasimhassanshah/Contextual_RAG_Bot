"""
Phoenix Prompt Playground for RAG System
Advanced prompt testing and comparison environment (Updated for current Phoenix)
"""

import os
import sys
from pathlib import Path
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import pandas as pd
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class PhoenixPromptPlayground:
    """Advanced prompt playground for RAG system testing"""
    
    def __init__(self):
        self.rag_chain = None
        self.session = None
        self.test_results = []
        
    def initialize_playground(self):
        """Initialize Phoenix prompt playground"""
        
        print("🔬 Initializing Phoenix Prompt Playground...")
        
        # Set Phoenix port
        os.environ["PHOENIX_PORT"] = "6006"
        
        # Launch Phoenix if not already running
        try:
            self.session = px.launch_app()
            print("✅ Phoenix launched for prompt playground")
        except:
            print("📍 Phoenix already running, connecting...")
        
        # Setup OpenInference instrumentation
        tracer_provider = register(
            project_name="Prompt_Playground_RAG",
            endpoint="http://localhost:6006/v1/traces"
        )
        
        # Enable LangChain instrumentation
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        
        # Load RAG system
        from Context_RAG.retrieval_generation import setup_rag_system_with_reranking
        self.rag_chain = setup_rag_system_with_reranking()
        
        print("✅ Prompt playground initialized with Phoenix tracing")
        
    def define_test_prompts(self):
        """Define different prompt variations for testing"""
        
        return {
            "standard": {
                "name": "Standard Procurement Assistant",
                "prompt": """You are an expert procurement assistant with access to Abu Dhabi government documents.
                
Context: {context}
Question: {question}

Provide a clear, accurate answer based on the context.""",
                "temperature": 0.3
            },
            
            "detailed": {
                "name": "Detailed Policy Expert",
                "prompt": """You are a detailed Abu Dhabi procurement and policy expert. Provide comprehensive answers with:
- Specific policy references
- Document citations
- Step-by-step guidance when applicable
- Compliance requirements

Context: {context}
Question: {question}

Detailed Expert Response:""",
                "temperature": 0.1
            },
            
            "concise": {
                "name": "Concise Assistant",
                "prompt": """Provide brief, direct answers to procurement questions.

Context: {context}
Question: {question}

Brief Answer:""",
                "temperature": 0.2
            },
            
            "structured": {
                "name": "Structured Response Assistant",
                "prompt": """Provide structured answers using this format:
## Summary
[Brief overview]

## Key Points
- Point 1
- Point 2

## References
[Document sources]

Context: {context}
Question: {question}

Structured Response:""",
                "temperature": 0.2
            },
            
            "conversational": {
                "name": "Conversational Assistant",
                "prompt": """You are a friendly, conversational procurement assistant. Explain complex policies in simple terms while maintaining accuracy.

Context: {context}
Question: {question}

Friendly Response:""",
                "temperature": 0.4
            }
        }
    
    def get_test_questions(self):
        """Get test questions for prompt evaluation"""
        
        return [
            {
                "question": "What are the key procurement standards for government entities?",
                "category": "Standards",
                "expected_elements": ["procurement standards", "government entities", "compliance"]
            },
            {
                "question": "How should suppliers be registered and classified?",
                "category": "Supplier Management", 
                "expected_elements": ["registration", "classification", "suppliers"]
            },
            {
                "question": "What security requirements apply to information systems?",
                "category": "Security",
                "expected_elements": ["security requirements", "information systems", "controls"]
            },
            {
                "question": "What are the HR policies for employee disciplinary actions?",
                "category": "HR Policies",
                "expected_elements": ["disciplinary actions", "HR policies", "procedures"]
            }
        ]
    
    def test_prompt_variation(self, prompt_config, question_data, session_id):
        """Test a specific prompt variation (automatic tracing via OpenInference)"""
        
        try:
            print(f"   🧪 Testing: {prompt_config['name']}")
            
            # Execute query with automatic tracing (no manual trace needed)
            response = self.rag_chain.invoke(
                {"input": question_data["question"]},
                config={"configurable": {"session_id": session_id}}
            )
            
            answer = response["answer"]
            context_docs = response.get("context", [])
            
            # Analyze response
            analysis = self.analyze_response(answer, question_data, context_docs)
            
            return {
                "prompt_name": prompt_config["name"],
                "question": question_data["question"],
                "answer": answer,
                "context_count": len(context_docs),
                "sources": [doc.metadata.get('source_file', 'Unknown') for doc in context_docs],
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error testing prompt '{prompt_config['name']}': {e}")
            return None
    
    def analyze_response(self, answer, question_data, context_docs):
        """Analyze response quality"""
        
        # Check if expected elements are present
        expected_elements = question_data.get("expected_elements", [])
        contains_expected = sum(1 for element in expected_elements 
                               if element.lower() in answer.lower())
        expected_coverage = contains_expected / len(expected_elements) if expected_elements else 0
        
        # Basic readability analysis
        sentences = answer.count('.') + answer.count('!') + answer.count('?')
        words = len(answer.split())
        avg_sentence_length = words / max(sentences, 1)
        readability_score = min(100, max(0, 100 - avg_sentence_length * 2))  # Simple metric
        
        # Source diversity
        sources = list(set([doc.metadata.get('source_file', 'Unknown') for doc in context_docs]))
        source_diversity = len(sources)
        
        return {
            "contains_expected": expected_coverage,
            "readability_score": readability_score,
            "word_count": words,
            "sentence_count": sentences,
            "avg_sentence_length": avg_sentence_length,
            "source_diversity": source_diversity,
            "sources": sources
        }
    
    def run_prompt_comparison(self):
        """Run comprehensive prompt comparison test"""
        
        print("🔬 Running Prompt Playground Comparison...")
        print("=" * 60)
        
        prompts = self.define_test_prompts()
        questions = self.get_test_questions()
        
        results = []
        total_tests = len(prompts) * len(questions)
        current_test = 0
        
        for prompt_name, prompt_config in prompts.items():
            print(f"\n🧪 Testing: {prompt_config['name']}")
            
            for question_data in questions:
                current_test += 1
                print(f"   📋 Question {current_test}/{total_tests}: {question_data['question'][:50]}...")
                
                session_id = f"playground_{prompt_name}_{current_test}"
                result = self.test_prompt_variation(prompt_config, question_data, session_id)
                
                if result:
                    results.append(result)
                    print(f"   ✅ Completed - {result['analysis']['word_count']} words")
        
        self.test_results = results
        print(f"\n✅ Prompt comparison completed! {len(results)} tests run")
        
        return results
    
    def generate_comparison_report(self, results):
        """Generate detailed comparison report"""
        
        print("\n📊 Generating Prompt Comparison Report...")
        
        # Convert to DataFrame for analysis
        df_data = []
        for result in results:
            df_data.append({
                "prompt_name": result["prompt_name"],
                "question": result["question"][:50] + "...",
                "word_count": result["analysis"]["word_count"],
                "readability_score": result["analysis"]["readability_score"],
                "expected_coverage": result["analysis"]["contains_expected"],
                "source_diversity": result["analysis"]["source_diversity"],
                "context_count": result["context_count"]
            })
        
        df = pd.DataFrame(df_data)
        
        # Generate summary statistics
        summary = df.groupby("prompt_name").agg({
            "word_count": ["mean", "std"],
            "readability_score": ["mean", "std"],
            "expected_coverage": ["mean", "std"],
            "source_diversity": ["mean", "std"],
            "context_count": ["mean", "std"]
        }).round(2)
        
        print("\n📈 Prompt Performance Summary:")
        print("=" * 80)
        print(summary)
        
        # Find best performing prompts
        avg_scores = df.groupby("prompt_name").agg({
            "readability_score": "mean",
            "expected_coverage": "mean", 
            "source_diversity": "mean"
        })
        
        # Calculate overall score (weighted average)
        avg_scores["overall_score"] = (
            avg_scores["readability_score"] * 0.3 +
            avg_scores["expected_coverage"] * 100 * 0.4 +  # Convert to 0-100 scale
            avg_scores["source_diversity"] * 10 * 0.3  # Scale up
        )
        
        best_prompt = avg_scores["overall_score"].idxmax()
        
        print(f"\n🏆 Best Performing Prompt: {best_prompt}")
        print(f"📊 Overall Score: {avg_scores.loc[best_prompt, 'overall_score']:.2f}")
        print(f"🌐 View detailed traces: http://localhost:6006")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(__file__).parent / f"prompt_comparison_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "best_prompt": best_prompt,
                "summary_stats": avg_scores.to_dict(),
                "detailed_results": results
            }, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        return summary, best_prompt
    
    def launch_interactive_playground(self):
        """Launch interactive prompt playground"""
        
        print("🚀 Launching Interactive Prompt Playground...")
        print("=" * 60)
        
        prompts = self.define_test_prompts()
        
        while True:
            print("\n🔬 Prompt Playground Menu:")
            print("1. Test single prompt")
            print("2. Compare all prompts")
            print("3. Run custom prompt")
            print("4. View results summary")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                self.test_single_prompt_interactive(prompts)
            elif choice == "2":
                results = self.run_prompt_comparison()
                self.generate_comparison_report(results)
            elif choice == "3":
                self.test_custom_prompt()
            elif choice == "4":
                if self.test_results:
                    self.generate_comparison_report(self.test_results)
                else:
                    print("⚠️ No results available. Run tests first.")
            elif choice == "5":
                print("👋 Exiting prompt playground...")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def test_single_prompt_interactive(self, prompts):
        """Interactive single prompt testing"""
        
        print("\n📝 Available Prompts:")
        for i, (key, config) in enumerate(prompts.items(), 1):
            print(f"{i}. {config['name']}")
        
        try:
            choice = int(input("\nSelect prompt (1-5): ")) - 1
            prompt_keys = list(prompts.keys())
            selected_prompt = prompts[prompt_keys[choice]]
            
            question = input("Enter your question: ").strip()
            
            if question:
                session_id = f"interactive_{datetime.now().strftime('%H%M%S')}"
                question_data = {"question": question, "category": "Interactive", "expected_elements": []}
                
                result = self.test_prompt_variation(selected_prompt, question_data, session_id)
                
                if result:
                    print(f"\n🤖 Response from '{selected_prompt['name']}':")
                    print("-" * 50)
                    print(result["answer"])
                    print(f"\n📊 Analysis:")
                    print(f"   Words: {result['analysis']['word_count']}")
                    print(f"   Readability: {result['analysis']['readability_score']:.1f}")
                    print(f"   Sources: {', '.join(result['sources'])}")
            
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    def test_custom_prompt(self):
        """Test custom user-defined prompt"""
        
        print("\n✏️ Custom Prompt Testing")
        print("Enter your custom prompt (use {context} and {question} placeholders):")
        
        custom_prompt = input().strip()
        if not custom_prompt:
            print("❌ Empty prompt")
            return
        
        question = input("Enter test question: ").strip()
        if not question:
            print("❌ Empty question")
            return
        
        # Create custom prompt config
        custom_config = {
            "name": "Custom Prompt",
            "prompt": custom_prompt,
            "temperature": 0.3
        }
        
        question_data = {"question": question, "category": "Custom", "expected_elements": []}
        session_id = f"custom_{datetime.now().strftime('%H%M%S')}"
        
        result = self.test_prompt_variation(custom_config, question_data, session_id)
        
        if result:
            print(f"\n🤖 Custom Prompt Response:")
            print("-" * 50)
            print(result["answer"])
            print(f"\n📊 Analysis:")
            print(f"   Words: {result['analysis']['word_count']}")
            print(f"   Readability: {result['analysis']['readability_score']:.1f}")

def main():
    """Main function to launch prompt playground"""
    
    playground = PhoenixPromptPlayground()
    
    try:
        playground.initialize_playground()
        playground.launch_interactive_playground()
        
    except KeyboardInterrupt:
        print("\n👋 Playground session ended")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()