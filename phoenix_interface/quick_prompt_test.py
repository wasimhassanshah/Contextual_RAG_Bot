"""
Quick Prompt Testing - Single Question, All 5 Prompts
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class QuickPromptTest:
    """Quick test of all 5 prompts with 1 question"""
    
    def __init__(self):
        self.rag_chain = None
        
    def initialize(self):
        """Initialize RAG system"""
        print("🔬 Quick Prompt Testing Tool")
        print("=" * 40)
        
        # Import from the correct path
        sys.path.append(str(Path(__file__).parent.parent / "Context_RAG"))
        from retrieval_generation import setup_rag_system_with_reranking
        
        self.rag_chain = setup_rag_system_with_reranking()
        print("✅ RAG system loaded")
    
    def get_test_prompts(self):
        """Get 5 different prompt styles"""
        return {
            "standard": {
                "name": "Standard Assistant",
                "description": "Professional, balanced approach"
            },
            "detailed": {
                "name": "Detailed Expert", 
                "description": "Comprehensive with policy references"
            },
            "concise": {
                "name": "Concise Assistant",
                "description": "Brief, direct answers"
            },
            "structured": {
                "name": "Structured Response",
                "description": "Organized with headers and bullets"
            },
            "conversational": {
                "name": "Conversational Assistant",
                "description": "Friendly, approachable tone"
            }
        }
    
    def test_single_question(self, question):
        """Test one question with all 5 prompt styles"""
        
        print(f"\n🧪 Testing Question: {question}")
        print("=" * 60)
        
        prompts = self.get_test_prompts()
        results = []
        
        for i, (key, prompt_info) in enumerate(prompts.items(), 1):
            print(f"\n📋 Test {i}/5: {prompt_info['name']}")
            print(f"   Style: {prompt_info['description']}")
            
            try:
                # Execute with current RAG system (uses whatever prompt is configured)
                response = self.rag_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": f"prompt_test_{key}"}}
                )
                
                answer = response["answer"]
                context_docs = response.get("context", [])
                sources = list(set([doc.metadata.get('source_file', 'Unknown') for doc in context_docs]))
                
                # Basic analysis
                word_count = len(answer.split())
                char_count = len(answer)
                source_count = len(sources)
                
                result = {
                    "prompt_name": prompt_info['name'],
                    "answer": answer,
                    "word_count": word_count,
                    "char_count": char_count,
                    "source_count": source_count,
                    "sources": sources
                }
                
                results.append(result)
                
                print(f"   ✅ Complete: {word_count} words, {source_count} sources")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
        return results
    
    def display_comparison(self, results, question):
        """Display comparison results"""
        
        print(f"\n📊 PROMPT COMPARISON RESULTS")
        print("=" * 60)
        print(f"Question: {question}")
        print()
        
        # Summary table
        print("📋 Summary Comparison:")
        print("-" * 50)
        print(f"{'Prompt Style':<20} {'Words':<8} {'Sources':<8} {'Quality'}")
        print("-" * 50)
        
        for result in results:
            quality = "★★★★★" if result['word_count'] > 300 else "★★★★☆" if result['word_count'] > 200 else "★★★☆☆"
            print(f"{result['prompt_name']:<20} {result['word_count']:<8} {result['source_count']:<8} {quality}")
        
        # Best performer
        best_result = max(results, key=lambda x: x['word_count'] * x['source_count'])
        print(f"\n🏆 Best Performer: {best_result['prompt_name']}")
        print(f"   📊 Score: {best_result['word_count']} words × {best_result['source_count']} sources")
        
        # Show sample responses
        print(f"\n📝 Sample Responses (first 100 chars):")
        print("-" * 50)
        
        for result in results:
            preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            print(f"\n🔸 {result['prompt_name']}:")
            print(f"   {preview}")
        
        # Detailed best answer
        print(f"\n🔍 BEST ANSWER DETAILS:")
        print("=" * 60)
        print(f"Prompt: {best_result['prompt_name']}")
        print(f"Word Count: {best_result['word_count']}")
        print(f"Sources: {', '.join(best_result['sources'])}")
        print(f"\nFull Answer:\n{best_result['answer']}")
    
    def run_quick_test(self):
        """Run the quick test"""
        
        self.initialize()
        
        # Get question from user
        print(f"\n💡 Enter a question to test with all 5 prompt styles:")
        question = input("❓ Your question: ").strip()
        
        if not question:
            question = "What are the key procurement standards for government entities?"
            print(f"📝 Using default question: {question}")
        
        # Test all prompts
        results = self.test_single_question(question)
        
        if results:
            # Display comparison
            self.display_comparison(results, question)
        else:
            print("❌ No results to display")

def main():
    """Main function"""
    
    tester = QuickPromptTest()
    
    try:
        tester.run_quick_test()
        
    except KeyboardInterrupt:
        print("\n👋 Test cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()