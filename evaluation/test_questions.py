"""
Test Questions for RAGAs Evaluation
Contains predefined questions for evaluating RAG performance
"""

# Test questions for RAG evaluation (reduced to 2 for rate limit testing)
TEST_QUESTIONS = [
    "How do the \"Delivery Terms\" and \"Payment Terms\" relate to a \"Purchase Order\" within the procurement process described in this document?",
    
    "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?"
]

# Full question set (can be used later when rate limits are resolved)
FULL_TEST_QUESTIONS = [
    "How do the \"Delivery Terms\" and \"Payment Terms\" relate to a \"Purchase Order\" within the procurement process described in this document?",
    
    "Infer the hierarchical relationship between Law No. (6) of 2016 and Decision No. (10) of 2020, based on their descriptions in the document.",
    
    "Based on the penalties section, what are the different levels of disciplinary actions for various infringements, and how do they relate to preserving pension or bonus rights?",
    
    "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?"
]

# Additional questions for comprehensive evaluation (can be expanded)
EXTENDED_TEST_QUESTIONS = [
    "What are the key procurement standards mentioned in Abu Dhabi documents?",
    "What security requirements are mentioned for supplier agreements?",
    "What HR policies should employees be aware of?",
    "What is the procurement process workflow according to the standards?",
    "What are the confidentiality requirements for suppliers?",
    "What are the information security guidelines for data protection?",
    "How should procurement practitioners manage vendor relationships?",
    "What are the compliance requirements for government procurement?"
]

def get_test_questions(extended=False, full_set=False):
    """
    Get test questions for evaluation
    
    Args:
        extended (bool): Whether to include extended question set
        full_set (bool): Whether to use all 4 original questions (may hit rate limits)
    
    Returns:
        list: List of test questions
    """
    if full_set:
        base_questions = FULL_TEST_QUESTIONS
    else:
        base_questions = TEST_QUESTIONS  # Just 2 questions to avoid rate limits
        
    if extended:
        return base_questions + EXTENDED_TEST_QUESTIONS
    return base_questions

def get_question_categories():
    """
    Categorize questions by domain for targeted evaluation
    
    Returns:
        dict: Questions categorized by domain
    """
    categories = {
        "procurement_process": [
            "How do the \"Delivery Terms\" and \"Payment Terms\" relate to a \"Purchase Order\" within the procurement process described in this document?",
            "What are the key procurement standards mentioned in Abu Dhabi documents?",
            "What is the procurement process workflow according to the standards?"
        ],
        "legal_hierarchy": [
            "Infer the hierarchical relationship between Law No. (6) of 2016 and Decision No. (10) of 2020, based on their descriptions in the document."
        ],
        "hr_policies": [
            "Based on the penalties section, what are the different levels of disciplinary actions for various infringements, and how do they relate to preserving pension or bonus rights?",
            "What HR policies should employees be aware of?"
        ],
        "security_controls": [
            "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?",
            "What are the information security guidelines for data protection?",
            "What security requirements are mentioned for supplier agreements?"
        ]
    }
    return categories

if __name__ == "__main__":
    print("📋 Test Questions for RAG Evaluation")
    print("=" * 50)
    
    print("🔥 Rate-Limit Safe Questions (Default):")
    questions = get_test_questions()
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    print(f"\nTotal questions: {len(questions)}")
    
    print("\n🚀 Full Question Set (May hit rate limits):")
    full_questions = get_test_questions(full_set=True)
    for i, question in enumerate(full_questions, 1):
        print(f"{i}. {question[:80]}...")
    
    print(f"Full set total: {len(full_questions)}")
    
    print("\n📊 Questions by Category:")
    categories = get_question_categories()
    for category, cat_questions in categories.items():
        print(f"\n{category.upper()}:")
        for question in cat_questions:
            print(f"  - {question[:80]}...")