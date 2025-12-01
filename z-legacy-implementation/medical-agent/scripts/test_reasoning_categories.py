"""
Test script for 10 medical reasoning categories
Measures reasoning steps for each type of medical query
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from orchestrator.main import Orchestrator


def main():
    """Test 10 reasoning categories and measure reasoning steps"""
    
    # Define 10 test queries for different reasoning categories
    test_queries = [
        {
            "category": "1. Diagnostic Pattern Reasoning",
            "description": "Connect findings with differential diagnoses",
            "query": "What are the possible diagnoses for a patient with elevated troponin levels and recent chest pain?"
        },
        {
            "category": "2. Mechanistic/Pathophysiologic Reasoning", 
            "description": "Link pathophysiology to clinical findings",
            "query": "Explain how insulin resistance leads to the clinical findings we see in diabetic patients"
        },
        {
            "category": "3. Comparative/Evidence-Based Selection",
            "description": "Evaluate optimal treatment options",
            "query": "Compare the effectiveness of aspirin versus warfarin for stroke prevention in atrial fibrillation"
        },
        {
            "category": "4. Temporal/Longitudinal Reasoning",
            "description": "Analyze sequential data and trends", 
            "query": "What trends should we monitor in patients taking amphetamine over the past 48 hours?"
        },
        {
            "category": "5. Multi-Modal Correlation",
            "description": "Integrate multiple data sources",
            "query": "Correlate recent admissions with medication prescriptions and patient demographics"
        },
        {
            "category": "6. Clinical Biomarker Reasoning",
            "description": "Interpret biomarkers and clinical indicators",
            "query": "What do elevated insulin levels combined with recent hospital admission suggest about patient management?"
        },
        {
            "category": "7. Report Summarization/Impression",
            "description": "Synthesize findings into clinical impressions",
            "query": "Summarize the clinical status of patients admitted in the last 24 hours with their medication profiles"
        },
        {
            "category": "8. Clinical Decision Support/Triage",
            "description": "Urgent vs non-urgent recommendations",
            "query": "Which patients taking warfarin require immediate attention based on their recent prescriptions?"
        },
        {
            "category": "9. Ambiguity Resolution",
            "description": "Handle conflicting or unclear findings",
            "query": "How should we interpret conflicting medication timing data for patients with multiple prescriptions?"
        },
        {
            "category": "10. Quantitative and Predictive Reasoning",
            "description": "Use numerical data for clinical predictions",
            "query": "Based on prescription patterns, predict which patients are at highest risk for medication interactions"
        }
    ]
    
    # Initialize orchestrator
    print("Initializing A2A Clinical Retrieval System...")
    print("=" * 80)
    
    try:
        orchestrator = Orchestrator(
            service_account_path='adsp-34002-ip09-team-2-e0cca2d396a9.json',
            structured_worker_url='http://localhost:8001',
            unstructured_worker_url='http://localhost:8002'
        )
        
        results = []
        
        print(f"Testing {len(test_queries)} reasoning categories:\n")
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"Test {i}/10: {test_case['category']}")
            print(f"Description: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            print("-" * 60)
            
            try:
                # Execute query
                result = orchestrator.query(test_case['query'])
                
                # Extract reasoning steps
                reasoning_steps = result.get('reasoning_steps', 'N/A')
                answer_length = len(result.get('answer', ''))
                has_structured = bool(result.get('structured_source'))
                has_unstructured = bool(result.get('unstructured_source'))
                
                print(f"✓ Reasoning Steps: {reasoning_steps}")
                print(f"  Answer Length: {answer_length} chars")
                print(f"  Used Structured Data: {'Yes' if has_structured else 'No'}")
                print(f"  Used Medical Knowledge: {'Yes' if has_unstructured else 'No'}")
                
                results.append({
                    'category': test_case['category'],
                    'query': test_case['query'],
                    'reasoning_steps': reasoning_steps,
                    'answer_length': answer_length,
                    'used_structured': has_structured,
                    'used_unstructured': has_unstructured,
                    'success': True
                })
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                results.append({
                    'category': test_case['category'],
                    'query': test_case['query'],
                    'reasoning_steps': 0,
                    'answer_length': 0,
                    'used_structured': False,
                    'used_unstructured': False,
                    'success': False,
                    'error': str(e)
                })
            
            print()
        
        # Summary
        print("=" * 80)
        print("REASONING ANALYSIS SUMMARY")
        print("=" * 80)
        
        successful_tests = [r for r in results if r['success']]
        
        if successful_tests:
            reasoning_steps = [r['reasoning_steps'] for r in successful_tests if isinstance(r['reasoning_steps'], int)]
            
            print(f"Successful Tests: {len(successful_tests)}/{len(test_queries)}")
            
            if reasoning_steps:
                print(f"Average Reasoning Steps: {sum(reasoning_steps) / len(reasoning_steps):.1f}")
                print(f"Reasoning Steps Range: {min(reasoning_steps)} - {max(reasoning_steps)}")
            
            print(f"Tests Using Structured Data: {sum(1 for r in successful_tests if r['used_structured'])}")
            print(f"Tests Using Medical Knowledge: {sum(1 for r in successful_tests if r['used_unstructured'])}")
            
            print("\nDetailed Results:")
            for i, result in enumerate(successful_tests, 1):
                steps = result['reasoning_steps']
                category = result['category'].split('.', 1)[1] if '.' in result['category'] else result['category']
                steps_str = f"{steps:2d}" if isinstance(steps, int) else str(steps)
                print(f"  {i:2d}. {category:30s} - {steps_str} steps")
        else:
            print("No successful tests to analyze.")
        
        # Save detailed results
        with open('reasoning_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: reasoning_test_results.json")
        
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        print("\nMake sure the workers are running:")
        print("  ./scripts/start_workers.sh")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
