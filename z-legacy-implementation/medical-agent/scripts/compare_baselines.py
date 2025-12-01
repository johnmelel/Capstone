"""
Comprehensive Baseline Comparison Script
Tests all orchestrator architectures and compares performance metrics for academic paper
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import argparse
from typing import Dict, Any, List
from datetime import datetime

# Import all orchestrator types
from orchestrator.main import Orchestrator as MCPOrchestrator
from baselines.orchestrator_no_mcp import NoMCPOrchestrator
from baselines.orchestrator_single_agent import SingleAgentOrchestrator
from baselines.orchestrator_vector_only import VectorOnlyOrchestrator


# 10 test queries designed for actual MIMIC-IV data availability
# Note: Dates in database are anonymized (2025-2034), queries avoid strict time windows
TEST_QUERIES = [
    {
        "category": "1. Diagnostic Pattern Reasoning",
        "description": "Connect EMR findings with differential diagnoses using domain knowledge",
        "query": "What are the key monitoring requirements and potential complications for patients diagnosed with diabetes mellitus (ICD-9 code 25000) who are on insulin therapy?"
    },
    {
        "category": "2. Mechanistic/Pathophysiologic Reasoning", 
        "description": "Link pathophysiology to clinical management",
        "query": "Explain why patients with atrial fibrillation (ICD-9 code 42731) in our database require careful medication management and what monitoring is essential"
    },
    {
        "category": "3. Comparative/Evidence-Based Selection",
        "description": "Evaluate treatment options based on EMR patterns and guidelines",
        "query": "Compare the use of Furosemide versus other diuretics in our patient population. What clinical scenarios favor Furosemide and what monitoring is required?"
    },
    {
        "category": "4. Temporal/Longitudinal Reasoning",
        "description": "Analyze medication patterns over time", 
        "query": "Analyze the prescription patterns for Metoprolol Tartrate in our database. What dosing patterns are most common and what do they suggest about patient management?"
    },
    {
        "category": "5. Multi-Modal Correlation",
        "description": "Integrate demographics, diagnoses, and medications",
        "query": "Correlate patient age, gender, and admission types with insulin prescriptions. What patterns emerge and what do they tell us about diabetes management?"
    },
    {
        "category": "6. Clinical Biomarker Reasoning",
        "description": "Interpret lab values with clinical context",
        "query": "What laboratory monitoring is essential for patients on Heparin therapy, and what abnormal findings would require immediate intervention?"
    },
    {
        "category": "7. Report Summarization/Impression",
        "description": "Synthesize multi-table findings into clinical impressions",
        "query": "Summarize the typical clinical profile of urgent admissions in our database including common diagnoses and initial medication regimens"
    },
    {
        "category": "8. Clinical Decision Support/Triage",
        "description": "Risk stratification and prioritization",
        "query": "Identify high-risk medication combinations in our prescriptions database. Which patients receiving Potassium Chloride need additional monitoring?"
    },
    {
        "category": "9. Medication Safety and Interactions",
        "description": "Handle polypharmacy and drug interactions",
        "query": "For patients receiving both Acetaminophen and HYDROmorphone (common in our data), what are the key safety considerations and monitoring requirements?"
    },
    {
        "category": "10. Population Health and Patterns",
        "description": "Analyze cohort-level trends and implications",
        "query": "Analyze the prevalence of hypertension (ICD codes 4019 and I10) in our patient population. What are the standard treatment approaches and how do they vary by patient characteristics?"
    }
]


def initialize_orchestrator(orchestrator_type: str, service_account_path: str):
    """Initialize the specified orchestrator type"""
    if orchestrator_type == "mcp_multi_agent":
        return MCPOrchestrator(
            service_account_path=service_account_path,
            structured_worker_url='http://localhost:8001',
            unstructured_worker_url='http://localhost:8002'
        )
    elif orchestrator_type == "no_mcp":
        return NoMCPOrchestrator(
            service_account_path=service_account_path
        )
    elif orchestrator_type == "single_agent":
        return SingleAgentOrchestrator(
            service_account_path=service_account_path
        )
    elif orchestrator_type == "vector_only":
        return VectorOnlyOrchestrator(
            service_account_path=service_account_path,
            unstructured_worker_url='http://localhost:8002'
        )
    else:
        raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")


def test_orchestrator(orchestrator, orchestrator_type: str, test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
    """Test an orchestrator with all queries and collect metrics"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {orchestrator_type.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}: {test_case['category']}")
        print(f"Query: {test_case['query'][:60]}...")
        
        try:
            # Measure latency
            start_time = time.time()
            result = orchestrator.query(test_case['query'])
            latency = time.time() - start_time
            
            # Extract metrics
            reasoning_steps = result.get('reasoning_steps', 0)
            answer = result.get('answer', '')
            answer_length = len(answer)
            
            # Check data source utilization
            has_structured = bool(result.get('structured_source', []))
            has_unstructured = bool(result.get('unstructured_source', [])) or bool(result.get('sources', []))
            
            # Determine success
            success = bool(answer and 'error' not in answer.lower()[:50])
            
            print(f"  Latency: {latency:.2f}s | Steps: {reasoning_steps} | Length: {answer_length} chars")
            print(f"  Success: {'Yes' if success else 'No'}")
            
            results.append({
                'category': test_case['category'],
                'query': test_case['query'],
                'reasoning_steps': reasoning_steps if isinstance(reasoning_steps, int) else 0,
                'answer_length': answer_length,
                'latency': latency,
                'used_structured': has_structured,
                'used_unstructured': has_unstructured,
                'success': success,
                'answer_preview': answer[:200] if answer else ''
            })
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                'category': test_case['category'],
                'query': test_case['query'],
                'reasoning_steps': 0,
                'answer_length': 0,
                'latency': 0,
                'used_structured': False,
                'used_unstructured': False,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Calculate aggregate metrics
    successful = [r for r in results if r['success']]
    
    aggregate = {
        'orchestrator_type': orchestrator_type,
        'total_tests': len(results),
        'successful_tests': len(successful),
        'success_rate': len(successful) / len(results) * 100 if results else 0,
        'avg_reasoning_steps': sum(r['reasoning_steps'] for r in successful) / len(successful) if successful else 0,
        'avg_latency': sum(r['latency'] for r in successful) / len(successful) if successful else 0,
        'avg_answer_length': sum(r['answer_length'] for r in successful) / len(successful) if successful else 0,
        'multi_modal_usage': sum(1 for r in successful if r['used_structured'] and r['used_unstructured']) / len(successful) * 100 if successful else 0,
        'structured_only': sum(1 for r in successful if r['used_structured'] and not r['used_unstructured']),
        'unstructured_only': sum(1 for r in successful if r['used_unstructured'] and not r['used_structured']),
        'detailed_results': results
    }
    
    return aggregate


def print_comparison_table(all_results: List[Dict[str, Any]]):
    """Print a formatted comparison table"""
    
    print(f"\n{'='*100}")
    print("BASELINE COMPARISON RESULTS")
    print(f"{'='*100}\n")
    
    # Header
    print(f"{'Metric':<30} | {'MCP Multi':<12} | {'No-MCP':<12} | {'Single':<12} | {'Vector-Only':<12}")
    print("-" * 100)
    
    # Find results for each type
    results_map = {r['orchestrator_type']: r for r in all_results}
    
    mcp = results_map.get('mcp_multi_agent', {})
    no_mcp = results_map.get('no_mcp', {})
    single = results_map.get('single_agent', {})
    vector = results_map.get('vector_only', {})
    
    # Print metrics
    metrics = [
        ('Success Rate (%)', 'success_rate', '.1f'),
        ('Avg Reasoning Steps', 'avg_reasoning_steps', '.1f'),
        ('Avg Latency (s)', 'avg_latency', '.2f'),
        ('Avg Answer Length', 'avg_answer_length', '.0f'),
        ('Multi-Modal Usage (%)', 'multi_modal_usage', '.1f'),
    ]
    
    for label, key, fmt in metrics:
        mcp_val = mcp.get(key, 0)
        no_mcp_val = no_mcp.get(key, 0)
        single_val = single.get(key, 0)
        vector_val = vector.get(key, 0)
        
        print(f"{label:<30} | {mcp_val:{fmt}} | {no_mcp_val:{fmt}} | {single_val:{fmt}} | {vector_val:{fmt}}")
    
    print("-" * 100)
    
    # Print key insights
    print("\nKEY INSIGHTS:")
    print(f"1. MCP Multi-Agent achieved {mcp.get('success_rate', 0):.1f}% success rate (control)")
    
    if no_mcp:
        step_diff = mcp.get('avg_reasoning_steps', 0) - no_mcp.get('avg_reasoning_steps', 0)
        print(f"2. No-MCP baseline: {abs(step_diff):.1f} {'fewer' if step_diff > 0 else 'more'} reasoning steps on average")
    
    if single:
        step_diff = mcp.get('avg_reasoning_steps', 0) - single.get('avg_reasoning_steps', 0)
        print(f"3. Single-Agent baseline: {abs(step_diff):.1f} {'fewer' if step_diff > 0 else 'more'} reasoning steps on average")
    
    if vector:
        success_diff = mcp.get('success_rate', 0) - vector.get('success_rate', 0)
        print(f"4. Vector-Only baseline: {success_diff:.1f}% lower success rate (demonstrates multi-modal value)")


def save_results(all_results: List[Dict[str, Any]], output_file: str):
    """Save detailed results to JSON file"""
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'test_configuration': {
            'num_queries': len(TEST_QUERIES),
            'orchestrator_types_tested': len(all_results)
        },
        'results': all_results,
        'comparison_summary': {
            r['orchestrator_type']: {
                'success_rate': r['success_rate'],
                'avg_reasoning_steps': r['avg_reasoning_steps'],
                'avg_latency': r['avg_latency'],
                'multi_modal_usage': r['multi_modal_usage']
            }
            for r in all_results
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline orchestrator architectures')
    parser.add_argument('--orchestrator-type', 
                       choices=['mcp_multi_agent', 'no_mcp', 'single_agent', 'vector_only', 'all'],
                       default='all',
                       help='Which orchestrator to test')
    parser.add_argument('--service-account', 
                       default='adsp-34002-ip09-team-2-e0cca2d396a9.json',
                       help='Path to service account JSON file')
    parser.add_argument('--output', 
                       default='baseline_comparison_results.json',
                       help='Output file for results')
    parser.add_argument('--quick',
                       action='store_true',
                       help='Run quick test with first 3 queries only')
    
    args = parser.parse_args()
    
    # Select test queries
    test_queries = TEST_QUERIES[:3] if args.quick else TEST_QUERIES
    
    print("="*80)
    print("BASELINE ORCHESTRATOR COMPARISON")
    print("="*80)
    print(f"Testing {len(test_queries)} queries across different architectures")
    print(f"Service Account: {args.service_account}")
    
    if args.quick:
        print("QUICK MODE: Testing first 3 queries only")
    
    # Determine which orchestrators to test
    if args.orchestrator_type == 'all':
        orchestrator_types = ['mcp_multi_agent', 'no_mcp', 'single_agent', 'vector_only']
    else:
        orchestrator_types = [args.orchestrator_type]
    
    print(f"Orchestrators to test: {', '.join(orchestrator_types)}")
    
    all_results = []
    
    for orch_type in orchestrator_types:
        try:
            print(f"\nInitializing {orch_type}...")
            orchestrator = initialize_orchestrator(orch_type, args.service_account)
            
            result = test_orchestrator(orchestrator, orch_type, test_queries)
            all_results.append(result)
            
        except Exception as e:
            print(f"\nFailed to test {orch_type}: {e}")
            print("Continuing with other orchestrators...")
    
    if all_results:
        # Print comparison table
        print_comparison_table(all_results)
        
        # Save results
        save_results(all_results, args.output)
        
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE")
        print(f"{'='*80}")
        
        return 0
    else:
        print("\nNo orchestrators could be tested successfully.")
        return 1


if __name__ == "__main__":
    exit(main())
