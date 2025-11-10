"""
Script to show the full reasoning output for a single test query
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from orchestrator.main import Orchestrator


def main():
    """Show full reasoning output for a single query"""
    
    parser = argparse.ArgumentParser(description='Show full reasoning output for a single query')
    parser.add_argument('--query', '-q', required=True, help='Query to test')
    parser.add_argument('--category', '-c', help='Category name for display')
    args = parser.parse_args()
    
    print("A2A Clinical Retrieval System - Full Reasoning Display")
    print("=" * 80)
    
    if args.category:
        print(f"Category: {args.category}")
    print(f"Query: {args.query}")
    print("=" * 80)
    
    try:
        orchestrator = Orchestrator(
            service_account_path='adsp-34002-ip09-team-2-e0cca2d396a9.json',
            structured_worker_url='http://localhost:8001',
            unstructured_worker_url='http://localhost:8002'
        )
        
        print("Processing query...")
        result = orchestrator.query(args.query)
        
        print("\n" + "=" * 80)
        print("FULL RESPONSE OUTPUT")
        print("=" * 80)
        
        print("\n📊 METADATA:")
        print(f"Reasoning Steps: {result.get('reasoning_steps', 'N/A')}")
        print(f"Answer Length: {len(result.get('answer', ''))} characters")
        
        print(f"\n📋 STRUCTURED SOURCES:")
        structured_sources = result.get('structured_source', [])
        if structured_sources:
            if isinstance(structured_sources, list):
                for source in structured_sources:
                    print(f"  • {source}")
            else:
                print(f"  • {structured_sources}")
        else:
            print("  (None)")
        
        print(f"\n📚 UNSTRUCTURED SOURCES:")
        unstructured_sources = result.get('unstructured_source', [])
        if unstructured_sources:
            if isinstance(unstructured_sources, list):
                for source in unstructured_sources:
                    print(f"  • {source}")
            else:
                print(f"  • {unstructured_sources}")
        else:
            print("  (None)")
        
        print(f"\n💡 FULL ANSWER:")
        print("-" * 40)
        answer = result.get('answer', 'No answer available')
        print(answer)
        print("-" * 40)
        
        if 'error' in result:
            print(f"\n⚠️  ERRORS:")
            print(result['error'])
        
        print(f"\n📄 COMPLETE JSON RESPONSE:")
        print("-" * 40)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
