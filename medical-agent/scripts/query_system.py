"""
Simple CLI interface for the A2A Clinical Retrieval System
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
from pathlib import Path

from orchestrator.main import Orchestrator


def main():
    """Simple CLI interface for medical queries"""
    parser = argparse.ArgumentParser(
        description='A2A Clinical Retrieval System - Query medical data using structured EMR and unstructured knowledge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/query_system.py "Get me all patients who took Amphetamine in the last 24 hours"
  python scripts/query_system.py "Show recent admissions in the past 12 hours"
  python scripts/query_system.py "What are the monitoring requirements for insulin patients?"

Note: Make sure the structured worker (port 8001) and unstructured worker (port 8002) are running.
""")
    
    parser.add_argument('query', help='Medical query to process')
    
    parser.add_argument('--service-account', 
                       default='adsp-34002-ip09-team-2-e0cca2d396a9.json',
                       help='Path to Google Cloud service account JSON file (default: %(default)s)')
    
    parser.add_argument('--structured-url', 
                       default='http://localhost:8001',
                       help='Structured worker URL (default: %(default)s)')
    
    parser.add_argument('--unstructured-url', 
                       default='http://localhost:8002',
                       help='Unstructured worker URL (default: %(default)s)')
    
    parser.add_argument('--output', '-o',
                       choices=['json', 'pretty', 'answer-only'],
                       default='pretty',
                       help='Output format (default: %(default)s)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Show detailed information including LLM prompts and responses')
    
    parser.add_argument('--debug-llm',
                       action='store_true',
    # Check if service account file exists
    if not Path(args.service_account).exists():
        print(f"Warning: Service account file '{args.service_account}' not found.")
        print("Continuing with mock Gemini responses...")
    
    try:
        # Create orchestrator
        orchestrator = Orchestrator(
            service_account_path=args.service_account,
            structured_worker_url=args.structured_url,
            unstructured_worker_url=args.unstructured_url
        )
        
        # Only print processing info for non-JSON output
        if args.output != 'json':
            print(f"Processing query: {args.query}")
            print("-" * 50)
        
        # Execute query
        result = orchestrator.query(args.query)
        
        # Format output based on user preference
        if args.output == 'json':
            print(json.dumps(result, indent=2))
            
        elif args.output == 'answer-only':
            print(result.get('answer', 'No answer available'))
            
        else:  # pretty format
            print("ANSWER:")
            print(result.get('answer', 'No answer available'))
            
            if args.verbose:
                print("\nSOURCES:")
                structured_sources = result.get('structured_source', [])
                unstructured_sources = result.get('unstructured_source', [])
                
                if structured_sources:
                    print(f"  Structured Data: {', '.join(structured_sources)}")
                    
                if unstructured_sources:
                    print(f"  Medical Knowledge: {', '.join(unstructured_sources)}")
                    
                if 'error' in result:
                    print(f"\nERRORS: {result['error']}")
            
    except KeyboardInterrupt:
        print("\nQuery cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error processing query: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
