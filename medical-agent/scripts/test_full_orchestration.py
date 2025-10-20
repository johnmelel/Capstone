#!/usr/bin/env python3
"""
Full Orchestration Testing - A2A Clinical Retrieval System
Tests complete end-to-end flow: Orchestrator -> Workers -> Reasoning -> Output
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import subprocess
import time
import requests
import json
import argparse
from typing import List, Dict, Any


def start_workers_background(verbose: bool = False):
    """Start both workers in background"""
    if verbose:
        print("INFO: Starting both workers in background...")
    
    # Kill any existing workers
    subprocess.run(["pkill", "-f", "structured_worker|unstructured_worker"], capture_output=True)
    time.sleep(2)
    
    # Start workers
    structured_proc = subprocess.Popen([
        sys.executable, "agents/structured_worker/main.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    unstructured_proc = subprocess.Popen([
        sys.executable, "agents/unstructured_worker/main.py"  
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if verbose:
        print("INFO: Waiting for workers to initialize (8 seconds)...")
    time.sleep(8)  # Give time for Gemini initialization
    
    # Check if workers are ready
    try:
        r1 = requests.get("http://localhost:8001/health", timeout=5)
        r2 = requests.get("http://localhost:8002/health", timeout=5)
        if r1.status_code == 200 and r2.status_code == 200:
            if verbose:
                print("INFO: Both workers ready and responding")
            return structured_proc, unstructured_proc
        else:
            print(f"ERROR: Workers not responding - structured: {r1.status_code}, unstructured: {r2.status_code}")
            return None, None
    except Exception as e:
        print(f"ERROR: Failed to connect to workers: {e}")
        return None, None


def run_single_query(query: str, query_num: int, verbose: bool = False) -> bool:
    """Run a single orchestration test query"""
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"QUERY {query_num}: {query}")
        print(f"{'=' * 80}")
        print("Running full orchestration with detailed output...")
    else:
        print(f"\nQuery {query_num}: {query}")
    
    try:
        if verbose:
            # For verbose mode, enable LLM logging and run with verbose flags
            env = os.environ.copy()
            env["LLM_VERBOSE_LOGGING"] = "true"
            
            result = subprocess.run([
                sys.executable, "scripts/query_system.py", query, "--output", "pretty", "--verbose"
            ], capture_output=True, text=True, timeout=60, env=env)
            
            # Show all outputs in verbose mode
            print("\nORCHESTRATOR OUTPUT:")
            print("-" * 60)
            
            if result.stdout:
                print(result.stdout)
            
            # Show worker logs in verbose mode - these contain the orchestration details
            if result.stderr and result.stderr.strip():
                print("\nWORKER ORCHESTRATION LOGS:")
                print("-" * 60)
                # Filter out only the Gemini deprecation warnings for cleaner output
                stderr_lines = result.stderr.split('\n')
                filtered_stderr = []
                skip_deprecation = False
                
                for line in stderr_lines:
                    # Skip Gemini deprecation warnings
                    if 'UserWarning: This feature is deprecated as of June 24, 2025' in line:
                        skip_deprecation = True
                        continue
                    elif skip_deprecation and 'warning_logs.show_deprecation_warning()' in line:
                        skip_deprecation = False
                        continue
                    elif skip_deprecation:
                        continue
                    # Keep all other logs including worker initialization and processing
                    else:
                        filtered_stderr.append(line)
                
                if filtered_stderr and any(line.strip() for line in filtered_stderr):
                    print('\n'.join(filtered_stderr))
            
            print("-" * 60)
            
        else:
            # For non-verbose mode, run with JSON output for proper error detection
            result = subprocess.run([
                sys.executable, "scripts/query_system.py", query, "--output", "json"
            ], capture_output=True, text=True, timeout=60)
            
            # Only show the final clean JSON output
            if result.stdout:
                print(result.stdout.strip())
        
        # NEW VALIDATION LOGIC: Check actual response content, not just exit code
        success = _validate_query_response(result, query_num, verbose)
        
        if verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"QUERY {query_num} STATUS: {status}")
        
        return success
            
    except subprocess.TimeoutExpired:
        print(f"ERROR: Query {query_num} timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"ERROR: Query {query_num} exception: {e}")
        return False


def _validate_query_response(result: subprocess.CompletedProcess, query_num: int, verbose: bool = False) -> bool:
    """Validate query response by checking actual content, not just exit code"""
    
    # First check if subprocess failed completely
    if result.returncode != 0:
        if not verbose:
            print(f"ERROR: Process failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"Details: {result.stderr.strip()}")
        return False
    
    # Check if we have stdout output
    if not result.stdout or not result.stdout.strip():
        if not verbose:
            print("ERROR: No output received")
        return False
    
    try:
        # Try to parse JSON response for detailed error checking
        response_data = json.loads(result.stdout.strip())
        
        # Check for explicit error field
        if 'error' in response_data:
            error_msg = response_data['error']
            if not verbose:
                print(f"ERROR: {error_msg}")
            return False
        
        # Check answer field for error messages
        answer = response_data.get('answer', '')
        if not answer:
            if not verbose:
                print("ERROR: Empty answer received")
            return False
        
        # Check for database errors in the answer
        error_patterns = [
            "Database Error:",
            "Could not generate SQL query",
            "Transport error: timed out",
            "SQL execution failed",
            "Processing error:",
            "Error processing query:"
        ]
        
        answer_lower = answer.lower()
        for pattern in error_patterns:
            if pattern.lower() in answer_lower:
                error_type = _categorize_error(pattern)
                if not verbose:
                    print(f"ERROR: {error_type} - {pattern}")
                return False
        
        # Check for suspicious data patterns
        if _has_data_quality_issues(response_data, query_num):
            if not verbose:
                print("ERROR: Data quality issues detected")
            return False
        
        # If we get here, the query appears to have succeeded
        return True
        
    except json.JSONDecodeError:
        # If not JSON, check stdout for error patterns in plain text
        stdout_lower = result.stdout.lower()
        error_patterns = [
            "error:",
            "failed:",
            "timeout",
            "exception:",
            "could not",
            "unable to"
        ]
        
        for pattern in error_patterns:
            if pattern in stdout_lower:
                if not verbose:
                    print(f"ERROR: Text output contains error pattern: {pattern}")
                return False
        
        # Non-JSON response but no obvious errors - assume success for now
        return True


def _categorize_error(error_pattern: str) -> str:
    """Categorize error types for better reporting"""
    if "Could not generate SQL query" in error_pattern:
        return "SQL Generation Failure"
    elif "timed out" in error_pattern:
        return "Database Timeout"
    elif "Transport error" in error_pattern:
        return "MCP Transport Error"
    elif "SQL execution failed" in error_pattern:
        return "SQL Execution Error"
    elif "Processing error" in error_pattern:
        return "Worker Processing Error"
    else:
        return "Unknown Error"


def _has_data_quality_issues(response_data: Dict[str, Any], query_num: int) -> bool:
    """Check for data quality issues (future dates are expected in mock data)"""
    answer = response_data.get('answer', '')
    
    # NOTE: Future dates (2201) are expected in mock data - not considered an error
    
    # For admission queries, check if we got reasonable responses
    if query_num in [1, 4, 7, 10]:  # Known admission queries
        # If answer is very short or generic, might indicate no real data
        if len(answer.strip()) < 30:
            return True
        
        # Check for generic "no results" type responses only if they seem suspicious
        no_data_patterns = [
            "no data available",
            "database connection failed",
            "query execution failed"
        ]
        
        answer_lower = answer.lower()
        for pattern in no_data_patterns:
            if pattern in answer_lower:
                return True
    
    return False


def get_test_queries() -> List[str]:
    """Get the standard set of test queries"""
    return [
        "Show hospital admissions from past 12 hours",
        "Get patients who took insulin in last 24 hours", 
        "List 5 medications people took in past 12 hours",
        "Recent admissions in last 24 hours",
        "Find patients who received warfarin",
        "Name 8 drugs from past 24 hours",
        "Hospital admissions in last 48 hours",
        "Patients taking amphetamine recently",
        "Show recent medications used",
        "Get recent hospital admissions"
    ]


def main():
    """Run full orchestration testing with configurable options"""
    parser = argparse.ArgumentParser(
        description='A2A Clinical Retrieval System - Full Orchestration Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                    # Run all queries with minimal output
  %(prog)s --verbose          # Run all queries with detailed output
  %(prog)s --test 3           # Run only test query #3
  %(prog)s --test 1 --verbose # Run test #1 with verbose output
        '''
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed orchestration flow'
    )
    
    parser.add_argument(
        '--test', '-t',
        type=int,
        metavar='N',
        help='Run specific test number (1-10) instead of all tests'
    )
    
    args = parser.parse_args()
    
    test_queries = get_test_queries()
    
    # Validate test number
    if args.test is not None:
        if args.test < 1 or args.test > len(test_queries):
            print(f"ERROR: Test number must be between 1 and {len(test_queries)}")
            sys.exit(1)
    
    # Print header
    print("A2A Clinical Retrieval System - Full Orchestration Testing")
    print("=" * 65)
    
    if args.verbose:
        print("Testing complete end-to-end flow:")
        print("1. Orchestrator task planning")
        print("2. Structured worker (SQL generation)")
        print("3. Unstructured worker (medical context)")  
        print("4. Orchestrator Gemini reasoning")
        print("5. Final JSON output")
        print("=" * 65)
    
    # Start workers
    structured_proc, unstructured_proc = start_workers_background(args.verbose)
    
    if not structured_proc or not unstructured_proc:
        print("ERROR: Failed to start workers")
        sys.exit(1)
    
    try:
        successful_queries = 0
        total_queries = 0
        
        if args.test is not None:
            # Run single test
            query = test_queries[args.test - 1]
            success = run_single_query(query, args.test, args.verbose)
            total_queries = 1
            if success:
                successful_queries = 1
        else:
            # Run all tests
            for i, query in enumerate(test_queries, 1):
                success = run_single_query(query, i, args.verbose)
                total_queries += 1
                if success:
                    successful_queries += 1
                
                # Small delay between queries (except for the last one)
                if i < len(test_queries) and not args.verbose:
                    time.sleep(1)
                elif args.verbose and i < len(test_queries):
                    time.sleep(2)
        
        # Print summary
        print("\n" + "=" * 65)
        print(f"TESTING COMPLETE: {successful_queries}/{total_queries} queries successful")
        
        if successful_queries == total_queries:
            print("STATUS: All tests passed")
            exit_code = 0
        else:
            print(f"STATUS: {total_queries - successful_queries} tests failed")
            exit_code = 1
        
        if args.verbose:
            print("\nComplete orchestration flow demonstrated for each query:")
            print("- Task planning by orchestrator")
            print("- Worker responses (structured + unstructured)")
            print("- Orchestrator reasoning over both sources")
            print("- Final JSON output")
        
    except KeyboardInterrupt:
        print("\nTesting cancelled by user")
        exit_code = 1
    finally:
        # Cleanup
        if args.verbose:
            print("\nCleaning up workers...")
        
        if structured_proc:
            structured_proc.terminate()
        if unstructured_proc:
            unstructured_proc.terminate()
        
        subprocess.run(["pkill", "-f", "structured_worker|unstructured_worker"], 
                      capture_output=True)
        
        if args.verbose:
            print("Cleanup complete")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
