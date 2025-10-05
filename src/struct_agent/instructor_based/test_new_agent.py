from __future__ import annotations

import traceback
from typing import List, Dict, Any, Tuple
from datetime import datetime

from struct_agent.instructor_based.new_agent import run_react_loop
from struct_agent.instructor_based.client_manager import build_client

test_queries = [
    "Find the current CEO of OpenAI, then search for their alma mater, and finally find the title and date of their most recent public talk.",
    "Find the current president of France, then look up their spouse's name, and finally find the spouse's profession.",
    "Find the current world record holder for the 100m sprint, then find their country of birth, and finally find their personal best time.",
    "Find the current Prime Minister of the United Kingdom, then find which political party they lead, and then find when that party was founded.",
    "Find the current richest person in the world according to Forbes, then find the name of their primary company, and finally find the year that company was founded.",
    "Find the current Secretary-General of the United Nations, then find their country of origin, and finally find the year they began their term.",
    "Find the current CEO of Google, then find which university they graduated from, and finally find when they joined Google.",
    "Find the capital city of Japan, then find who the current mayor of that city is, and then find when they took office.",
    "Find who won the most recent FIFA World Cup, then find the team's coach, and finally find which country that coach was born in.",
    "Find the current U.S. Secretary of State, then find their previous government position, and finally find when they were appointed Secretary of State."
]

def run_test_suite() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run the agent test suite on all queries.

    Returns:
        Tuple of (summary_stats, detailed_results)
    """
    print("Starting agent test suite...")
    print(f"Total queries to test: {len(test_queries)}")
    print("=" * 80)

    # Initialize client
    try:
        client = build_client(use_lmstudio=True)
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return {"error": "Client initialization failed"}, []

    # Track results
    results = []
    success_count = 0
    failure_count = 0

    # Run each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)} ---")
        print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        result = {
            "query": query,
            "query_number": i,
            "success": False,
            "answer": None,
            "error": None,
            "execution_time": None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            start_time = datetime.now()
            answer = run_react_loop(query, client)
            end_time = datetime.now()

            result["success"] = True
            result["answer"] = answer
            result["execution_time"] = (end_time - start_time).total_seconds()

            success_count += 1
            print(f"✓ Success ({result['execution_time']:.2f}s)")

        except Exception as e:
            end_time = datetime.now()
            result["error"] = str(e)
            result["execution_time"] = (end_time - start_time).total_seconds()
            result["traceback"] = traceback.format_exc()

            failure_count += 1
            print(f"✗ Failed: {e}")

        results.append(result)

    # Create summary
    summary = {
        "total_queries": len(test_queries),
        "successful_queries": success_count,
        "failed_queries": failure_count,
        "success_rate": (success_count / len(test_queries)) * 100,
        "total_execution_time": sum(r["execution_time"] for r in results if r["execution_time"]),
        "average_execution_time": sum(r["execution_time"] for r in results if r["execution_time"]) / len(results) if results else 0
    }

    return summary, results

def print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary of test results."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total queries:     {summary['total_queries']}")
    print(f"Successful:        {summary['successful_queries']}")
    print(f"Failed:            {summary['failed_queries']}")
    print(f"Success rate:      {summary['success_rate']:.1f}%")
    print(f"Total time:        {summary['total_execution_time']:.2f}s")
    print(f"Average time:      {summary['average_execution_time']:.2f}s")

def print_detailed_results(results: List[Dict[str, Any]]) -> None:
    """Print detailed Q/A pairs for all test results."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n--- Query {result['query_number']} ---")
        print(f"Status: {'✓ SUCCESS' if result['success'] else '✗ FAILED'}")
        print(f"Time: {result['execution_time']:.2f}s")
        print(f"Query: {result['query']}")

        if result['success']:
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")
            if 'traceback' in result:
                print("Traceback:")
                # Print only last few lines of traceback for readability
                tb_lines = result['traceback'].split('\n')
                for line in tb_lines[-5:]:
                    if line.strip():
                        print(f"  {line}")

        print("-" * 60)

def main():
    """Main test execution function."""
    print("Agent Test Suite")
    print("=" * 80)

    # Run tests
    summary, results = run_test_suite()

    # Print summary
    print_summary(summary)

    # Print detailed results
    print_detailed_results(results)

    # Return for programmatic use if needed
    return summary, results

if __name__ == "__main__":
    main()
