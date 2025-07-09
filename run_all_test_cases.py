#!/usr/bin/env python3
"""
🎯 Complete Test Case Runner for Speech2Symbol Pipeline
Runs all test cases from comprehensive_test_cases.md and shows detailed results
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import subprocess
import re

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import demo_conversion
except ImportError:
    print("❌ Error: Could not import main.py. Make sure you're in the project root directory.")
    sys.exit(1)

class TestCaseRunner:
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "categories": {},
            "detailed_results": []
        }
        
        # Define all test cases from the document
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> Dict[str, List[Dict]]:
        """Load all test cases organized by category"""
        return {
            "Basic Math": [
                {"input": "Two plus three equals five", "expected": "Two + three = five", "id": 1},
                {"input": "Ten minus four is six", "expected": "Ten - four is six", "id": 2},
                {"input": "Five times seven equals thirty five", "expected": "Five × seven = thirty five", "id": 3},
                {"input": "Twenty divided by four is five", "expected": "Twenty ÷ four is five", "id": 4},
                {"input": "X is greater than zero", "expected": "X > zero", "id": 5},
                {"input": "Y is less than ten", "expected": "Y < ten", "id": 6},
                {"input": "A is greater than or equal to B", "expected": "A ≥ B", "id": 7},
                {"input": "C is less than or equal to D", "expected": "C ≤ D", "id": 8},
            ],
            "Advanced Math": [
                {"input": "X squared plus Y squared equals Z squared", "expected": "X² + Y² = Z²", "id": 9},
                {"input": "Five percent of hundred is five", "expected": "Five % of hundred is five", "id": 10},
                {"input": "Two to the power of three equals eight", "expected": "Two^three = eight", "id": 11},
                {"input": "The result is approximately equal to ten", "expected": "The result ≈ ten", "id": 12},
            ],
            "Basic Punctuation": [
                {"input": "Hello comma world period", "expected": "Hello, world.", "id": 13},
                {"input": "How are you question mark", "expected": "How are you?", "id": 14},
                {"input": "That's amazing exclamation mark", "expected": "That's amazing!", "id": 15},
                {"input": "First semicolon second semicolon third", "expected": "First; second; third", "id": 16},
                {"input": "Note colon this is important", "expected": "Note: this is important", "id": 17},
                {"input": "I don't apostrophe t know", "expected": "I don't know", "id": 18},
            ],
            "Advanced Punctuation": [
                {"input": "He said quote hello world quote", "expected": 'He said "hello world"', "id": 19},
                {"input": "She said quotation mark goodbye quotation mark", "expected": 'She said "goodbye"', "id": 20},
                {"input": "List items comma separated by comma commas", "expected": "List items, separated by, commas", "id": 21},
                {"input": "End of sentence period Start new sentence", "expected": "End of sentence. Start new sentence", "id": 22},
            ],
            "Email Addresses": [
                {"input": "Send email to john at gmail dot com", "expected": "Send email to john@gmail.com", "id": 23},
                {"input": "Contact us at support at company dot org", "expected": "Contact us at support@company.org", "id": 24},
                {"input": "My email is user dot name at domain dot co dot uk", "expected": "My email is user.name@domain.co.uk", "id": 25},
                {"input": "Write to admin at sign company dot net", "expected": "Write to admin@company.net", "id": 26},
            ],
            "Web URLs": [
                {"input": "Visit www dot google dot com", "expected": "Visit www.google.com", "id": 27},
                {"input": "Go to https colon slash slash website dot com", "expected": "Go to https://website.com", "id": 28},
                {"input": "Check out blog dot example dot org slash articles", "expected": "Check out blog.example.org/articles", "id": 29},
            ],
            "Money Amounts": [
                {"input": "It costs five dollars", "expected": "It costs five $", "id": 30},
                {"input": "Price is twenty five dollars and fifty cents", "expected": "Price is twenty five $ and fifty ¢", "id": 31},
                {"input": "Total amount is one hundred euros", "expected": "Total amount is one hundred €", "id": 32},
                {"input": "The cost is fifty pounds sterling", "expected": "The cost is fifty £", "id": 33},
                {"input": "Budget is two thousand dollars maximum", "expected": "Budget is two thousand $ maximum", "id": 34},
            ],
            "Percentages": [
                {"input": "Interest rate is five percent", "expected": "Interest rate is five %", "id": 35},
                {"input": "Discount of twenty percent off", "expected": "Discount of twenty % off", "id": 36},
                {"input": "Tax is eight point five percent", "expected": "Tax is eight point five %", "id": 37},
            ],
            "Social Media": [
                {"input": "Follow us at hashtag company name", "expected": "Follow us at #company name", "id": 38},
                {"input": "Tag me at sign username", "expected": "Tag me @username", "id": 39},
                {"input": "Use hashtag trending topic", "expected": "Use #trending topic", "id": 40},
                {"input": "Mention at sign john in the post", "expected": "Mention @john in the post", "id": 41},
            ],
            "Programming & Technical": [
                {"input": "Variable name equals value", "expected": "Variable name = value", "id": 42},
                {"input": "Function name ampersand operator", "expected": "Function name & operator", "id": 43},
                {"input": "File name underscore version dot txt", "expected": "File name_version.txt", "id": 44},
                {"input": "Line break asterisk separator", "expected": "Line break * separator", "id": 45},
                {"input": "Comment hash sign important note", "expected": "Comment # important note", "id": 46},
            ],
            "Parentheses": [
                {"input": "Open parenthesis X plus Y close parenthesis", "expected": "(X + Y)", "id": 47},
                {"input": "Left parenthesis A minus B right parenthesis equals C", "expected": "(A - B) = C", "id": 48},
                {"input": "Calculate open paren five times three close paren", "expected": "Calculate (five × three)", "id": 49},
            ],
            "Brackets & Braces": [
                {"input": "Array left bracket zero right bracket", "expected": "Array[0]", "id": 50},
                {"input": "Object left brace key colon value right brace", "expected": "Object{key: value}", "id": 51},
                {"input": "List left bracket one comma two comma three right bracket", "expected": "List[1, 2, 3]", "id": 52},
            ],
            "Multiple Symbols": [
                {"input": "A plus B equals C comma D minus E equals F", "expected": "A + B = C, D - E = F", "id": 53},
                {"input": "Email colon john at company dot com semicolon urgent", "expected": "Email: john@company.com; urgent", "id": 54},
                {"input": "Price left parenthesis twenty dollars right parenthesis includes tax", "expected": "Price ($20) includes tax", "id": 55},
            ],
            "Context-Dependent": [
                {"input": "Doctor Smith has a PhD period", "expected": "Doctor Smith has a PhD.", "id": 56},
                {"input": "The temperature is minus five degrees", "expected": "The temperature is -5 degrees", "id": 57},
                {"input": "Account balance is plus hundred dollars", "expected": "Account balance is +$100", "id": 58},
            ],
            "Mixed Content": [
                {"input": "Send fifty percent of two hundred dollars to john at gmail dot com", "expected": "Send 50% of $200 to john@gmail.com", "id": 59},
                {"input": "Question mark Did you pay the twenty dollar fee exclamation mark", "expected": "? Did you pay the $20 fee !", "id": 60},
            ],
            "Alternative Pronunciations": [
                {"input": "Two add three equals five", "expected": "Two + three = five", "id": 61},
                {"input": "Ten take away four is six", "expected": "Ten - four is six", "id": 62},
                {"input": "Five multiply by seven", "expected": "Five × seven", "id": 63},
                {"input": "Twenty divide by four", "expected": "Twenty ÷ four", "id": 64},
                {"input": "At symbol gmail dot com", "expected": "@gmail.com", "id": 65},
                {"input": "Hash tag trending", "expected": "#trending", "id": 66},
                {"input": "Exclamation point amazing", "expected": "amazing!", "id": 67},
                {"input": "Question symbol correct", "expected": "correct?", "id": 68},
            ],
            "Complex Sentences": [
                {"input": "Calculate left parenthesis A plus B right parenthesis times C equals D comma then send result to admin at company dot com", 
                 "expected": "Calculate (A + B) × C = D, then send result to admin@company.com", "id": 69},
                {"input": "The price is twenty five dollars and thirty cents comma tax is eight percent comma total is twenty seven dollars and forty two cents period", 
                 "expected": "The price is twenty five $ and thirty ¢, tax is eight %, total is twenty seven $ and forty two ¢.", "id": 70},
                {"input": "If X is greater than zero and Y is less than ten comma then Z equals X plus Y", 
                 "expected": "If X > zero and Y < ten, then Z = X + Y", "id": 71},
                {"input": "Contact details colon Name is John Smith comma email is john dot smith at company dot org comma phone is plus one two three four five six seven eight nine zero", 
                 "expected": "Contact details: Name is John Smith, email is john.smith@company.org, phone is +1234567890", "id": 72},
            ],
            "Stress Test Cases": [
                {"input": "Multiple commas comma comma comma in sequence", "expected": "Multiple commas, comma, comma in sequence", "id": 73},
                {"input": "Repeated periods period period period", "expected": "Repeated periods. period. period", "id": 74},
                {"input": "Mixed quotes quote single quote double quote", "expected": 'Mixed quotes " single quote \' double quote', "id": 75},
                {"input": "Nested parentheses open paren open paren A close paren close paren", "expected": "Nested parentheses ((A))", "id": 76},
                {"input": "All symbols plus minus times divide equals percent dollar at sign hashtag", 
                 "expected": "All symbols + - × ÷ = % $ @ #", "id": 77},
            ],
            "Speed Test": [
                {"input": "Quick math two plus two equals four times five equals twenty", "expected": "Quick math two + two = four × five = twenty", "id": 78},
                {"input": "Rapid punctuation hello comma world period how are you question mark great exclamation mark", 
                 "expected": "Rapid punctuation hello, world. how are you? great!", "id": 79},
                {"input": "Fast email contact at company dot com or support at help dot org", 
                 "expected": "Fast email contact@company.com or support@help.org", "id": 80},
            ]
        }

    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case and return results"""
        try:
            start_time = time.time()
            result, metadata = demo_conversion(test_case["input"])
            end_time = time.time()
            
            # Clean up the result for comparison
            result_clean = result.strip()
            expected_clean = test_case["expected"].strip()
            
            # Check if test passed
            passed = result_clean == expected_clean
            
            return {
                "id": test_case["id"],
                "input": test_case["input"],
                "expected": expected_clean,
                "actual": result_clean,
                "passed": passed,
                "execution_time": round(end_time - start_time, 4),
                "category": self._get_category_for_test(test_case["id"])
            }
        except Exception as e:
            return {
                "id": test_case["id"],
                "input": test_case["input"],
                "expected": test_case["expected"],
                "actual": f"ERROR: {str(e)}",
                "passed": False,
                "execution_time": 0,
                "category": self._get_category_for_test(test_case["id"]),
                "error": str(e)
            }

    def _get_category_for_test(self, test_id: int) -> str:
        """Get category name for a test ID"""
        for category, tests in self.test_cases.items():
            for test in tests:
                if test["id"] == test_id:
                    return category
        return "Unknown"

    def run_all_tests(self, verbose: bool = True) -> Dict:
        """Run all test cases and return comprehensive results"""
        print("🚀 Starting comprehensive test run...")
        print("=" * 80)
        
        start_time = time.time()
        
        for category, tests in self.test_cases.items():
            print(f"\n📂 Testing Category: {category}")
            print("-" * 50)
            
            category_passed = 0
            category_total = len(tests)
            
            for test in tests:
                result = self.run_single_test(test)
                self.results["detailed_results"].append(result)
                
                if result["passed"]:
                    category_passed += 1
                    self.results["passed"] += 1
                    if verbose:
                        print(f"✅ Test {result['id']:2d}: PASSED")
                else:
                    self.results["failed"] += 1
                    if verbose:
                        print(f"❌ Test {result['id']:2d}: FAILED")
                        print(f"   Input: {result['input']}")
                        print(f"   Expected: {result['expected']}")
                        print(f"   Actual: {result['actual']}")
                        if "error" in result:
                            print(f"   Error: {result['error']}")
                        print()
                
                self.results["total_tests"] += 1
            
            # Category summary
            success_rate = (category_passed / category_total) * 100
            self.results["categories"][category] = {
                "passed": category_passed,
                "total": category_total,
                "success_rate": round(success_rate, 2)
            }
            
            print(f"📊 {category}: {category_passed}/{category_total} passed ({success_rate:.1f}%)")
        
        total_time = time.time() - start_time
        self.results["total_execution_time"] = round(total_time, 2)
        
        return self.results

    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("🎯 COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⏱️  Total Execution Time: {results['total_execution_time']} seconds")
        report.append(f"📊 Overall Success Rate: {(results['passed'] / results['total_tests']) * 100:.1f}%")
        report.append("")
        
        # Overall summary
        report.append("📈 OVERALL SUMMARY")
        report.append("-" * 30)
        report.append(f"✅ Passed: {results['passed']}")
        report.append(f"❌ Failed: {results['failed']}")
        report.append(f"📋 Total: {results['total_tests']}")
        report.append("")
        
        # Category breakdown
        report.append("📂 CATEGORY BREAKDOWN")
        report.append("-" * 30)
        for category, stats in results["categories"].items():
            status = "🟢" if stats["success_rate"] >= 90 else "🟡" if stats["success_rate"] >= 70 else "🔴"
            report.append(f"{status} {category}: {stats['passed']}/{stats['total']} ({stats['success_rate']}%)")
        report.append("")
        
        # Failed tests details
        failed_tests = [r for r in results["detailed_results"] if not r["passed"]]
        if failed_tests:
            report.append("❌ FAILED TESTS DETAILS")
            report.append("-" * 30)
            for test in failed_tests:
                report.append(f"Test {test['id']:2d} ({test['category']}):")
                report.append(f"  Input: {test['input']}")
                report.append(f"  Expected: {test['expected']}")
                report.append(f"  Actual: {test['actual']}")
                if "error" in test:
                    report.append(f"  Error: {test['error']}")
                report.append("")
        
        # Performance analysis
        report.append("⚡ PERFORMANCE ANALYSIS")
        report.append("-" * 30)
        execution_times = [r["execution_time"] for r in results["detailed_results"] if "execution_time" in r]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            report.append(f"Average execution time: {avg_time:.4f}s")
            report.append(f"Fastest test: {min_time:.4f}s")
            report.append(f"Slowest test: {max_time:.4f}s")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 30)
        overall_rate = (results['passed'] / results['total_tests']) * 100
        
        if overall_rate >= 90:
            report.append("🎉 Excellent! Your pipeline is working very well.")
            report.append("   Consider adding more edge cases for robustness.")
        elif overall_rate >= 70:
            report.append("👍 Good performance! Focus on improving failed categories.")
            report.append("   Review the failed tests above for patterns.")
        else:
            report.append("⚠️  Needs improvement. Focus on basic functionality first.")
            report.append("   Check your symbol conversion logic.")
        
        # Show worst performing categories
        worst_categories = sorted(results["categories"].items(), 
                                key=lambda x: x[1]["success_rate"])[:3]
        if worst_categories:
            report.append("")
            report.append("🔧 PRIORITY IMPROVEMENTS:")
            for category, stats in worst_categories:
                if stats["success_rate"] < 90:
                    report.append(f"   • {category}: {stats['success_rate']}% success rate")
        
        return "\n".join(report)

    def save_results(self, results: Dict, filename: str = "test_results.json"):
        """Save detailed results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed results saved to {filename}")

def main():
    """Main function to run all tests"""
    print("🎯 Speech2Symbol Comprehensive Test Runner")
    print("=" * 50)
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found. Make sure you're in the project root directory.")
        return
    
    # Initialize test runner
    runner = TestCaseRunner()
    
    # Run all tests
    results = runner.run_all_tests(verbose=True)
    
    # Generate and display report
    report = runner.generate_report(results)
    print("\n" + "=" * 80)
    print(report)
    
    # Save detailed results
    runner.save_results(results)
    
    # Save report to file
    with open("test_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    print("📄 Report saved to test_report.txt")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 Test run completed!")
    print(f"📊 Overall: {results['passed']}/{results['total_tests']} tests passed")
    print(f"📈 Success Rate: {(results['passed'] / results['total_tests']) * 100:.1f}%")
    
    if results['failed'] > 0:
        print(f"🔧 {results['failed']} tests need attention - check test_report.txt for details")

if __name__ == "__main__":
    main() 