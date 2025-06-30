#!/usr/bin/env python3
"""
Quick Batch Test - Selected Test Cases
Tests key examples from our comprehensive test list
"""

import subprocess
import sys
from pathlib import Path

# Selected test cases from comprehensive_test_cases.md
QUICK_TEST_CASES = [
    # Basic Math (Cases 1-4)
    "Two plus three equals five",
    "Ten minus four is six", 
    "Five times seven equals thirty five",
    "Twenty divided by four is five",
    
    # Basic Punctuation (Cases 13-15)
    "Hello comma world period",
    "How are you question mark",
    "That's amazing exclamation mark",
    
    # Email & Web (Cases 23, 27)
    "Send email to john at gmail dot com",
    "Visit www dot google dot com",
    
    # Currency (Cases 30, 35)
    "It costs five dollars",
    "Interest rate is five percent",
    
    # Special Characters (Cases 38, 39)
    "Follow us at hashtag company name",
    "Tag me at sign username",
    
    # Edge Cases (Cases 53, 59)
    "A plus B equals C comma D minus E equals F",
    "Send fifty percent of two hundred dollars to john at gmail dot com",
    
    # Stress Test (Case 78)
    "Quick math two plus two equals four times five equals twenty",
    
    # Alternative Pronunciations (Cases 61-62)
    "Two add three equals five",
    "Ten take away four is six",
]

def run_test_case(text, case_number):
    """Run a single test case"""
    print(f"\n🧪 **Test Case {case_number}:**")
    print(f"📝 Input: '{text}'")
    
    try:
        # Run the demo command
        result = subprocess.run([
            sys.executable, "main.py", "demo", "--text", text
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            # Print full output for debugging and show conversions
            output = result.stdout.strip()
            print("📤 **Output:**")
            
            # Look for conversion lines
            lines = output.split('\n')
            conversion_found = False
            
            for line in lines:
                if "Converted text:" in line:
                    converted = line.split("Converted text:")[-1].strip().replace("'", "")
                    print(f"   🔄 Result: {converted}")
                    conversion_found = True
                elif "INFO:__main__:" in line and "->" in line:
                    conversion = line.split("INFO:__main__:")[-1].strip()
                    print(f"   ✅ {conversion}")
                    conversion_found = True
            
            if not conversion_found:
                print("   ❌ No conversions detected")
                # Show some output for debugging
                for line in lines[-3:]:  # Show last 3 lines
                    if line.strip():
                        print(f"   🔍 {line}")
        else:
            print(f"❌ **Error:** {result.stderr}")
            
    except Exception as e:
        print(f"❌ **Exception:** {e}")
    
    print("-" * 60)

def run_all_tests():
    """Run all test cases"""
    print("🎯 **Quick Batch Test - Speech-to-Symbol Pipeline**")
    print("=" * 60)
    
    for i, test_case in enumerate(QUICK_TEST_CASES, 1):
        run_test_case(test_case, i)
    
    print(f"\n🏆 **Batch Test Complete!**")
    print(f"📊 **Total Test Cases:** {len(QUICK_TEST_CASES)}")
    print(f"📋 **Full Test List:** See comprehensive_test_cases.md")

def run_audio_test_example():
    """Show example of how to test with audio"""
    print("\n🎵 **Audio Testing Example:**")
    print("=" * 40)
    print("1. Record yourself saying: 'Two plus three equals five'")
    print("2. Save as test.mp3")
    print("3. Run: python main.py audio --file test.mp3")
    print("4. Expected: 'Two + three = five'")
    print("\n💡 **Tips:**")
    print("- Speak clearly and at normal pace")
    print("- Use natural pauses")
    print("- Test simple cases first")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "audio":
            run_audio_test_example()
        elif sys.argv[1] == "single":
            if len(sys.argv) > 2:
                test_text = " ".join(sys.argv[2:])
                run_test_case(test_text, 1)
            else:
                print("Usage: python quick_test_batch.py single 'your test text'")
        else:
            print("Usage: python quick_test_batch.py [audio|single 'text']")
    else:
        run_all_tests() 