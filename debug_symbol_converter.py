"""
Debug Symbol Converter
Test the symbol converter in isolation to identify the issue
"""

import sys
import os

def test_old_converter():
    """Test the old symbol converter"""
    print("🔍 Testing OLD Symbol Converter")
    print("=" * 40)
    
    try:
        # Add speech2symbol to path
        sys.path.append('speech2symbol')
        from postprocessing.symbol_converter import ContextAwareSymbolConverter
        
        converter = ContextAwareSymbolConverter(use_spacy=True)
        print("✅ Old converter created successfully")
        
        # Test conversion
        result, metadata = converter.convert_text("two plus three")
        print(f"Old converter result: '{result}'")
        print(f"Has conversion_rules: {hasattr(converter, 'conversion_rules')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Old converter failed: {e}")
        return False

def test_new_converter():
    """Test the new symbol converter"""
    print("\n🔍 Testing NEW Symbol Converter")
    print("=" * 40)
    
    try:
        from ml_models.symbol_converter import create_advanced_converter
        
        converter = create_advanced_converter()
        print("✅ New converter created successfully")
        
        # Test conversion
        result, metadata = converter.convert("two plus three")
        print(f"New converter result: '{result}'")
        print(f"Has conversion_rules: {hasattr(converter, 'conversion_rules')}")
        
        return True
        
    except Exception as e:
        print(f"❌ New converter failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_asr_pipeline():
    """Test ASR pipeline creation"""
    print("\n🔍 Testing ASR Pipeline")
    print("=" * 40)
    
    try:
        from ml_models.asr_pipeline import create_api_pipeline
        
        pipeline = create_api_pipeline()
        print("✅ ASR pipeline created successfully")
        
        # Test text processing
        result = pipeline.process_text("two plus three")
        print(f"Pipeline result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ ASR pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🐛 Symbol Converter Debug Test")
    print("=" * 50)
    
    # Test old converter
    old_works = test_old_converter()
    
    # Test new converter
    new_works = test_new_converter()
    
    # Test ASR pipeline
    pipeline_works = test_asr_pipeline()
    
    print("\n" + "=" * 50)
    print("📊 Results:")
    print(f"  Old Converter: {'✅' if old_works else '❌'}")
    print(f"  New Converter: {'✅' if new_works else '❌'}")
    print(f"  ASR Pipeline: {'✅' if pipeline_works else '❌'}")
    
    if not new_works:
        print("\n🔧 Fixing new converter...")
        fix_new_converter()

def fix_new_converter():
    """Try to fix the new converter"""
    print("🔧 Attempting to fix new converter...")
    
    try:
        # Check if the file exists and has the right content
        with open('ml_models/symbol_converter.py', 'r') as f:
            content = f.read()
            
        if 'conversion_rules' not in content:
            print("❌ conversion_rules not found in file")
            return
            
        if 'def _create_conversion_rules' not in content:
            print("❌ _create_conversion_rules method not found")
            return
            
        print("✅ File content looks correct")
        
        # Try importing again with explicit path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "symbol_converter", 
            "ml_models/symbol_converter.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        converter = module.create_advanced_converter()
        print("✅ Direct import works!")
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")

if __name__ == "__main__":
    main() 