"""
Demo Script: ML Pipeline Usage
Demonstrates how to use the refactored Speech-to-Symbol ML Pipeline
"""

import os
import sys
from pathlib import Path

# Add ml_models to path
sys.path.append(str(Path(__file__).parent))

def demo_symbol_converter():
    """Demonstrate standalone symbol converter"""
    print("🔄 Symbol Converter Demo")
    print("=" * 40)
    
    from ml_models.symbol_converter import create_advanced_converter, create_basic_converter
    
    # Create converter instances
    basic_converter = create_basic_converter()
    advanced_converter = create_advanced_converter()
    
    test_texts = [
        "two plus three equals five",
        "ten percent of fifty",
        "send email to john at company dot com",
        "list items comma separated by comma",
        "is this correct question mark",
        "A is greater than or equal to B"
    ]
    
    print("\n📝 Basic Converter Results:")
    for text in test_texts:
        result, metadata = basic_converter.convert(text)
        print(f"  '{text}' → '{result}' ({metadata['total_conversions']} conversions)")
    
    print("\n🧠 Advanced Converter Results (with NLP):")
    for text in test_texts:
        result, metadata = advanced_converter.convert(text)
        confidence = metadata.get('average_confidence', 0)
        print(f"  '{text}' → '{result}' ({metadata['total_conversions']} conversions, {confidence:.2f} confidence)")
    
    # Show converter statistics
    print(f"\n📊 Converter Stats:")
    stats = advanced_converter.get_statistics()
    print(f"  Total Rules: {stats['total_rules']}")
    print(f"  Rule Categories: {stats['rule_categories']}")
    print(f"  spaCy Enabled: {stats['spacy_enabled']}")

def demo_whisper_wrapper():
    """Demonstrate Whisper wrapper (text simulation)"""
    print("\n🎤 Whisper Wrapper Demo")
    print("=" * 40)
    
    try:
        from ml_models.whisper_wrapper import create_base_whisper
        
        # Create Whisper instance
        print("📥 Loading Whisper model...")
        whisper = create_base_whisper("small", use_gpu=False)
        
        print("✅ Whisper loaded successfully!")
        print(f"Model Info: {whisper.get_model_info()}")
        
        # Simulate audio processing (would normally use actual audio files)
        print("\n🎵 Audio Processing Simulation:")
        print("  (In real usage, you would pass audio file paths)")
        print("  Example: result = whisper.transcribe('audio.wav')")
        
    except Exception as e:
        print(f"⚠️  Whisper demo skipped: {e}")
        print("   (This is normal if Whisper dependencies aren't fully installed)")

def demo_asr_pipeline():
    """Demonstrate complete ASR pipeline"""
    print("\n🔄 Complete ASR Pipeline Demo")
    print("=" * 40)
    
    try:
        from ml_models.asr_pipeline import create_basic_pipeline, create_api_pipeline
        
        print("📥 Creating pipeline...")
        
        # Create a basic pipeline (will fallback to text-only if Whisper fails)
        try:
            pipeline = create_api_pipeline()
            print("✅ Full pipeline created successfully!")
            pipeline_type = "Full (ASR + Symbol Conversion)"
        except Exception as e:
            print(f"⚠️  Full pipeline failed: {e}")
            print("🔄 Creating text-only pipeline...")
            # Create text-only version by mocking ASR
            from ml_models.symbol_converter import create_advanced_converter
            from ml_models.asr_pipeline import ASRPipeline
            
            # Create a minimal pipeline for text processing
            converter = create_advanced_converter()
            pipeline = ASRPipeline(
                whisper_model="mock",
                symbol_converter=converter
            )
            # Override the ASR component for text-only demo
            pipeline.asr = None
            pipeline_type = "Text-Only (Symbol Conversion)"
        
        print(f"Pipeline Type: {pipeline_type}")
        
        # Demonstrate text processing
        print("\n📝 Text Processing Examples:")
        test_texts = [
            "two plus three equals five",
            "The result is fifty percent",
            "Send email to user at domain dot com",
            "List items comma first item comma second item",
            "Is this working correctly question mark"
        ]
        
        for text in test_texts:
            if pipeline.asr:
                # Full pipeline
                result = pipeline.process_text(text, return_metadata=False)
            else:
                # Text-only pipeline
                result = pipeline.converter.convert(text)
                if isinstance(result, tuple):
                    converted_text, metadata = result
                    result = {
                        'status': 'success',
                        'final_output': converted_text,
                        'total_conversions': metadata['total_conversions']
                    }
            
            if isinstance(result, dict) and result.get('status') == 'success':
                output = result['final_output']
                conversions = result.get('total_conversions', 0)
                print(f"  '{text}'")
                print(f"  → '{output}' ({conversions} conversions)")
            else:
                print(f"  '{text}' → Error processing")
        
        # Show pipeline statistics
        print(f"\n📊 Pipeline Statistics:")
        if hasattr(pipeline, 'get_stats'):
            try:
                stats = pipeline.get_stats()
                print(f"  Total Processed: {stats.get('total_processed', 0)}")
                print(f"  Average Processing Time: {stats.get('average_processing_time', 0):.3f}s")
                if 'converter_info' in stats:
                    print(f"  Converter Rules: {stats['converter_info'].get('total_rules', 0)}")
            except:
                print("  Stats not available")
        else:
            print("  Text-only mode - limited stats")
        
    except Exception as e:
        print(f"❌ Pipeline demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_flask_api():
    """Demonstrate Flask API usage"""
    print("\n🌐 Flask API Demo")
    print("=" * 40)
    
    print("🚀 Flask API Features:")
    print("  • POST /convert/text - Convert text with symbols")
    print("  • POST /convert/audio - Upload and process audio")
    print("  • POST /convert/batch - Batch process multiple texts")
    print("  • GET  /health - Check API health")
    print("  • GET  /stats - Get pipeline statistics")
    print("  • GET  /config - Get/update configuration")
    
    print("\n💡 To start the API server:")
    print("  python app.py")
    print("\n📝 Example API calls:")
    print("  # Text conversion")
    print("  curl -X POST http://localhost:5000/convert/text \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"text\": \"two plus three equals five\"}'")
    print()
    print("  # Audio upload")
    print("  curl -X POST http://localhost:5000/convert/audio \\")
    print("    -F 'audio=@test.wav'")
    
    # Try to check if Flask dependencies are available
    try:
        import flask
        import flask_cors
        print(f"\n✅ Flask dependencies available (v{flask.__version__})")
    except ImportError as e:
        print(f"\n⚠️  Flask dependencies missing: {e}")
        print("   Install with: pip install flask flask-cors")

def demo_model_serialization():
    """Demonstrate model saving and loading"""
    print("\n💾 Model Serialization Demo")
    print("=" * 40)
    
    import tempfile
    import os
    
    from ml_models.symbol_converter import create_advanced_converter
    
    # Create and configure a converter
    converter = create_advanced_converter()
    
    # Add a custom rule
    from ml_models.symbol_converter import ConversionRule
    custom_rule = ConversionRule(
        pattern=r'\btest\b',
        replacement='✓',
        priority=10,
        confidence_base=1.0
    )
    converter.add_custom_rule(custom_rule)
    
    # Test the custom rule
    result, _ = converter.convert("this is a test")
    print(f"Original converter: 'this is a test' → '{result}'")
    
    # Save the model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        converter.save_model(temp_path)
        print(f"✅ Model saved to: {temp_path}")
        
        # Load the model
        from ml_models.symbol_converter import SymbolConverter
        loaded_converter = SymbolConverter.load_model(temp_path)
        
        # Test loaded model
        loaded_result, _ = loaded_converter.convert("this is a test")
        print(f"Loaded converter: 'this is a test' → '{loaded_result}'")
        
        if result == loaded_result:
            print("✅ Model serialization successful!")
        else:
            print("❌ Model serialization failed!")
            
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def main():
    """Run all demos"""
    print("🎭 Speech-to-Symbol ML Pipeline Demo")
    print("=" * 50)
    print("This demo shows the refactored ML pipeline components")
    print("=" * 50)
    
    # Run demos
    demo_symbol_converter()
    demo_whisper_wrapper()
    demo_asr_pipeline()
    demo_flask_api()
    demo_model_serialization()
    
    print("\n" + "=" * 50)
    print("✨ Demo Complete!")
    print("🔗 Next Steps:")
    print("  1. Run tests: python test_pipeline.py")
    print("  2. Start API: python app.py")
    print("  3. Use in your projects: from ml_models.asr_pipeline import create_api_pipeline")
    print("=" * 50)

if __name__ == "__main__":
    main() 