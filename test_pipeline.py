"""
Comprehensive Test Suite for Speech-to-Symbol ML Pipeline
Tests for WhisperWrapper, SymbolConverter, ASRPipeline, and Flask API
"""

import unittest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import ML components
from ml_models.symbol_converter import SymbolConverter, ConversionRule, create_basic_converter
from ml_models.asr_pipeline import ASRPipeline, create_basic_pipeline, create_api_pipeline

class TestSymbolConverter(unittest.TestCase):
    """Test cases for SymbolConverter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.converter = create_basic_converter()
    
    def test_basic_conversions(self):
        """Test basic mathematical operator conversions"""
        test_cases = [
            ("two plus three", "two + three"),
            ("five minus two", "five - two"),
            ("ten percent", "ten %"),
            ("send email at company dot com", "send email at company. com"),
            ("list items comma separated", "list items, separated"),
            ("is this correct question mark", "is this correct?"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result, metadata = self.converter.convert(input_text, confidence_threshold=0.6)
                self.assertEqual(result, expected)
                self.assertGreater(metadata['total_conversions'], 0)
    
    def test_confidence_thresholds(self):
        """Test different confidence thresholds"""
        text = "two plus three equals five"
        
        # High confidence should have fewer conversions
        high_conf_result, high_meta = self.converter.convert(text, confidence_threshold=0.9)
        low_conf_result, low_meta = self.converter.convert(text, confidence_threshold=0.3)
        
        # Lower threshold should result in more or equal conversions
        self.assertGreaterEqual(low_meta['total_conversions'], high_meta['total_conversions'])
    
    def test_no_conversions(self):
        """Test text that should not be converted"""
        text = "hello world this is a normal sentence"
        result, metadata = self.converter.convert(text)
        
        self.assertEqual(result, text)  # Should remain unchanged
        self.assertEqual(metadata['total_conversions'], 0)
    
    def test_batch_conversion(self):
        """Test batch processing"""
        texts = [
            "two plus three",
            "ten percent",
            "hello world"
        ]
        
        results = self.converter.batch_convert(texts)
        self.assertEqual(len(results), len(texts))
        
        # First two should have conversions, last should not
        self.assertGreater(results[0][1]['total_conversions'], 0)
        self.assertGreater(results[1][1]['total_conversions'], 0)
        self.assertEqual(results[2][1]['total_conversions'], 0)
    
    def test_custom_rules(self):
        """Test adding custom conversion rules"""
        custom_rule = ConversionRule(
            pattern=r'\bhello\b',
            replacement='hi',
            priority=10
        )
        
        self.converter.add_custom_rule(custom_rule)
        
        result, metadata = self.converter.convert("hello world")
        self.assertEqual(result, "hi world")
        self.assertEqual(metadata['total_conversions'], 1)
    
    def test_save_load_model(self):
        """Test model serialization"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save model
            self.converter.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load model
            loaded_converter = SymbolConverter.load_model(temp_path)
            
            # Test that loaded model works the same
            test_text = "two plus three"
            original_result = self.converter.convert(test_text)
            loaded_result = loaded_converter.convert(test_text)
            
            self.assertEqual(original_result[0], loaded_result[0])
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestASRPipeline(unittest.TestCase):
    """Test cases for ASRPipeline"""
    
    def setUp(self):
        """Set up test fixtures with mocked Whisper"""
        # Mock the WhisperWrapper to avoid loading actual models
        with patch('ml_models.asr_pipeline.create_base_whisper') as mock_whisper:
            mock_asr = Mock()
            mock_asr.transcribe.return_value = {
                "transcription": "two plus three equals five",
                "status": "success",
                "model_type": "base",
                "device": "cpu"
            }
            mock_asr.get_model_info.return_value = {
                "model_name": "openai/whisper-small",
                "model_type": "base",
                "device": "cpu"
            }
            mock_whisper.return_value = mock_asr
            
            self.pipeline = create_basic_pipeline()
    
    def test_text_processing(self):
        """Test text-only processing"""
        test_text = "two plus three equals five"
        result = self.pipeline.process_text(test_text, return_metadata=True)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['input_text'], test_text)
        self.assertEqual(result['final_output'], "two + three = five")
        self.assertGreater(result['total_conversions'], 0)
        self.assertIn('conversions_made', result)
    
    def test_audio_processing_mock(self):
        """Test audio processing with mocked audio input"""
        # Create a dummy audio file path
        audio_path = "dummy_audio.wav"
        
        result = self.pipeline.process_audio(audio_path, return_metadata=True)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['raw_transcription'], "two plus three equals five")
        self.assertEqual(result['final_output'], "two + three = five")
        self.assertGreater(result['total_conversions'], 0)
    
    def test_batch_text_processing(self):
        """Test batch text processing"""
        texts = [
            "two plus three",
            "ten percent",
            "normal text"
        ]
        
        results = self.pipeline.batch_process_text(texts, return_metadata=False)
        
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertEqual(result['status'], 'success')
    
    def test_evaluation(self):
        """Test pipeline evaluation"""
        test_samples = [
            {"input": "two plus three", "expected": "two + three"},
            {"input": "ten percent", "expected": "ten %"},
            {"input": "hello world", "expected": "hello world"}
        ]
        
        evaluation = self.pipeline.evaluate_on_samples(test_samples)
        
        self.assertIn('accuracy', evaluation)
        self.assertIn('total_samples', evaluation)
        self.assertEqual(evaluation['total_samples'], len(test_samples))
        self.assertGreaterEqual(evaluation['accuracy'], 0.5)  # Should get at least 50% right
    
    def test_confidence_threshold_update(self):
        """Test updating confidence threshold"""
        original_threshold = self.pipeline.confidence_threshold
        new_threshold = 0.9
        
        self.pipeline.update_confidence_threshold(new_threshold)
        
        self.assertEqual(self.pipeline.confidence_threshold, new_threshold)
        self.assertEqual(self.pipeline.converter.confidence_threshold, new_threshold)
    
    def test_stats_collection(self):
        """Test statistics collection"""
        # Process some text to generate stats
        self.pipeline.process_text("test text")
        
        stats = self.pipeline.get_stats()
        
        self.assertIn('total_processed', stats)
        self.assertIn('asr_info', stats)
        self.assertIn('converter_info', stats)
        self.assertIn('pipeline_config', stats)
        self.assertGreater(stats['total_processed'], 0)

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up integration test environment"""
        # Create pipeline with mocked components
        with patch('ml_models.asr_pipeline.create_base_whisper') as mock_whisper:
            mock_asr = Mock()
            mock_asr.transcribe.return_value = {
                "transcription": "two plus three equals five",
                "status": "success"
            }
            mock_asr.get_model_info.return_value = {
                "model_name": "whisper-small"
            }
            mock_whisper.return_value = mock_asr
            
            self.pipeline = create_api_pipeline()
    
    def test_end_to_end_text_flow(self):
        """Test complete text processing flow"""
        input_text = "calculate two plus three times four"
        result = self.pipeline.process_text(input_text)
        
        # Should convert mathematical operators
        self.assertIn('+', result['final_output'])
        self.assertIn('×', result['final_output'])
        self.assertEqual(result['status'], 'success')
    
    def test_pipeline_performance(self):
        """Test pipeline performance characteristics"""
        texts = ["two plus three"] * 10
        
        start_time = time.time()
        results = self.pipeline.batch_process_text(texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process 10 short texts in reasonable time (< 5 seconds)
        self.assertLess(processing_time, 5.0)
        
        # All should succeed
        successful = [r for r in results if r['status'] == 'success']
        self.assertEqual(len(successful), len(texts))
    
    def test_error_handling(self):
        """Test error handling in pipeline"""
        # Test with empty text
        result = self.pipeline.process_text("")
        self.assertIn('status', result)  # Should handle gracefully
        
        # Test with very long text
        long_text = "word " * 1000
        result = self.pipeline.process_text(long_text)
        self.assertEqual(result['status'], 'success')  # Should handle long text

class TestFlaskAPI(unittest.TestCase):
    """Test cases for Flask API (if app is available)"""
    
    def setUp(self):
        """Set up Flask test client"""
        try:
            # Try to import and set up Flask app
            from app import create_app
            
            # Create test app with mocked pipeline
            with patch('app.init_pipeline') as mock_init:
                mock_init.return_value = True
                self.app = create_app({'TESTING': True})
                self.client = self.app.test_client()
                
                # Mock the global pipeline
                with patch('app.pipeline') as mock_pipeline:
                    mock_pipeline.process_text.return_value = {
                        'status': 'success',
                        'final_output': 'two + three',
                        'total_conversions': 1
                    }
                    mock_pipeline.get_stats.return_value = {
                        'total_processed': 0,
                        'asr_info': {'model_name': 'whisper-small'},
                        'converter_info': {'total_rules': 50}
                    }
                    
                    self.mock_pipeline = mock_pipeline
            self.api_available = True
            
        except ImportError:
            self.api_available = False
            self.skipTest("Flask API not available")
    
    def test_health_endpoint(self):
        """Test API health check"""
        if not self.api_available:
            self.skipTest("Flask API not available")
            
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
    
    def test_text_conversion_endpoint(self):
        """Test text conversion API endpoint"""
        if not self.api_available:
            self.skipTest("Flask API not available")
            
        payload = {'text': 'two plus three'}
        
        response = self.client.post('/convert/text',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('final_output', data)
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        if not self.api_available:
            self.skipTest("Flask API not available")
            
        response = self.client.get('/stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('pipeline_stats', data)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for the pipeline"""
    
    def setUp(self):
        """Set up performance testing"""
        self.converter = create_basic_converter()
    
    def test_conversion_speed(self):
        """Benchmark conversion speed"""
        texts = [
            "two plus three equals five",
            "ten percent of fifty dollars",
            "send email to user at domain dot com"
        ] * 100  # 300 texts total
        
        start_time = time.time()
        results = self.converter.batch_convert(texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        per_text_time = total_time / len(texts)
        
        print(f"\nPerformance Benchmark:")
        print(f"Processed {len(texts)} texts in {total_time:.2f} seconds")
        print(f"Average time per text: {per_text_time*1000:.2f} ms")
        
        # Should process each text in under 10ms
        self.assertLess(per_text_time, 0.01)
    
    def test_memory_usage(self):
        """Test memory usage characteristics"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many texts
        texts = ["test text"] * 1000
        self.converter.batch_convert(texts)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage:")
        print(f"Initial: {initial_memory:.1f} MB")
        print(f"Final: {final_memory:.1f} MB")
        print(f"Increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (< 100MB for 1000 texts)
        self.assertLess(memory_increase, 100)

def run_all_tests():
    """Run all test suites"""
    test_classes = [
        TestSymbolConverter,
        TestASRPipeline,
        TestPipelineIntegration,
        TestFlaskAPI,
        TestPerformanceBenchmarks
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_quick_tests():
    """Run only essential tests (faster)"""
    test_classes = [
        TestSymbolConverter,
        TestASRPipeline
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🧪 Running Speech-to-Symbol Pipeline Tests")
    print("=" * 50)
    
    # Check if user wants quick tests
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick test suite...")
        success = run_quick_tests()
    else:
        print("Running full test suite...")
        success = run_all_tests()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 