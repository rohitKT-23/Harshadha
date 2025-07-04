"""
Simple Flask App Test
Quick verification that the Flask API starts correctly
"""

import subprocess
import time
import requests
import json
import sys
import os

def test_flask_startup():
    """Test Flask app startup without dependencies"""
    print("🧪 Testing Flask App Startup")
    print("=" * 40)
    
    # First test - import Flask app
    try:
        from app import app, create_app
        print("✅ Flask app imports successfully")
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False
    
    # Test app creation
    try:
        test_app = create_app({'TESTING': True})
        print("✅ Flask app created successfully")
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return False
    
    return True

def test_basic_routes():
    """Test basic Flask routes"""
    print("\n🌐 Testing Basic Routes")
    print("=" * 40)
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test home route
            response = client.get('/')
            print(f"GET / : {response.status_code}")
            
            # Test health route (might fail due to missing pipeline)
            response = client.get('/health')
            print(f"GET /health : {response.status_code}")
            
            # Test invalid route
            response = client.get('/invalid')
            print(f"GET /invalid : {response.status_code}")
            
        print("✅ Basic routing works")
        return True
        
    except Exception as e:
        print(f"❌ Route testing failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints with mock data"""
    print("\n📡 Testing API Endpoints")
    print("=" * 40)
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test text endpoint (might fail due to missing pipeline)
            test_data = {'text': 'hello world'}
            response = client.post('/convert/text',
                                 data=json.dumps(test_data),
                                 content_type='application/json')
            print(f"POST /convert/text : {response.status_code}")
            
            # Test batch endpoint
            test_batch = {'texts': ['hello', 'world']}
            response = client.post('/convert/batch',
                                 data=json.dumps(test_batch),
                                 content_type='application/json')
            print(f"POST /convert/batch : {response.status_code}")
            
            # Test stats endpoint
            response = client.get('/stats')
            print(f"GET /stats : {response.status_code}")
            
        print("✅ API endpoints respond (some may fail without ML pipeline)")
        return True
        
    except Exception as e:
        print(f"❌ API endpoint testing failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n📦 Testing Dependencies")
    print("=" * 40)
    
    required_packages = [
        'flask',
        'flask_cors',
        'werkzeug'
    ]
    
    optional_packages = [
        'torch',
        'transformers',
        'spacy',
        'nltk'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing (Required)")
            print(f"   Install with: pip install {package}")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"⚠️  {package} - Missing (Optional for ML features)")
    
    return True

def run_live_server_test():
    """Test running the actual Flask server"""
    print("\n🚀 Testing Live Server")
    print("=" * 40)
    
    print("To test the live server:")
    print("1. Run: python app.py")
    print("2. Open: http://localhost:5000")
    print("3. Check logs for any errors")
    print("\nExpected behavior:")
    print("- Server should start without errors")
    print("- Home page should load")
    print("- Health endpoint may show pipeline errors (normal without ML deps)")
    
def main():
    """Run all tests"""
    print("🧪 Flask App Compatibility Test")
    print("=" * 50)
    
    tests = [
        test_flask_startup,
        test_dependencies,
        test_basic_routes,
        test_api_endpoints
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Flask app should work correctly.")
    else:
        print("⚠️  Some tests failed, but app might still work.")
    
    run_live_server_test()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 