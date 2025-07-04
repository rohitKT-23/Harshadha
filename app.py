"""
Production-Ready Flask API for Speech-to-Symbol Conversion
RESTful API wrapper for the ASR Pipeline with file upload, real-time processing, and batch operations
"""

from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS
import os
import sys
import tempfile
import logging
from typing import Dict, List
import traceback
import time
import json
from werkzeug.utils import secure_filename
from pathlib import Path

# Import our ML pipeline
try:
    from ml_models.asr_pipeline import ASRPipeline, create_production_pipeline, create_api_pipeline
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    logger.error(f"ML models not available: {e}")
    logger.error("Please install required dependencies: pip install -r requirements.txt")
    ML_MODELS_AVAILABLE = False
    
    # Create dummy functions for graceful failure
    ASRPipeline = None
    create_production_pipeline = None
    create_api_pipeline = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB max file size
    'UPLOAD_FOLDER': 'uploads',
    'ALLOWED_EXTENSIONS': {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'},
    'SECRET_KEY': 'speech2symbol-api-key'  # Change in production
})

# Global pipeline instance
pipeline = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_pipeline():
    """Initialize the ML pipeline"""
    global pipeline
    
    if not ML_MODELS_AVAILABLE:
        logger.error("ML models not available - cannot initialize pipeline")
        logger.error("Please install dependencies: pip install torch transformers flask flask-cors")
        return False
    
    try:
        # Try to load custom model if available
        model_path = os.environ.get('CUSTOM_MODEL_PATH')
        confidence = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.7'))
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading custom model from: {model_path}")
            pipeline = create_production_pipeline(model_path, confidence)
        else:
            logger.info("Loading default pipeline")
            pipeline = create_api_pipeline(confidence=confidence)
        
        logger.info("Pipeline initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.error("This might be due to missing dependencies. Please run: pip install -r requirements.txt")
        return False

def handle_error(error_msg: str, status_code: int = 500) -> tuple:
    """Standard error response"""
    return jsonify({
        'status': 'error',
        'error': error_msg,
        'timestamp': time.time()
    }), status_code

# API Routes

@app.route('/', methods=['GET'])
def home():
    """Web interface for Speech-to-Symbol conversion"""
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def api_docs():
    """API documentation page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Speech-to-Symbol API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #007bff; }
            .url { font-family: monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
            pre { background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🎤 Speech-to-Symbol Conversion API</h1>
        <p>Intelligent speech-to-text with context-aware symbol conversion</p>
        
        <h2>📋 Available Endpoints</h2>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/health</div>
            <p>Check API health status</p>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/convert/text</div>
            <p>Convert text with symbol substitution</p>
            <pre>{"text": "two plus three equals five"}</pre>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/convert/audio</div>
            <p>Upload and process audio file</p>
            <p>Supports: WAV, MP3, FLAC, M4A, OGG, WebM</p>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/convert/batch</div>
            <p>Batch process multiple texts</p>
            <pre>{"texts": ["text1", "text2", "text3"]}</pre>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/stats</div>
            <p>Get pipeline statistics and model info</p>
        </div>
        
        <h2>💡 Example Usage</h2>
        <pre>
# Text conversion
curl -X POST http://localhost:5000/convert/text \\
  -H "Content-Type: application/json" \\
  -d '{"text": "two plus three equals five"}'

# Audio upload
curl -X POST http://localhost:5000/convert/audio \\
  -F "audio=@audio.wav"
        </pre>
        
        <h2>📊 Status: <span style="color: green;">{{ status }}</span></h2>
        <p>Pipeline Model: {{ model_info }}</p>
    </body>
    </html>
    """
    
    model_info = "Unknown"
    status = "Unknown"
    
    try:
        if pipeline:
            stats = pipeline.get_stats()
            model_info = stats.get('asr_info', {}).get('model_name', 'Unknown')
            status = "Ready"
        else:
            status = "Not Initialized"
    except:
        status = "Error"
    
    return render_template_string(html_template, status=status, model_info=model_info)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if pipeline is None:
            return handle_error("Pipeline not initialized", 503)
        
        # Quick test
        test_result = pipeline.process_text("test", return_metadata=False)
        
        if test_result.get('status') == 'success':
            return jsonify({
                'status': 'healthy',
                'pipeline_ready': True,
                'timestamp': time.time(),
                'version': '1.0.0'
            })
        else:
            return handle_error("Pipeline health check failed", 503)
            
    except Exception as e:
        return handle_error(f"Health check error: {str(e)}", 503)

@app.route('/convert/text', methods=['POST'])
def convert_text():
    """Convert text with symbol substitution"""
    try:
        # Validate request
        if not request.is_json:
            return handle_error("Content-Type must be application/json", 400)
        
        data = request.get_json()
        if not data or 'text' not in data:
            return handle_error("Missing 'text' field in request body", 400)
        
        text = data['text']
        if not text or not isinstance(text, str):
            return handle_error("Text must be a non-empty string", 400)
        
        # Optional parameters
        confidence = data.get('confidence', None)
        include_metadata = data.get('include_metadata', True)
        
        # Process text
        logger.info(f"Processing text: '{text[:50]}...'")
        result = pipeline.process_text(
            text, 
            custom_confidence=confidence,
            return_metadata=include_metadata
        )
        
        # Add API-specific metadata
        result['api_version'] = '1.0.0'
        result['endpoint'] = 'text'
        result['timestamp'] = time.time()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Text conversion error: {e}")
        return handle_error(f"Text conversion failed: {str(e)}", 500)

@app.route('/convert/audio', methods=['POST'])
def convert_audio():
    """Upload and process audio file"""
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return handle_error("No audio file provided", 400)
        
        file = request.files['audio']
        if file.filename == '':
            return handle_error("No file selected", 400)
        
        if not allowed_file(file.filename):
            return handle_error(
                f"Invalid file type. Allowed: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", 
                400
            )
        
        # Optional parameters from form data
        confidence = request.form.get('confidence', type=float)
        include_metadata = request.form.get('include_metadata', 'true').lower() == 'true'
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{int(time.time())}_{filename}")
        file.save(temp_path)
        
        try:
            # Process audio
            logger.info(f"Processing audio file: {filename}")
            result = pipeline.process_audio(
                temp_path,
                return_metadata=include_metadata,
                custom_confidence=confidence
            )
            
            # Add API-specific metadata
            result['api_version'] = '1.0.0'
            result['endpoint'] = 'audio'
            result['timestamp'] = time.time()
            result['original_filename'] = filename
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return handle_error(f"Audio conversion failed: {str(e)}", 500)

@app.route('/convert/batch', methods=['POST'])
def convert_batch():
    """Batch process multiple texts"""
    try:
        # Validate request
        if not request.is_json:
            return handle_error("Content-Type must be application/json", 400)
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return handle_error("Missing 'texts' field in request body", 400)
        
        texts = data['texts']
        if not isinstance(texts, list) or not texts:
            return handle_error("'texts' must be a non-empty list", 400)
        
        # Validate each text
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                return handle_error(f"Text at index {i} must be a non-empty string", 400)
        
        # Optional parameters
        confidence = data.get('confidence', None)
        include_metadata = data.get('include_metadata', False)  # Default false for batch
        
        # Process batch
        logger.info(f"Batch processing {len(texts)} texts")
        results = pipeline.batch_process_text(texts, return_metadata=include_metadata)
        
        # Update each result with API metadata
        for result in results:
            result['api_version'] = '1.0.0'
            result['endpoint'] = 'batch'
        
        # Add batch summary
        successful = [r for r in results if r.get('status') == 'success']
        total_conversions = sum(r.get('total_conversions', 0) for r in successful)
        
        response = {
            'status': 'success',
            'batch_size': len(texts),
            'successful': len(successful),
            'failed': len(texts) - len(successful),
            'total_conversions': total_conversions,
            'results': results,
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch conversion error: {e}")
        return handle_error(f"Batch conversion failed: {str(e)}", 500)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get pipeline statistics and model information"""
    try:
        if pipeline is None:
            return handle_error("Pipeline not initialized", 503)
        
        stats = pipeline.get_stats()
        
        # Add API-specific info
        api_stats = {
            'api_version': '1.0.0',
            'uptime': time.time(),  # Could track actual uptime
            'timestamp': time.time(),
            'pipeline_stats': stats
        }
        
        return jsonify(api_stats)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return handle_error(f"Failed to get stats: {str(e)}", 500)

@app.route('/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update pipeline configuration"""
    if request.method == 'GET':
        # Get current configuration
        try:
            config = {
                'confidence_threshold': pipeline.confidence_threshold,
                'whisper_model': pipeline.whisper_model,
                'language': pipeline.language,
                'converter_stats': pipeline.converter.get_statistics()
            }
            return jsonify(config)
        except Exception as e:
            return handle_error(f"Failed to get config: {str(e)}", 500)
    
    elif request.method == 'POST':
        # Update configuration
        try:
            if not request.is_json:
                return handle_error("Content-Type must be application/json", 400)
            
            data = request.get_json()
            
            # Update confidence threshold
            if 'confidence_threshold' in data:
                new_threshold = float(data['confidence_threshold'])
                if 0.0 <= new_threshold <= 1.0:
                    pipeline.update_confidence_threshold(new_threshold)
                else:
                    return handle_error("Confidence threshold must be between 0.0 and 1.0", 400)
            
            # Add custom conversion rules
            if 'custom_rules' in data:
                for rule in data['custom_rules']:
                    pipeline.add_custom_conversion_rule(
                        rule['pattern'], 
                        rule['replacement'], 
                        rule.get('priority', 5)
                    )
            
            return jsonify({
                'status': 'success',
                'message': 'Configuration updated',
                'timestamp': time.time()
            })
            
        except Exception as e:
            return handle_error(f"Failed to update config: {str(e)}", 500)

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return handle_error("File too large. Maximum size is 50MB.", 413)

@app.errorhandler(404)
def not_found(e):
    return handle_error("Endpoint not found", 404)

@app.errorhandler(500)
def internal_error(e):
    return handle_error("Internal server error", 500)

def create_app(config=None):
    """Factory function to create Flask app"""
    if config:
        app.config.update(config)
    
    # Initialize pipeline immediately
    logger.info("Initializing Speech-to-Symbol API...")
    if not init_pipeline():
        logger.error("Failed to initialize pipeline!")
    else:
        logger.info("API ready to serve requests!")
    
    return app

def startup_check():
    """Check if pipeline is initialized"""
    if pipeline is None:
        logger.warning("Pipeline not initialized, attempting to initialize...")
        if not init_pipeline():
            logger.error("Failed to initialize pipeline on startup!")
            return False
    return True

# Initialize pipeline when module is imported (for Gunicorn)
logger.info("Initializing Speech-to-Symbol pipeline...")
if not init_pipeline():
    logger.error("Failed to initialize pipeline!")

if __name__ == '__main__':
    # Development server
    logger.info("Starting Speech-to-Symbol API...")
    
    # Initialize pipeline before starting server
    if not startup_check():
        logger.error("Failed to initialize pipeline! Exiting...")
        sys.exit(1)
    
    # Run server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting Speech-to-Symbol API on port {port}")
    logger.info(f"📱 API Documentation: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug) 