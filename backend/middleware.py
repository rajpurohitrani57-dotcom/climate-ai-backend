from flask import request, jsonify
from functools import wraps
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        logger.info(f"📥 {request.method} {request.path} from {request.remote_addr}")
        response = f(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"✅ Completed in {elapsed:.3f}s")
        return response
    return decorated_function

def rate_limit(limit_per_minute=60):
    requests = {}
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            if client_ip in requests:
                requests[client_ip] = [t for t in requests[client_ip] if t > current_time - 60]
            else:
                requests[client_ip] = []
            if len(requests[client_ip]) >= limit_per_minute:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {limit_per_minute} requests per minute',
                    'retry_after': 60
                }), 429
            requests[client_ip].append(current_time)
            return f(*args, **kwargs)
        return decorated_function
    return decorator