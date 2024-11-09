from flask import Flask, request, jsonify, render_template
from bot import ChatAgentWithRL
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from dataclasses import dataclass
from typing import Optional
import json

# Data classes for structured data handling
@dataclass
class ChatRequest:
    query: str
    timestamp: float = time.time()
    session_id: Optional[str] = None

@dataclass
class ChatResponse:
    response: str
    confidence: float
    processing_time: float
    timestamp: float = time.time()

@dataclass
class FeedbackData:
    query: str
    response: str
    feedback: int
    timestamp: float = time.time()

class ChatApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.chat_agent = ChatAgentWithRL()
        self.setup_logging()
        self.setup_routes()
        
        # Analytics storage
        self.response_times = []
        self.feedback_data = []
        
    def setup_logging(self):
        """Configure application logging"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        file_handler = RotatingFileHandler(
            'logs/app.log',
            maxBytes=1024 * 1024,  # 1MB
            backupCount=10
        )
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.app.logger.addHandler(file_handler)
        self.app.logger.setLevel(logging.INFO)
        
    def setup_routes(self):
        """Setup Flask routes"""
        self.app.route('/')(self.home)
        self.app.route('/chat', methods=['POST'])(self.chat)
        self.app.route('/feedback', methods=['POST'])(self.feedback)
        self.app.route('/stats')(self.get_stats)
        
    def home(self):
        """Render the chat interface"""
        return render_template('index.html')
        
    def chat(self):
        """Handle chat requests"""
        try:
            # Parse and validate request
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({"error": "Invalid request"}), 400
                
            chat_request = ChatRequest(
                query=data['query'],
                session_id=request.headers.get('X-Session-ID')
            )
            
            # Log request
            self.app.logger.info(
                f'Chat request received - Session: {chat_request.session_id}, '
                f'Query: {chat_request.query[:50]}...'
            )
            
            # Process request and time it
            start_time = time.time()
            response = self.chat_agent.chat(chat_request.query)
            processing_time = time.time() - start_time
            
            # Calculate confidence (this is a placeholder - implement based on your model)
            confidence = self.chat_agent.get_confidence() if hasattr(self.chat_agent, 'get_confidence') else 0.7
            
            # Create response object
            chat_response = ChatResponse(
                response=response,
                confidence=confidence,
                processing_time=processing_time
            )
            
            # Update analytics
            self.response_times.append(processing_time)
            
            # Log response
            self.app.logger.info(
                f'Response sent - Time: {processing_time:.2f}s, '
                f'Confidence: {confidence:.2f}'
            )
            
            return jsonify({
                "response": chat_response.response,
                "confidence": chat_response.confidence,
                "processing_time": chat_response.processing_time * 1000  # in milliseconds
            })
        except Exception as e:
            self.app.logger.error(f"Error in /chat route: {str(e)}")
            return jsonify({"error": "An error occurred while processing your request"}), 500
        
    def feedback(self):
        """Handle feedback submissions"""
        try:
            # Parse and validate request
            data = request.get_json()
            if not data or not all(key in data for key in ['query', 'response', 'feedback']):
                return jsonify({"error": "Invalid feedback data"}), 400
                
            feedback_data = FeedbackData(
                query=data['query'],
                response=data['response'],
                feedback=data['feedback']
            )
            
            # Log feedback
            self.app.logger.info(
                f'Feedback received - Query: {feedback_data.query[:50]}..., '
                f'Response: {feedback_data.response[:50]}..., '
                f'Feedback: {feedback_data.feedback}'
            )
            
            # Store feedback (this could be used for training or improving the model)
            self.feedback_data.append(feedback_data)
            
            # Optionally: process feedback to improve model here
            
            return jsonify({"status": "Feedback received"})
        except Exception as e:
            self.app.logger.error(f"Error in /feedback route: {str(e)}")
            return jsonify({"error": "An error occurred while processing feedback"}), 500
            
    def get_stats(self):
        """Get system and performance statistics"""
        try:
            # Calculate average response time
            avg_response_time = (sum(self.response_times) / len(self.response_times)) if self.response_times else 0
            avg_response_time_ms = round(avg_response_time * 1000, 2)  # Convert to ms
            
            # Gather feedback statistics
            helpful_count = sum(1 for feedback in self.feedback_data if feedback.feedback == 1)
            not_helpful_count = sum(1 for feedback in self.feedback_data if feedback.feedback == 0)
            
            return jsonify({
                "average_response_time": avg_response_time_ms,
                "total_feedback": len(self.feedback_data),
                "helpful_feedback": helpful_count,
                "not_helpful_feedback": not_helpful_count
            })
        except Exception as e:
            self.app.logger.error(f"Error in /stats route: {str(e)}")
            return jsonify({"error": "An error occurred while fetching statistics"}), 500

# Initialize and run the application
if __name__ == '__main__':
    chat_app = ChatApp()
    chat_app.app.run(debug=True)
