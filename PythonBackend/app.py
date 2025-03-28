from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from datetime import datetime
import json
from speech_to_isl import TextToSignConverter
from gesture_to_speech import GestureRecognizer

app = Flask(__name__)
CORS(app)

# Initialize converters
text_to_sign = TextToSignConverter()
gesture_recognizer = GestureRecognizer()

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/text-to-sign', methods=['POST'])
def text_to_sign_endpoint():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate video file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'text_to_sign_{timestamp}.mp4'
        output_path = os.path.join(UPLOAD_FOLDER, output_file)

        # Convert text to sign language video
        text_to_sign.convert_text_to_sign(text, output_path)

        return jsonify({
            'success': True,
            'video_url': f'/uploads/{output_file}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speech-to-sign', methods=['POST'])
def speech_to_sign_endpoint():
    try:
        data = request.json
        text = data.get('speech_text', '')
        if not text:
            return jsonify({'error': 'No speech text provided'}), 400

        # Generate video file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'speech_to_sign_{timestamp}.mp4'
        output_path = os.path.join(UPLOAD_FOLDER, output_file)

        # Convert speech text to sign language video
        text_to_sign.convert_text_to_sign(text, output_path)

        return jsonify({
            'success': True,
            'video_url': f'/uploads/{output_file}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sign-to-text', methods=['POST'])
def sign_to_text_endpoint():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected video file'}), 400

        # Save the uploaded video temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(UPLOAD_FOLDER, f'sign_video_{timestamp}.mp4')
        video_file.save(video_path)

        # Process the video and get text
        text = gesture_recognizer.process_video(video_path)

        # Clean up the temporary video file
        os.remove(video_path)

        return jsonify({
            'success': True,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True) 