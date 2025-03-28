import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import pyttsx3
import mediapipe as mp
import pickle

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language Recognition")
        self.window.geometry("1200x800")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85,
            max_num_hands=1
        )
        
        # Load the model
        try:
            print("Loading model...")
            self.model_dict = pickle.load(open('./model.p', 'rb'))
            self.scaler = self.model_dict['scaler']
            self.classifier = self.model_dict['classifier']
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            exit()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            exit()
        print("Camera initialized successfully")
        
        # Initialize prediction tracking
        self.text_content = ""
        self.last_prediction = None
        self.prediction_count = 0
        self.stable_threshold = 20
        self.stable_predictions = []
        self.max_stable_predictions = 20
        self.min_confidence = 0.45
        self.min_stability = 0.75
        self.no_hand_frames = 0
        self.MAX_NO_HAND_FRAMES = 10
        self.last_stable_prediction = None
        self.stable_prediction_count = 0
        self.MIN_STABLE_COUNT = 5
        
        # Add moving average for confidence scores
        self.confidence_window = []
        self.confidence_window_size = 20
        
        # Add weighted voting system
        self.prediction_weights = {}
        self.weight_decay = 0.85
        self.min_weight = 0.2
        self.max_weight = 1.0
        
        # Create GUI elements
        self.create_widgets()
        
        # Start video stream
        self.update_video()
    
    def create_widgets(self):
        # Create main container
        main_container = ttk.Frame(self.window, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        video_frame = ttk.LabelFrame(main_container, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        
        # Hand skeleton frame
        skeleton_frame = ttk.LabelFrame(main_container, text="Hand Skeleton", padding="5")
        skeleton_frame.grid(row=0, column=1, padx=5, pady=5)
        
        self.skeleton_label = ttk.Label(skeleton_frame)
        self.skeleton_label.pack()
        
        # Text display frame
        text_frame = ttk.LabelFrame(main_container, text="Recognized Text", padding="5")
        text_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.text_display = tk.Text(text_frame, wrap=tk.WORD, width=40, height=10)
        self.text_display.pack(expand=True, fill=tk.BOTH)
        
        # Button frame
        button_frame = ttk.Frame(main_container)
        button_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        self.speak_button = ttk.Button(button_frame, text="Speak Text", command=self.speak_text)
        self.speak_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear Text", command=self.clear_text)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_container, textvariable=self.status_var)
        status_bar.grid(row=3, column=0, columnspan=2, pady=5)
    
    def update_video(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                # Create skeleton image
                skeleton_image = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
                
                # Draw stability indicator
                stability_height = 30
                cv2.rectangle(frame, (10, 10), (200, 10 + stability_height), (0, 0, 0), -1)
                if len(self.stable_predictions) > 0:
                    current_char = self.stable_predictions[-1]
                    stability = self.stable_predictions.count(current_char) / len(self.stable_predictions)
                    width = int(190 * stability)
                    color = (0, 255, 0) if stability >= self.min_stability else (0, 165, 255)
                    cv2.rectangle(frame, (10, 10), (10 + width, 10 + stability_height), color, -1)
                    cv2.putText(frame, f"Stability: {stability:.2f}", (15, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if results.multi_hand_landmarks:
                    self.no_hand_frames = 0
                    try:
                        data_aux = []
                        x_ = []
                        y_ = []
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Draw on original frame
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                            
                            # Draw on skeleton image
                            self.mp_drawing.draw_landmarks(
                                skeleton_image,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                            
                            # Extract landmarks for bounding box and prediction
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                x_.append(x)
                                y_.append(y)
                            
                            # Normalize coordinates relative to the first landmark
                            base_x = hand_landmarks.landmark[0].x
                            base_y = hand_landmarks.landmark[0].y
                            data_aux = []
                            
                            # Collect both x and y coordinates normalized to the first landmark
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                data_aux.append(x - base_x)
                                data_aux.append(y - base_y)
                            
                            # Process landmarks for prediction
                            if len(data_aux) > 0:
                                data_aux = np.array(data_aux)  # This will be 42 features (x and y coordinates)
                                
                                # Make prediction
                                data_input = data_aux.reshape(1, -1)
                                try:
                                    # Scale the input data
                                    data_scaled = self.scaler.transform(data_input)
                                    # Make prediction
                                    predicted_character = str(self.classifier.predict(data_scaled)[0])
                                    probabilities = self.classifier.predict_proba(data_scaled)[0]
                                    confidence = np.max(probabilities)
                                    
                                    # Update confidence moving average
                                    self.confidence_window.append(confidence)
                                    if len(self.confidence_window) > self.confidence_window_size:
                                        self.confidence_window.pop(0)
                                    avg_confidence = np.mean(self.confidence_window)
                                    
                                    # Update prediction weights
                                    self.update_prediction_weights(predicted_character, confidence)
                                    
                                    # Get the most stable prediction
                                    max_weight_label, max_weight = self.update_prediction_weights(predicted_character, confidence)
                                    
                                    print(f"Predicted: {max_weight_label}, Confidence: {max_weight:.2f}, Avg Confidence: {avg_confidence:.2f}")
                                    
                                    # Only update predictions if average confidence is high enough
                                    if avg_confidence >= self.min_confidence:
                                        # Update stable predictions list
                                        self.stable_predictions.append(max_weight_label)
                                        if len(self.stable_predictions) > self.max_stable_predictions:
                                            self.stable_predictions.pop(0)
                                        
                                        # Calculate stability
                                        current_stability = 0
                                        if len(self.stable_predictions) >= 5:
                                            current_stability = self.stable_predictions.count(max_weight_label) / len(self.stable_predictions)
                                            
                                            # Check for stable prediction
                                            if current_stability >= self.min_stability:
                                                if max_weight_label == self.last_stable_prediction:
                                                    self.stable_prediction_count += 1
                                                else:
                                                    self.last_stable_prediction = max_weight_label
                                                    self.stable_prediction_count = 1
                                                
                                                # Add character after consistent predictions
                                                if self.stable_prediction_count >= self.MIN_STABLE_COUNT:
                                                    self.text_display.insert(tk.END, max_weight_label)
                                                    self.text_content += max_weight_label
                                                    self.stable_prediction_count = 0  # Reset count after adding character
                                            else:
                                                self.stable_prediction_count = 0
                                        
                                        # Draw prediction on frame
                                        H, W = frame.shape[:2]
                                        x1 = int(min(x_) * W) - 20
                                        y1 = int(min(y_) * H) - 20
                                        x2 = int(max(x_) * W) + 20
                                        y2 = int(max(y_) * H) + 20
                                        
                                        x1, y1 = max(0, x1), max(0, y1)
                                        x2, y2 = min(W, x2), min(H, y2)
                                        
                                        # Draw rectangle with color based on stability
                                        color = (0, 255, 0) if current_stability >= self.min_stability else (0, 165, 255)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        # Add prediction text with background
                                        display_text = f"{max_weight_label} ({max_weight:.2f})"
                                        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                                        cv2.rectangle(frame, 
                                                    (x1, y1 - text_size[1] - 10),
                                                    (x1 + text_size[0], y1),
                                                    (0, 0, 0),
                                                    -1)
                                        
                                        cv2.putText(frame, display_text,
                                                  (x1, y1 - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                                
                                except Exception as e:
                                    print(f"Error during prediction: {str(e)}")
                    
                    except Exception as e:
                        print(f"Error processing hand: {e}")
                else:
                    self.no_hand_frames += 1
                    if self.no_hand_frames >= self.MAX_NO_HAND_FRAMES:
                        self.stable_predictions = []  # Clear predictions when no hand is detected
                        self.stable_prediction_count = 0
                        self.last_stable_prediction = None
                
                # Update displays
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                skeleton_rgb = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2RGB)
                
                # Update video feed
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
                # Update skeleton display
                skeleton = Image.fromarray(skeleton_rgb)
                skeleton_photo = ImageTk.PhotoImage(image=skeleton)
                self.skeleton_label.configure(image=skeleton_photo)
                self.skeleton_label.image = skeleton_photo
            
            # Schedule next update
            self.window.after(10, self.update_video)
    
    def speak_text(self):
        text = self.text_display.get("1.0", tk.END).strip()
        if text:
            self.speak_button.configure(state='disabled')
            self.status_var.set("Speaking...")
            
            def speak():
                self.engine.say(text)
                self.engine.runAndWait()
                self.window.after(0, self.speak_complete)
            
            threading.Thread(target=speak, daemon=True).start()
    
    def speak_complete(self):
        self.speak_button.configure(state='normal')
        self.status_var.set("Ready")
    
    def clear_text(self):
        self.text_display.delete("1.0", tk.END)
        self.text_content = ""
        self.stable_predictions = []
        self.stable_prediction_count = 0
        self.last_stable_prediction = None
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        self.engine.stop()

    def update_prediction_weights(self, prediction, confidence):
        # Decay existing weights
        for label in self.prediction_weights:
            self.prediction_weights[label] *= self.weight_decay
            if self.prediction_weights[label] < self.min_weight:
                self.prediction_weights[label] = self.min_weight

        # Update weight for current prediction
        current_weight = self.prediction_weights.get(prediction, 0)
        new_weight = min(current_weight + confidence * 0.5, self.max_weight)
        self.prediction_weights[prediction] = new_weight

        # Normalize weights
        total_weight = sum(self.prediction_weights.values())
        if total_weight > 0:
            for label in self.prediction_weights:
                self.prediction_weights[label] /= total_weight

        # Get prediction with highest weight
        max_weight_label = max(self.prediction_weights.items(), key=lambda x: x[1])[0]
        max_weight = self.prediction_weights[max_weight_label]

        return max_weight_label, max_weight

def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 