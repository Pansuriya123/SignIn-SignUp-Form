import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import ttk, scrolledtext
import joblib
import os
import threading
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import pickle

class GestureToSpeech:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Gesture to Speech/Text")
        
        # Set window size and position
        window_width = 800
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Load the trained model and scaler
        self.load_model()
        
        # Initialize video capture
        self.cap = None
        self.is_capturing = False
        self.capture_thread = None
        
        # Initialize prediction variables
        self.current_gesture = None
        self.gesture_count = 0
        self.min_gesture_count = 5
        self.last_spoken = None
        self.recognized_text = ""
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title = ttk.Label(main_frame, 
                         text="Gesture to Speech/Text Converter",
                         font=('Arial', 20, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Camera frame
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Input", padding="10")
        camera_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Start/Stop camera button
        self.camera_btn = ttk.Button(camera_frame,
                                   text="Start Camera",
                                   command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=0, pady=5)
        
        # Status label
        self.status_label = ttk.Label(camera_frame,
                                    text="Status: Camera off",
                                    font=('Arial', 10))
        self.status_label.grid(row=0, column=1, pady=5, padx=10)
        
        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Recognized Text", padding="10")
        output_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Output text
        self.output_text = scrolledtext.ScrolledText(output_frame,
                                                   wrap=tk.WORD,
                                                   width=50,
                                                   height=8,
                                                   state='disabled')
        self.output_text.grid(row=0, column=0, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Clear text button
        self.clear_btn = ttk.Button(control_frame,
                                  text="Clear Text",
                                  command=self.clear_output)
        self.clear_btn.grid(row=0, column=0, padx=5)
        
        # Speak text button
        self.speak_btn = ttk.Button(control_frame,
                                  text="Speak Text",
                                  command=self.speak_output)
        self.speak_btn.grid(row=0, column=1, padx=5)
        
        # Exit button
        exit_btn = ttk.Button(main_frame,
                            text="Exit",
                            command=self.cleanup)
        exit_btn.grid(row=4, column=0, columnspan=2, pady=20)
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load the model using pickle
            model_path = 'model.p'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_dict = pickle.load(f)
                self.scaler = model_dict['scaler']
                self.classifier = model_dict['classifier']
                self.update_output("Model and scaler loaded successfully")
            else:
                self.update_output("Model file not found. Please ensure model.p exists in the root directory.", True)
                self.classifier = None
                self.scaler = None
        except Exception as e:
            self.update_output(f"Error loading model: {str(e)}", True)
            self.classifier = None
            self.scaler = None
    
    def update_output(self, message, is_error=False):
        """Update output text widget"""
        self.output_text.configure(state='normal')
        self.output_text.insert(tk.END, message + '\n')
        if is_error:
            self.output_text.tag_add("error", "end-2c linestart", "end-1c")
            self.output_text.tag_configure("error", foreground="red")
        self.output_text.configure(state='disabled')
        self.output_text.see(tk.END)
    
    def clear_output(self):
        """Clear the output text"""
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state='disabled')
        self.recognized_text = ""
    
    def speak_output(self):
        """Speak the recognized text"""
        if self.recognized_text:
            try:
                self.engine.say(self.recognized_text)
                self.engine.runAndWait()
            except Exception as e:
                self.update_output(f"Error speaking text: {str(e)}", True)
        else:
            self.update_output("No text to speak", True)
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_capturing:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.is_capturing = True
                self.camera_btn.configure(text="Stop Camera")
                self.status_label.configure(text="Status: Camera on")
                
                # Start capture thread
                self.capture_thread = threading.Thread(target=self.process_camera)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                
            except Exception as e:
                self.update_output(f"Error starting camera: {str(e)}", True)
        else:
            self.is_capturing = False
            self.camera_btn.configure(text="Start Camera")
            self.status_label.configure(text="Status: Camera off")
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
    
    def extract_landmarks(self, results):
        """Extract hand landmarks and normalize them"""
        if not results.multi_hand_landmarks:
            return None
        
        landmarks = results.multi_hand_landmarks[0].landmark
        data = []
        
        # Get reference point (wrist)
        ref_x = landmarks[0].x
        ref_y = landmarks[0].y
        
        # Extract normalized coordinates
        for lm in landmarks:
            data.extend([
                (lm.x - ref_x),
                (lm.y - ref_y)
            ])
        
        return np.array(data)
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        try:
            if self.classifier is None or self.scaler is None:
                return None
            
            # Preprocess landmarks
            landmarks_scaled = self.scaler.transform(landmarks.reshape(1, -42))
            
            # Make prediction
            prediction = self.classifier.predict(landmarks_scaled)[0]
            probabilities = self.classifier.predict_proba(landmarks_scaled)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            self.update_output(f"Error during prediction: {str(e)}", True)
            return None
    
    def process_camera(self):
        """Process camera feed and recognize gestures"""
        try:
            while self.is_capturing:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    # Draw hand landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                    
                    # Extract landmarks and predict gesture
                    landmarks = self.extract_landmarks(results)
                    if landmarks is not None:
                        prediction = self.predict_gesture(landmarks)
                        if prediction:
                            gesture, confidence = prediction
                            
                            # Update current gesture if confidence is high enough
                            if confidence > 0.6:  # Lowered threshold for better detection
                                if gesture == self.current_gesture:
                                    self.gesture_count += 1
                                else:
                                    self.current_gesture = gesture
                                    self.gesture_count = 1
                                
                                # Add gesture to text if it's stable
                                if self.gesture_count >= self.min_gesture_count:
                                    self.recognized_text += str(gesture)
                                    self.output_text.configure(state='normal')
                                    self.output_text.delete(1.0, tk.END)
                                    self.output_text.insert(tk.END, self.recognized_text)
                                    self.output_text.configure(state='disabled')
                                    self.gesture_count = 0
                            
                            # Display prediction
                            cv2.putText(
                                frame,
                                f"Gesture: {gesture} ({confidence:.2f})",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )
                
                # Display frame
                cv2.imshow('Gesture Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            self.update_output(f"Error in gesture recognition: {str(e)}", True)
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def cleanup(self):
        """Clean up resources and exit"""
        self.is_capturing = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.engine.stop()
        self.root.quit()
    
    def run(self):
        """Run the application"""
        try:
            # Create models directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
                self.update_output("Created models directory. Please add model files.")
            
            # Start the GUI event loop
            self.root.mainloop()
            
        except Exception as e:
            print(f"Error running application: {str(e)}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    converter = GestureToSpeech()
    converter.run() 