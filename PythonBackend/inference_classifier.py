import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3
import threading

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language To Text Conversion")
        self.window.geometry("1200x800")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Get available voices and set to a female voice if available
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        # Configure the grid
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)
        
        # Create main title
        title_label = ttk.Label(window, text="Sign Language To Text Conversion", 
                              font=('Arial', 24, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Create left frame for camera feed
        self.left_frame = ttk.Frame(window)
        self.left_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        # Create camera label
        self.camera_label = ttk.Label(self.left_frame)
        self.camera_label.pack(padx=10, pady=10)

        # Create right frame for hand skeleton
        self.right_frame = ttk.Frame(window)
        self.right_frame.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        
        # Create skeleton label
        self.skeleton_label = ttk.Label(self.right_frame)
        self.skeleton_label.pack(padx=10, pady=10)

        # Create bottom frame for text display
        self.bottom_frame = ttk.Frame(window)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # Create character label
        self.char_label = ttk.Label(self.bottom_frame, text="Character : ", font=('Arial', 14))
        self.char_label.grid(row=0, column=0, padx=5, pady=5)
        self.current_char = ttk.Label(self.bottom_frame, text="", font=('Arial', 14))
        self.current_char.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create sentence label
        self.sentence_label = ttk.Label(self.bottom_frame, text="Sentence : ", font=('Arial', 14))
        self.sentence_label.grid(row=1, column=0, padx=5, pady=5)
        self.current_sentence = ttk.Label(self.bottom_frame, text="", font=('Arial', 14))
        self.current_sentence.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Create suggestions frame
        self.suggestions_frame = ttk.Frame(self.bottom_frame)
        self.suggestions_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Create suggestions label
        self.suggestions_label = ttk.Label(self.suggestions_frame, text="Suggestions : ", 
                                         font=('Arial', 14, 'bold'), foreground='red')
        self.suggestions_label.pack(side=tk.LEFT, padx=5)

        # Create buttons frame
        self.buttons_frame = ttk.Frame(self.bottom_frame)
        self.buttons_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Create Clear and Speak buttons
        self.clear_button = ttk.Button(self.buttons_frame, text="Clear", command=self.clear_text)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.speak_button = ttk.Button(self.buttons_frame, text="Speak", command=self.speak_text)
        self.speak_button.pack(side=tk.LEFT, padx=5)

        # Initialize OpenCV and MediaPipe
        self.init_cv()
        
        # Start video loop
        self.update_video()

    def init_cv(self):
        try:
            print("Loading model...")
            self.model_dict = pickle.load(open('./model.p', 'rb'))
            self.pipeline = self.model_dict['pipeline']
            print("Model loaded successfully")
            print("Pipeline steps:", [name for name, _ in self.pipeline.steps])
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            exit()

        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            exit()
        print("Camera initialized successfully")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True,  # Changed to True for more stable detection
                                       min_detection_confidence=0.7,  # Lowered slightly
                                       min_tracking_confidence=0.7,   # Added tracking confidence
                                       max_num_hands=1)

        # Initialize prediction tracking
        self.text_content = ""
        self.last_prediction = None
        self.prediction_count = 0
        self.PREDICTION_THRESHOLD = 8  # Reduced for faster response
        self.stable_predictions = []
        self.max_stable_predictions = 8  # Reduced window size
        self.min_confidence = 0.15  # Lowered confidence threshold
        self.min_stability = 0.6  # Lowered stability threshold
        self.no_hand_frames = 0
        self.MAX_NO_HAND_FRAMES = 10
        self.last_stable_prediction = None
        self.stable_prediction_count = 0
        self.MIN_STABLE_COUNT = 2  # Reduced required consecutive predictions

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Create blank skeleton image
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
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        
                        # Draw on skeleton image
                        self.mp_drawing.draw_landmarks(
                            skeleton_image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

                    if len(data_aux) > 0:
                        # Ensure data is properly formatted
                        data_aux = np.array(data_aux)
        if len(data_aux) < 84:
            data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), 'constant')
                        elif len(data_aux) > 84:
                            data_aux = data_aux[:84]

                        # Use pipeline for prediction
                        data_input = data_aux.reshape(1, -1)
                        try:
                            # Get prediction and probabilities
                            predicted_character = str(self.pipeline.predict(data_input)[0])
                            probabilities = self.pipeline.named_steps['classifier'].predict_proba(data_input)[0]
                            confidence = np.max(probabilities)
                            
                            print(f"Predicted: {predicted_character}, Confidence: {confidence:.2f}")
                            
                            # Only update predictions if confidence is high enough
                            if confidence >= self.min_confidence:
                                # Update stable predictions list
                                self.stable_predictions.append(predicted_character)
                                if len(self.stable_predictions) > self.max_stable_predictions:
                                    self.stable_predictions.pop(0)
                                
                                # Calculate stability
                                if len(self.stable_predictions) >= 5:
                                    current_stability = self.stable_predictions.count(predicted_character) / len(self.stable_predictions)
                                    
                                    # Check for stable prediction
                                    if current_stability >= self.min_stability:
                                        if predicted_character == self.last_stable_prediction:
                                            self.stable_prediction_count += 1
                                        else:
                                            self.last_stable_prediction = predicted_character
                                            self.stable_prediction_count = 1
                                        
                                        # Update UI only after consecutive stable predictions
                                        if self.stable_prediction_count >= self.MIN_STABLE_COUNT:
                                            self.current_char.config(text=predicted_character)
                                            if not self.text_content or self.text_content[-1] != predicted_character:
                                                self.text_content += predicted_character
                                                self.current_sentence.config(text=self.text_content)
                                    else:
                                        self.stable_prediction_count = 0
                                else:
                                    current_stability = 0

                                # Draw bounding box and prediction
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
                                display_text = f"{predicted_character} ({confidence*100:.1f}%)"
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
                            import traceback
                            traceback.print_exc()

                except Exception as e:
                    print(f"Error processing hand: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                self.no_hand_frames += 1
                if self.no_hand_frames >= self.MAX_NO_HAND_FRAMES:
                    self.stable_predictions = []  # Clear predictions when no hand is detected
                    self.stable_prediction_count = 0  # Reset stable prediction count
                    self.last_stable_prediction = None  # Reset last stable prediction

            # Convert frames to PhotoImage
            frame = cv2.resize(frame, (480, 360))
            skeleton_image = cv2.resize(skeleton_image, (480, 360))
            
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            skeleton_image = Image.fromarray(cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2RGB))
            
            self.photo = ImageTk.PhotoImage(image=frame_image)
            self.skeleton_photo = ImageTk.PhotoImage(image=skeleton_image)
            
            self.camera_label.configure(image=self.photo)
            self.skeleton_label.configure(image=self.skeleton_photo)

        self.window.after(10, self.update_video)

    def clear_text(self):
        self.text_content = ""
        self.current_char.config(text="")
        self.current_sentence.config(text="")
        self.stable_predictions = []

    def speak_text(self):
        # Get the current sentence
        text = self.text_content.strip()
        if not text:
            # If no text, speak a message
            text = "No text to speak"
        
        # Create a thread for speaking to avoid freezing the UI
        def speak_thread():
            try:
                self.speak_button.config(state='disabled')  # Disable button while speaking
                self.engine.say(text)
                self.engine.runAndWait()
            finally:
                self.speak_button.config(state='normal')  # Re-enable button after speaking
        
        # Start speaking in a separate thread
        threading.Thread(target=speak_thread, daemon=True).start()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'engine'):
            self.engine.stop()

def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
