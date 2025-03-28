import tkinter as tk
from tkinter import ttk, scrolledtext
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import cv2
import threading
import queue
import json
import os
import time
from PIL import Image, ImageTk

class SpeechToISL:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Speech/Text to Sign Language")
        
        # Set window size and position
        window_width = 1200
        window_height = 800
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 28, 'bold'), foreground='#2C3E50')
        self.style.configure('SubTitle.TLabel', font=('Helvetica', 12), foreground='#34495E')
        self.style.configure('Custom.TButton', font=('Helvetica', 11), padding=10)
        self.style.configure('Custom.TEntry', font=('Helvetica', 12), padding=5)
        
        # Set window background
        self.root.configure(bg='#ECF0F1')
        
        # Initialize other components
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Load ISL word mappings
        self.word_to_gesture = self.load_word_mappings()
        
        # Initialize video display
        self.current_video = None
        self.is_playing = False
        self.display_queue = queue.Queue()
        self.display_thread = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Start display thread
        self.start_display_thread()
    
    def create_widgets(self):
        # Create main frame with padding and background
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)  # Give more weight to the right column
        main_frame.rowconfigure(0, weight=1)     # Make rows expandable
        
        # Left frame for input controls
        left_frame = ttk.Frame(main_frame, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 20))
        
        # Title with enhanced styling
        title = ttk.Label(left_frame, 
                         text="Sign Language Converter",
                         style='Title.TLabel')
        title.grid(row=0, column=0, pady=(0, 5))
        
        # Subtitle
        subtitle = ttk.Label(left_frame,
                           text="Convert text or speech into sign language gestures",
                           style='SubTitle.TLabel')
        subtitle.grid(row=1, column=0, pady=(0, 20))
        
        # Text input frame with enhanced styling
        text_frame = ttk.LabelFrame(left_frame, text="Enter Text", padding="15")
        text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Text input field with custom style
        self.text_input = ttk.Entry(text_frame, width=40, style='Custom.TEntry')
        self.text_input.grid(row=0, column=0, pady=10, padx=5)
        
        # Speech status frame
        speech_status_frame = ttk.LabelFrame(text_frame, text="Speech Recognition Status", padding="10")
        speech_status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Speech status indicator (dot)
        self.status_indicator = tk.Canvas(speech_status_frame, width=20, height=20, bg='#ECF0F1', highlightthickness=0)
        self.status_indicator.grid(row=0, column=0, padx=5)
        self.status_dot = self.status_indicator.create_oval(5, 5, 15, 15, fill='gray')
        
        # Speech status text
        self.speech_status_var = tk.StringVar(value="Ready")
        self.speech_status = ttk.Label(speech_status_frame, 
                                     textvariable=self.speech_status_var,
                                     style='SubTitle.TLabel')
        self.speech_status.grid(row=0, column=1, sticky=tk.W)
        
        # Transcript field
        self.transcript_text = scrolledtext.ScrolledText(speech_status_frame, 
                                                       wrap=tk.WORD, 
                                                       height=3, 
                                                       font=('Helvetica', 10))
        self.transcript_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Button frame
        button_frame = ttk.Frame(text_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        # Convert text button with custom style
        self.convert_text_btn = ttk.Button(button_frame,
                                         text="Convert Text",
                                         style='Custom.TButton',
                                         command=self.convert_text)
        self.convert_text_btn.grid(row=0, column=0, padx=5)
        
        # Speech input button
        self.speech_btn = ttk.Button(button_frame,
                                   text="🎤 Speech Input",
                                   style='Custom.TButton',
                                   command=self.start_speech_recognition)
        self.speech_btn.grid(row=0, column=1, padx=5)
        
        # Stop button (hidden by default)
        self.stop_btn = ttk.Button(button_frame,
                                 text="⏹ Stop",
                                 style='Custom.TButton',
                                 command=self.stop_speech_recognition,
                                 state='disabled')
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Clear text button with custom style
        self.clear_text_btn = ttk.Button(button_frame,
                                       text="Clear Text",
                                       style='Custom.TButton',
                                       command=lambda: self.text_input.delete(0, tk.END))
        self.clear_text_btn.grid(row=0, column=3, padx=5)
        
        # Right frame for gesture display
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Display frame title
        display_title = ttk.Label(right_frame,
                                text="Sign Language Display",
                                style='Title.TLabel')
        display_title.grid(row=0, column=0, pady=(0, 10))
        
        # Display frame with enhanced styling
        self.display_frame = ttk.LabelFrame(right_frame, padding="10")
        self.display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_rowconfigure(0, weight=1)
        
        # Canvas with border and background
        self.canvas = tk.Canvas(self.display_frame, width=500, height=400,  # Reduced size
                              bg='white', highlightthickness=1, 
                              highlightbackground='#BDC3C7')
        self.canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
        
        # Create a frame inside canvas for centering
        self.canvas_frame = ttk.Frame(self.canvas)
        self.canvas.create_window(250, 200, window=self.canvas_frame, anchor='center')  # Adjusted coordinates
        
        # Display label inside canvas frame
        self.display_label = ttk.Label(self.canvas_frame)
        self.display_label.pack(expand=True, fill='both')
        
        # Progress frame at the bottom
        progress_frame = ttk.Frame(right_frame)
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar with custom style
        self.style.configure("Custom.Horizontal.TProgressbar", 
                           thickness=12,  # Slightly reduced thickness
                           troughcolor='#ECF0F1',
                           background='#3498DB')
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                          style="Custom.Horizontal.TProgressbar",
                                          variable=self.progress_var,
                                          maximum=100,
                                          length=400)  # Reduced length
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Status label with custom style
        self.status_var = tk.StringVar(value="Ready to convert")
        self.status_label = ttk.Label(progress_frame, 
                                    textvariable=self.status_var,
                                    style='SubTitle.TLabel')
        self.status_label.grid(row=1, column=0, pady=5)
    
    def load_word_mappings(self):
        """Load word to ISL gesture mappings from JSON file"""
        mapping_file = 'isl_mappings.json'
        if not os.path.exists(mapping_file):
            # Create default mappings if file doesn't exist
            default_mappings = {
                "hello": "gestures/hello.mp4",
                "thank": "gestures/thank_you.mp4",
                "you": "gestures/you.mp4",
                "yes": "gestures/yes.mp4",
                "no": "gestures/no.mp4",
                "please": "gestures/please.mp4",
                # Add more mappings as needed
            }
            with open(mapping_file, 'w') as f:
                json.dump(default_mappings, f, indent=4)
            return default_mappings
        
        with open(mapping_file, 'r') as f:
            return json.load(f)
    
    def preprocess_text(self, text):
        """Process text using NLP techniques"""
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords (except essential ones)
        stop_words = set(stopwords.words('english')) - {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'no', 'yes'}
        tokens = [word for word in tokens if word not in stop_words]
        
        return tokens
    
    def update_output(self, message, is_error=False):
        """Update output text widget"""
        self.output_text.configure(state='normal')
        self.output_text.insert(tk.END, message + '\n')
        if is_error:
            self.output_text.tag_add("error", "end-2c linestart", "end-1c")
            self.output_text.tag_configure("error", foreground="red")
        self.output_text.configure(state='disabled')
        self.output_text.see(tk.END)
    
    def convert_text(self):
        """Convert text input to gestures"""
        text = self.text_input.get().strip().upper()
        if not text:
            self.status_var.set("Please enter some text!")
            return
        
        # Clear previous items in queue
        while not self.display_queue.empty():
            self.display_queue.get()
        
        # Add each letter to the display queue
        for letter in text:
            if letter.isalnum() or letter in [' ', '.', '?', '!']:
                self.display_queue.put(letter)
        
        self.status_var.set(f"Converting: {text}")
        self.progress_var.set(0)
    
    def start_display_thread(self):
        """Start the display thread"""
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()
    
    def display_loop(self):
        """Main display loop"""
        while True:
            try:
                # Get next letter from queue
                letter = self.display_queue.get()
                
                if letter == ' ':
                    # Shorter pause for space
                    time.sleep(0.3)
                    continue
                
                # Get image path for letter
                if letter.isdigit():
                    image_path = f"GestureAtoZ1to0/{letter}.png"
                else:
                    image_path = f"GestureAtoZ1to0/{letter}.png"
                
                if not os.path.exists(image_path):
                    self.status_var.set(f"No gesture found for: {letter}")
                    time.sleep(0.3)
                    continue
                
                try:
                    # Load and display image
                    image = Image.open(image_path)
                    
                    # Calculate resize dimensions while maintaining aspect ratio
                    canvas_width = 500  # Match canvas width
                    canvas_height = 400  # Match canvas height
                    
                    # Calculate scaling factor to fit image within canvas
                    width_ratio = canvas_width / image.size[0]
                    height_ratio = canvas_height / image.size[1]
                    scale_factor = min(width_ratio, height_ratio) * 0.85  # Slightly smaller for better visibility
                    
                    new_width = int(image.size[0] * scale_factor)
                    new_height = int(image.size[1] * scale_factor)
                    
                    # Resize image
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Fade out current image (if any)
                    if hasattr(self.display_label, 'image') and self.display_label.image:
                        for alpha in range(100, 0, -20):
                            self.display_label.configure(image="")
                            self.root.update()
                            time.sleep(0.01)
                    
                    # Update display with fade-in effect
                    self.display_label.configure(image=photo)
                    self.display_label.image = photo
                    
                    # Update status with letter and progress
                    self.status_var.set(f"Showing gesture for: {letter}")
                    
                    # Update progress
                    total_items = self.display_queue.qsize() + 1
                    progress = (1 - (self.display_queue.qsize() / total_items)) * 100
                    self.progress_var.set(progress)
                    
                    # Display each letter for 0.8 seconds (even faster)
                    time.sleep(0.8)
                    
                except Exception as e:
                    print(f"Error displaying image: {str(e)}")
                    self.status_var.set(f"Error displaying gesture for: {letter}")
                    time.sleep(0.3)
                
            except queue.Empty:
                # Reset display when queue is empty
                self.display_label.configure(image="")
                self.status_var.set("Ready to convert")
                self.progress_var.set(0)
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                time.sleep(0.5)
    
    def start_speech_recognition(self):
        """Start speech recognition"""
        try:
            # Update status
            self.status_var.set("Listening... Speak now")
            self.speech_status_var.set("🎤 Listening...")
            self.speech_btn.configure(state='disabled')
            self.stop_btn.configure(state='normal')  # Enable stop button
            self.is_listening = True  # Set listening flag
            
            # Update status indicator
            self.status_indicator.itemconfig(self.status_dot, fill='#e74c3c')  # Red for recording
            self.transcript_text.delete(1.0, tk.END)
            self.transcript_text.insert(tk.END, "Listening...\n")
            self.root.update()
            
            # Initialize recognizer
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.transcript_text.insert(tk.END, "Adjusted for ambient noise...\n")
                self.root.update()
                
                try:
                    # Listen for speech with stop check
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    if not self.is_listening:  # Check if stopped
                        raise sr.WaitTimeoutError()
                    
                    # Update status
                    self.speech_status_var.set("Processing speech...")
                    self.status_indicator.itemconfig(self.status_dot, fill='#f1c40f')  # Yellow for processing
                    self.transcript_text.insert(tk.END, "Processing speech...\n")
                    self.root.update()
                    
                    # Convert speech to text
                    text = self.recognizer.recognize_google(audio)
                    
                    # Update text input and transcript
                    self.text_input.delete(0, tk.END)
                    self.text_input.insert(0, text.upper())
                    self.transcript_text.insert(tk.END, f"Recognized: {text}\n")
                    
                    # Update status
                    self.speech_status_var.set("✓ Speech recognized")
                    self.status_indicator.itemconfig(self.status_dot, fill='#2ecc71')  # Green for success
                    
                    # Convert the recognized text
                    self.convert_text()
                    
                except sr.WaitTimeoutError:
                    if self.is_listening:  # Only show timeout message if not manually stopped
                        self.status_var.set("No speech detected. Please try again.")
                        self.speech_status_var.set("⚠ No speech detected")
                        self.status_indicator.itemconfig(self.status_dot, fill='#e67e22')  # Orange for warning
                        self.transcript_text.insert(tk.END, "Error: No speech detected\n")
                except sr.UnknownValueError:
                    self.status_var.set("Could not understand audio. Please try again.")
                    self.speech_status_var.set("⚠ Could not understand")
                    self.status_indicator.itemconfig(self.status_dot, fill='#e67e22')  # Orange for warning
                    self.transcript_text.insert(tk.END, "Error: Could not understand audio\n")
                except sr.RequestError as e:
                    self.status_var.set(f"Could not request results; {str(e)}")
                    self.speech_status_var.set("⚠ Service error")
                    self.status_indicator.itemconfig(self.status_dot, fill='#e74c3c')  # Red for error
                    self.transcript_text.insert(tk.END, f"Error: Could not request results - {str(e)}\n")
                
            self.cleanup_speech_recognition()
            
        except Exception as e:
            self.status_var.set(f"Error with speech recognition: {str(e)}")
            self.speech_status_var.set("⚠ Error occurred")
            self.status_indicator.itemconfig(self.status_dot, fill='#e74c3c')  # Red for error
            self.transcript_text.insert(tk.END, f"Error: {str(e)}\n")
            self.cleanup_speech_recognition()
        
        self.root.update()
    
    def stop_speech_recognition(self):
        """Stop speech recognition"""
        self.is_listening = False
        self.status_var.set("Speech recognition stopped")
        self.speech_status_var.set("⏹ Stopped")
        self.status_indicator.itemconfig(self.status_dot, fill='gray')
        self.transcript_text.insert(tk.END, "Speech recognition stopped by user\n")
        self.cleanup_speech_recognition()
    
    def cleanup_speech_recognition(self):
        """Clean up speech recognition UI state"""
        self.speech_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.is_listening = False
        self.root.update()
    
    def cleanup(self):
        """Clean up resources and exit"""
        cv2.destroyAllWindows()
        self.root.quit()
    
    def run(self):
        """Run the application"""
        try:
            # Check if GestureAtoZ1to0 directory exists
            if not os.path.exists('GestureAtoZ1to0'):
                self.status_var.set("Error: GestureAtoZ1to0 directory not found!")
                return
            
            # Start the GUI event loop
            self.root.mainloop()
            
        except Exception as e:
            print(f"Error running application: {str(e)}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    app = SpeechToISL()
    app.run() 