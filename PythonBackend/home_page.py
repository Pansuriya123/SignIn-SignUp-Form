import tkinter as tk
from tkinter import ttk
import subprocess
import os
import sys

class HomePage:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sign Language Converter")
        
        # Set window size and position it in center
        window_width = 800
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 12))
        self.style.configure('Title.TLabel', font=('Arial', 24, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Arial', 14))
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title = ttk.Label(main_frame, 
                         text="Sign Language Conversion System",
                         style='Title.TLabel')
        title.grid(row=0, column=0, pady=20)
        
        # Subtitle
        subtitle = ttk.Label(main_frame,
                           text="Choose a conversion mode:",
                           style='Subtitle.TLabel')
        subtitle.grid(row=1, column=0, pady=10)
        
        # Button Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=40)
        
        # Speech/Text to Gesture Button
        speech_to_gesture_btn = ttk.Button(
            button_frame,
            text="Speech/Text to Gesture",
            command=self.open_speech_to_gesture,
            width=30
        )
        speech_to_gesture_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Add description
        ttk.Label(button_frame,
                 text="Convert spoken words or text into sign language gestures",
                 style='Subtitle.TLabel').grid(row=1, column=0, pady=5)
        
        # Gesture to Speech Button
        gesture_to_speech_btn = ttk.Button(
            button_frame,
            text="Gesture to Speech/Text",
            command=self.open_gesture_to_speech,
            width=30
        )
        gesture_to_speech_btn.grid(row=2, column=0, padx=10, pady=10)
        
        # Add description
        ttk.Label(button_frame,
                 text="Convert sign language gestures into spoken words or text",
                 style='Subtitle.TLabel').grid(row=3, column=0, pady=5)
        
        # Exit Button
        exit_btn = ttk.Button(
            main_frame,
            text="Exit",
            command=self.root.quit,
            width=15
        )
        exit_btn.grid(row=3, column=0, pady=20)
        
    def open_speech_to_gesture(self):
        """Open the speech to gesture conversion window"""
        self.root.withdraw()  # Hide main window
        try:
            subprocess.run([sys.executable, 'speech_to_isl.py'])
            self.root.deiconify()  # Show main window again
        except Exception as e:
            print(f"Error opening speech to gesture converter: {str(e)}")
            self.root.deiconify()
    
    def open_gesture_to_speech(self):
        """Open the gesture to speech conversion window"""
        self.root.withdraw()  # Hide main window
        try:
            subprocess.run([sys.executable, 'gesture_to_speech.py'])
            self.root.deiconify()  # Show main window again
        except Exception as e:
            print(f"Error opening gesture to speech converter: {str(e)}")
            self.root.deiconify()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = HomePage()
    app.run() 