import tkinter as tk
from tkinter import messagebox, ttk
import threading
import subprocess
import time
import cv2
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='sign_language_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class SignLanguageConverter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sign Language to Text Converter")
        self.root.geometry("800x600")
        self.is_recording = False
        self.setup_gui()
        self.setup_camera()

    def setup_camera(self):
        """Initialize the camera and gesture recognition system"""
        self.cap = None
        self.camera_thread = None

    def setup_gui(self):
        """Set up the GUI components"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Text display area with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.text_box = tk.Text(text_frame, height=15, width=70, font=("Arial", 14))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_box.yview)
        self.text_box.configure(yscrollcommand=scrollbar.set)

        self.text_box.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Start/Stop button
        self.toggle_button = ttk.Button(
            button_frame,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.toggle_button.pack(side=tk.LEFT, padx=5)

        # Clear button
        ttk.Button(
            button_frame,
            text="Clear Text",
            command=self.clear_text
        ).pack(side=tk.LEFT, padx=5)

        # Save button
        ttk.Button(
            button_frame,
            text="Save Text",
            command=self.save_text
        ).pack(side=tk.LEFT, padx=5)

        # Quit button
        ttk.Button(
            button_frame,
            text="Quit",
            command=self.quit_application
        ).pack(side=tk.LEFT, padx=5)

    def toggle_recording(self):
        """Toggle between starting and stopping the gesture recognition"""
        if not self.is_recording:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        """Start the gesture recognition process"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise ValueError("Could not open camera")

            self.is_recording = True
            self.toggle_button.configure(text="Stop Recording")
            self.status_var.set("Recording...")

            self.camera_thread = threading.Thread(target=self.gesture_recognition_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()

            logging.info("Started gesture recognition")
        except Exception as e:
            self.show_error(f"Failed to start camera: {str(e)}")
            logging.error(f"Camera start error: {str(e)}")

    def stop_recognition(self):
        """Stop the gesture recognition process"""
        self.is_recording = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.toggle_button.configure(text="Start Recording")
        self.status_var.set("Ready")
        logging.info("Stopped gesture recognition")

    def gesture_recognition_loop(self):
        """Main loop for gesture recognition"""
        try:
            while self.is_recording:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process the frame here
                # TODO: Add your gesture recognition model implementation
                processed_text = self.process_frame(frame)
                if processed_text:
                    self.update_text(processed_text)

                cv2.imshow('Gesture Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.show_error(f"Error in gesture recognition: {str(e)}")
            logging.error(f"Gesture recognition error: {str(e)}")
        finally:
            self.stop_recognition()

    def process_frame(self, frame):
        """Process a single frame and return recognized text"""
        # TODO: Implement your gesture recognition logic here
        # This is a placeholder for the actual implementation
        return None

    def update_text(self, new_text):
        """Update the text display with new recognized text"""
        self.text_box.insert(tk.END, f"{new_text}\n")
        self.text_box.see(tk.END)

    def clear_text(self):
        """Clear the text display"""
        self.text_box.delete("1.0", tk.END)
        logging.info("Cleared text display")

    def save_text(self):
        """Save the recognized text to a file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recognized_text_{timestamp}.txt"

            with open(filename, "w") as file:
                file.write(self.text_box.get("1.0", tk.END))

            messagebox.showinfo("Success", f"Text saved to '{filename}'")
            logging.info(f"Saved text to {filename}")
        except Exception as e:
            self.show_error(f"Failed to save file: {str(e)}")
            logging.error(f"File save error: {str(e)}")

    def show_error(self, message):
        """Display error message to user"""
        messagebox.showerror("Error", message)

    def quit_application(self):
        """Clean up and quit the application"""
        self.stop_recognition()
        self.root.quit()

    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = SignLanguageConverter()
    app.run()