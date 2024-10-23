import cv2
import numpy as np
import tkinter as tk
from tkinter import Frame, Label, Button, messagebox
from PIL import Image, ImageTk

class VideoCallApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Create a frame for video display
        self.frame = Frame(window)
        self.frame.pack()

        # Create a label for video
        self.video_label = Label(self.frame)
        self.video_label.pack()

        # Create a button to start video
        self.start_button = Button(window, text="Start Video", command=self.start_video)
        self.start_button.pack()

        # Initialize video capture
        self.vid = None

        # Start the GUI loop
        self.window.mainloop()

    def start_video(self):
        # Initialize video capture from webcam
        self.vid = cv2.VideoCapture(0)

        if not self.vid.isOpened():
            messagebox.showerror("Error", "Unable to access the webcam.")
            return

        # Start updating the video frame
        self.update_video()

    def update_video(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Convert the frame to RGB (OpenCV uses BGR by default)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to a PIL image
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)

                # Update the label with the new frame
                self.video_label.img = img  # Keep a reference to avoid garbage collection
                self.video_label.configure(image=img)

            # Call this method again after 10 ms
            self.window.after(10, self.update_video)
        else:
            self.vid.release()

if __name__ == "__main__":
    root = tk.Tk()
    VideoCallApp(root, "Video Call App")
