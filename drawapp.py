import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import mediapipe as mp

class DrawingApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Hand Gesture Drawing App")

        # Panel 1 for drawing
        self.panel1 = tk.LabelFrame(self.root, text="Drawing Panel", width=500, height=500)
        self.panel1.grid(row=0, column=0, padx=10, pady=10)
        self.canvas = tk.Canvas(self.panel1, bg="black", width=500, height=500)
        self.canvas.pack()
        self.coord_label = tk.Label(self.panel1, text="Coordinates: ", bg="black", fg="white")
        self.coord_label.pack(anchor="nw")

        # Panel 2 for webcam feed and hand tracking
        self.panel2 = tk.LabelFrame(self.root, text="Webcam Panel", width=500, height=500)
        self.panel2.grid(row=0, column=1, padx=10, pady=10)
        self.video_label = tk.Label(self.panel2)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.width = 500
        self.height = 500

        # Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.frame_sequence = []

        # Hand tracking variables
        self.hand_coords = []
        self.gesture_state = 'idle'

        # Gesture confirmation variables
        self.current_gesture = None
        self.gesture_counter = 0
        self.gesture_threshold = 5  # Number of consistent frames to confirm gesture

        self.frame_count = 0
        self.update_video()
        self.root.mainloop()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.track_hand(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.video_label.after(10, self.update_video)

    def track_hand(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.width)
                y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.height)
                self.hand_coords.append((x, y))
                self.coord_label.config(text=f"Coordinates: ({x}, {y})")

                # Always capture frames for gesture prediction
                preprocessed_frame = self.preprocess_frame(frame)
                self.frame_sequence.append(preprocessed_frame)
                if len(self.frame_sequence) > 10:
                    self.frame_sequence.pop(0)

                # Skip frames for gesture prediction to improve performance
                self.frame_count += 1
                if self.frame_count % 5 == 0:
                    self.predict_gesture(preprocessed_frame)

                if self.gesture_state == 'drawing':
                    self.draw_on_canvas(x, y)
                elif self.gesture_state == 'erasing':
                    self.erase_on_canvas(x, y)

    def preprocess_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        normalized_frame = resized_frame / 255.0  # Normalize pixel values
        return normalized_frame[..., np.newaxis]  # Add channel dimension

    def predict_gesture(self, frame):
        frames = np.expand_dims(frame, axis=0)  # Create batch dimension
        prediction = self.model.predict(frames)
        gesture = np.argmax(prediction)
        print(f"Prediction: {prediction}, Gesture: {gesture}")

        if gesture == self.current_gesture:
            self.gesture_counter += 1
        else:
            self.current_gesture = gesture
            self.gesture_counter = 1

        if self.gesture_counter >= self.gesture_threshold:
            if gesture == 0:  # Drawing gesture (index finger)
                if self.gesture_state != 'drawing':
                    self.start_drawing()
            elif gesture == 1:  # Erasing gesture (thumb)
                if self.gesture_state != 'erasing':
                    self.start_erasing()
            else:
                self.stop_drawing()
                self.stop_erasing()

    def draw_on_canvas(self, x, y):
        if len(self.hand_coords) > 1:
            prev_x, prev_y = self.hand_coords[-2]
            self.canvas.create_line(prev_x, prev_y, x, y, fill="white", width=2)

    def erase_on_canvas(self, x, y):
        items = self.canvas.find_overlapping(x-5, y-5, x+5, y+5)
        for item in items:
            self.canvas.delete(item)

    def start_drawing(self):
        print("Started drawing")
        self.gesture_state = 'drawing'

    def stop_drawing(self):
        print("Stopped drawing")
        self.gesture_state = 'idle'

    def start_erasing(self):
        print("Started erasing")
        self.gesture_state = 'erasing'

    def stop_erasing(self):
        print("Stopped erasing")
        self.gesture_state = 'idle'

# Initialize the application
root = tk.Tk()
model_path = 'model/gesture_recognition_model.h5'  # Replace with the actual path to your trained model
app = DrawingApp(root, model_path)
