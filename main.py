import kivy
kivy.require('2.1.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import os
import time
import random
import numpy as np
import subprocess
import sys

import tensorflow as tf

# --- Constants ---
PENDING_DIR = "pending_review"
REVIEWED_DIR = "reviewed_images"
NORMAL_DIR = os.path.join(REVIEWED_DIR, "normal")
DANGEROUS_DIR = os.path.join(REVIEWED_DIR, "dangerous")
ASSETS_DIR = "assets"
MODEL_FILE = os.path.join(ASSETS_DIR, "model.keras") # Use the .keras model now
SOUND_FILE = os.path.join(ASSETS_DIR, "alarm.mp3")
# CAMERA_SOURCE = "http://192.168.100.51:8080/video"
# CAMERA_SOURCE = 0 # Use 0 for default system camera
CAMERA_SOURCE = "assets/testt.mp4" # Use a simple string path

# --- Main Application Screen ---
class MainScreen(Screen):
    def on_enter(self, *args):
        self.app = App.get_running_app()
        # Always release and reopen the capture when entering the screen
        # This ensures the video restarts correctly and handles any previous state issues.
        if self.app.capture.isOpened():
            self.app.capture.release()
        self.app.capture.open(CAMERA_SOURCE)
        if not self.app.capture.isOpened():
            print(f"Error: Could not open video source: {CAMERA_SOURCE}")
            self.status_label.text = "Status: ERROR - VIDEO NOT FOUND"
        self.camera_update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)

    def on_leave(self, *args):
        Clock.unschedule(self.camera_update_event)
        if self.app.capture.isOpened():
            self.app.capture.release()

    def update(self, dt):
        app = self.app
        app.dangerous_cooldown = max(0, app.dangerous_cooldown - dt)
        ret, frame = app.capture.read()
        if ret:
            self.last_frame = frame.copy()  # Store the latest frame for capture
            self.process_frame(frame)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_view.texture = texture
        else:
            # If the video ends, reset it to the beginning to loop it
            print("Video ended. Looping...")
            app.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_frame(self, frame):
        app = self.app
        if not app.model or app.dangerous_cooldown > 0:
            self.confidence_label.text = "Confidence: ---"
            return

        try:
            # Preprocess frame to match model input
            # The Keras model expects input shape (height, width, 3)
            input_shape = app.model.input_shape
            height, width = input_shape[1], input_shape[2] # e.g., (None, 224, 224, 3)

            resized_frame = cv2.resize(frame, (width, height))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb_frame, axis=0) # Add batch dimension
            input_data = input_data / 255.0 # Rescale

            # Run Keras prediction
            prediction = app.model.predict(input_data)
            dangerous_probability = prediction[0][0]

            self.confidence_label.text = f"Confidence: {dangerous_probability:.2f}"

            if dangerous_probability > 0.7:
                self.trigger_dangerous_event(frame)

        except Exception as e:
            print(f"Error during model inference: {e}")
            app.model = None # Stop trying to process
            self.status_label.text = "Status: Model Error"

    def trigger_dangerous_event(self, frame):
        app = self.app
        self.status_label.text = "Status: DANGEROUS!"
        if app.sound:
            app.sound.play()

        timestamp = int(time.time())
        filename = os.path.join(PENDING_DIR, f"dangerous_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved screenshot: {filename}")

        app.dangerous_cooldown = 5
        Clock.schedule_once(self.reset_status, 5)

    def reset_status(self, dt):
        self.status_label.text = "Status: Normal"

    def capture_image(self, *args):
        """Capture the current frame and save to pending_review."""
        if hasattr(self, 'last_frame'):
            timestamp = int(time.time())
            filename = os.path.join(PENDING_DIR, f"manual_{timestamp}.png")
            cv2.imwrite(filename, self.last_frame)
            self.status_label.text = "Status: Image Captured!"
            print(f"Manual capture saved: {filename}")
            Clock.schedule_once(self.reset_status, 2)
        else:
            self.status_label.text = "Status: No frame to capture"
            Clock.schedule_once(self.reset_status, 2)

# --- Image Review Screen ---
class ReviewScreen(Screen):
    def on_enter(self, *args):
        self.pending_files = os.listdir(PENDING_DIR)
        self.current_file = None
        self.load_next_image()

    def load_next_image(self):
        if self.pending_files:
            self.current_file = self.pending_files.pop(0)
            self.review_image.source = os.path.join(PENDING_DIR, self.current_file)
            self.image_label.text = f"Reviewing: {self.current_file}"
            self.classify_buttons.disabled = False
        else:
            self.current_file = None
            self.review_image.source = ''
            self.image_label.text = "No more images to review."
            self.classify_buttons.disabled = True

    def classify_image(self, classification):
        if not self.current_file:
            return

        source_path = os.path.join(PENDING_DIR, self.current_file)
        dest_dir = NORMAL_DIR if classification == 'normal' else DANGEROUS_DIR
        dest_path = os.path.join(dest_dir, self.current_file)

        os.rename(source_path, dest_path)
        print(f"Moved {self.current_file} to {dest_dir}")

        self.load_next_image()

# --- Main App Class ---
class MonitoringApp(App):
    def build(self):
        # Create directories
        for d in [PENDING_DIR, NORMAL_DIR, DANGEROUS_DIR]:
            os.makedirs(d, exist_ok=True)

        # Initialize capture here so it's available early
        self.capture = cv2.VideoCapture()
        self.frame_count = 0
        self.dangerous_cooldown = 0

        # UI Setup
        sm = ScreenManager()

        # Main Screen UI
        main_screen = MainScreen(name='main')
        main_layout = BoxLayout(orientation='vertical')
        main_screen.camera_view = Image()
        
        # Status labels layout
        status_layout = BoxLayout(size_hint_y=0.1)
        main_screen.status_label = Label(text='Status: Normal')
        main_screen.confidence_label = Label(text='Confidence: ---')
        status_layout.add_widget(main_screen.status_label)
        status_layout.add_widget(main_screen.confidence_label)

        # Button layout for main screen
        main_button_layout = BoxLayout(size_hint_y=0.1)
        review_button = Button(text='Review Images')
        review_button.bind(on_press=lambda x: setattr(sm, 'current', 'review'))
        retrain_button = Button(text='Retrain Model')
        retrain_button.bind(on_press=lambda x: self.retrain_model())
        capture_button = Button(text='Capture')
        capture_button.bind(on_press=main_screen.capture_image)

        main_button_layout.add_widget(review_button)
        main_button_layout.add_widget(retrain_button)
        main_button_layout.add_widget(capture_button)

        main_layout.add_widget(main_screen.camera_view)
        main_layout.add_widget(status_layout)
        main_layout.add_widget(main_button_layout)
        main_screen.add_widget(main_layout)
        sm.add_widget(main_screen)

        # Review Screen UI
        review_screen = ReviewScreen(name='review')
        review_layout = BoxLayout(orientation='vertical')
        review_screen.image_label = Label(size_hint_y=0.1)
        review_screen.review_image = Image(allow_stretch=True)
        review_screen.classify_buttons = BoxLayout(size_hint_y=0.2)
        normal_button = Button(text='Normal')
        normal_button.bind(on_press=lambda x: review_screen.classify_image('normal'))
        dangerous_button = Button(text='Dangerous')
        dangerous_button.bind(on_press=lambda x: review_screen.classify_image('dangerous'))
        review_screen.classify_buttons.add_widget(normal_button)
        review_screen.classify_buttons.add_widget(dangerous_button)
        back_button = Button(text='Back to Main', size_hint_y=0.1)
        back_button.bind(on_press=lambda x: setattr(sm, 'current', 'main'))
        review_layout.add_widget(review_screen.image_label)
        review_layout.add_widget(review_screen.review_image)
        review_layout.add_widget(review_screen.classify_buttons)
        review_layout.add_widget(back_button)
        review_screen.add_widget(review_layout)
        sm.add_widget(review_screen)

        return sm

    def on_start(self):
        self.sound = SoundLoader.load(SOUND_FILE)
        self.model = self.load_model()
        if not self.model:
            # Access the label on the main screen and update it
            main_screen = self.root.get_screen('main')
            main_screen.status_label.text = "Status: ERROR - MODEL NOT LOADED"
            main_screen.confidence_label.text = "Confidence: Check Logs"

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                # Load the Keras model. `custom_objects` is not needed for this simple model.
                model = tf.keras.models.load_model(MODEL_FILE)
                print("Keras model loaded successfully.")
                # You can print model summary for verification
                model.summary()
                return model
            except Exception as e:
                import traceback
                print(f"Failed to load Keras model: {e}")
                traceback.print_exc()
        else:
            print(f"Model file not found at {MODEL_FILE}")
        return None

    def retrain_model(self, *args):
        """
        Launches the external trainer script and closes the app.
        """
        main_screen = self.root.get_screen('main')
        main_screen.status_label.text = "Status: Starting trainer..."
        print("Starting external training script...")

        try:
            # Launch trainer.py as a separate process
            # sys.executable ensures we use the same python interpreter
            subprocess.Popen([sys.executable, "trainer.py"])
            
            # Inform the user and schedule the app to close
            main_screen.status_label.text = "Trainer started. Please restart app when done."
            Clock.schedule_once(self.stop, 5) # Close the app after 5 seconds

        except FileNotFoundError:
            print("Error: trainer.py not found.")
            main_screen.status_label.text = "Error: trainer.py not found."
        except Exception as e:
            print(f"Failed to start trainer: {e}")
            main_screen.status_label.text = "Error: Failed to start trainer."

if __name__ == '__main__':
    MonitoringApp().run()
