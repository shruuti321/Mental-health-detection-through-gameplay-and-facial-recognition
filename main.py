import cv2
import pygetwindow as gw
import pyautogui
import csv
import datetime
from datetime import datetime
import numpy as np
import os
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
import calendar
from openpyxl import Workbook
import warnings
import matplotlib.pyplot as plt
import time

# Set up video writer with H.264 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file_path = 'screen_record_with_timestamp.mp4'

# Get the currently active window
active_window = gw.getActiveWindow()

if active_window is None:
    print('Error: No active window found.')
    exit()

# Get the window size
width, height = active_window.size

# Initialize video writer
out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (width, height))

# Create a directory to store individual frames
frames_directory = 'frames'
os.makedirs(frames_directory, exist_ok=True)

# Open a CSV file for writing timestamps and frame paths
csv_file_path = 'timestamps_and_frames.csv'
csv_writer_frames = csv.writer(open(csv_file_path, 'w', newline=''))
csv_writer_frames.writerow(['Timestamp', 'Frame Path'])

# Function to write CSV header
def write_to_csvhead(file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['timestamp', 'Intensity', 'emotion'])

# Function to write to CSV
def write_to_csv(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(data)

# Load emotion detection model
model = load_model("best_model.h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a new CSV file for emotion detection
file_path_emotion = 'emotion_detection.csv'
write_to_csvhead(file_path_emotion)

cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture the screen of the active window
        screenshot = pyautogui.screenshot(region=(active_window.left, active_window.top, width, height))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Save the frame as an image file
        frame_filename = f'{frames_directory}/{timestamp.replace(":", "_").replace(".", "_")}.png'
        cv2.imwrite(frame_filename, frame)

        # Write the timestamp and frame path to the CSV file
        csv_writer_frames.writerow([timestamp, frame_filename])

        # Write the frame to the video file
        out.write(frame)

        # Process the frame for emotion detection
        ret, test_img = cap.read()
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            data_to_write = [
                (datetime.now(), np.max(predictions[0]), predicted_emotion)
            ]
            write_to_csv(file_path_emotion, data_to_write)

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        if cv2.waitKey(10) == ord('q'):
            break

except KeyboardInterrupt:
    # Release resources on keyboard interrupt
    out.release()
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Recording stopped.")
