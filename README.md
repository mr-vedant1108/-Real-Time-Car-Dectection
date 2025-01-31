# -Real-Time-Car-Dectection
import tkinter as tk
from tkinter import ttk
import torch
import cv2
import numpy as np
import geopy.distance
 
# YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# define the range of the camera in meter
range_m = 1000

# Camera index
camera_index = 0

# GPS coordinates of your device
device_lat = 18.637526  # Latitude
device_lon = 73.189493  # Longitude

def calculate_distance(pixel_width, focal_length, object_width):
    """Calculates distance to the object based on pixel width."""
    return (object_width * focal_length) / pixel_width

def detect_and_calculate(focal_length, object_width):
    """Detects objects in the video stream and calculates distances."""
    cap = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)  # Run the YOLOv5 model on the frame
        detections = results.pandas().xyxy[0]  # Get the detections as a pandas DataFrame

        for index, row in detections.iterrows():
            # Detect both cars and persons
            if row['name'] in ['car','bottle' , 'person']:  
                x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
                pixel_width = x2 - x1
                distance = calculate_distance(pixel_width, focal_length, object_width)

                # Placeholder GPS coordinates of the object
                object_lat = device_lat
                object_lon = device_lon

                gps_distance = geopy.distance.geodesic((device_lat, device_lon), (object_lat, object_lon)).meters

                # Set color and label based on the object type
                label = f"{row['name'].capitalize()} Distance: {distance:.2f} meters"
                color = (0, 255, 0) if row['name'] == 'car' else (255, 0, 0)

                # Draw bounding box and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"GPS Distance: {gps_distance:.2f} meters", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Bottle,Object and Person Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tkinter GUI
def create_gui():
    """Creates the Tkinter GUI for inputs."""
    root = tk.Tk()
    root.title(" Bottle,Object and Person Detection")

    focal_length_label = ttk.Label(root, text="Focal Length:")
    focal_length_label.grid(row=0, column=0, padx=10, pady=5)
    focal_length_entry = ttk.Entry(root)
    focal_length_entry.grid(row=0, column=1, padx=10, pady=5)

    object_width_label = ttk.Label(root, text="Object Width (in meters):")
    object_width_label.grid(row=1, column=0, padx=10, pady=5)
    object_width_entry = ttk.Entry(root)
    object_width_entry.grid(row=1, column=1, padx=10, pady=5)

    def start_detection():
        focal_length = float(focal_length_entry.get())
        object_width = float(object_width_entry.get())
        detect_and_calculate(focal_length, object_width)

    start_button = ttk.Button(root, text="Start Detection", command=start_detection)
    start_button.grid(row=2, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()

