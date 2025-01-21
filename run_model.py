import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import webbrowser  # To open the web browser automatically
import cv2
import pandas as pd
from flask import Flask, render_template, Response
import torch
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model (ensure this path matches your `best.pt` location)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='annot_data/weights/best.pt', force_reload=True)

weights_path = Path('annot_data/weights/best.pt').resolve()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), force_reload=True)

# Class labels
class_labels = ['Manifold Installed', 'Empty Panel', 'Battery Installed', 'Battery Not Installed', 'Battery Cushion Installed', 'Battery Cover Installed', 'Screws Installed', 'Screws Not Installed', 'U Clamp Installed']

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam, 1 for external camera

# Function to generate frames for the video feed
def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Perform detection
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Get the detection results as a pandas DataFrame

        for _, row in detections.iterrows():
            x1, y1, x2, y2, confidence, class_id = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class'])

            # Validate class ID
            if class_id < len(class_labels):
                label = f"{class_labels[class_id]} ({confidence * 100:.2f}%)"
                color = (0, 255, 0)  # Green color for bounding box
            else:
                label = "Unknown Class"
                color = (0, 0, 255)  # Red color for unknown classes

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode the frame as JPEG to send over HTTP
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of a multipart response for the web browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    # Automatically open the web browser to the Flask app URL
    webbrowser.open("http://127.0.0.1:5000/")

    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
