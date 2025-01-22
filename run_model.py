import os
import pandas as pd
from flask import Flask, render_template, Response, jsonify
import webbrowser
import cv2
import torch
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
weights_path = Path('annot_data/weights/best.pt').resolve()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), force_reload=True)

# Class labels
class_labels = [
    'Manifold Installed', 'Empty Panel', 'Battery Installed', 'Battery Not Installed',
    'Battery Cushion Installed', 'Battery Cover Installed', 'Screws Installed',
    'Screws Not Installed', 'U Clamp Installed', 'M8 x 35 Screw', 
    'M_F Spacer Screw', '1by4 x 1by2 Screw', '1by4 x 1 Screw'
]

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam, 1 for external camera

# Load instructions from Excel
def load_instructions():
    instructions_df = pd.read_excel("instructions.xlsx", header=0)
    instruction_texts = instructions_df['Instruction']
    numbered_instructions = [f"{i + 1}. {instruction}" for i, instruction in enumerate(instruction_texts)]
    return numbered_instructions

# Global variable to store predictions
current_prediction = {"class": "Waiting for Prediction..."}

# Generate video frames
def generate_frames():
    global current_prediction
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Perform detection
        results = model(frame)
        detections = results.pandas().xyxy[0]

        if not detections.empty:
            detected_class_id = int(detections.iloc[0]['class'])
            detected_class = class_labels[detected_class_id] if detected_class_id < len(class_labels) else "Unknown Class"
            current_prediction["class"] = detected_class
        else:
            current_prediction["class"] = "No Detection"

        # Annotate frame
        for _, row in detections.iterrows():
            x1, y1, x2, y2, confidence, class_id = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class'])
            label = f"{class_labels[class_id]} ({confidence * 100:.2f}%)" if class_id < len(class_labels) else "Unknown Class"
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for HTTP streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for homepage
@app.route('/')
def index():
    instructions = load_instructions()
    return render_template('index.html', instructions=instructions)

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to fetch current prediction
@app.route('/current_prediction')
def current_prediction_route():
    return jsonify(current_prediction)

# Run Flask app
if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")  # Open web browser automatically
    app.run(debug=False, host='0.0.0.0', port=5000)
