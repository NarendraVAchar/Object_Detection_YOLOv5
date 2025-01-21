import os
import webbrowser
import cv2
import torch
from flask import Flask, render_template, Response, jsonify
from pathlib import Path
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model (ensure this path matches your `best.pt` location)
weights_path = Path('annot_data/weights/best.pt').resolve()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), force_reload=True)

# Class labels
class_labels = [
    'Manifold Installed', 'Empty Panel', 'Battery Installed', 'Battery Not Installed', 'Battery Cushion Installed', 
    'Battery Cover Installed', 'Screws Installed', 'Screws Not Installed', 'U Clamp Installed'
]

# Define the expected flow of steps
step_flow = [
    "Empty Panel", "U Clamp Installed", "Manifold Installed", 
    "Screws Not Installed", "Screws Installed", "Battery Not Installed", 
    "Battery Installed", "Battery Cushion Installed", "Battery Cover Installed"
]

# Shared state for instructions and results
state_data = {
    "current_instruction": "Connect Battery",
    "completed_steps": [],
    "missed_steps": step_flow.copy()
}

# Lock for thread-safe state updates
state_lock = threading.Lock()

def update_state(detected_label):
    with state_lock:
        if detected_label in step_flow:
            # Mark all previous steps as completed
            index = step_flow.index(detected_label)
            completed_steps = step_flow[:index + 1]

            # Add all completed steps to completed_steps and remove them from missed_steps
            for step in completed_steps:
                if step not in state_data["completed_steps"]:
                    state_data["completed_steps"].append(step)
                    if step in state_data["missed_steps"]:
                        state_data["missed_steps"].remove(step)

            # Update current instruction
            if index < len(step_flow) - 1:
                state_data["current_instruction"] = step_flow[index + 1]
            else:
                state_data["current_instruction"] = "Process Complete"

# Initialize video capture
video_capture = cv2.VideoCapture(1)  # Use 0 for default webcam, 1 for external camera

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
            class_id = int(row['class'])
            if class_id < len(class_labels):
                detected_label = class_labels[class_id]

                # Update the state with the detected label
                update_state(detected_label)

                # Draw bounding box and label on the frame
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                confidence = row['confidence']
                label = f"{detected_label} ({confidence * 100:.2f}%)"
                color = (0, 255, 0) if detected_label in state_data["completed_steps"] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_instructions')
def get_instructions():
    with state_lock:
        return jsonify({"instruction": state_data["current_instruction"]})

@app.route('/get_results')
def get_results():
    with state_lock:
        return jsonify({
            "completed_steps": state_data["completed_steps"],
            "missed_steps": state_data["missed_steps"]
        })

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False, host='0.0.0.0', port=5000)
