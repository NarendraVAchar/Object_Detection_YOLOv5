import os
import webbrowser
import cv2
import torch
from flask import Flask, render_template, Response, jsonify
from pathlib import Path
import threading
import pandas as pd  # Ensure pandas is imported
import openpyxl  # Ensure openpyxl is installed to read Excel files

# Initialize Flask app
app = Flask(__name__)
model_path = "exp3/weights/best.pt"

# Load the YOLO model (ensure this path matches your `best.pt` location)
weights_path = Path(model_path).resolve()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), force_reload=True)

# Load instructions from the Excel file
instructions_df = pd.read_excel('instructions.xlsx')

# Create a dictionary to hold the instructions in the correct format
instructions = {
    row['Instruction Number']: row['Instruction'] 
    for _, row in instructions_df.iterrows()
}

# Also, load associated images (from the 'inst_img' folder with '.png' extension)
instruction_images = {
    row['Instruction Number']: f'inst_img/{row["Instruction Number"]}.png'  # Assuming images are in the 'inst_img' folder
    for _, row in instructions_df.iterrows()
}

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
    "current_instruction": instructions.get(11, "Connect Battery"),
    "current_image": instruction_images.get(11, ""),
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
                state_data["current_image"] = instruction_images.get(step_flow[index + 1], "")
            else:
                state_data["current_instruction"] = "Process Complete"
                state_data["current_image"] = ""

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

@app.route('/get_all_instructions')
def get_all_instructions():
    # Return all instructions from the excel sheet as a list of tuples
    all_instructions = [{"number": num, "instruction": instr} for num, instr in instructions.items()]
    return jsonify(all_instructions)

@app.route('/get_results')
def get_results():
    with state_lock:
        return jsonify({
            "completed_steps": state_data["completed_steps"],
            "missed_steps": state_data["missed_steps"]
        })

@app.route('/get_instruction_image')
def get_instruction_image():
    with state_lock:
        # Return the image associated with the current instruction
        return jsonify({"inst_img": state_data["current_image"]})

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False, host='0.0.0.0', port=5000)
