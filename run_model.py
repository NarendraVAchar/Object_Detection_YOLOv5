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

# Load class labels and their corresponding instruction numbers
class_labels_df = pd.read_excel("class_labels.xlsx")
instructions_df = pd.read_excel("instructions.xlsx")

class_labels = class_labels_df['Class'].tolist()
instruction_map = dict(zip(class_labels_df['Class'], class_labels_df['Instruction Number']))

# Initialize video capture
video_capture = cv2.VideoCapture(0)

current_prediction = {"class": "None", "instruction": "None", "instruction_number": -1}

def generate_frames():
    global current_prediction
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Perform detection
        results = model(frame)
        detections = results.pandas().xyxy[0]

        # detected_class = "Battery Installed"
        # # Map detected class to instruction number
        # instruction_number = instruction_map.get(detected_class, -1)

        # # Fetch the instruction text
        # instruction_text = "None"
        # if instruction_number > 0:
        #     instruction_text = instructions_df.iloc[instruction_number - 1]['Instruction']

        # # Update the current prediction
        # current_prediction = {
        #     "class": detected_class,
        #     "instruction": instruction_text,
        #     "instruction_number": instruction_number
        # }

        if not detections.empty:
            # Take the first detected class (you can modify this to handle multiple detections)
            detected_class_id = int(detections.iloc[0]['class'])
            detected_class = class_labels[detected_class_id]

            # Map detected class to instruction number
            instruction_number = instruction_map.get(detected_class, -1)

            # Fetch the instruction text
            instruction_text = "None"
            if instruction_number > 0:
                instruction_text = instructions_df.iloc[instruction_number - 1]['Instruction']

            # Update the current prediction
            current_prediction = {
                "class": detected_class,
                "instruction": instruction_text,
                "instruction_number": instruction_number
            }

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    instructions = instructions_df['Instruction'].tolist()
    highlighted_instruction = current_prediction["instruction_number"]  # Use the highlighted instruction from the prediction
    return render_template('index.html', instructions=instructions, highlighted_instruction=highlighted_instruction)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def current_prediction_route():
    return jsonify(current_prediction)

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False, host='0.0.0.0', port=5000)
