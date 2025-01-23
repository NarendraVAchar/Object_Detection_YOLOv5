import os
import pandas as pd
from flask import Flask, render_template, Response, jsonify, request
import webbrowser
import cv2
import torch
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
weights_path = Path('exp10/weights/best.pt').resolve()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), force_reload=True)

# Load class labels and their corresponding instruction numbers
class_labels_df = pd.read_excel("class_labels.xlsx")
instructions_df = pd.read_excel("instructions.xlsx")

class_labels = class_labels_df['Class'].tolist()
instruction_map = dict(zip(class_labels_df['Class'], class_labels_df['Instruction Number']))
    
# Initialize video capture
video_capture = cv2.VideoCapture(0)

# State variables to track progress
current_instruction_index = 0
completed_instructions = set()

def generate_frames():
    global current_instruction_index, completed_instructions

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Perform detection
        results = model(frame)
        detections = results.pandas().xyxy[0]

        if not detections.empty:
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

                # Check if detected class matches the current instruction
                detected_class = class_labels[class_id]
                instruction_number = instruction_map.get(detected_class, -1)

                if instruction_number == instructions_df.iloc[current_instruction_index]['Instruction Number']:
                    completed_instructions.add(current_instruction_index)
                    current_instruction_index += 1

                    # Ensure we don't go out of bounds
                    if current_instruction_index >= len(instructions_df):
                        current_instruction_index = len(instructions_df) - 1

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Prepare instructions with highlight status
    instructions = instructions_df['Instruction'].tolist()
    highlighted_instruction = current_instruction_index  # Highlight the current instruction
    return render_template(
        'index.html',
        instructions=instructions,
        highlighted_instruction=highlighted_instruction,
        completed_instructions=completed_instructions
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def current_prediction_route():
    current_instruction_text = "None"
    if 0 <= current_instruction_index < len(instructions_df):
        current_instruction_text = instructions_df.iloc[current_instruction_index]['Instruction']

    return jsonify({
        "current_instruction_index": current_instruction_index,
        "current_instruction": current_instruction_text,
        "completed_instructions": list(completed_instructions),
    })

@app.route('/mark_instruction_done', methods=['POST'])
def mark_instruction_done():
    """
    Mark the current instruction as done and move to the next instruction.
    """
    global current_instruction_index, completed_instructions

    if 0 <= current_instruction_index < len(instructions_df):
        completed_instructions.add(current_instruction_index)
        current_instruction_index += 1

        # Ensure we don't exceed the total number of instructions
        if current_instruction_index >= len(instructions_df):
            current_instruction_index = len(instructions_df) - 1

    return jsonify({"status": "success", "current_instruction_index": current_instruction_index})


if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False, host='0.0.0.0', port=5000)
