import os
import pandas as pd
from flask import Flask, render_template, Response, jsonify, request
import webbrowser
import cv2
import torch
from pathlib import Path
import time
 
# Initialize Flask app
app = Flask(__name__)
 
# Video loop to record 1 minute of footage in each iteration
video_name_flag=True
exit_flag=False
start_instruction=1
end_instruction=7
 
# Check if the variable exists and is initialized
if 'video_name_flag' not in locals():
    video_name_flag = True
   
# Create the directory to store the videos if it doesn't exist
output_folder = "recorded_videos"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
 
# Load the YOLO model
weights_path = Path('exp10/weights/best.pt').resolve()
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), force_reload=True)
 
# Load class labels and their corresponding instruction numbers
class_labels_df = pd.read_excel("class_labels.xlsx")
instructions_df = pd.read_excel("instructions.xlsx")
 
class_labels = class_labels_df['Class'].tolist()
instruction_map = dict(zip(class_labels_df['Class'], class_labels_df['Instruction Number']))
   
# Initialize video capture
video_capture = cv2.VideoCapture(1)
 
# State variables to track progress
current_instruction_index = 0
completed_instructions = set()
 
# Get the frame width and height for video size
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# Set the video codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
# Function to get the last video count
def get_last_video_count():
    # Get all files in the output folder
    video_files = [f for f in os.listdir(output_folder) if f.endswith('.avi')]
   
    # If the folder is empty, return 0
    if not video_files:
        return 0
   
    # Extract the numbers from filenames and get the max
    video_numbers = []
    for file in video_files:
        # Assuming the format video_<count>_inst_<start>_<end>.avi
        filename_parts = file.split('_')
        try:
            video_numbers.append(int(filename_parts[1]))
        except ValueError:
            continue
   
    if video_numbers:
        return max(video_numbers)
    else:
        return 0
 
# Get the last recorded video count
video_count = get_last_video_count() + 1
 
def generate_frames():
    global current_instruction_index, completed_instructions
    global video_name_flag
    global video_count
    global exit_flag
    global start_instruction
    global end_instruction
    global instruction_number
    global frame_1
       
    while True:
 
        success, frame = video_capture.read()
        if not success:
            break
        frame_1 = frame.copy()
 
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
        print(video_name_flag)
        if video_name_flag == True:
            video_name_flag = False
            # Define video output filename with a unique name
            end_instruction =  current_instruction_index
            video_filename = os.path.join(output_folder, f"video_{video_count}_inst_{start_instruction}_{end_instruction}.avi")
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))
            start_time = time.time()
        else:
            # Read frame from the camera
            #success, frame = video_capture.read()
            #if not success:
             #   print("Error: Failed to capture image.")
              #  break
 
            # Write the frame to the output video file
            out.write(frame_1)
 
            # Display the frame in a window
##            cv2.imshow("Recording", frame)
 
            # Check if the 'q' key is pressed to stop recording
##            if cv2.waitKey(1) & 0xFF == ord('q'):
##                exit_flag=True
##                print("Recording stopped.")
##                break
 
            # Stop recording after 60 seconds (1 minute)
            if time.time() - start_time > 600:
                video_name_flag=True
                # Release the VideoWriter object for the current video
                start_instruction =  current_instruction_index
                out.release()
 
                # Increment the video count for the next video
                video_count += 1
               
           
           
 
 
 
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