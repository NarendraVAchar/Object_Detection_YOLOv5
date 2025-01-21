import cv2
import os

# Define the video path and output directory
video_path = 'dataset_video/Screws_installed.mp4'  # Path to your video file
output_dir = 'dataset/train/Screws_installed'  # Directory to save extracted images

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
frame_interval = 1  # Extract one frame every 30 frames, adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Save every 30th frame
    if frame_count % frame_interval == 0:
        image_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(image_filename, frame)  # Save the frame as an image

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted and saved to {output_dir}")