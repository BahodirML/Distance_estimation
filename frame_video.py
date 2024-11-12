import cv2
import os
from pathlib import Path

def create_video_from_frames(frame_dir, output_video_path, frame_rate=10):
    # Get all image files in the directory and sort them by filename
    frames = sorted(Path(frame_dir).glob("*.jpg"))
    
    if not frames:
        print("No frames found in the directory.")
        return

    # Read the first frame to get video dimensions
    sample_frame = cv2.imread(str(frames[0]))
    height, width, _ = sample_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'avc1' if 'mp4v' doesn't work
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    print(f"Creating video at {output_video_path} with frame rate {frame_rate} FPS...")

    # Add each frame to the video
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipping frame {frame_path}: could not read image.")
            continue
        out.write(frame)  # Write the frame to the video

    # Release the VideoWriter
    out.release()
    print(f"Video created successfully at {output_video_path}")

# Usage with 10 FPS
frame_dir = 'output_with_distances'        # Directory containing the annotated frames
output_video_path = 'output_with_distances_video.mp4'  # Path for the output video file
create_video_from_frames(frame_dir, output_video_path, frame_rate=10)
