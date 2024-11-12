import cv2
import os

def video_to_images(video_path, output_folder, fps=10):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the video's frames per second (FPS)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)  # Calculate interval to match desired FPS
    
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video_capture.read()
        
        if not success:
            break
        
        # Save frames at specified intervals
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1

    # Release the video capture
    video_capture.release()
    print(f"Saved {saved_count} images to '{output_folder}'")

# Example usage
video_to_images("Factory3/실내2-Video-20241101_144655.avi", "C:/Users/bahod/Desktop/Factory/Factory3/frames", fps=10)
