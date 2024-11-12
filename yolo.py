import torch
import cv2
import os
from pathlib import Path
from glob import glob
from urllib.request import urlretrieve
import zipfile

# Function to check if input is an image, video, or folder, and extract frames if it's a video
def check_and_extract_frames(input_path, frame_output_dir="frames"):
    if os.path.isdir(input_path):
        image_files = glob(os.path.join(input_path, '*.jpg'))
        print(f"Found {len(image_files)} images in directory {input_path}.")
        return sorted(image_files)
    elif os.path.isfile(input_path):
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in ['.jpg', '.png', '.jpeg']:
            return [input_path]
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            os.makedirs(frame_output_dir, exist_ok=True)
            video_capture = cv2.VideoCapture(input_path)
            frame_count = 0
            frame_files = []
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame_filename = os.path.join(frame_output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_files.append(frame_filename)
                frame_count += 1
            video_capture.release()
            print(f"Extracted {frame_count} frames from the video {input_path}.")
            return frame_files
    else:
        print("Invalid input path.")
        return []

# Download and set up YOLOv5
def download_yolov5():
    if not Path("yolov5").exists():
        print("Downloading YOLOv5 repository...")
        urlretrieve("https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip", "yolov5.zip")
        with zipfile.ZipFile("yolov5.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.rename("yolov5-master", "yolov5")
        os.remove("yolov5.zip")
        print("YOLOv5 downloaded successfully.")
    else:
        print("YOLOv5 is already available.")

# Detect objects in frames, save output images, and labels
def detect_objects_and_save(frames, output_frame_dir="output_frames", output_label_dir="output_labels"):
    download_yolov5()

    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Running YOLOv5 on GPU.")
    else:
        print("Running YOLOv5 on CPU.")
    
    # Create output directories
    Path(output_frame_dir).mkdir(parents=True, exist_ok=True)
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)
    
    # Detect objects and save results
    for frame_path in frames:
        image = cv2.imread(frame_path)
        results = model(image)
        boxes = results.xyxy[0].cpu().numpy()  # Get bounding boxes
        
        # Prepare label output
        frame_name = Path(frame_path).stem
        label_file_path = os.path.join(output_label_dir, f"{frame_name}.txt")
        
        with open(label_file_path, "w") as label_file:
            for box in boxes:
                # YOLO output format: [x_min, y_min, x_max, y_max, confidence, class]
                x_min, y_min, x_max, y_max, conf, cls = box
                cls = int(cls)
                
                # Save label in YOLO format with normalized coordinates
                x_center = (x_min + x_max) / 2 / image.shape[1]
                y_center = (y_min + y_max) / 2 / image.shape[0]
                width = (x_max - x_min) / image.shape[1]
                height = (y_max - y_min) / image.shape[0]
                label_file.write(f"{cls} {x_center} {y_center} {width} {height}\n")
                
                # Draw bounding box on the image
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(image, f"{int(cls)}: {conf:.2f}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Save the image with bounding boxes
        output_image_path = os.path.join(output_frame_dir, f"{frame_name}.jpg")
        cv2.imwrite(output_image_path, image)
        print(f"Processed and saved frame: {output_image_path} with labels at {label_file_path}")

# Example usage
image_path = 'Factory3/frames'  # Directory containing frames or video file
frames = check_and_extract_frames(image_path)  # Extract frames from video or load images
detect_objects_and_save(frames)
