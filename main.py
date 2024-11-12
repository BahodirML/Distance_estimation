import os
import cv2
import numpy as np
import open3d as o3d
import json
from pathlib import Path

# Load calibration parameters (using default/rough calibration values)
def load_calibration():
    intrinsic = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    rotation = np.eye(3)
    translation = np.array([0.2, 0, 1.5])
    return intrinsic, rotation, translation

# Load the PCD file using Open3D
def load_pcd_file(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

# Project a 3D point to the 2D image plane
def project_point(point, intrinsic, rotation, translation):
    point_camera = rotation @ point + translation
    if point_camera[2] <= 0:
        return None
    uv = intrinsic @ point_camera[:3]
    u = int(uv[0] / point_camera[2])
    v = int(uv[1] / point_camera[2])
    return (u, v)

# Find the closest point to the bounding box center
def find_closest_point_to_bbox_center(points, bbox_center, intrinsic, rotation, translation):
    center_x, center_y = bbox_center
    closest_distance = float('inf')
    closest_point = None

    for point in points:
        projected_point = project_point(point, intrinsic, rotation, translation)
        if projected_point:
            u, v = projected_point
            distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = point

    closest_distance_3d = np.linalg.norm(closest_point) if closest_point is not None else float('inf')
    return closest_point, closest_distance_3d

# Main function to process frames and labels with PCD files
def process_output_frames_and_labels(frame_dir, label_dir, pcd_dir, output_frame_dir="output_with_distances", output_json_dir="output_distances_json"):
    # Load calibration parameters
    intrinsic, rotation, translation = load_calibration()

    # Create output directories
    Path(output_frame_dir).mkdir(parents=True, exist_ok=True)
    Path(output_json_dir).mkdir(parents=True, exist_ok=True)

    # Load all PCD files and log their names for verification
    pcd_files = sorted(Path(pcd_dir).glob("*.pcd"))
    print(f"Found {len(pcd_files)} PCD files:")
    for pcd_file in pcd_files:
        print(f"PCD file: {pcd_file}")

    # Process each frame and corresponding label
    for idx, label_file in enumerate(Path(label_dir).glob("*.txt")):
        frame_file = Path(frame_dir) / f"{label_file.stem}.jpg"

        # Use sequential matching for PCD files if filenames don’t align
        pcd_file = pcd_files[idx] if idx < len(pcd_files) else None
        if not frame_file.exists():
            print(f"Skipping {label_file.stem}: Frame not found at {frame_file}")
            continue
        if not pcd_file or not pcd_file.exists():
            print(f"Skipping {label_file.stem}: PCD file not found or out of range")
            continue

        print(f"Processing {frame_file} with {label_file} and {pcd_file}...")
        
        # Load frame and PCD file
        image = cv2.imread(str(frame_file))
        points = load_pcd_file(str(pcd_file))
        height, width = image.shape[:2]

        # Open label file
        with open(label_file, 'r') as f:
            labels = f.readlines()
        
        object_data = []  # For JSON output
        
        for label in labels:
            parts = label.strip().split()
            cls, x_center, y_center, box_width, box_height = map(float, parts[:5])

            # Convert normalized coordinates to actual pixel coordinates
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            box_width = int(box_width * width)
            box_height = int(box_height * height)
            x_min = x_center - box_width // 2
            y_min = y_center - box_width // 2
            x_max = x_center + box_width // 2
            y_max = y_center + box_width // 2

            # Draw bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Find the closest point to the bounding box center
            bbox_center = (x_center, y_center)
            closest_point, closest_distance = find_closest_point_to_bbox_center(points, bbox_center, intrinsic, rotation, translation)

            # Add distance text above the bounding box
            if closest_distance < float('inf'):
                distance_text = f"Distance: {closest_distance:.2f} meters"
                print(f"Object class {int(cls)}, distance: {closest_distance:.2f} meters")  # Print to console
                cv2.putText(image, distance_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                distance_text = "Distance: N/A"
                print(f"Object class {int(cls)}, distance not available")
                cv2.putText(image, distance_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Prepare data for JSON output
            object_info = {
                "class_id": int(cls),
                "bounding_box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                "distance": closest_distance if closest_distance < float('inf') else None
            }
            object_data.append(object_info)

        # Save annotated image with distances
        output_image_path = Path(output_frame_dir) / f"{frame_file.stem}_with_distance.jpg"
        cv2.imwrite(str(output_image_path), image)
        print(f"Saved frame with distance: {output_image_path}")

        # Save JSON file with distances
        output_json_path = Path(output_json_dir) / f"{frame_file.stem}_distance_data.json"
        with open(output_json_path, 'w') as json_file:
            json.dump(object_data, json_file, indent=4)
        print(f"Saved JSON with distances: {output_json_path}")

# Example usage
frame_dir = 'output_frames'        # Directory containing output frames from YOLOv detection
label_dir = 'output_labels'        # Directory containing labels from YOLOv detection
pcd_dir = 'test_lidar'             # Directory containing corresponding PCD files
process_output_frames_and_labels(frame_dir, label_dir, pcd_dir)
