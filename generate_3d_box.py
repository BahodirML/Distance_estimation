import open3d as o3d
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

def train_classifier():
    """Mock training data with realistic features for 'person' and 'non-person'."""
    # Features: [width, height, depth]
    human_features = np.random.normal(loc=[0.6, 1.7, 0.5], scale=0.1, size=(50, 3))
    non_human_features = np.random.normal(loc=[2.0, 0.3, 2.0], scale=0.5, size=(50, 3))

    features = np.vstack((human_features, non_human_features))
    labels = np.array([1] * 50 + [0] * 50)  # 1 = person, 0 = non-person

    clf.fit(features, labels)

train_classifier()

def load_point_cloud(file_path):
    """Load the point cloud data from a .pcd file."""
    return o3d.io.read_point_cloud(file_path)

def voxel_downsample(point_cloud, voxel_size=0.05):
    """Downsample the point cloud for efficient processing."""
    return point_cloud.voxel_down_sample(voxel_size)

def segment_objects(point_cloud):
    """Segment objects in the point cloud using DBSCAN clustering."""
    points = np.asarray(point_cloud.points)
    db = DBSCAN(eps=0.5, min_samples=10).fit(points)
    labels = db.labels_

    segmented_objects = []
    for label in set(labels):
        if label == -1:  # Noise
            continue
        object_points = points[labels == label]
        if len(object_points) > 10:  # Filter small clusters
            segmented_objects.append(object_points)
    return segmented_objects

def classify_and_draw_boxes(objects, visualizer):
    """Classify objects, draw bounding boxes, and collect data for export."""
    human_boxes = []
    data_for_export = []
    for obj_points in objects:
        features = extract_features(obj_points)
        
        # Define size constraints for humans with flexible depth
        min_human_height, max_human_height = 0.5, 2.0
        min_human_width, max_human_width = 0.3, 1.5
        max_human_depth = 2.5

        if clf.predict([features]) == 1:  # 1 = 'person'
            width, height, depth = features
            
            if (min_human_width <= width <= max_human_width and
                min_human_height <= height <= max_human_height and
                depth <= max_human_depth):
                
                # Create bounding box and calculate center and distance
                min_bound = np.min(obj_points, axis=0)
                max_bound = np.max(obj_points, axis=0)
                box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                box.color = (1, 0, 0)  # Red for humans
                human_boxes.append(box)
                visualizer.add_geometry(box)
                
                # Calculate center of the bounding box
                center = box.get_center()
                distance = np.linalg.norm(center)  # Distance from origin (0, 0, 0)

                # Collect data for export
                data_for_export.append({
                    "width": width,
                    "height": height,
                    "depth": depth,
                    "center": center.tolist(),
                    "distance": distance
                })
    
    # Export data to JSON file
    export_to_json(data_for_export)
    return human_boxes

def extract_features(object_points):
    """Extract size features for classification."""
    size = np.max(object_points, axis=0) - np.min(object_points, axis=0)
    return size  # Width, Height, Depth

def export_to_json(data):
    """Export bounding box data and distances to a JSON file."""
    with open("bounding_boxes.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Bounding box data exported to bounding_boxes.json")

# Main pipeline
pcd_file = "Factory3/test_lidar/pcd_files_0.pcd"  # Replace with your .pcd file path
point_cloud = load_point_cloud(pcd_file)

# Step 1: Voxel downsample
downsampled_pcd = voxel_downsample(point_cloud)

# Step 2: Segment objects
segmented_objects = segment_objects(downsampled_pcd)

# Step 3: Visualization setup
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(downsampled_pcd)

# Step 4: Classify, draw bounding boxes, and export data
human_boxes = classify_and_draw_boxes(segmented_objects, vis)

# Step 5: Visualization
for box in human_boxes:
    vis.update_geometry(box)
vis.update_geometry(downsampled_pcd)
vis.poll_events()
vis.update_renderer()
vis.run()
vis.destroy_window()
