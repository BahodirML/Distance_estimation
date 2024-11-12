import open3d as o3d

# Load the PCD file
pcd = o3d.io.read_point_cloud("Factory3/test_lidar/pcd_files_0.pcd")

# Set up the visualizer with a black background
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().background_color = [0, 0, 0]  # Set background to black
vis.add_geometry(pcd)

# Run the visualizer
vis.run()
vis.destroy_window()
