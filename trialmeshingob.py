import cv2
import torch
import numpy as np
import open3d as o3d
import sys
sys.path.append("MiDaS")
from midas.model_loader import load_model
from torchvision.transforms import Compose

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "MiDaS/weights/dpt_hybrid_384.pt"
model_type = "dpt_hybrid_384"

# Load MiDaS model
depth_model, transform, net_w, net_h = load_model(device, model_path, model_type)

# Open webcam
ip_camera_url = 'http://172.19.152.62:4747/video'  # Replace with your actual IP address
cap = cv2.VideoCapture(ip_camera_url)

# Camera intrinsics (assumed)
intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.set_intrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240)

prev_pcd = None  # Store previous point cloud for ICP
global_pcd = o3d.geometry.PointCloud()  # Global map

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # MiDaS input
    sample = {"image": img_rgb}
    transformed = transform(sample)
    img_input = torch.from_numpy(transformed["image"]).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = depth_model.forward(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()

    # Normalize for display
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # Show RGB + depth
    cv2.imshow("RGB + Depth", np.hstack((frame, depth_color)))

    # Create mesh every 30 frames (adjust if needed)
    if frame_idx % 30 == 0:
        print(f"[INFO] Generating mesh from frame {frame_idx}...")

        # Convert depth to Open3D format
        rgb = o3d.geometry.Image(img_rgb)
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # Scale depth to mm
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb,
            depth=depth_o3d,
            convert_rgb_to_intensity=False,
            depth_trunc=3000.0
        )

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd.estimate_normals()

        # Align with previous frame using ICP
        if prev_pcd is not None:
            print("[INFO] Aligning with previous frame using ICP...")
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd, prev_pcd, max_correspondence_distance=0.02,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            pcd.transform(reg_p2p.transformation)

        # Update global point cloud
        global_pcd += pcd
        prev_pcd = pcd  # Store current frame for next ICP

        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(global_pcd, depth=9)
        mesh.compute_vertex_normals()

        # Visualize updated mesh
        o3d.visualization.draw_geometries([mesh])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
