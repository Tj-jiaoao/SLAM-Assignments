
# https://blog.csdn.net/qq_46082765/article/details/130856214
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Helper function to draw matches
def draw_matches(img1, kp1, img2, kp2, matches):
    return cv.drawMatches(img1, kp1, img2, kp2, matches, None)

# Example image and depth data loading
def load_images_and_depths(image_paths, depth_paths):
    images = [cv.imread(p) for p in image_paths]
    depths = [cv.imread(p, cv.IMREAD_UNCHANGED).astype(np.uint16) for p in depth_paths]
    depths = [(depth) * 0.001 for depth in depths]  # Adjust depth values
    return images, depths

# Extract 3D points from the first image using depth map
def extract_3d_points(image, depth, K):
    surf = cv.xfeatures2d.SURF_create(400)
    keypoints, descriptors = surf.detectAndCompute(image, None)
    
    points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    depths = np.array([depth[int(pt[1]), int(pt[0])] for pt in points_2d])
    
    # Convert 2D points to 3D points
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    points_3d = np.array([[(pt[0] - cx) * z / fx, (pt[1] - cy) * z / fy, z] for pt, z in zip(points_2d, depths)])
    
    return keypoints, descriptors, points_2d, np.array(points_3d)

# Calculate relative pose using PnP
def calculate_pnp_pose(pts_2d, pts_3d, K):
    _, R_vec, t, inliers = cv.solvePnPRansac(pts_3d, pts_2d, K, None)
    R, _ = cv.Rodrigues(R_vec)
    return R, t

# Main function to compute camera poses and depths using triangulation
def main():
    # Load images and depth data
    image_paths = ['/home/jiaoao/SLAMHW/scanImage/0_resized.jpg',
                   '/home/jiaoao/SLAMHW/scanImage/1_resized.jpg']
    depth_paths = ['/home/jiaoao/SLAMHW/scanDepth/0.png',
                   '/home/jiaoao/SLAMHW/scanDepth/1.png']
    images, depths = load_images_and_depths(image_paths, depth_paths)

    # Camera intrinsic matrix
    # K = np.array([[570.342224, 0, 320.0], 
    #               [0, 570.342224, 240.0], 
    #               [0, 0, 1]])
    K = np.array([[1165.723022 / 2,0,649.094971 / 2],
                 [0,1165.738037 / 2, 484.765015 / 2],
                 [0,0,1]])
    # Extract 3D points from the first image using depth map
    kp1, des1, pts_2d_1, pts_3d_1 = extract_3d_points(images[0], depths[0], K)
    
    # SURF feature extraction for the second image and matching
    surf = cv.xfeatures2d.SURF_create(400)
    kp2, des2 = surf.detectAndCompute(images[1], None)
    
    # FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f'Number of good matches: {len(good_matches)}')
    
    # Draw matches
    matched_img = draw_matches(images[0], kp1, images[1], kp2, good_matches)
    plt.imshow(matched_img[:,:,::-1])
    plt.show()
    
    # Get matched 2D-3D points from the first image and 2D points from the second image
    pts_2d_1_matched = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    pts_3d_1_matched = np.array([pts_3d_1[m.queryIdx] for m in good_matches], dtype=np.float32)
    pts_2d_2_matched = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    
    # Ensure the matched points have the same number of points
    if len(pts_2d_1_matched) != len(pts_3d_1_matched) or len(pts_2d_1_matched) != len(pts_2d_2_matched):
        raise ValueError("The number of 2D points and 3D points must be the same.")
    
    # Calculate pose using PnP
    R, t = calculate_pnp_pose(pts_2d_2_matched, pts_3d_1_matched, K)
    print(f'Rotation matrix:\n{R}')
    print(f'Translation vector:\n{t}')
    
    # Triangulation to find depth in the second image
    projMatr1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projMatr2 = K @ np.hstack((R, t))
    
    # Triangulate points
    points4D_hom = cv.triangulatePoints(projMatr1, projMatr2, pts_2d_1_matched.T, pts_2d_2_matched.T)
    points4D = points4D_hom / points4D_hom[3]
    points4D = points4D[:3].T
    
    # Estimated depths from triangulated points
    depths_estimated = points4D[:, 2]
    # Visualize estimated depths
    for i, pt in enumerate(pts_2d_2_matched):
        color = (0, 0, int(255 * depths_estimated[i] / np.max(depths_estimated)))
        images[1] = cv.circle(images[1], tuple(pt.astype(int)), 5, color, -1)
    
    plt.imshow(images[1][:,:,::-1])
    plt.show()

    # Calculate real depths from the second depth image
    real_depths = np.array([depths[1][int(pt[1]), int(pt[0])] for pt in pts_2d_2_matched])

    # Filter out non-positive depths
    valid_indices = np.where((depths_estimated > 0) & (real_depths > 0))
    depths_estimated = depths_estimated[valid_indices]
    real_depths = real_depths[valid_indices]

    # Calculate error metrics
    abs_error = np.mean(np.abs(real_depths - depths_estimated))
    rmse = np.sqrt(np.mean((real_depths - depths_estimated) ** 2))
    rmse_log = np.sqrt(np.mean((np.log1p(real_depths) - np.log1p(depths_estimated)) ** 2))

    print(f'Absolute Error (Abs): {abs_error}')
    print(f'Root Mean Square Error (RMSE): {rmse}')
    print(f'Root Mean Square Error of Logarithms (RMSE log): {rmse_log}')

if __name__ == "__main__":
    main()
