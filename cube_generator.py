import cv2
import numpy as np
import pyAprilTag

# Camera intrinsic matrix (K)
K = np.array([[958.71843813, 0, 622.4762461], 
              [0, 959.42584804, 362.01675535], 
              [0, 0, 1]])

# Distortion coefficients
distortion_coeffs = np.array([5.63198758e-02, -2.24161921e-01, 1.65855782e-03, 1.24012548e-04, 2.76610743e-01])

# Define the real-world dimensions of the AprilTag (e.g., 1 unit by 1 unit)
tag_size = 1.7  # Tag size in real-world units (e.g., cm)

# Load the image containing the AprilTag
image = cv2.imread('data/tag_1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the tags
ids, corners, centers, Hs = pyAprilTag.find(gray_image)
print("Ids: ", ids)
# print("corners: ", corners)

if len(ids) > 0:
    # Draw the detected tag's boundary
    for tag in range(len(ids)):
        # Define the 3D points of the cube in the real world
        half_size = tag_size / 2
        cube_points_3D = np.array([
            [-half_size, -half_size, 0],   # Bottom-left of the tag (z=0 plane)
            [half_size, -half_size, 0],    # Bottom-right of the tag (z=0 plane)
            [half_size, half_size, 0],     # Top-right of the tag (z=0 plane)
            [-half_size, half_size, 0],    # Top-left of the tag (z=0 plane)
            [-half_size, -half_size, tag_size],  # Bottom-left of the top face
            [half_size, -half_size, tag_size],   # Bottom-right of the top face
            [half_size, half_size, tag_size],    # Top-right of the top face
            [-half_size, half_size, tag_size],   # Top-left of the top face
        ])

        # Get the 2D image points from the AprilTag detection (corners)
        image_points_2D = np.array(corners[tag], dtype=np.float32)

        # Solve for the pose of the tag (rotation and translation vectors)
        ret, rvec, tvec = cv2.solvePnP(cube_points_3D[:4], image_points_2D, K, distortion_coeffs)

        # Project the 3D points of the cube onto the image plane
        cube_points_2D, _ = cv2.projectPoints(cube_points_3D, rvec, tvec, K, distortion_coeffs)

        # Convert to integer points
        cube_points_2D = np.int32(cube_points_2D).reshape(-1, 2)

        # Draw the cube edges on the image
        image = cv2.drawContours(image, [cube_points_2D[:4]], -1, (0, 0, 255), 2)  # Bottom face
        image = cv2.drawContours(image, [cube_points_2D[4:]], -1, (0, 0, 255), 2)  # Top face

        for i in range(4):
            # Draw vertical lines connecting the bottom and top faces
            image = cv2.line(image, tuple(cube_points_2D[i]), tuple(cube_points_2D[i + 4]), (0, 0, 255), 2)

    # Show the result
    cv2.imwrite('result/3D_cube_on_AprilTag.jpg', image)
    cv2.imshow('3D Cube on AprilTag', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No AprilTags detected.")
