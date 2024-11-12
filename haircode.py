import cv2
import numpy as np
from sklearn.cluster import KMeans
import mediapipe as mp

def detect_hair_color(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the image")
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define hair region points (approximately above the forehead)
    hair_points = []
    for i in [10, 108, 69, 104, 151, 337, 338, 338]:  # Key points around hairline
        landmark = landmarks[i]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        hair_points.append([x, y])
    
    # Create convex hull for hair region
    hair_points = np.array(hair_points)
    cv2.fillConvexPoly(mask, hair_points, 255)
    
    # Extend the hair region upward
    top_point = np.min(hair_points[:, 1])
    mask[0:top_point, :] = 255
    
    # Apply mask to image
    hair_region = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
    
    # Extract non-black pixels
    hair_pixels = hair_region[np.where((hair_region != [0,0,0]).all(axis=2))]
    
    if len(hair_pixels) == 0:
        raise ValueError("No hair region detected")
    
    # Perform k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(hair_pixels)
    
    # Get the dominant color (the cluster center with the most assigned pixels)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_cluster = np.argmax(counts)
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)
    
    return '#%02x%02x%02x' % tuple(dominant_color)

print(detect_hair_color('image.jpg'))