import cv2
import numpy as np
from collections import Counter
import face_recognition  
from sklearn.cluster import KMeans
import mediapipe as mp


def detect_skin_color(image_path):
    # detecting skin pixels
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)

    image = cv2.imread(image_path)

    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

    # major colourcode
    image = cv2.cvtColor(skinYCrCb,cv2.COLOR_BGR2RGB)
    flat_pixels = [tuple(pixel) for line in image for pixel in line]
    color_counts = Counter(flat_pixels)
    top_colors = color_counts.most_common(2)
    if top_colors[0][0] == (0,0,0):
        return '#%02x%02x%02x' % top_colors[1][0]
    else:
        return '#%02x%02x%02x' % top_colors[0][0]


def detect_iris_color(image_path):
    image = cv2.imread(image_path)
    
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    if not face_landmarks_list:
        return "No face detected"
    
    for face_landmarks in face_landmarks_list:
        left_eye = np.array(face_landmarks['left_eye'])
        right_eye = np.array(face_landmarks['right_eye'])
        
        for eye in [left_eye, right_eye]:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [eye], 255)
            
            eye_region = cv2.bitwise_and(image, image, mask=mask)
            
            rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
            
            flat_pixels = [tuple(pixel) for line in rgb for pixel in line]
            color_counts = Counter(flat_pixels)
            top_colors = color_counts.most_common(5)
            if top_colors[0][0] == (0,0,0):
                return '#%02x%02x%02x' % top_colors[1][0]
            else:
                return '#%02x%02x%02x' % top_colors[0][0]
    
    return "Could not determine iris color"


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


def hexcodes(image_path):
    hex = {
        'skin':(detect_skin_color(image_path)),
        'iris':(detect_iris_color(image_path)),
        'hair':(detect_hair_color(image_path))
    }
    return hex

print(hexcodes('image.jpg'))