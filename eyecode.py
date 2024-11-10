import cv2
import numpy as np
from collections import Counter
import face_recognition  

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

result = detect_iris_color("image.jpg")
print(result)