import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as std_trnsf
import face_recognition  
from networks import get_network


# def detect_skin_color(image):
#     # detecting skin pixels
#     min_YCrCb = np.array([0,133,77],np.uint8)
#     max_YCrCb = np.array([235,173,127],np.uint8)

#     imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
#     skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
#     skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

#     cv2.imshow('Skin Detected', skinYCrCb)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # major colourcode
#     image = cv2.cvtColor(skinYCrCb,cv2.COLOR_BGR2RGB)
#     flat_pixels = [tuple(pixel) for line in image for pixel in line]
#     color_counts = Counter(flat_pixels)
#     top_colors = color_counts.most_common(2)
#     if top_colors[0][0] == (0,0,0):
#         return '#%02x%02x%02x' % top_colors[1][0]
#     else:
#         return '#%02x%02x%02x' % top_colors[0][0]


# def detect_iris_color(image):
#     face_landmarks_list = face_recognition.face_landmarks(image)
    
#     if not face_landmarks_list:
#         return "No face detected"
    
#     for face_landmarks in face_landmarks_list:
#         left_eye = np.array(face_landmarks['left_eye'])
#         right_eye = np.array(face_landmarks['right_eye'])
        
#         for eye in [left_eye, right_eye]:
#             # Get eye center and approximate radius
#             x_coords = eye[:, 0]
#             y_coords = eye[:, 1]
            
#             # Calculate center point
#             center_x = int(np.mean(x_coords))
#             center_y = int(np.mean(y_coords))
            
#             # Calculate approximate iris radius (about 1/3 of eye width)
#             eye_width = int(np.max(x_coords) - np.min(x_coords))
#             iris_radius = int(eye_width * 0.15)  # Adjust this factor as needed
            
#             # Create circular mask for iris
#             mask = np.zeros(image.shape[:2], dtype=np.uint8)
#             cv2.circle(mask, (center_x, center_y), iris_radius, 255, -1)
            
#             eye_region = cv2.bitwise_and(image, image, mask=mask)
            
#             rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
#             flat_pixels = [tuple(pixel) for line in rgb for pixel in line]
#             color_counts = Counter(flat_pixels)
#             top_colors = color_counts.most_common(5)
#             if top_colors[0][0] == (0,0,0):
#                 return '#%02x%02x%02x' % top_colors[1][0]
#             else:
#                 return '#%02x%02x%02x' % top_colors[0][0]
    
#     return "Could not determine iris color"


def detect_hair_color(image, checkpoint_path='networks/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained network
    net = get_network('pspnet_resnet101').to(device)
    state = torch.load(checkpoint_path)
    net.load_state_dict(state['weight'])
    net.eval()

    # Image transformations
    transform = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = net(input_tensor)
        pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
        
    mask = pred >= 0.5 
    mask_color = input_image * mask[:, :, np.newaxis]

    pixels = mask_color.reshape(-1, 3)
    pixels = pixels[np.any(pixels > 0, axis=1)]

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    colors = []
    for color in dominant_colors:
        colors.append('#%02x%02x%02x' % tuple(color))
    return colors 


# def hexcodes(image_path):
#     image = cv2.imread(image_path)
#     hex = {
#         'skin':(detect_skin_color(image)),
#         # 'iris':(detect_iris_color(image)),
#         'hair':(detect_hair_color(image))
#     }
#     return hex

# print(hexcodes('image.jpg'))
print(detect_hair_color(cv2.imread('image.jpg')))