import cv2
import mediapipe as mp

def detect_facial_landmarks(image_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return

    # Convert to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_mesh.process(rgb_img)
    
    # Draw facial landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            # mp_drawing.draw_landmarks(
            #     image=img,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            
    # print(results)
    
    # Display result
    cv2.imshow('MediaPipe Face Mesh', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Clean up
    face_mesh.close()

# Use the function
detect_facial_landmarks('image.jpg')
