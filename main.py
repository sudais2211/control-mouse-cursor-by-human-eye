import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye landmarks (e.g., index 33, 133 for right eye)
            right_eye_x = face_landmarks.landmark[33].x
            right_eye_y = face_landmarks.landmark[33].y
            
            # Map to screen size
            screen_width, screen_height = pyautogui.size()
            cursor_x = int(right_eye_x * screen_width)
            cursor_y = int(right_eye_y * screen_height)
            
            # Move the cursor
            pyautogui.moveTo(cursor_x, cursor_y)

    cv2.imshow('Eye Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
